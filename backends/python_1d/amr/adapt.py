"""Mesh adaptation routines for Adaptive Mesh Refinement.

This module provides functions for marking elements for refinement/coarsening,
adapting the mesh structure, and projecting the solution between meshes.

Key Functions:
    mark: Mark elements for refinement or coarsening based on solution.
    adapt_mesh: Perform mesh refinement/coarsening based on marks.
    adapt_sol: Project solution to adapted mesh using scatter/gather.
    check_balance: Verify mesh satisfies 2:1 balance constraint.
    enforce_balance: Iteratively refine to restore 2:1 balance.

Note:
    The 2:1 balance constraint requires that neighboring elements differ
    by at most one refinement level. This prevents excessive jumps in
    resolution across element boundaries.
"""

import numpy as np

from ..grid.mesh import create_grid_us
from .forest import get_active_levels

#~~~~~~~~~~~~~~~~~~~ single refine/coarsen routine ~~~~~~~~~~~~~~~~~~~~~~~~

def mark(active_grid, label_mat, intma, q, criterion, threshold=0.5):
    """Mark elements for refinement or coarsening based on solution criteria.
    
    Examines solution values on each element and marks for adaptation.
    Coarsening only occurs when both siblings qualify.
    
    Args:
        active_grid: Currently active element IDs (1-indexed), shape (n_active,).
        label_mat: Element relationships from forest(), shape (total_elements, 5).
        intma: Element-node connectivity, shape (ngl, n_active).
        q: Solution values at nodes, shape (npoin_dg,).
        criterion: Marking criterion to use:
            - 1: Threshold-based (refine if max(q) >= threshold)
            - 2: Reserved for future use
        threshold: Solution threshold for refinement. Defaults to 0.5.
        
    Returns:
        marks: Adaptation markers, shape (n_active,).
            -1 = coarsen, 0 = no change, 1 = refine.
    
    Note:
        Coarsening requires both siblings to be marked. If only one sibling
        qualifies, neither is coarsened.
    """
    n_active = len(active_grid)
    marks = np.zeros(n_active, dtype=int)
    refs = [] # Track elements marked for refinement
    defs = [] # Track elements marked for coarsening (to avoid double-marking)
    
    # Pre-compute label_mat lookups for all active elements
    parents = label_mat[active_grid - 1, 1]    # Parent IDs (0 = no parent)
    children = label_mat[active_grid - 1, 2:4] # [child1, child2] IDs (0 = no children)
    
    # Process each active element
    for idx, (elem, parent) in enumerate(zip(active_grid, parents, strict=False)):
        # Get solution values on this element
        elem_nodes = intma[:, idx]
        elem_sols = q[elem_nodes]
        max_sol = np.max(elem_sols)
        
        # Check refinement criteria
        if (criterion == 1):
            # Threshold-based marking: refine if max solution >= threshold
            
            # Check refinement: element must have children (not at max level)
            if max_sol >= threshold and children[idx, 0] != 0:
            # if max_sol >= - 0.5 and children[idx, 0] != 0:
                refs.append(elem)
                marks[idx] = 1
                continue
                
            # Check coarsening: element must have a parent (not at base level)
            if max_sol < threshold and parent != 0:
                # Find sibling by checking neighbors with same parent
                sibling = None
                sib_idx = None
                if elem > 1 and label_mat[elem-2, 1] == parent:
                    sibling = elem - 1
                    sib_idx = idx - 1
                elif elem < len(label_mat) and label_mat[elem, 1] == parent:
                    sibling = elem + 1
                    sib_idx = idx + 1
                    
                # Coarsen only if sibling is also active and qualifies
                if sibling in active_grid:
                    sib_nodes = intma[:, sib_idx]
                    sib_sols = q[sib_nodes]
                    
                    # Mark both siblings if sibling also below threshold
                    if np.max(sib_sols) < threshold and sibling not in defs:
                        marks[idx] = marks[sib_idx] = -1
                        defs.extend([elem, sibling])
        

    
    return  marks

# ==================== Single-Element AMR Primitives ====================
#
# These functions perform individual refine/coarsen operations on the
# mesh topology and solution vector. They are the atomic building blocks
# used by:
#   - adapt_mesh() / adapt_sol() for batch operations (marks array)
#   - solver.refine_element() / solver.coarsen_element() for RL actions
#
# Design: stateless free functions. They take arrays, return modified
# arrays. No solver reference, no marks array, no side effects beyond
# the returned values. label_mat is read-only (forest pre-computes all
# possible elements).
# ======================================================================


def refine_single(nop, grid, active, label_mat, info_mat, active_idx, max_level):
    """Refine a single element, splitting it into two children.

    Inserts a midpoint into the grid and replaces the parent element
    with its two children in the active array. The forest structure
    (label_mat, info_mat) already contains the children's IDs and
    coordinates — this function just activates them.

    Args:
        nop: Polynomial order.
        grid: Element boundary coordinates, shape (n_active + 1,).
        active: Active element IDs (1-indexed), shape (n_active,).
        label_mat: Element relationships from forest(), shape (total_elements, 5).
        info_mat: Element metadata from forest(), shape (total_elements, 5).
        active_idx: Index into the active array (0-based) of the element
            to refine.
        max_level: Maximum allowed refinement level.

    Returns:
        Tuple of (new_grid, new_active, success):
            - new_grid: Updated element boundaries, shape (n_active + 2,).
            - new_active: Updated active element IDs, shape (n_active + 1,).
            - success: True if refinement was performed, False if element
              is already at max_level.

    Note:
        After refinement, the two children occupy indices active_idx and
        active_idx + 1 in the new active array. All subsequent indices
        shift by +1. The caller must account for this when iterating.
    """
    elem = active[active_idx]
    level = label_mat[elem - 1][4]

    if level >= max_level:
        return grid, active, False

    # Get children IDs and midpoint from pre-computed forest
    c1, c2 = label_mat[elem - 1][2:4]
    c1_r = info_mat[c1 - 1][4]  # right edge of child 1 = midpoint

    # Insert midpoint into grid (splits parent element)
    grid = np.insert(grid, active_idx + 1, c1_r)

    # Replace parent with two children in active array
    active = np.concatenate([
        active[:active_idx],
        [c1, c2],
        active[active_idx + 1:]
    ])

    return grid, active, True


def coarsen_pair(nop, grid, active, label_mat, left_idx, right_idx):
    """Coarsen a sibling pair, merging two children into their parent.

    Removes the midpoint between two siblings from the grid and replaces
    both children with their parent in the active array.

    Args:
        nop: Polynomial order.
        grid: Element boundary coordinates, shape (n_active + 1,).
        active: Active element IDs (1-indexed), shape (n_active,).
        label_mat: Element relationships from forest(), shape (total_elements, 5).
        left_idx: Index in active array of the left sibling (0-based).
            Must be less than right_idx.
        right_idx: Index in active array of the right sibling (0-based).
            Must equal left_idx + 1 (siblings are adjacent).

    Returns:
        Tuple of (new_grid, new_active, success):
            - new_grid: Updated element boundaries, shape (n_active,).
            - new_active: Updated active element IDs, shape (n_active - 1,).
            - success: True if coarsening was performed, False if elements
              don't share a parent or indices are invalid.

    Note:
        After coarsening, the parent occupies index left_idx in the new
        active array. All subsequent indices shift by -1. The caller must
        account for this when iterating.
    """
    # Validate indices
    if left_idx < 0 or right_idx >= len(active) or right_idx != left_idx + 1:
        return grid, active, False

    left_elem = active[left_idx]
    right_elem = active[right_idx]

    # Validate shared parent
    left_parent = label_mat[left_elem - 1][1]
    right_parent = label_mat[right_elem - 1][1]

    if left_parent != right_parent or left_parent == 0:
        return grid, active, False

    parent = left_parent

    # Remove midpoint from grid (merges two children)
    grid = np.delete(grid, left_idx + 1)

    # Replace both children with parent in active array
    active = np.concatenate([
        active[:left_idx],
        [parent],
        active[left_idx + 2:]
    ])

    return grid, active, True


def project_refine_single(parent_vals, PS1, PS2):
    """Project parent solution onto two children (scatter).

    Uses pre-computed projection matrices to map the parent element's
    polynomial solution to its two children's reference domains.

    Args:
        parent_vals: Solution values on parent element's LGL nodes,
            shape (ngl,).
        PS1: Scatter matrix for left child, shape (ngl, ngl).
        PS2: Scatter matrix for right child, shape (ngl, ngl).

    Returns:
        Tuple of (child1_vals, child2_vals), each shape (ngl,).
    """
    return PS1 @ parent_vals, PS2 @ parent_vals


def project_coarsen_single(child1_vals, child2_vals, PG1, PG2):
    """Project two children solutions onto parent (gather).

    Uses pre-computed projection matrices to combine two sibling
    elements' polynomial solutions into the parent element's solution.

    Args:
        child1_vals: Solution values on left child's LGL nodes, shape (ngl,).
        child2_vals: Solution values on right child's LGL nodes, shape (ngl,).
        PG1: Gather matrix for left child, shape (ngl, ngl).
        PG2: Gather matrix for right child, shape (ngl, ngl).

    Returns:
        Parent solution values at LGL nodes, shape (ngl,).
    """
    return PG1 @ child1_vals + PG2 @ child2_vals

def check_balance(active_grid, label_mat, periodic=True):
    """Check if mesh satisfies 2:1 balance constraint.
    
    A mesh is balanced if neighboring elements differ by at most one
    refinement level.
    
    Args:
        active_grid: Active element IDs (1-indexed), shape (n_active,).
        label_mat: Element relationships from forest(), shape (total_elements, 5).
        periodic: If True, also check last-vs-first element (periodic BCs).
        
    Returns:
        True if mesh is balanced, False otherwise.
    """
    levels = get_active_levels(active_grid, label_mat)
    # Check all adjacent pairs (interior)
    interior_balanced = np.all(np.abs(np.diff(levels)) <= 1)
    # Check periodic wrap-around (last element vs first element)
    if periodic:
        periodic_balanced = abs(int(levels[-1]) - int(levels[0])) <= 1
        return interior_balanced and periodic_balanced
    return interior_balanced


def balance_mark(active, label_mat, periodic=True):
    """Mark elements that need refinement to restore 2:1 balance.
    
    Identifies element pairs where the level difference exceeds 1 and
    marks the coarser element for refinement.
    
    Args:
        active: Active element IDs (1-indexed), shape (n_active,).
        label_mat: Element relationships from forest(), shape (total_elements, 5).
        periodic: If True, check last-vs-first element (periodic BCs).
        
    Returns:
        Refinement markers, shape (n_active,). 1 = needs refinement, 0 = ok.
    
    Note:
        Only marks for refinement, never coarsening. Called iteratively
        by enforce_balance() until mesh is fully balanced.
    """
    n_active = len(active)
    balance_marks = np.zeros(n_active, dtype=int)
    levels = get_active_levels(active, label_mat)

    # For non-periodic: start at 1 to skip wrap-around comparison
    start = 0 if periodic else 1
    for e in range(start, n_active):
        # Compare this element's level to left neighbor (e-1 wraps for periodic)
        elem_level = levels[e]
        left_level = levels[e-1]
        level_difference = abs(elem_level - left_level)

        if level_difference > 1:
            if elem_level > left_level:
                balance_marks[e-1] = 1
            elif elem_level < left_level:
                balance_marks[e] = 1

    return balance_marks


def enforce_balance(
    active, label_mat, grid, info_mat, nop, coord, PS1, PS2, PG1, PG2,
    ngl, xgl, qp, max_level, periodic=True,
):
    """Iteratively enforce 2:1 balance constraint on the mesh.
    
    Repeatedly marks and refines elements until no neighboring elements
    differ by more than one refinement level.
    
    Args:
        active: Active element IDs (1-indexed), shape (n_active,).
        label_mat: Element relationships from forest(), shape (total_elements, 5).
        grid: Element boundary coordinates, shape (n_active + 1,).
        info_mat: Element metadata from forest(), shape (total_elements, 5).
        nop: Polynomial order.
        coord: Node coordinates, shape (npoin_dg,).
        PS1, PS2: Scatter matrices for refinement projection, shape (ngl, ngl).
        PG1, PG2: Gather matrices for coarsening projection, shape (ngl, ngl).
        ngl: Number of LGL nodes per element (nop + 1).
        xgl: LGL node positions in reference element, shape (ngl,).
        qp: Solution values, shape (npoin_dg,).
        max_level: Maximum allowed refinement level.
        
    Returns:
        bal_q: Balanced solution, shape (new_npoin_dg,).
        bal_active: Balanced active elements, shape (new_n_active,).
        bal_nelem: New element count.
        bal_intma: New connectivity, shape (ngl, new_n_active).
        bal_coord: New node coordinates, shape (new_npoin_dg,).
        bal_grid: New element boundaries, shape (new_n_active + 1,).
        bal_npoin_dg: New DG node count.
        bal_periodicity: New periodicity array.
    
    Note:
        May require multiple iterations (up to max_level) to fully balance.
    """
    bal_ctr = 0
    while (bal_ctr <= max_level):
        if check_balance(active, label_mat, periodic=periodic):
            bal_ctr = max_level + 1

        else:


            bal_marks = balance_mark(active, label_mat, periodic=periodic)
            pre_active = active  
            pre_coord = coord

            bal_grid, bal_active, ref_marks, bal_nelem, npoin_cg, bal_npoin_dg = adapt_mesh(
                nop, grid, active, label_mat, info_mat, bal_marks, max_level
            )
            bal_coord, bal_intma, bal_periodicity = create_grid_us(
                ngl, bal_nelem, npoin_cg, bal_npoin_dg, xgl, bal_grid
            )
            bal_q = adapt_sol(qp, pre_coord, bal_marks, pre_active, label_mat, PS1, PS2, PG1, PG2, ngl)

            # Update for next level
            qp = bal_q
            active = bal_active
            coord = bal_coord
            grid = bal_grid

            bal_ctr += 1

    return bal_q, bal_active, bal_nelem, bal_intma, bal_coord, bal_grid, bal_npoin_dg, bal_periodicity


def adapt_mesh(nop, cur_grid, active, label_mat, info_mat, marks, max_level):
    """Perform mesh adaptation based on refinement/coarsening markers.
    
    Processes marks sequentially: refining elements (replacing parent with
    two children) or coarsening element pairs (replacing siblings with parent).
    
    Delegates actual mesh operations to refine_single() and coarsen_pair()
    primitives. This function handles marks-array iteration and bookkeeping.
    
    Args:
        nop: Polynomial order.
        cur_grid: Current element boundaries, shape (n_active + 1,).
        active: Current active element IDs (1-indexed), shape (n_active,).
        label_mat: Element relationships from forest(), shape (total_elements, 5).
        info_mat: Element metadata from forest(), shape (total_elements, 5).
        marks: Adaptation markers, shape (n_active,).
            -1 = coarsen, 0 = no change, 1 = refine.
        max_level: Maximum allowed refinement level.
        
    Returns:
        cur_grid: Updated element boundaries, shape (new_n_active + 1,).
        active: Updated active element IDs, shape (new_n_active,).
        marks: Updated markers (all zeros after processing).
        new_nelem: New element count.
        new_npoin_cg: New CG node count.
        new_npoin_dg: New DG node count.
    
    Note:
        Elements at max_level cannot be refined further (warning printed).
        Coarsening requires both siblings to be marked.
    """
    ngl = nop + 1
    
    # Early exit if no adaptation needed
    if not np.any(marks):
        new_nelem = len(active)
        return (cur_grid, active, marks,  new_nelem, 
                nop * new_nelem + 1, ngl * new_nelem)
    
    # Process marks sequentially (indices shift as we modify arrays)
    i = 0
    while i < len(marks):
        if marks[i] == 0:
            # No change — skip to next element
            i += 1
            continue
            
        if marks[i] > 0:
            # === REFINEMENT via primitive ===
            cur_grid, active, success = refine_single(
                nop, cur_grid, active, label_mat, info_mat, i, max_level
            )
            
            if success:
                # Update marks: replace one mark with two zeros (children)
                marks = np.concatenate([marks[:i], [0, 0], marks[i+1:]])
                # Skip past both children
                i += 2
            else:
                marks[i] = 0
                i += 1
            
        elif marks[i] < 0:  
            # === COARSENING ===
            # Find valid sibling pair: both must be marked for coarsening
            # and share the same parent in label_mat.
            elem = active[i]
            parent = label_mat[elem-1][1]
            
            # Check left neighbor: same parent AND marked for coarsening
            if label_mat[elem-2][1] == parent and i > 0 and marks[i-1] < 0:
                left_idx = i - 1
                right_idx = i
            # Check right neighbor: same parent AND marked for coarsening
            elif i + 1 < len(marks) and label_mat[elem][1] == parent and marks[i+1] < 0:
                left_idx = i
                right_idx = i + 1
            else:
                # No valid sibling pair — skip
                i += 1
                continue
            
            # Coarsen via primitive
            cur_grid, active, success = coarsen_pair(
                nop, cur_grid, active, label_mat, left_idx, right_idx
            )
            
            if success:
                # Update marks: replace two marks with one zero (parent)
                marks = np.concatenate([
                    marks[:left_idx], [0], marks[left_idx+2:]
                ])
                # Continue from merged element position
                i = left_idx + 1
            else:
                i += 1
    
    # Calculate new mesh dimensions
    new_nelem = len(active)
    new_npoin_cg = nop * new_nelem + 1
    new_npoin_dg = ngl * new_nelem
    
    return cur_grid, active, marks, new_nelem, new_npoin_cg, new_npoin_dg

def adapt_sol(q, coord, marks, active, label_mat, PS1, PS2, PG1, PG2, ngl):
    """Project solution onto adapted mesh using scatter/gather operations.
    
    Delegates projection operations to project_refine_single() and
    project_coarsen_single() primitives. This function handles the
    marks-based iteration and solution array assembly.
    
    Args:
        q: Current solution values, shape (npoin_dg,).
        coord: Current node coordinates (unused, kept for API compatibility).
        marks: Adaptation markers, shape (n_active,).
            -1 = coarsen, 0 = no change, 1 = refine.
        active: Active element IDs before adaptation (1-indexed), shape (n_active,).
        label_mat: Element relationships from forest(), shape (total_elements, 5).
        PS1, PS2: Scatter matrices for children 1 and 2, shape (ngl, ngl).
        PG1, PG2: Gather matrices for children 1 and 2, shape (ngl, ngl).
        ngl: Number of LGL nodes per element.
        
    Returns:
        Adapted solution values, shape (new_npoin_dg,).
    
    Note:
        The `active` array must correspond to the pre-adaptation mesh
        (same ordering as `marks`), not the post-adaptation mesh.
    """
    
    new_q = []
    
    i = 0
    while i < len(marks):
        
        if marks[i] == 0:
            # === NO CHANGE ===
            # Copy solution values directly
            elem_vals = q[i*ngl:(i+1)*ngl]
            new_q.extend(elem_vals)
            i += 1
            
        elif marks[i] == 1:
            # === REFINEMENT ===
            # Scatter parent solution to two children via primitive
            parent_vals = q[i*ngl:(i+1)*ngl]
            child1_vals, child2_vals = project_refine_single(
                parent_vals, PS1, PS2
            )
            new_q.extend(child1_vals)
            new_q.extend(child2_vals)
            i += 1
            
        else:  # marks[i] == -1
            # === COARSENING ===
            # Gather two children solutions to parent via primitive

            # Verify we have a pair of coarsening marks
            if i + 1 < len(marks) and marks[i+1] == -1:
                child1_vals = q[i*ngl:(i+1)*ngl]
                child2_vals = q[(i+1)*ngl:(i+2)*ngl]
                parent_vals = project_coarsen_single(
                    child1_vals, child2_vals, PG1, PG2
                )
                new_q.extend(parent_vals)
                i += 2  # Skip both children
            else:
                # Unpaired coarsening mark — copy as-is (shouldn't happen)
                elem_vals = q[i*ngl:(i+1)*ngl]
                new_q.extend(elem_vals)
                i += 1
    
    result = np.array(new_q)
    return result





