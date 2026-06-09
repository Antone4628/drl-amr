"""Hierarchical mesh forest structure for Adaptive Mesh Refinement.

This module provides the data structures that track element relationships
(parent/child) and refinement levels in the AMR hierarchy. The 'forest'
represents the complete hierarchy of possible elements across all refinement
levels.

Key Data Structures:
    label_mat: Element family relationships [elem_id, parent_id, child1_id, child2_id, level].
    info_mat: Element metadata [elem_id, parent_id, level, left_coord, right_coord].
    active: Array of currently active (leaf) element IDs.

Key Functions:
    forest: Create the hierarchical mesh structure.
    get_active_levels: Get refinement levels for active elements.

Note:
    Element IDs are 1-indexed. The forest contains ALL possible elements
    up to max_level; the 'active' array tracks which are currently in use.
"""
import numpy as np


def next_level(xelem):
    """Generate next refinement level by inserting midpoints.
    
    Args:
        xelem: Element boundary coordinates, shape (n,).
        
    Returns:
        Refined coordinates with midpoints, shape (2n - 1,).
    """
    m = len(xelem)
    out = np.zeros((2*m-1),dtype=xelem.dtype)
    out[::2] = xelem
    midpoints = (xelem[:-1] + xelem[1:]) / 2
    out[1::2]=midpoints
    return out

def level_arrays(xelem, max_level: int):
    """Generate coordinate arrays for all refinement levels.
    
    Args:
        xelem: Initial element boundary coordinates, shape (n,).
        max_level: Maximum refinement level (0 = no refinement).
        
    Returns:
        List of coordinate arrays, one per level. Level 0 has shape (n,),
        level k has shape (2^k * (n-1) + 1,).
    """
    levels = []
    levels.append(xelem)
    for i in range(max_level):
        next_lev = next_level(levels[i])
        levels.append(next_lev)

    return levels

def stacker(level):
    """Create element boundary pairs from level coordinates.
    
    Args:
        level: Grid coordinates at one refinement level, shape (m,).
        
    Returns:
        Element boundary pairs, shape (m - 1, 2). Each row is [left, right].
    """
    m = int(len(level))
    out = np.zeros(2*m-1)
    out[::2] = level
    out[1::2]= out[2::2]
    stacker = out[:-1].reshape(int(m-1),2)
    return stacker

def vstacker(levels):
    """Stack element boundary pairs from all levels vertically.
    
    Args:
        levels: List of coordinate arrays from level_arrays().
        
    Returns:
        All element boundaries stacked vertically, shape (total_elements, 2).
    """
    stacks=[]
    for level in levels:
        stacks.append(stacker(level))
    vstack = np.vstack(stacks)
    return vstack


def forest(xelem0, max_level: int):
    """Create hierarchical mesh structure for AMR operations.
    
    Builds the complete forest of possible elements across all refinement
    levels, tracking parent/child relationships and coordinates.
    
    Args:
        xelem0: Initial element boundary coordinates, shape (nelem0 + 1,).
        max_level: Maximum allowed refinement level.
        
    Returns:
        label_mat: Element relationships, shape (total_elements, 5).
            Columns: [elem_id, parent_id, child1_id, child2_id, level].
            IDs are 1-indexed; 0 means no parent/child.
        info_mat: Element metadata, shape (total_elements, 5).
            Columns: [elem_id, parent_id, level, left_coord, right_coord].
        active: Currently active element IDs, shape (nelem0,).
            Initially contains base level elements [1, 2, ..., nelem0].
    
    Note:
        Total elements = nelem0 * (2^(max_level+1) - 1).
        The forest contains ALL possible elements; 'active' tracks which
        are currently leaf nodes in the mesh.
    """
    num_levels = max_level + 1
    elems0 = len(xelem0) - 1  # Number of base (level 0) elements
    total_elements = elems0   # Will accumulate total across all levels
    last_parent_idx = 0       # Last element index that can have children (not at max_level)
    
    
    # Calculate total elements and find last element that can have children
    # Level k has elems0 * 2^k elements
    elems = np.zeros(num_levels, dtype=int)
    elems[0] = elems0
    for i in range(num_levels-1):
        a = 2**(i+1) * elems0
        total_elements += a
        last_parent_idx = total_elements - a
    
    # Initialize matrices
    label_mat = np.zeros([total_elements, 5], dtype=int)  # [elem_id, parent_id, child1, child2, level]
    info_mat = np.zeros([total_elements, 3])              # [elem_id, parent_id, level] — coords added later
    
    # Build list of element counts per level for level lookup
    elems_per_level = [elems0]
    for _i in range(max_level):
        elems_per_level.append(elems_per_level[-1] * 2)
    
    # Fill matrices row by row
    # parent_counter tracks position for parent ID calculation (non-base elements)
    parent_counter = 2
    for j in range(total_elements):
        div = elems0
        
        # Set element ID (1-indexed)
        label_mat[j][0] = j + 1
        info_mat[j][0] = j + 1
        

        # Set parent ID
        # Base elements (level 0) have no parent (parent_id = 0)
        # Other elements: parent_id = parent_counter // 2, then increment counter
        if j < div:
            label_mat[j][1] = j//div
            info_mat[j][1] = int(j//div)
        else:
            label_mat[j][1] = (parent_counter)//2
            info_mat[j][1] = int(parent_counter//2)
            parent_counter += 1
            
        # Set children IDs (only for elements not at max_level)
        if j < last_parent_idx:
            label_mat[j][2] = div + (2*j+1)  # Left child
            label_mat[j][3] = div + 2*(j+1)  # Right child
        
        # Calculate and store element's refinement level
        cum_sum = 0
        for lvl, num_elems in enumerate(elems_per_level):
            if j < (cum_sum + num_elems):
                label_mat[j][4] = lvl  # Store level in label_mat's new column
                info_mat[j][2] = lvl   # Keep level in info_mat for compatibility
                break
            cum_sum += num_elems
    
    # Add coordinate information from level_arrays
    levels_arr = level_arrays(xelem0, max_level)
    coord_pairs = vstacker(levels_arr)             # [left, right] for each element
    info_mat = np.hstack((info_mat, coord_pairs))  # Add coordinates to info_mat
    
    # Initialize active grid with base elements
    active_grid = np.arange(1, len(xelem0))
    
    return label_mat, info_mat, active_grid


def get_active_levels(active, label_mat):
    """Get refinement levels for active elements.
    
    Args:
        active: Active element IDs (1-indexed), shape (n_active,).
        label_mat: Element relationships from forest(), shape (total_elements, 5).
        
    Returns:
        Refinement level for each active element, shape (n_active,).
    """
    active_levels = np.zeros(len(active), dtype=int)
    
    for i, elem in enumerate(active):
        active_levels[i] = label_mat[elem-1][4]  # Get level from label_mat
        
    return active_levels

def print_active_levels(active, label_mat):
    """Print active elements and their levels (debug utility)."""
    levels = get_active_levels(active, label_mat)
    print("\nActive elements and their levels:")
    print("Element  Level")
    print("---------------")
    for elem, level in zip(active, levels, strict=False):
        print(f"{elem:7d}  {level:5d}")



def elem_info(elem: int, label_mat):
    """Print element's parent, children, and sibling (debug utility)."""
    parent = label_mat[elem-1][1]
    c1 = label_mat[elem-1][2]
    c2 = label_mat[elem-1][3]
    print(f'\n\n element number {elem} has parent {parent} and children {c1} and {c2}')
    if (parent != 0):
      # find sibling
        if label_mat[elem-2][1] == parent:
            sib = elem-1
        elif label_mat[elem][1] == parent:
            sib = elem+1
        print(f'elemnt {elem} has sibling {sib}')


