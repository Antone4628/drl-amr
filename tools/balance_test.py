"""Balance enforcement exploration for 2:1 mesh constraint.

Diagnostic script to understand how enforce_balance() behaves with
cascading refinement and coarsening near level boundaries. Results
inform the action masking and cascade handling design for the
multiround environment (Phase 2, Tasks 2.3–2.4).

Usage:
    python tests/amr/balance_test.py

Output:
    - Console diagnostics at each step (active list, levels, cascade info)
    - Multi-panel figure showing mesh state at each stage
"""

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from backends.python_1d.amr.adapt import adapt_mesh, adapt_sol, check_balance
from backends.python_1d.amr.forest import get_active_levels
from backends.python_1d.grid.mesh import create_grid_us
from backends.python_1d.solvers.dg_advection_solver_multiround import DGAdvectionSolver


def print_mesh_state(solver, step_name):
    """Print diagnostic info about current mesh state."""
    levels = get_active_levels(solver.active, solver.label_mat)
    balanced = check_balance(solver.active, solver.label_mat)
    print(f"\n{'='*60}")
    print(f"  {step_name}")
    print(f"{'='*60}")
    print(f"  Active elements: {list(solver.active)}")
    print(f"  Levels:          {list(levels)}")
    print(f"  Element count:   {len(solver.active)}")
    print(f"  Balanced:        {balanced}")
    print(f"  xelem:           {np.round(solver.xelem, 4)}")
    return levels, balanced


def diff_active(pre_active, post_active):
    """Identify elements added and removed between two active lists."""
    pre_set = set(pre_active)
    post_set = set(post_active)
    added = post_set - pre_set
    removed = pre_set - post_set
    return added, removed


def refine_element(solver, active_idx):
    """Refine a single element by its index in the active list.
    
    Performs refinement, solution projection, and matrix rebuild.
    Does NOT enforce balance — caller handles that separately so
    we can observe the cascade.
    
    Args:
        solver: DGAdvectionSolver instance.
        active_idx: Index into solver.active (0-based).
    
    Returns:
        marks: The marks array used (for reference).
    """
    marks = np.zeros(len(solver.active), dtype=int)
    marks[active_idx] = 1
    
    pre_coord = solver.coord.copy()
    pre_active = solver.active.copy()
    
    new_grid, new_active, _, new_nelem, npoin_cg, new_npoin_dg = adapt_mesh(
        solver.nop, solver.xelem, solver.active, solver.label_mat,
        solver.info_mat, marks, solver.max_level
    )
    new_coord, new_intma, new_periodicity = create_grid_us(
        solver.ngl, new_nelem, npoin_cg, new_npoin_dg,
        solver.xgl, new_grid
    )
    q_new = adapt_sol(
        solver.q, pre_coord, marks, pre_active, solver.label_mat,
        solver.PS1, solver.PS2, solver.PG1, solver.PG2, solver.ngl
    )
    
    solver.q = q_new
    solver.active = new_active
    solver.nelem = new_nelem
    solver.intma = new_intma
    solver.coord = new_coord
    solver.xelem = new_grid
    solver.npoin_dg = new_npoin_dg
    solver.periodicity = new_periodicity
    solver._update_matrices()
    
    return marks


def plot_mesh_panel(ax, solver, title, highlight_ids=None):
    """Plot mesh state on a single axes panel.
    
    Shows elements as colored rectangles with height = refinement level,
    element IDs labeled, and optional highlighting of specific elements.
    
    Args:
        ax: Matplotlib axes.
        solver: DGAdvectionSolver instance.
        title: Panel title string.
        highlight_ids: Set of element IDs to highlight in orange.
    """
    if highlight_ids is None:
        highlight_ids = set()
    
    levels = get_active_levels(solver.active, solver.label_mat)
    balanced = check_balance(solver.active, solver.label_mat)
    
    for i, elem_id in enumerate(solver.active):
        level = levels[i]
        x_left = solver.info_mat[elem_id - 1][-2]
        x_right = solver.info_mat[elem_id - 1][-1]
        width = x_right - x_left
        
        color = 'orange' if elem_id in highlight_ids else 'lightblue'
        
        rect = mpatches.FancyBboxPatch(
            (x_left, 0), width, max(level, 0.15),
            boxstyle="round,pad=0.01",
            linewidth=1.5, edgecolor='navy', facecolor=color, alpha=0.8
        )
        ax.add_patch(rect)
        
        # Element ID label
        label_y = max(level, 0.15) + 0.1
        ax.text((x_left + x_right) / 2, label_y, str(elem_id),
                ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        # Level label inside rectangle
        if level > 0:
            ax.text((x_left + x_right) / 2, max(level, 0.15) / 2, f'L{level}',
                    ha='center', va='center', fontsize=7, color='navy')
    
    status = "BALANCED" if balanced else "UNBALANCED"
    ax.set_title(f"{title}\n[{len(solver.active)} elements, {status}]", fontsize=10)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-0.1, 4.5)
    ax.set_xlabel('x')
    ax.set_ylabel('Level')
    ax.axhline(y=0, color='black', linewidth=0.5)


def main():
    # =========================================================================
    # Setup: 8 base elements, max_level=3, balance=True
    # =========================================================================
    xelem = np.linspace(-1, 1, 9)  # 8 elements
    solver = DGAdvectionSolver(
        nop=4, xelem=xelem, max_elements=64, max_level=3,
        icase=1, balance=True, verbose=False
    )
    
    # We'll collect snapshots for plotting
    snapshots = []
    
    # =========================================================================
    # Step 0: Base mesh
    # =========================================================================
    print_mesh_state(solver, "Step 0: Base mesh (8 elements, all level 0)")
    snapshots.append(("Step 0: Base mesh", set(), solver.active.copy(),
                       get_active_levels(solver.active, solver.label_mat).copy(),
                       solver.xelem.copy(), solver.info_mat.copy()))
    
    # =========================================================================
    # Step 1: Refine element at index 3 (middle-ish) to level 1
    # No cascade expected — level 1 next to level 0 is fine
    # =========================================================================
    pre_active = solver.active.copy()
    target_idx = 3
    target_id = solver.active[target_idx]
    print(f"\n>>> Refining active index {target_idx} (element ID {target_id})")
    
    refine_element(solver, target_idx)
    levels, balanced = print_mesh_state(solver, "Step 1: After refining one element to level 1")
    added, removed = diff_active(pre_active, solver.active)
    print(f"  Added elements:   {added}")
    print(f"  Removed elements: {removed}")
    
    # Check balance and enforce if needed
    if not balanced:
        print("  >> Balance violated! Enforcing...")
        pre_bal = solver.active.copy()
        solver.balance_mesh(balance=True)
        bal_added, _ = diff_active(pre_bal, solver.active)
        print_mesh_state(solver, "Step 1b: After balance enforcement")
        print(f"  Cascade-created elements: {bal_added}")
        snapshots.append(("Step 1: Refine to L1 + cascade", bal_added,
                           solver.active.copy(),
                           get_active_levels(solver.active, solver.label_mat).copy(),
                           solver.xelem.copy(), solver.info_mat.copy()))
    else:
        snapshots.append(("Step 1: Refine to L1 (no cascade)", added,
                           solver.active.copy(),
                           get_active_levels(solver.active, solver.label_mat).copy(),
                           solver.xelem.copy(), solver.info_mat.copy()))
    
    # =========================================================================
    # Step 2: Refine one of the level-1 children to level 2
    # Find a level-1 element to refine
    # =========================================================================
    levels = get_active_levels(solver.active, solver.label_mat)
    level1_indices = np.where(levels == 1)[0]
    if len(level1_indices) > 0:
        target_idx = level1_indices[0]
        target_id = solver.active[target_idx]
        print(f"\n>>> Refining active index {target_idx} (element ID {target_id}, level 1 -> 2)")
        
        pre_active = solver.active.copy()
        refine_element(solver, target_idx)
        levels, balanced = print_mesh_state(solver, "Step 2: After refining to level 2")
        added, removed = diff_active(pre_active, solver.active)
        print(f"  Added elements:   {added}")
        print(f"  Removed elements: {removed}")
        
        if not balanced:
            print("  >> Balance violated! Enforcing...")
            pre_bal = solver.active.copy()
            solver.balance_mesh(balance=True)
            bal_added, _ = diff_active(pre_bal, solver.active)
            print_mesh_state(solver, "Step 2b: After balance enforcement")
            print(f"  Cascade-created elements: {bal_added}")
            snapshots.append(("Step 2: Refine to L2 + cascade", bal_added,
                               solver.active.copy(),
                               get_active_levels(solver.active, solver.label_mat).copy(),
                               solver.xelem.copy(), solver.info_mat.copy()))
        else:
            snapshots.append(("Step 2: Refine to L2 (no cascade)", added,
                               solver.active.copy(),
                               get_active_levels(solver.active, solver.label_mat).copy(),
                               solver.xelem.copy(), solver.info_mat.copy()))
    
    # =========================================================================
    # Step 3: Refine a level-2 element to level 3
    # This should create a level-3 next to level-1 → cascade expected
    # =========================================================================
    levels = get_active_levels(solver.active, solver.label_mat)
    level2_indices = np.where(levels == 2)[0]
    if len(level2_indices) > 0:
        target_idx = level2_indices[0]
        target_id = solver.active[target_idx]
        print(f"\n>>> Refining active index {target_idx} (element ID {target_id}, level 2 -> 3)")
        
        pre_active = solver.active.copy()
        refine_element(solver, target_idx)
        levels, balanced = print_mesh_state(solver, "Step 3: After refining to level 3 (pre-balance)")
        added, removed = diff_active(pre_active, solver.active)
        print(f"  Added elements:   {added}")
        print(f"  Removed elements: {removed}")
        
        if not balanced:
            print("  >> Balance violated! Enforcing...")
            pre_bal = solver.active.copy()
            solver.balance_mesh(balance=True)
            bal_added, _ = diff_active(pre_bal, solver.active)
            levels_post, _ = print_mesh_state(solver, "Step 3b: After balance enforcement")
            print(f"  Cascade-created elements: {bal_added}")
            snapshots.append(("Step 3: Refine to L3 + cascade", bal_added,
                               solver.active.copy(),
                               get_active_levels(solver.active, solver.label_mat).copy(),
                               solver.xelem.copy(), solver.info_mat.copy()))
        else:
            snapshots.append(("Step 3: Refine to L3 (no cascade)", added,
                               solver.active.copy(),
                               get_active_levels(solver.active, solver.label_mat).copy(),
                               solver.xelem.copy(), solver.info_mat.copy()))
    
    # =========================================================================
    # Step 4: Coarsen test — try coarsening a cascade-created element
    # Find a level-1 element that was created by cascade and coarsen it
    # =========================================================================
    levels = get_active_levels(solver.active, solver.label_mat)
    level1_indices = np.where(levels == 1)[0]
    
    if len(level1_indices) >= 2:
        # Find a sibling pair at level 1
        coarsen_pair = None
        for idx in level1_indices:
            elem = solver.active[idx]
            parent = solver.label_mat[elem - 1][1]
            if parent == 0:
                continue
            # Check if sibling is also active at level 1
            for idx2 in level1_indices:
                if idx2 == idx:
                    continue
                elem2 = solver.active[idx2]
                if solver.label_mat[elem2 - 1][1] == parent:
                    coarsen_pair = (idx, idx2)
                    break
            if coarsen_pair:
                break
        
        if coarsen_pair:
            idx_a, idx_b = coarsen_pair
            elem_a = solver.active[idx_a]
            elem_b = solver.active[idx_b]
            parent = solver.label_mat[elem_a - 1][1]
            print(f"\n>>> Coarsening elements {elem_a} and {elem_b} (parent {parent})")
            
            pre_active = solver.active.copy()
            marks = np.zeros(len(solver.active), dtype=int)
            marks[idx_a] = -1
            marks[idx_b] = -1
            
            pre_coord = solver.coord.copy()
            new_grid, new_active, _, new_nelem, npoin_cg, new_npoin_dg = adapt_mesh(
                solver.nop, solver.xelem, solver.active, solver.label_mat,
                solver.info_mat, marks, solver.max_level
            )
            new_coord, new_intma, new_periodicity = create_grid_us(
                solver.ngl, new_nelem, npoin_cg, new_npoin_dg,
                solver.xgl, new_grid
            )
            q_new = adapt_sol(
                solver.q, pre_coord, marks, pre_active, solver.label_mat,
                solver.PS1, solver.PS2, solver.PG1, solver.PG2, solver.ngl
            )
            
            solver.q = q_new
            solver.active = new_active
            solver.nelem = new_nelem
            solver.intma = new_intma
            solver.coord = new_coord
            solver.xelem = new_grid
            solver.npoin_dg = new_npoin_dg
            solver.periodicity = new_periodicity
            solver._update_matrices()
            
            levels, balanced = print_mesh_state(solver, "Step 4: After coarsening (pre-balance)")
            added, removed = diff_active(pre_active, solver.active)
            print(f"  Added elements:   {added}")
            print(f"  Removed elements: {removed}")
            
            if not balanced:
                print("  >> Balance violated after coarsening! Enforcing...")
                pre_bal = solver.active.copy()
                solver.balance_mesh(balance=True)
                bal_added, _ = diff_active(pre_bal, solver.active)
                print_mesh_state(solver, "Step 4b: After balance enforcement")
                print(f"  Cascade-created elements: {bal_added}")
                snapshots.append(("Step 4: Coarsen + cascade", bal_added,
                                   solver.active.copy(),
                                   get_active_levels(solver.active, solver.label_mat).copy(),
                                   solver.xelem.copy(), solver.info_mat.copy()))
            else:
                snapshots.append(("Step 4: Coarsen (no cascade)", set(),
                                   solver.active.copy(),
                                   get_active_levels(solver.active, solver.label_mat).copy(),
                                   solver.xelem.copy(), solver.info_mat.copy()))
        else:
            print("\n>>> No valid sibling pair found for coarsening test")
    else:
        print("\n>>> Not enough level-1 elements for coarsening test")

    # =========================================================================
    # Run coarsening-induced violation test
    # =========================================================================
    coarsen_test()
    # =========================================================================
    # Run periodic boundary balance test
    # =========================================================================
    periodic_balance_test()
    
    # =========================================================================
    # Plot all snapshots
    # =========================================================================
    n_panels = len(snapshots)
    fig, axes = plt.subplots(n_panels, 1, figsize=(12, 3.5 * n_panels))
    if n_panels == 1:
        axes = [axes]
    
    for ax, (title, highlight_ids, active, levels, _xelem, info_mat) in zip(axes, snapshots, strict=False):
        # Temporarily set solver state to snapshot for plotting
        # (We only need active, info_mat, and label_mat for the plot)
        for i, elem_id in enumerate(active):
            level = levels[i]
            x_left = info_mat[elem_id - 1][-2]
            x_right = info_mat[elem_id - 1][-1]
            width = x_right - x_left
            
            color = 'orange' if elem_id in highlight_ids else 'lightblue'
            
            rect = mpatches.FancyBboxPatch(
                (x_left, 0), width, max(level, 0.15),
                boxstyle="round,pad=0.01",
                linewidth=1.5, edgecolor='navy', facecolor=color, alpha=0.8
            )
            ax.add_patch(rect)
            
            label_y = max(level, 0.15) + 0.1
            ax.text((x_left + x_right) / 2, label_y, str(elem_id),
                    ha='center', va='bottom', fontsize=8, fontweight='bold')
            
            if level > 0:
                ax.text((x_left + x_right) / 2, max(level, 0.15) / 2, f'L{level}',
                        ha='center', va='center', fontsize=7, color='navy')
        
        balanced = np.all(np.abs(np.diff(levels)) <= 1)
        status = "BALANCED" if balanced else "UNBALANCED"
        ax.set_title(f"{title}  [{len(active)} elements, {status}]", fontsize=10)
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-0.1, 4.5)
        ax.set_xlabel('x')
        ax.set_ylabel('Level')
        ax.axhline(y=0, color='black', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig('tools/balance_test_output.png', dpi=150, bbox_inches='tight')
    print("\n\nFigure saved to tools/balance_test_output.png")
    plt.show()

def coarsen_test():
    """Test coarsening-induced balance violations.
    
    Constructs a balanced staircase mesh, then coarsens buffer elements
    to create a 2-level gap. Tests whether enforce_balance detects and
    fixes the violation (and whether it effectively undoes the coarsening).
    """
    print("\n" + "="*60)
    print("  COARSENING-INDUCED BALANCE VIOLATION TEST")
    print("="*60)
    
    # =========================================================================
    # Setup: 8 base elements, max_level=3, balance=True
    # =========================================================================
    xelem = np.linspace(-1, 1, 9)
    solver = DGAdvectionSolver(
        nop=4, xelem=xelem, max_elements=64, max_level=3,
        icase=1, balance=True, verbose=False
    )
    
    snapshots = []
    
    # Step 0: Base mesh
    print_mesh_state(solver, "Coarsen Step 0: Base mesh")
    snapshots.append(("C0: Base mesh", set()))
    
    # Step 1: Refine element 4 → children 15, 16 (L1)
    target_idx = 3  # element 4
    print(f"\n>>> Refining element {solver.active[target_idx]} (L0 -> L1)")
    refine_element(solver, target_idx)
    print_mesh_state(solver, "Coarsen Step 1: Refine elem 4 to L1")
    snapshots.append(("C1: Refine elem 4 → L1", set()))
    
    # Step 2: Refine element 3 → children 13, 14 (L1)
    # This creates a buffer zone of L1 elements next to the L1 pair
    target_idx = 2  # element 3
    print(f"\n>>> Refining element {solver.active[target_idx]} (L0 -> L1, creating buffer)")
    refine_element(solver, target_idx)
    print_mesh_state(solver, "Coarsen Step 2: Refine elem 3 to L1 (buffer)")
    snapshots.append(("C2: Refine elem 3 → L1 (buffer)", set()))
    
    # Step 3: Refine element 15 → children 37, 38 (L2)
    # Find element 15 in active list
    idx_15 = np.where(solver.active == 15)[0]
    if len(idx_15) == 0:
        print("ERROR: Element 15 not found in active list")
        return
    target_idx = idx_15[0]
    print("\n>>> Refining element 15 (L1 -> L2)")
    refine_element(solver, target_idx)
    levels, balanced = print_mesh_state(solver, "Coarsen Step 3: Refine elem 15 to L2")
    
    # Balance if needed (L2 next to L0 on the left side of elem 3's children)
    if not balanced:
        print("  >> Balance violated after L2 refine, enforcing...")
        pre_bal = solver.active.copy()
        solver.balance_mesh(balance=True)
        bal_added, _ = diff_active(pre_bal, solver.active)
        print_mesh_state(solver, "Coarsen Step 3b: After balance enforcement")
        print(f"  Cascade-created elements: {bal_added}")
        snapshots.append(("C3: Refine 15 → L2 + cascade", bal_added))
    else:
        snapshots.append(("C3: Refine 15 → L2 (no cascade)", set()))
    
    # At this point we should have something like:
    # [1, 2, 13, 14, 37, 38, 16, 5, 6, 7, 8]
    # [0, 0,  1,  1,  2,  2,  1, 0, 0, 0, 0]
    # This is balanced: 0-0-1-1-2-2-1-0-0-0-0
    
    print("\n" + "-"*60)
    print("  Now attempting coarsening to induce balance violation")
    print("-"*60)
    
    # Step 4: Coarsen elements 13 and 14 (siblings, parent = 3)
    # This should create L0 next to L2 → violation
    idx_13 = np.where(solver.active == 13)[0]
    idx_14 = np.where(solver.active == 14)[0]
    
    if len(idx_13) == 0 or len(idx_14) == 0:
        print("ERROR: Elements 13 or 14 not found — mesh structure differs from expected")
        print(f"  Active: {list(solver.active)}")
        print(f"  Levels: {list(get_active_levels(solver.active, solver.label_mat))}")
        return
    
    idx_a, idx_b = idx_13[0], idx_14[0]
    print("\n>>> Coarsening elements 13 and 14 (parent = element 3)")
    print("  This should create L0 adjacent to L2 → balance violation")
    
    pre_active = solver.active.copy()
    
    # Manually apply coarsening marks
    marks = np.zeros(len(solver.active), dtype=int)
    marks[idx_a] = -1
    marks[idx_b] = -1
    
    pre_coord = solver.coord.copy()
    new_grid, new_active, _, new_nelem, npoin_cg, new_npoin_dg = adapt_mesh(
        solver.nop, solver.xelem, solver.active, solver.label_mat,
        solver.info_mat, marks, solver.max_level
    )
    new_coord, new_intma, new_periodicity = create_grid_us(
        solver.ngl, new_nelem, npoin_cg, new_npoin_dg,
        solver.xgl, new_grid
    )
    q_new = adapt_sol(
        solver.q, pre_coord, marks, pre_active, solver.label_mat,
        solver.PS1, solver.PS2, solver.PG1, solver.PG2, solver.ngl
    )
    
    solver.q = q_new
    solver.active = new_active
    solver.nelem = new_nelem
    solver.intma = new_intma
    solver.coord = new_coord
    solver.xelem = new_grid
    solver.npoin_dg = new_npoin_dg
    solver.periodicity = new_periodicity
    solver._update_matrices()

    
    levels, balanced = print_mesh_state(solver, "Coarsen Step 4: After coarsening 13+14 (pre-balance)")
    added, removed = diff_active(pre_active, solver.active)
    print(f"  Added elements:   {added}")
    print(f"  Removed elements: {removed}")
    snapshots.append(("C4: Coarsen 13+14 (pre-balance)", set()))
    
    # Step 5: Enforce balance — this should re-refine element 3
    if not balanced:
        print("\n  >> Balance violated after coarsening! Enforcing...")
        pre_bal = solver.active.copy()
        solver.balance_mesh(balance=True)
        bal_added, bal_removed = diff_active(pre_bal, solver.active)
        print_mesh_state(solver, "Coarsen Step 5: After balance enforcement")
        print(f"  Cascade-created elements: {bal_added}")
        print(f"  Cascade-removed elements: {bal_removed}")
        
        # Check if enforcement effectively undid the coarsening
        if 13 in solver.active and 14 in solver.active:
            print("\n  ** FINDING: enforce_balance re-refined element 3 back into 13+14")
            print("  ** The coarsening was effectively UNDONE by balance enforcement")
            print("  ** This confirms: action masking should PREVENT this coarsening")
        elif 3 not in solver.active:
            print("\n  ** FINDING: element 3 was re-refined into different children")
        else:
            print("\n  ** FINDING: unexpected post-balance state")
        
        snapshots.append(("C5: After balance enforcement", bal_added))
    else:
        print("\n  ** UNEXPECTED: Mesh is balanced after coarsening L1 pair next to L2")
        print("  ** Check mesh structure — this shouldn't happen")
        snapshots.append(("C5: Balanced (unexpected)", set()))
    
    # =========================================================================
    # Plot all coarsening test snapshots
    # =========================================================================
    # Rebuild snapshots with full state info for plotting
    # (We need to re-run to capture states, or store them — for simplicity,
    # just re-run the whole sequence storing plot data)
    print("\n\nCoarsening test complete. See console output above for findings.")
    print("The key question was: does coarsening create violations that")
    print("enforce_balance fixes by undoing the coarsening?")
    print("If yes → action masking should prevent such coarsening (D-025).")

def periodic_balance_test():
    """Test that check_balance detects violations at periodic boundaries.
    
    Constructs a mesh where the first and last active elements differ by
    2 refinement levels, which is only a violation if the periodic
    wrap-around is checked. Verifies the fix to check_balance() that
    adds the last-vs-first comparison.
    
    Pre-fix behavior: check_balance uses np.diff(levels) which only
    compares adjacent pairs in the array — misses the wrap-around.
    Post-fix behavior: check_balance also compares levels[-1] vs levels[0].
    """
    print("\n" + "="*60)
    print("  PERIODIC BOUNDARY BALANCE TEST")
    print("="*60)
    
    # =========================================================================
    # Setup: 8 base elements, balance=False so we control everything
    # =========================================================================
    xelem = np.linspace(-1, 1, 9)
    solver = DGAdvectionSolver(
        nop=4, xelem=xelem, max_elements=64, max_level=3,
        icase=1, balance=True, verbose=False
    )
    
    print_mesh_state(solver, "Periodic Step 0: Base mesh (all L0)")
    
    # =========================================================================
    # Step 1: Refine the LAST element (index 7, rightmost)
    # Creates L1 at the right boundary, L0 at the left boundary.
    # Across the periodic wrap: L1 vs L0 → balanced (diff = 1).
    # =========================================================================
    target_idx = len(solver.active) - 1
    target_id = solver.active[target_idx]
    print(f"\n>>> Refining last element (active_idx={target_idx}, elem ID={target_id})")
    refine_element(solver, target_idx)
    levels, balanced = print_mesh_state(solver, "Periodic Step 1: Last element refined to L1")
    print(f"  Periodic boundary: levels[-1]={levels[-1]}, levels[0]={levels[0]}")
    print(f"  check_balance says: {balanced}")
    assert balanced, "ERROR: L1 vs L0 across boundary should be balanced!"
    print("  ✓ Correctly identified as balanced (diff = 1)")
    
    # =========================================================================
    # Step 2: Refine one of the L1 children at the right boundary to L2
    # Now the rightmost active element is L2, leftmost is L0.
    # Across the periodic wrap: L2 vs L0 → UNBALANCED (diff = 2).
    # This is the case that the old check_balance missed.
    # =========================================================================
    levels = get_active_levels(solver.active, solver.label_mat)
    # Find the last L1 element (rightmost in the active list)
    last_l1_idx = None
    for i in range(len(levels) - 1, -1, -1):
        if levels[i] == 1:
            last_l1_idx = i
            break
    
    if last_l1_idx is None:
        print("ERROR: No L1 element found at right boundary")
        return
    
    target_id = solver.active[last_l1_idx]
    print(f"\n>>> Refining rightmost L1 element (active_idx={last_l1_idx}, elem ID={target_id})")
    refine_element(solver, last_l1_idx)
    levels, balanced = print_mesh_state(solver, "Periodic Step 2: Rightmost L1 refined to L2 (pre-balance)")
    
    print("\n  --- CRITICAL CHECK ---")
    print(f"  Periodic boundary: levels[-1]={levels[-1]}, levels[0]={levels[0]}")
    print(f"  Interior diffs:    {list(np.abs(np.diff(levels)))}")
    interior_ok = np.all(np.abs(np.diff(levels)) <= 1)
    print(f"  Interior balanced: {interior_ok}")
    print(f"  check_balance says: {balanced}")
    
    if interior_ok and not balanced:
        print("  ✓ PASS: check_balance correctly catches periodic boundary violation!")
        print("  ✓ Interior is fine but the wrap-around (L2 vs L0) is caught.")
    elif interior_ok and balanced:
        print("  ✗ FAIL: check_balance missed the periodic boundary violation!")
        print("  ✗ The old np.diff-only bug is still present.")
    else:
        print("  Interior violation also present — periodic test inconclusive.")
    
    # =========================================================================
    # Step 3: Enforce balance and verify it fixes the periodic violation
    # =========================================================================
    if not balanced:
        print("\n>>> Enforcing balance to fix periodic violation...")
        pre_bal = solver.active.copy()
        solver.balance_mesh(balance=True)
        bal_added, bal_removed = diff_active(pre_bal, solver.active)
        levels, balanced = print_mesh_state(solver, "Periodic Step 3: After balance enforcement")
        print(f"  Cascade-created elements: {bal_added}")
        print(f"  Periodic boundary: levels[-1]={levels[-1]}, levels[0]={levels[0]}")
        
        if balanced:
            print("  ✓ Balance enforcement fixed the periodic violation")
        else:
            print("  ✗ Balance enforcement did NOT fix the periodic violation")
    
    print("\nPeriodic balance test complete.")

if __name__ == '__main__':
    main()