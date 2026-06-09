
# =========================================================================
# Public Helper: Neighbor lookup
# =========================================================================

# find_neighbor_index(solver, active_idx, direction)
# Find index of left/right neighbor in solver.active with periodic wrapping.


def find_neighbor_index(solver, active_idx, direction='left'):
    """Find the index of a neighbor element in the active list.

    Uses active-list-based modular wrap for periodic boundary conditions.
    Always succeeds — every active element has both neighbors under periodic BC.

    Args:
        solver: DGAdvectionSolver instance.
        active_idx: Index of the current element in solver.active.
        direction: 'left' or 'right'.

    Returns:
        Index of the neighbor in solver.active.
    """
    n_active = len(solver.active)
    if direction == 'left':
        return (active_idx - 1) % n_active
    else:
        return (active_idx + 1) % n_active