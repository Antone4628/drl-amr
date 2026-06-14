"""Priority-magnitude queue construction for the agent core (RESTRUCTURE Phase 5).

Pure function of a SolverState snapshot + the per-element error vector. Orders
the active elements for one adaptation round by distance from the neutral zone
(farthest first), so the highest-impact elements are presented first
(Architecture Spec §5.3):

    priority(k) = log10(e_k / e_max)   if e_k > e_max   (under-refined)
                  log10(e_min / e_k)   if e_k < e_min   (over-refined)
                  0                    if neutral

Sort descending by priority; neutral elements (priority 0) fall to the end.
This is a presentation-efficiency heuristic — it does NOT make decisions. The
queue stores element IDs (not active positions) so it stays valid as mesh
mutations shift the active ordering within a round.
"""
from __future__ import annotations

import numpy as np

from contract.element_state import SolverState


def build_queue(
    state: SolverState, errors: np.ndarray, e_max: float, e_min: float
) -> list[int]:
    """Build the priority-sorted queue of element IDs for one round.

    Args:
        state: Current mesh snapshot. Reads n_active and element_id.
        errors: Raw per-element error indicators (shape (n_active,)), aligned
            with state's active ordering.
        e_max: Upper threshold (alpha * ||e||_inf); above it is under-refined.
        e_min: Lower threshold (e_max ** beta); below it is over-refined.

    Returns:
        Element IDs in descending priority order. Length == n_active.
    """
    n_active = state.n_active
    eps = 1e-30

    # Priority magnitude per element. Both under- and over-refined produce
    # positive priorities; neutral stays 0. eps guards both log(0) and the
    # degenerate-threshold case (e_max/e_min <= eps disables the branches,
    # leaving everything neutral — matches the env at t=0).
    priorities = np.zeros(n_active)
    for i in range(n_active):
        e_k = max(errors[i], eps)
        if e_k > e_max and e_max > eps:
            priorities[i] = np.log10(e_k / e_max)
        elif e_k < e_min and e_min > eps:
            priorities[i] = np.log10(e_min / e_k)
        # else: neutral zone -> priority stays 0.0

    # Descending by priority (argsort is ascending, so negate). Return element
    # IDs for stability across within-round mesh changes.
    sorted_indices = np.argsort(-priorities)
    return [int(state.element_id[i]) for i in sorted_indices]