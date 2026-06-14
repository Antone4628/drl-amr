"""8-component observation assembly for the agent core (RESTRUCTURE Phase 5).

Pure function of a SolverState snapshot + the per-element error vector. Builds
the observation the policy sees for one element, identical to the vector the
training environment and deployment adapter produced from raw solver state
(parity target, D-038) — but reading topology (levels, neighbors) from the
contract's SolverState rather than poking solver internals.

Observation layout (Architecture Spec §6.2):
    [0] alpha-normalized log-error, current element
    [1] alpha-normalized log-error, left neighbor
    [2] alpha-normalized log-error, right neighbor
    [3] current refinement level / max_level
    [4] left neighbor level / max_level
    [5] right neighbor level / max_level
    [6] resource_usage = n_active / element_budget   (can exceed 1.0)
    [7] round_progress = round_number / max_level
"""
from __future__ import annotations

import numpy as np

from agent.normalization import normalized_error
from contract.element_state import SolverState

OBS_DIM = 8


def build_observation(
    state: SolverState,
    errors: np.ndarray,
    active_idx: int,
    alpha: float,
    element_budget: int,
    round_number: int,
) -> np.ndarray:
    """Construct the 8-component observation vector for one element.

    Args:
        state: Current mesh snapshot (topology, levels, globals). Per-element
            arrays are indexed by active position; position i pairs with
            errors[i].
        errors: Raw per-element error indicators (shape (n_active,)), aligned
            with state's active ordering — i.e. contract.compute_error(...).
        active_idx: Active position of the element being observed.
        alpha: Error tolerance for alpha-normalization (matches training).
        element_budget: Budget for the resource_usage component.
        round_number: Current adaptation round within the remesh interval
            (1..max_level), for round_progress.

    Returns:
        np.ndarray of shape (OBS_DIM,), dtype float32.
    """
    max_level = state.max_level
    e_inf = np.max(errors) if len(errors) > 0 else 0.0

    # Current element normalized error.
    obs_error = normalized_error(errors[active_idx], alpha, e_inf)

    # Neighbor active positions (periodic 1D: always valid; the >= 0 guard is
    # defensive and keeps the assembly correct if a backend ever reports an
    # absent neighbor as -1).
    left_idx = int(state.left[active_idx])
    right_idx = int(state.right[active_idx])

    obs_left_error = (
        normalized_error(errors[left_idx], alpha, e_inf) if left_idx >= 0 else 0.0
    )
    obs_right_error = (
        normalized_error(errors[right_idx], alpha, e_inf) if right_idx >= 0 else 0.0
    )

    # Refinement levels normalized to [0, 1].
    obs_level = state.level[active_idx] / max_level
    obs_left_level = state.level[left_idx] / max_level if left_idx >= 0 else 0.0
    obs_right_level = state.level[right_idx] / max_level if right_idx >= 0 else 0.0

    # Global context scalars. resource_usage can exceed 1.0 after cascades —
    # informative, not an error (Spec §6.4).
    resource_usage = state.n_active / element_budget
    round_progress = round_number / max_level if max_level > 0 else 0.0

    return np.array([
        obs_error,
        obs_left_error,
        obs_right_error,
        obs_level,
        obs_left_level,
        obs_right_level,
        resource_usage,
        round_progress,
    ], dtype=np.float32)