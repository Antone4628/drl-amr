"""Action masking for the agent core (RESTRUCTURE Phase 5).

Pure function over a SolverState snapshot. MaskablePPO queries this before
action selection to forbid structurally invalid actions; budget is NOT masked
(D-025) — the agent learns conservation through resource_usage + reward.

Action space (gym Discrete(3)):
    0 = coarsen, 1 = do-nothing (hold), 2 = refine

The heavy structural checks now live in the backend's SolverState.can_coarsen
(sibling is an active leaf AND post-coarsen 2:1 balance holds) and .can_refine
(level < max_level) — the old env _can_coarsen conditions 1, 2, 4. The agent
adds only the round-state exclusion the backend can't know: a sibling created by
a balance cascade this round is off-limits (condition 3), so the agent cannot
undo a cascade within the same round.
"""
from __future__ import annotations

import numpy as np

from contract.element_state import SolverState
from contract.solver_contract import COARSEN, HOLD, REFINE

# Action-index vocabulary (the mask is indexed in this space).
ACTION_COARSEN = 0
ACTION_HOLD = 1
ACTION_REFINE = 2

# Bridge from action index to the contract's mark vocabulary (-1/0/+1).
ACTION_TO_MARK = {
    ACTION_COARSEN: COARSEN,
    ACTION_HOLD: HOLD,
    ACTION_REFINE: REFINE,
}


def action_masks(
    state: SolverState, active_idx: int, consumed_elements: set[int]
) -> np.ndarray:
    """Return the boolean mask of valid actions for one element.

    Args:
        state: Current mesh snapshot. Reads can_refine, can_coarsen, sibling,
            element_id at active_idx.
        active_idx: Active position of the element being masked.
        consumed_elements: Element IDs created by balance cascades in the
            current round (the agent's per-round exclusion set).

    Returns:
        np.ndarray of shape (3,), dtype bool, indexed [coarsen, hold, refine].
        Hold is always valid; refine is valid below max_level; coarsen requires
        backend structural eligibility AND a non-cascade sibling.
    """
    mask = np.array([False, True, False], dtype=bool)

    # Refine: structural refinability (level < max_level), backend-owned.
    mask[ACTION_REFINE] = bool(state.can_refine[active_idx])

    # Coarsen: backend structural eligibility AND the round-state exclusion.
    # Only inspect the sibling when the backend reports the element coarsenable
    # — otherwise state.sibling[active_idx] is -1 (no active sibling).
    if state.can_coarsen[active_idx]:
        sibling_pos = int(state.sibling[active_idx])
        sibling_id = int(state.element_id[sibling_pos])
        mask[ACTION_COARSEN] = sibling_id not in consumed_elements

    return mask