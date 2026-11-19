"""The action mask of the 2D agent core, and the rule a decision follows.

Within a round the mesh is frozen, so the backend's can_coarsen cannot know
what the round has already decided. The round's decision record
(RoundState.decisions) holds that: one entry per position, UNDECIDED or the
mark committed for it. action_mask reads the record; decisions_after writes
the next one.

Actions (the policy's three):
    0 = coarsen, 1 = hold, 2 = refine

The mask (SOLVER_CONTRACT_V2 section 7.1): hold always; refine from the
state's can_refine; coarsen from can_coarsen and the coarsen rule. No
balance check and no budget term.

The coarsen rules, for an element whose sibling family is complete:

    first_decider   The first sibling reached decides for the family. Its
                    coarsen marks every sibling COARSEN; its hold or refine
                    closes coarsen to the others for the round.
    any_coarsen     The 1D rule carried over: any sibling's coarsen marks
                    every sibling COARSEN, unless one has already refined.
                    An earlier hold does not bind and is overwritten.
    unanimous       One vote each: a coarsen marks the element alone, and
                    the backend merges the family only if every sibling
                    voted for it.

A position whose entry is no longer UNDECIDED is skipped (is_decided). Under
the first two rules that is how a family coarsen reaches the siblings the
queue has not come to yet.

This module belongs to the 2D core (AGENT_2D_ROADMAP item 2.3) and stands
beside the sequential core's masking.py, which is not on this path.
"""
from __future__ import annotations

from numbers import Integral

import numpy as np

from agent.round_state import UNDECIDED, RoundState
from contract.solver_contract import COARSEN, HOLD, REFINE
from contract.state import SolverState

# The policy's action indices; the mask is indexed by them. The same values
# as the sequential core's (agent/masking.py), defined again so that this
# module does not import the 1D state.
ACTION_COARSEN = 0
ACTION_HOLD = 1
ACTION_REFINE = 2

# From an action index to the contract's mark.
ACTION_TO_MARK = {ACTION_COARSEN: COARSEN, ACTION_HOLD: HOLD, ACTION_REFINE: REFINE}

COARSEN_RULES = ("first_decider", "any_coarsen", "unanimous")


def _record(state: SolverState, round_state: RoundState) -> np.ndarray:
    """The round's record as an array: all UNDECIDED when none is held yet.

    Raises:
        ValueError: If the record's length is not the state's n_active.
    """
    record = round_state.decisions
    if record is None:
        return np.full(state.n_active, UNDECIDED, dtype=np.int8)
    if record.shape[0] != state.n_active:
        raise ValueError(f"decisions: length {record.shape[0]}, expected n_active = {state.n_active}")
    return record


def _check_position(state: SolverState, position: int) -> None:
    """Reject a position that names no element of the state.

    Raises:
        ValueError: Starting with "position".
    """
    if isinstance(position, bool) or not isinstance(position, Integral) or not 0 <= position < state.n_active:
        raise ValueError(f"position: {position!r}, expected an integer in [0, {state.n_active})")


def is_decided(state: SolverState, round_state: RoundState, position: int) -> bool:
    """Whether the round has already decided this position.

    True once the element's own decision is recorded, or a sibling's family
    coarsen has marked it. A decided position is skipped: it is not asked
    for a mask or an action.

    Raises:
        ValueError: If position names no element, or the record's length is
            not the state's n_active.
    """
    _check_position(state, position)
    return bool(_record(state, round_state)[position] != UNDECIDED)


def _family(state: SolverState, position: int) -> np.ndarray:
    """The positions of an element's siblings, itself included.

    Called only where the state says the element can coarsen, which the
    contract defines as its family row being complete. A row that is not
    would index by -1 and read the last element, so it is rejected.

    Raises:
        ValueError: Starting with "family".
    """
    family = state.family[position]
    if (family < 0).any():
        raise ValueError(f"family: position {position} can coarsen, but its family row is {family.tolist()}")
    return family


def action_mask(state: SolverState, round_state: RoundState, position: int, rule: str) -> np.ndarray:
    """The actions open to one undecided element.

    Hold is always open. Refine is the state's can_refine. Coarsen is the
    state's can_coarsen and the rule's term over the family's entries in the
    round's record: under first_decider, no sibling has decided; under
    any_coarsen, no sibling has refined; under unanimous, nothing more.
    There is no balance check and no budget term.

    Args:
        state: The round's frozen state.
        round_state: Holds the round's decision record.
        position: The element's position in state.
        rule: One of COARSEN_RULES.

    Returns:
        (3,) bool, a new array, indexed by action: coarsen, hold, refine.

    Raises:
        ValueError: If rule is not a coarsen rule, position names no
            element, the record's length is not the state's n_active, the
            position is already decided, or the state reports can_coarsen
            on an incomplete family.
    """
    if rule not in COARSEN_RULES:
        raise ValueError(f"rule: {rule!r}, expected one of {list(COARSEN_RULES)}")
    _check_position(state, position)
    record = _record(state, round_state)
    if record[position] != UNDECIDED:
        raise ValueError(f"position: {position} is already decided this round")

    mask = np.array([False, True, False], dtype=bool)
    mask[ACTION_REFINE] = bool(state.can_refine[position])
    if state.can_coarsen[position]:
        siblings = record[_family(state, position)]
        if rule == "first_decider":
            mask[ACTION_COARSEN] = bool((siblings == UNDECIDED).all())
        elif rule == "any_coarsen":
            mask[ACTION_COARSEN] = not (siblings == REFINE).any()
        else:
            mask[ACTION_COARSEN] = True
    return mask


def decisions_after(
    state: SolverState, round_state: RoundState, position: int, action: int, rule: str
) -> np.ndarray:
    """The round's record after one element's decision.

    The rule's other half: what a decision writes. Hold and refine record
    the element's own mark. Coarsen records COARSEN for every sibling under
    first_decider and any_coarsen, the element's own entry among them, and
    for the element alone under unanimous. Under any_coarsen that overwrites
    a sibling's earlier hold.

    Nothing handed in is changed. The committed count is not this
    function's: a RoundState for the next decision is built by the caller.

    Args:
        state: The round's frozen state.
        round_state: Holds the record before this decision.
        position: The deciding element's position in state.
        action: ACTION_COARSEN, ACTION_HOLD or ACTION_REFINE.
        rule: One of COARSEN_RULES.

    Returns:
        (n_active,) int8, a new read-only array.

    Raises:
        ValueError: As action_mask raises; or if action is not an action
            index, or is closed to this element by the mask.
    """
    mask = action_mask(state, round_state, position, rule)
    if isinstance(action, bool) or not isinstance(action, Integral) or action not in ACTION_TO_MARK:
        raise ValueError(f"action: {action!r}, expected 0 (coarsen), 1 (hold) or 2 (refine)")
    if not mask[action]:
        raise ValueError(f"action: {action} is masked for position {position} under {rule!r}")

    record = _record(state, round_state).copy()
    if action == ACTION_COARSEN and rule != "unanimous":
        record[_family(state, position)] = COARSEN
    else:
        record[position] = ACTION_TO_MARK[int(action)]
    record.flags.writeable = False
    return record