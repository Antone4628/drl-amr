"""The commit of the 2D agent core: one decision, recorded and counted.

Within a round two things move from one decision to the next: the round's
decision record and the committed element count. commit advances both
together, so no RoundState carries a new record beside a stale count.
start_round makes the RoundState a round begins with.

The committed count (SOLVER_CONTRACT_V2 section 7.1) starts at n_active and
moves once per family decision, never once per mark:

    refine                               + (2**ndim - 1)
    a coarsen that completes a family    - (2**ndim - 1)
    anything else                        0

A coarsen completes a family when, in the record after it, every sibling's
entry is COARSEN. Under first_decider and any_coarsen every coarsen does,
because it marks the whole family. Under unanimous only the last sibling's
vote does: earlier votes, and votes in a family that never becomes
unanimous, cost nothing, as on the mesh.

The count is exact for the requested marks and not for the realized mesh: a
refine that balance adds, or a family coarsen that balance undoes, leaves
the mesh above the count.

This module belongs to the 2D core (AGENT_2D_ROADMAP item 2.4) and stands
beside the sequential core's counter in core.py, which is not on this path.
"""
from __future__ import annotations

import dataclasses

import numpy as np

from agent.round_mask import ACTION_COARSEN, ACTION_REFINE, decisions_after
from agent.round_state import UNDECIDED, RoundState
from contract.solver_contract import COARSEN
from contract.state import SolverState


def start_round(state: SolverState, round_index: int, n_rounds: int) -> RoundState:
    """The RoundState a round begins with.

    The committed count is the state's n_active, and the record is a real
    array with every position UNDECIDED, so a driver that starts here and
    moves on with commit never holds a RoundState without a record.

    Args:
        state: The round's frozen state.
        round_index: The round within the remesh interval, 1 to n_rounds.
        n_rounds: Rounds per remesh interval.

    Returns:
        A new RoundState. Its record is (n_active,) int8, read-only.

    Raises:
        ValueError: As RoundState raises, for a round index or a round
            count that is not an integer or is out of range.
    """
    return RoundState(
        round_index=round_index,
        n_rounds=n_rounds,
        committed=state.n_active,
        decisions=np.full(state.n_active, UNDECIDED, dtype=np.int8),
    )


def commit(state: SolverState, round_state: RoundState, position: int, action: int, rule: str) -> RoundState:
    """The RoundState after one element's decision: recorded and counted.

    The record is decisions_after's. The committed count moves by
    2**ndim - 1: up for a refine, and down for a coarsen that leaves every
    sibling's entry COARSEN in the new record. The deciding position was
    undecided before, so that is the decision that completed the family.
    Under first_decider and any_coarsen every coarsen completes it; under
    unanimous only the last sibling's vote does. A hold, and a vote that
    completes nothing, leave the count where it was.

    Nothing handed in is changed, and the count handed in is not checked
    against the record.

    Args:
        state: The round's frozen state.
        round_state: The round before this decision. Its record may be
            None, a round's start.
        position: The deciding element's position in state.
        action: ACTION_COARSEN, ACTION_HOLD or ACTION_REFINE.
        rule: One of COARSEN_RULES.

    Returns:
        A new RoundState with the same round index and round count. Its
        record is always an array.

    Raises:
        ValueError: As decisions_after raises. Nothing is counted for a
            decision that is refused.
    """
    record = decisions_after(state, round_state, position, action, rule)
    arity = 2**state.ndim - 1
    change = 0
    if action == ACTION_REFINE:
        change = arity
    elif action == ACTION_COARSEN and (record[state.family[position]] == COARSEN).all():
        change = -arity
    return dataclasses.replace(round_state, committed=int(round_state.committed) + change, decisions=record)