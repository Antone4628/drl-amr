"""The reward of the 2D agent core: what one decision was worth, and what
one interval's mesh was worth.

Both rewards read one number per element, zone_distance
(agent/round_queue.py): how many decades its error lies above the neutral
zone (positive, under-refined) or below it (negative, over-refined), and 0
inside. The thresholds behind that number are the interval's: computed
once, from the first round's cache, and held for every round's queue,
every local reward and the global reward.

The local reward is paid at the commit, from the requested mark, weight
times distance:

                        coarsen        hold    refine
    under-refined       -p_ur * d      0       0
    over-refined        +p_cr * |d|    0       -p_or * |d|
    neutral             0              0       0

A hold is never paid, and a correct refine is not rewarded. The distance
is the element's own, except for a family coarsen, where it is the largest
over the family: the decider answers for siblings it cannot all see, and
is paid once for the family.

The global reward is paid once per interval, after the advance, on the
mesh the rounds left: the mean over its elements of p_ur * d for an
under-refined element that could still refine and p_or * |d| for an
over-refined one above level 0, negated, with d the distance of the
element's largest error over the interval. It is never positive.

The reward is training-only and is not part of the logic implemented
twice. This module belongs to the 2D core (AGENT_2D_ROADMAP item 2.6) and
stands beside the sequential core's reward.py, which is not on this path.
"""
from __future__ import annotations

import math
from numbers import Integral, Real

import numpy as np

from agent.round_mask import ACTION_COARSEN, ACTION_REFINE, ACTION_TO_MARK, COARSEN_RULES
from agent.round_observation import ObservationConfig
from agent.round_queue import zone_distance
from contract.backend import BackendFailure
from contract.state import SolverState


def _check_weights(**weights: float) -> None:
    """Reject a reward weight that is not a finite number, zero or above.

    Raises:
        ValueError: Naming the weight.
    """
    for name, value in weights.items():
        if isinstance(value, bool) or not isinstance(value, Real) or not 0 <= value < math.inf:
            raise ValueError(f"{name}: {value!r}, expected a finite number, not negative")


def local_reward(
    state: SolverState,
    distance: np.ndarray,
    position: int,
    action: int,
    rule: str,
    *,
    p_ur: float,
    p_or: float,
    p_cr: float,
) -> float:
    """The local reward of one element's decision, from the requested mark.

    A family coarsen is a coarsen under first_decider or any_coarsen, where
    one decision marks every sibling: it is paid on the largest distance
    over the family, which is the distance of the family's largest errors.
    Every other decision is paid on the element's own distance: a hold, a
    refine, and under unanimous a coarsen, which is one sibling's vote.

    Nothing here moves within a round, so the reward of a decision can be
    computed before or after its commit. The mask is not checked: commit
    refuses a masked action.

    Args:
        state: The round's frozen state.
        distance: zone_distance of the round's errors under the interval's
            thresholds, (n_active,), by position.
        position: The deciding element's position in state.
        action: ACTION_COARSEN, ACTION_HOLD or ACTION_REFINE.
        rule: One of COARSEN_RULES.
        p_ur: Weight of a coarsen above the zone, a penalty.
        p_or: Weight of a refine below the zone, a penalty.
        p_cr: Weight of a coarsen below the zone, a reward.

    Returns:
        The reward, a Python float: negative for a penalty.

    Raises:
        ValueError: Naming rule, position, action, distance or a weight; or
            starting with "family", for a family coarsen at an element with
            no complete family.
    """
    if rule not in COARSEN_RULES:
        raise ValueError(f"rule: {rule!r}, expected one of {list(COARSEN_RULES)}")
    if isinstance(position, bool) or not isinstance(position, Integral) or not 0 <= position < state.n_active:
        raise ValueError(f"position: {position!r}, expected an integer in [0, {state.n_active})")
    if isinstance(action, bool) or not isinstance(action, Integral) or action not in ACTION_TO_MARK:
        raise ValueError(f"action: {action!r}, expected 0 (coarsen), 1 (hold) or 2 (refine)")
    if not isinstance(distance, np.ndarray) or distance.shape != (state.n_active,):
        raise ValueError(f"distance: expected a numpy array of shape ({state.n_active},)")
    _check_weights(p_ur=p_ur, p_or=p_or, p_cr=p_cr)

    if action == ACTION_COARSEN and rule != "unanimous":
        family = state.family[position]
        if (family < 0).any():
            raise ValueError(f"family: position {position} has no complete family, its row is {family.tolist()}")
        d = float(distance[family].max())
    else:
        d = float(distance[position])

    if d > 0.0 and action == ACTION_COARSEN:
        return float(-p_ur * d)
    if d < 0.0 and action == ACTION_COARSEN:
        return float(p_cr * -d)
    if d < 0.0 and action == ACTION_REFINE:
        return float(-p_or * -d)
    return 0.0


def global_reward(
    errors: np.ndarray,
    level: np.ndarray,
    max_level: int,
    config: ObservationConfig,
    e_max: np.ndarray,
    e_min: np.ndarray,
    *,
    p_ur: float,
    p_or: float,
) -> float:
    """The global reward of one interval: how well the mesh fit the errors.

    A function of arrays alone, with no backend and no report in its
    signature: whoever drives the interval hands in the configured
    indicator's array from IntervalReport.acc and the level array of the
    state the last apply_marks returned, which step_interval leaves in
    place. Row p of errors and entry p of level must be the same element;
    only their lengths are checked.

    An element above the zone is counted only below max_level, where a
    refine was still possible, and one below the zone only above level 0.
    The sum is divided by the element count, so the reward measures the
    average element and a smaller mesh is not better for being smaller.

    Args:
        errors: Each element's largest error over the interval, (n, n_comp)
            float64.
        level: Each element's level after the interval, (n,) integers.
        max_level: The finest level an element may reach.
        config: The normalization rule and its floor.
        e_max: The interval's upper thresholds, (n_comp,).
        e_min: The interval's lower thresholds, (n_comp,).
        p_ur: Weight of an element left above the zone.
        p_or: Weight of an element left below the zone.

    Returns:
        The reward, a Python float, zero or below; 0.0 for no elements.

    Raises:
        BackendFailure: If an error is not finite.
        ValueError: As zone_distance raises; or naming level, max_level or
            a weight.
    """
    if isinstance(errors, np.ndarray) and errors.dtype == np.float64 and not np.isfinite(errors).all():
        raise BackendFailure("errors: not finite")
    distance = zone_distance(errors, config, e_max, e_min)
    n = distance.shape[0]
    if not isinstance(level, np.ndarray) or level.dtype.kind not in "iu" or level.shape != (n,):
        raise ValueError(f"level: expected an integer numpy array of shape ({n},)")
    if isinstance(max_level, bool) or not isinstance(max_level, Integral) or max_level < 0:
        raise ValueError(f"max_level: {max_level!r}, expected an integer, not negative")
    _check_weights(p_ur=p_ur, p_or=p_or)
    if n == 0:
        return 0.0

    under = (distance > 0.0) & (level < max_level)
    over = (distance < 0.0) & (level > 0)
    total = p_ur * float(distance[under].sum()) - p_or * float(distance[over].sum())
    return 0.0 - float(total) / n