"""The queue of the 2D agent core: the order a round's positions are asked in.

Within a round every element is decided against one frozen read, the
RoundCache, so the order is built once per round and never re-resolved. It
is a presentation order and decides nothing: elements farthest from the
neutral zone are asked first (priority magnitude, as the sequential core's
queue.py), and under the first_decider coarsen rule that makes each
family's decider its member farthest from the zone.

Thresholds, one pair per column of the indicator array, by the
observation's normalization rule:

    dynamo      e_max = alpha * e_inf, e_min = e_max ** beta: the 1D rule,
                agent.normalization.alpha_thresholds. One column only.
    relative    e_max = alpha * e_inf, e_min = alpha ** beta * e_inf:
                unchanged in effect when every error is multiplied by one
                constant, and equal to dynamo's when e_inf is 1.

e_inf is the column's largest error in the cache.

This module belongs to the 2D core (AGENT_2D_ROADMAP item 2.5) and stands
beside the sequential core's queue.py, which is not on this path.
"""
from __future__ import annotations

import math
from numbers import Integral, Real

import numpy as np

from agent.normalization import alpha_thresholds
from agent.round_cache import RoundCache
from agent.round_mask import is_decided
from agent.round_observation import ObservationConfig
from agent.round_state import RoundState
from contract.state import SolverState


def _errors(cache: RoundCache, config: ObservationConfig) -> np.ndarray:
    """The configured indicator's array, checked as the observation checks it.

    Raises:
        ValueError: If the configured indicator is not in the cache, the
            reductions differ, or the rule is dynamo and the cache has more
            than one column per indicator.
    """
    if config.indicator not in cache.errors:
        raise ValueError(f"indicator: {config.indicator!r} is not in the cache, which holds {list(cache.indicators)}")
    if config.reduction != cache.reduction:
        raise ValueError(
            f"reduction: the cache was read under {cache.reduction!r}, the configuration says {config.reduction!r}"
        )
    if cache.n_comp > 1 and config.normalization == "dynamo":
        raise ValueError(
            f"normalization: 'dynamo' takes one column per indicator, and the cache has n_comp = {cache.n_comp}"
        )
    return cache.errors[config.indicator]


def round_thresholds(cache: RoundCache, config: ObservationConfig, beta: float) -> tuple[np.ndarray, np.ndarray]:
    """The classification thresholds of one cache, one pair per column.

    An error above its column's e_max is under-refined, one below e_min is
    over-refined, and between them is the neutral zone. Under relative the
    zone is (beta - 1) * -log10(alpha) decades wide in every column.

    Whether a round uses the thresholds of its own cache or those of the
    interval's first round is the caller's choice: round_priority and
    round_order take them as arguments.

    Args:
        cache: The round's frozen state and indicator arrays.
        config: The indicator, alpha and the normalization rule.
        beta: The hysteresis exponent, above 1.

    Returns:
        e_max and e_min, each (n_comp,) float64, new read-only arrays. Both
        are 0 in a column whose errors are all zero.

    Raises:
        ValueError: As the observation raises for the cache and the
            configuration; or starting with "beta".
    """
    errors = _errors(cache, config)
    if isinstance(beta, bool) or not isinstance(beta, Real) or not 1 < beta < math.inf:
        raise ValueError(f"beta: {beta!r}, expected a finite number above 1")
    if config.normalization == "dynamo":
        pair = alpha_thresholds(errors[:, 0], config.alpha, beta)
        e_max, e_min = (np.array([value], dtype=np.float64) for value in pair)
    else:
        e_inf = errors.max(axis=0)
        e_max = config.alpha * e_inf
        e_min = config.alpha**beta * e_inf
    e_max.flags.writeable = False
    e_min.flags.writeable = False
    return e_max, e_min


def zone_distance(
    errors: np.ndarray, config: ObservationConfig, e_max: np.ndarray, e_min: np.ndarray
) -> np.ndarray:
    """Each row's signed distance from the neutral zone, in decades.

    The sequential queue's formula, column by column, then one value per
    row by the observation's rule, an element being as bad as its worst
    variable:

        under-refined   some column is above its e_max; the distance is
                        the largest log10(e / e_max) over columns, positive
        over-refined    otherwise, every column is below its e_min; the
                        distance is minus the smallest log10(e_min / e)
                        over columns, the margin of the worst one
        neutral         otherwise; 0

    The class is decided by comparison and only the size by logarithm.
    Errors are floored first, so a zero error has a finite distance: at
    1e-30 under dynamo, as the sequential queue floors them, and at
    e_max * 10**-floor_decades under relative, the observation's floor. A
    column whose threshold is zero (at most 1e-30 under dynamo) takes no
    part in the test it belongs to; when no column has a live e_min, no
    row is over-refined.

    The queue reads the size (round_priority) and the reward reads the
    sign as well (agent/round_reward.py). The rows need not be a cache's:
    the global reward hands in an interval's accumulated errors.

    Args:
        errors: Raw errors, (n, n_comp) float64, one row per element.
        config: The normalization rule and its floor.
        e_max: Upper thresholds, (n_comp,), as round_thresholds returns.
        e_min: Lower thresholds, (n_comp,).

    Returns:
        (n,) float64, a new array, by row.

    Raises:
        ValueError: Naming errors, for an array of the wrong form;
            starting with "normalization", for dynamo at more than one
            column; or naming e_max or e_min, for an array of the wrong
            form or with a negative or non-finite value.
    """
    if not isinstance(errors, np.ndarray) or errors.dtype != np.float64 or errors.ndim != 2 or errors.shape[1] < 1:
        raise ValueError("errors: expected a 2-D float64 numpy array with at least one column")
    n_comp = errors.shape[1]
    if n_comp > 1 and config.normalization == "dynamo":
        raise ValueError(f"normalization: 'dynamo' takes one column, and errors has {n_comp}")
    for name, value in (("e_max", e_max), ("e_min", e_min)):
        if not isinstance(value, np.ndarray) or value.shape != (n_comp,):
            raise ValueError(f"{name}: expected a numpy array of shape ({n_comp},)")
        if not (np.isfinite(value).all() and (value >= 0).all()):
            raise ValueError(f"{name}: {value.tolist()}, expected finite values, none negative")

    if config.normalization == "dynamo":
        tiny = 1e-30
        floor = np.full(n_comp, 1e-30)
    else:
        tiny = 0.0
        floor = e_max * 10.0**-config.floor_decades
    floored = np.maximum(errors, floor)
    up, down = e_max > tiny, e_min > tiny

    # Decades above e_max and below e_min, column by column. A column that
    # takes no part is left at the value the max or the min ignores.
    above = np.full(errors.shape, -np.inf)
    above[:, up] = np.log10(floored[:, up] / e_max[up])
    below = np.full(errors.shape, np.inf)
    below[:, down] = np.log10(e_min[down] / floored[:, down])

    under = ((floored > e_max) & up).any(axis=1)
    over = ((floored < e_min) | ~down).all(axis=1) & down.any()
    return np.where(under, above.max(axis=1), np.where(over, -below.min(axis=1), 0.0))


def round_priority(
    cache: RoundCache, config: ObservationConfig, e_max: np.ndarray, e_min: np.ndarray
) -> np.ndarray:
    """Each element's distance from the neutral zone, in decades.

    The size of zone_distance on the cache's indicator array, without its
    sign: how far above its e_max an under-refined element's worst column
    is, how far below its e_min an over-refined element's nearest column
    is, and 0 for a neutral one.

    Args:
        cache: The round's frozen state and indicator arrays.
        config: The indicator, the normalization rule and its floor.
        e_max: Upper thresholds, (n_comp,), as round_thresholds returns.
        e_min: Lower thresholds, (n_comp,).

    Returns:
        (n_active,) float64, a new array, by position; never negative.

    Raises:
        ValueError: As round_thresholds raises for the cache and the
            configuration; or naming e_max or e_min, for an array of the
            wrong form or with a negative or non-finite value.
    """
    return np.abs(zone_distance(_errors(cache, config), config, e_max, e_min))


def round_order(cache: RoundCache, config: ObservationConfig, e_max: np.ndarray, e_min: np.ndarray) -> np.ndarray:
    """The round's queue: every position once, farthest from the zone first.

    A stable sort of round_priority, descending, so elements of equal
    priority, the neutral ones among them, keep their position order. Built
    once per round; within a round positions do not move.

    Returns:
        (n_active,) integers, a new read-only array: a permutation of the
        positions of cache.state.

    Raises:
        ValueError: As round_priority raises.
    """
    order = np.argsort(-round_priority(cache, config, e_max, e_min), kind="stable")
    order.flags.writeable = False
    return order


def next_undecided(
    state: SolverState, round_state: RoundState, order: np.ndarray, cursor: int
) -> tuple[int | None, int, int]:
    """The next position of the queue that the round has not decided.

    Starts at order[cursor] and passes over every position whose entry in
    the round's record is no longer UNDECIDED: under the family rules, the
    siblings a family coarsen has already marked. One round is

        round_state = start_round(state, round_index, n_rounds)
        position, cursor, skipped = next_undecided(state, round_state, order, 0)
        while position is not None:
            mask = action_mask(state, round_state, position, rule)
            round_state = commit(state, round_state, position, action, rule)
            position, cursor, passed = next_undecided(state, round_state, order, cursor)
            skipped += passed

    with the action chosen from the mask. At the end, the positions asked
    and the positions skipped add up to n_active.

    Args:
        state: The round's frozen state.
        round_state: Holds the round's decision record.
        order: The round's queue, as round_order returns it.
        cursor: How many entries of order are already behind: 0 at the
            round's start, then the cursor this function last returned.

    Returns:
        The position to ask next, or None when the queue is exhausted; the
        cursor to pass in next time; and how many decided positions this
        call passed over.

    Raises:
        ValueError: Starting with "order" or "cursor", for an order that is
            not an array of n_active entries or a cursor outside
            [0, n_active]; or as is_decided raises, for an entry of order
            that names no element or a record of the wrong length.
    """
    if not isinstance(order, np.ndarray) or order.shape != (state.n_active,):
        raise ValueError(f"order: expected a numpy array of shape ({state.n_active},)")
    if isinstance(cursor, bool) or not isinstance(cursor, Integral) or not 0 <= cursor <= state.n_active:
        raise ValueError(f"cursor: {cursor!r}, expected an integer in [0, {state.n_active}]")
    cursor = int(cursor)
    skipped = 0
    while cursor < state.n_active:
        position = int(order[cursor])
        cursor += 1
        if not is_decided(state, round_state, position):
            return position, cursor, skipped
        skipped += 1
    return None, cursor, skipped