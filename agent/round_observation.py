"""The observation of the 2D agent core: what the policy sees of one element.

actor_observation builds one element's feature vector from a RoundCache (the
frozen mesh and indicator arrays of one round), a RoundState (what moves
within the round) and an ObservationConfig. It is the one function shared
with deployment (SOLVER_CONTRACT_V2 section 9): pure, with no backend, and
with no training-only type in its signature.

Features, for an element with F = 2*ndim faces (feature_names gives them):

    error                   the element's own normalized error
    error_<face> (F)        the face neighbor's; the larger of two finer halves
    error_family_max        the largest over the element's sibling family
    level                   level / max_level
    level_<face> (F)        the face neighbor's
    wall_<face> (F)         1 on a wall face, else 0
    resource                committed count / element budget; can exceed 1
    round_progress          round index / rounds per interval

A wall face repeats the element's own error and level, and an element with
no complete family is its own family: no jump is shown where there is
nothing across. A periodic wrap is not a feature; a neighbor through it is
an ordinary neighbor.

critic_observation and observation are training-only: the critic's features
(in this build, the actor's) and the dict an environment returns, keyed
ACTOR_KEY and CRITIC_KEY. feature_bounds and observation_bounds give the
ranges an environment declares; no gymnasium type is built here.

This module belongs to the 2D core (AGENT_2D_ROADMAP item 2.2) and stands
beside the sequential core's observation.py, which is not on this path.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np

from agent.normalization import normalized_error
from agent.round_cache import RoundCache
from agent.round_state import RoundState
from contract.reports import AuxFields

# dynamo: the 1D rule, agent.normalization.normalized_error (DynAMO Eq. 15).
# relative: log10(e / (alpha * e_inf)), floored; unchanged when every error
# is multiplied by one constant.
NORMALIZATIONS = ("dynamo", "relative")
# The keys of the observation dict (SOLVER_CONTRACT_V2 section 9).
ACTOR_KEY = "actor"
CRITIC_KEY = "critic"


@dataclass(frozen=True, slots=True, kw_only=True)
class ObservationConfig:
    """What the observation is built with. Saved with every checkpoint.

    Fields:
        indicator: The indicator the error features read, by name.
        reduction: The reduction its array was computed under.
        alpha: Error tolerance, 0 < alpha < 1; the threshold is
            alpha * (the round's largest error).
        element_budget: The element budget the resource feature divides by.
        normalization: One of NORMALIZATIONS.
        floor_decades: Under relative, the lowest value an error feature
            takes is -floor_decades. Not read under dynamo.
    """

    indicator: str
    reduction: str
    alpha: float
    element_budget: int
    normalization: str
    floor_decades: float

    def __post_init__(self) -> None:
        """Check every field.

        Raises:
            ValueError: Naming the field.
        """
        for name in ("indicator", "reduction"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name}: {value!r}, expected a name")
        for name in ("alpha", "floor_decades"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError(f"{name}: {value!r}, expected a number")
        if not 0 < self.alpha < 1:
            raise ValueError(f"alpha: {self.alpha}, expected between 0 and 1")
        if not 0 < self.floor_decades < math.inf:
            raise ValueError(f"floor_decades: {self.floor_decades}, expected a positive finite number")
        budget = self.element_budget
        if isinstance(budget, bool) or not isinstance(budget, Integral) or budget < 1:
            raise ValueError(f"element_budget: {budget!r}, expected an integer of at least 1")
        if self.normalization not in NORMALIZATIONS:
            raise ValueError(f"normalization: {self.normalization!r}, expected one of {list(NORMALIZATIONS)}")


def feature_names(ndim: int) -> tuple[str, ...]:
    """The actor features in order, 6 * ndim + 5 of them.

    Faces are named in the contract's order, f = 2 * axis + side: x_min,
    x_max, y_min, y_max, ...

    Raises:
        ValueError: If ndim is not 1, 2 or 3.
    """
    if ndim not in (1, 2, 3):
        raise ValueError(f"ndim: {ndim!r}, expected 1, 2 or 3")
    faces = [f"{axis}_{side}" for axis in "xyz"[:ndim] for side in ("min", "max")]
    return (
        "error",
        *(f"error_{face}" for face in faces),
        "error_family_max",
        "level",
        *(f"level_{face}" for face in faces),
        *(f"wall_{face}" for face in faces),
        "resource",
        "round_progress",
    )


def feature_bounds(ndim: int, config: ObservationConfig) -> tuple[np.ndarray, np.ndarray]:
    """The lowest and highest value each actor feature can take.

    For the environment's observation space, and kept beside the builder so
    the two cannot drift apart. Under relative an error feature lies between
    the floor and -log10(alpha), the value of a round's largest error; under
    dynamo it is unbounded. Levels, wall bits and round progress lie in
    [0, 1]. The resource feature has no upper bound: the committed count can
    pass the budget.

    Returns:
        low and high, each (6 * ndim + 5,) float32, new arrays, in
        feature_names' order.

    Raises:
        ValueError: If ndim is not 1, 2 or 3.
    """
    names = feature_names(ndim)
    if config.normalization == "relative":
        error_low, error_high = -config.floor_decades, -math.log10(config.alpha)
    else:
        error_low, error_high = -math.inf, math.inf
    low = np.zeros(len(names), dtype=np.float32)
    high = np.ones(len(names), dtype=np.float32)
    for i, name in enumerate(names):
        if name.startswith("error"):
            low[i], high[i] = error_low, error_high
    high[names.index("resource")] = np.inf
    return low, high


def _normalize(e: float, e_inf: float, config: ObservationConfig) -> float:
    """Normalize one raw error against the round's largest.

    Args:
        e: The raw error, not negative.
        e_inf: The largest raw error of the round.
        config: The rule and its parameters.
    """
    if config.normalization == "dynamo":
        return float(normalized_error(e, config.alpha, e_inf))
    floor = -config.floor_decades
    e_max = config.alpha * e_inf
    if e <= 0.0 or e_max <= 0.0:
        return floor
    # The ratio can underflow to zero, which log10 does not accept.
    ratio = e / e_max
    return max(math.log10(ratio), floor) if ratio > 0.0 else floor


def _feature(raw: np.ndarray, e_inf: np.ndarray, config: ObservationConfig) -> float:
    """One error feature from one raw value per column.

    Each column is normalized against its own largest error of the round,
    and the feature is the largest of the normalized values: the element's
    worst variable, each variable measured on its own scale.

    Args:
        raw: The raw errors, one per column. (n_comp,).
        e_inf: The round's largest raw error in each column. (n_comp,).
        config: The rule and its parameters.
    """
    return max(_normalize(float(e), float(m), config) for e, m in zip(raw, e_inf, strict=True))


def actor_observation(
    cache: RoundCache, round_state: RoundState, position: int, config: ObservationConfig
) -> np.ndarray:
    """Build the actor's feature vector for one element.

    Pure: the same arguments give the same vector, and nothing is called or
    changed. Every error feature is normalized against the largest error of
    the round, so all of them move together when the indicator is switched.
    With more than one column per indicator (the per_variable reduction),
    each column is normalized against its own largest error and an error
    feature is the largest of the normalized values.

    Args:
        cache: The round's frozen state and indicator arrays.
        round_state: The round index and the committed count.
        position: The element's position in cache.state.
        config: The indicator, the normalization and the budget.

    Returns:
        (6 * ndim + 5,) float32, a new array, in feature_names' order.

    Raises:
        ValueError: If the configured indicator is not in the cache, the
            reductions differ, the rule is dynamo and the cache has more
            than one column per indicator, position names no element, or
            max_level is below 1.
    """
    state = cache.state
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
    if isinstance(position, bool) or not isinstance(position, Integral) or not 0 <= position < state.n_active:
        raise ValueError(f"position: {position!r}, expected an integer in [0, {state.n_active})")
    if state.max_level < 1:
        raise ValueError(f"max_level: {state.max_level}, expected at least 1")

    # One row per element, one column per component; e_inf is per column.
    errors = cache.errors[config.indicator]
    e_inf = errors.max(axis=0)
    own_error = errors[position]
    own_level = int(state.level[position])

    face_error, face_level, wall = [], [], []
    for face in range(2 * state.ndim):
        slots = state.neighbors[position, face]
        filled = slots[slots >= 0]
        if filled.size == 0:
            # A wall: nothing across, so no jump is shown; the bit says why.
            face_error.append(own_error)
            face_level.append(own_level)
            wall.append(1.0)
        else:
            # One neighbor, or two finer halves: the larger error, column by
            # column. Slot 0 is always filled, and under 2:1 both halves are
            # at its level.
            face_error.append(errors[filled].max(axis=0))
            face_level.append(int(state.level[filled[0]]))
            wall.append(0.0)

    family = state.family[position]
    family_error = errors[family].max(axis=0) if (family >= 0).all() else own_error

    return np.array(
        [
            *(_feature(raw, e_inf, config) for raw in (own_error, *face_error, family_error)),
            own_level / state.max_level,
            *(level / state.max_level for level in face_level),
            *wall,
            round_state.committed / config.element_budget,
            round_state.round_index / round_state.n_rounds,
        ],
        dtype=np.float32,
    )


def critic_feature_names(ndim: int) -> tuple[str, ...]:
    """The critic features in order. In this build, the actor's."""
    return feature_names(ndim)


def critic_observation(
    cache: RoundCache,
    round_state: RoundState,
    position: int,
    config: ObservationConfig,
    aux: AuxFields | None = None,
) -> np.ndarray:
    """Build the critic's feature vector for one element. Training-only.

    The critic is not deployed, so unlike the actor builder this one may
    take the auxiliary channel. In this build no privileged feature is
    defined and it returns the actor's features.

    Args:
        cache: The round's frozen state and indicator arrays.
        round_state: The round index and the committed count.
        position: The element's position in cache.state.
        config: The indicator, the normalization and the budget.
        aux: The auxiliary fields; None until a privileged feature exists.

    Returns:
        (6 * ndim + 5,) float32, a new array, in critic_feature_names' order.

    Raises:
        ValueError: If aux is not None; and as actor_observation raises.
    """
    if aux is not None:
        raise ValueError("aux: no privileged critic feature is defined yet, expected None")
    return actor_observation(cache, round_state, position, config)


def observation(
    cache: RoundCache,
    round_state: RoundState,
    position: int,
    config: ObservationConfig,
    aux: AuxFields | None = None,
) -> dict[str, np.ndarray]:
    """Build the observation dict for one element. Training-only.

    Returns:
        ACTOR_KEY -> actor_observation's vector and CRITIC_KEY ->
        critic_observation's, two separate arrays.

    Raises:
        ValueError: As the two builders raise.
    """
    return {
        ACTOR_KEY: actor_observation(cache, round_state, position, config),
        CRITIC_KEY: critic_observation(cache, round_state, position, config, aux),
    }


def observation_bounds(ndim: int, config: ObservationConfig) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """feature_bounds by key, for the dict that observation returns.

    Raises:
        ValueError: If ndim is not 1, 2 or 3.
    """
    return {ACTOR_KEY: feature_bounds(ndim, config), CRITIC_KEY: feature_bounds(ndim, config)}