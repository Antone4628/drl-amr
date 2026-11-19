"""Tests for agent/round_observation.py and agent/round_state.py: the actor
observation of the 2D agent core and the two nouns it takes.

AGENT_2D_ROADMAP item 2.2: sitting 1's signals S1-S12 and sitting 2's T1-T7.
Six groups:

  - the two nouns: what RoundState and ObservationConfig accept and reject
    (S10), and the feature names (S1);
  - on a literal state: the five-element instance of the explanation file's
    section 2.2.2, every number hand-set, so each feature can be read off
    (S2-S9), and what the builder rejects (S10);
  - more than one column per indicator: the same instance with a second
    variable, section 2.2.9 (T2-T5), and the bounds every vector lies
    within (T7);
  - on the mock quad backend: the builder against a slow reference that
    finds every neighbor and family from the centroids and sizes alone,
    never from the neighbor or family arrays (S11), and per_variable
    against scalar at nvar = 3 (T6);
  - at ndim = 1: the builder against the sequential core's
    build_observation on the same four-element mesh (S12).

The critic builder, the dict and the bounds themselves are in
tests/agent/test_round_dict.py.

Run: pytest tests/agent/test_round_observation.py -v
"""
from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from agent.normalization import normalized_error
from agent.observation import build_observation
from agent.round_cache import RoundCache, read_round
from agent.round_observation import (
    NORMALIZATIONS,
    ObservationConfig,
    actor_observation,
    feature_bounds,
    feature_names,
)
from agent.round_state import RoundState
from backends.mock_quad.backend import MockQuadBackend
from contract.checks import training_only_params
from contract.element_state import SolverState as SequentialState
from contract.solver_contract import HOLD, REFINE
from contract.state import SolverState

ALPHA = 0.1
BUDGET = 8
FLOOR = 8.0
# The instance's raw errors, by position: cell A's four children, then cell B.
ERRORS = np.array([1e-3, 1e-1, 1e-2, 1.0, 1e-4])
# A second variable for the instance (section 2.2.9): about a thousand times
# larger, with a different worst element.
SECOND = np.array([1e3, 1.0, 10.0, 1e-1, 1e2])
# Feature indices at 2D, in feature_names' order.
ERROR_FEATURES = slice(0, 6)
OWN_ERROR, FACE_ERRORS, FAMILY_ERROR = 0, slice(1, 5), 5
OWN_LEVEL, FACE_LEVELS, WALLS = 6, slice(7, 11), slice(11, 15)
RESOURCE, PROGRESS = 15, 16

NAMES_2D = (
    "error",
    "error_x_min",
    "error_x_max",
    "error_y_min",
    "error_y_max",
    "error_family_max",
    "level",
    "level_x_min",
    "level_x_max",
    "level_y_min",
    "level_y_max",
    "wall_x_min",
    "wall_x_max",
    "wall_y_min",
    "wall_y_max",
    "resource",
    "round_progress",
)


def config(normalization="relative", **changes) -> ObservationConfig:
    """The instance's configuration, with any field replaced."""
    fields = {
        "indicator": "ind",
        "reduction": "scalar",
        "alpha": ALPHA,
        "element_budget": BUDGET,
        "normalization": normalization,
        "floor_decades": FLOOR,
    }
    return ObservationConfig(**{**fields, **changes})


def at(committed=5, round_index=1, n_rounds=2) -> RoundState:
    """Round 1 of 2, with the committed count where it is asked."""
    return RoundState(round_index=round_index, n_rounds=n_rounds, committed=committed)


def instance_state(*, wrap=True, max_level=2) -> SolverState:
    """The five-element mesh, built by hand: cell A's children at level 1
    (positions 0-3 in child order) and cell B at level 0 (position 4), on
    [0,2]x[0,1], periodic in x and walled in y."""
    neighbors = np.array(
        [
            [[4, -1], [1, -1], [-1, -1], [2, -1]],
            [[0, -1], [4, -1], [-1, -1], [3, -1]],
            [[4, -1], [3, -1], [0, -1], [-1, -1]],
            [[2, -1], [4, -1], [1, -1], [-1, -1]],
            [[1, 3], [0, 2], [-1, -1], [-1, -1]],
        ],
        dtype=np.int32,
    )
    neighbor_wrap = np.zeros((5, 4), dtype=bool)
    if wrap:
        neighbor_wrap[0, 0] = neighbor_wrap[2, 0] = neighbor_wrap[4, 1] = True
    return SolverState(
        level=np.array([1, 1, 1, 1, 0], dtype=np.int32),
        neighbors=neighbors,
        neighbor_wrap=neighbor_wrap,
        family=np.array([[0, 1, 2, 3]] * 4 + [[-1, -1, -1, -1]], dtype=np.int32),
        can_refine=np.ones(5, dtype=bool),
        can_coarsen=np.array([True, True, True, True, False]),
        centroid=np.array([[0.25, 0.25], [0.75, 0.25], [0.25, 0.75], [0.75, 0.75], [1.5, 0.5]]),
        h=np.array([[0.5, 0.5]] * 4 + [[1.0, 1.0]]),
        attributes=np.zeros((5, 0)),
        n_active=5,
        ndim=2,
        nvar=1,
        max_level=max_level,
        mesh_epoch=2,
        time=0.0,
        interval_T=1.0,
        dt_base=0.25,
        amr_freq=4,
        dt_stable=0.125,
        domain_lo=np.array([0.0, 0.0]),
        domain_hi=np.array([2.0, 1.0]),
        periodic=np.array([True, False]),
    )


def instance(errors=ERRORS, **state_kwargs) -> RoundCache:
    """A cache on the five-element mesh holding one indicator, "ind"."""
    values = np.array(errors, dtype=np.float64).reshape(5, 1)
    return RoundCache(
        state=instance_state(**state_kwargs), errors={"ind": values}, indicators=("ind",), reduction="scalar"
    )


def columns(*variables) -> RoundCache:
    """A cache on the five-element mesh holding "ind" with one column per
    argument, read under per_variable."""
    values = np.column_stack([np.asarray(v, dtype=np.float64) for v in variables])
    return RoundCache(
        state=instance_state(), errors={"ind": values}, indicators=("ind",), reduction="per_variable"
    )


def per_variable(normalization="relative", **changes) -> ObservationConfig:
    """The instance's configuration for a cache read under per_variable."""
    return config(normalization, reduction="per_variable", **changes)


def inside(obs: np.ndarray, cfg: ObservationConfig) -> bool:
    """Whether a vector lies within feature_bounds, inclusive, no tolerance."""
    low, high = feature_bounds(2, cfg)
    return bool(((low <= obs) & (obs <= high)).all())


# --- RoundState (S10) ------------------------------------------------------------

def test_round_state_holds_what_it_was_given():
    state = RoundState(round_index=2, n_rounds=3, committed=11)
    assert (state.round_index, state.n_rounds, state.committed) == (2, 3, 11)


def test_round_state_accepts_numpy_integers():
    state = RoundState(round_index=np.int64(1), n_rounds=np.int32(2), committed=np.int64(5))
    assert state.committed == 5


@pytest.mark.parametrize(
    "changes, pattern",
    [
        ({"round_index": 0}, r"^round_index: 0, expected 1 to n_rounds = 2"),
        ({"round_index": 3}, r"^round_index: 3, expected 1 to n_rounds = 2"),
        ({"n_rounds": 0}, r"^n_rounds: 0, expected at least 1"),
        ({"committed": 0}, r"^committed: 0, expected at least 1"),
        ({"round_index": 1.0}, r"^round_index: 1\.0, expected an integer"),
        ({"committed": True}, r"^committed: True, expected an integer"),
        ({"n_rounds": "2"}, r"^n_rounds: '2', expected an integer"),
    ],
    ids=["round 0", "past the last round", "no rounds", "nothing committed", "a float", "a bool", "a string"],
)
def test_round_state_rejections(changes, pattern):
    fields = {"round_index": 1, "n_rounds": 2, "committed": 5}
    with pytest.raises(ValueError, match=pattern):
        RoundState(**{**fields, **changes})


@pytest.mark.parametrize("field", ["round_index", "n_rounds", "committed"])
def test_round_state_is_frozen(field):
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(at(), field, 1)


# --- ObservationConfig (S10) -----------------------------------------------------

def test_config_holds_what_it_was_given():
    cfg = config()
    assert dataclasses.asdict(cfg) == {
        "indicator": "ind",
        "reduction": "scalar",
        "alpha": 0.1,
        "element_budget": 8,
        "normalization": "relative",
        "floor_decades": 8.0,
    }


def test_the_two_normalizations():
    assert NORMALIZATIONS == ("dynamo", "relative")
    for name in NORMALIZATIONS:
        assert config(name).normalization == name


def test_config_has_no_defaults():
    with pytest.raises(TypeError):
        ObservationConfig(indicator="ind", reduction="scalar", alpha=ALPHA, element_budget=BUDGET)


@pytest.mark.parametrize(
    "changes, pattern",
    [
        ({"indicator": ""}, r"^indicator: '', expected a name"),
        ({"reduction": None}, r"^reduction: None, expected a name"),
        ({"alpha": 0}, r"^alpha: 0, expected between 0 and 1"),
        ({"alpha": 1.0}, r"^alpha: 1\.0, expected between 0 and 1"),
        ({"alpha": "0.1"}, r"^alpha: '0\.1', expected a number"),
        ({"alpha": math.nan}, r"^alpha: nan, expected between 0 and 1"),
        ({"element_budget": 0}, r"^element_budget: 0, expected an integer of at least 1"),
        ({"element_budget": 8.0}, r"^element_budget: 8\.0, expected an integer of at least 1"),
        ({"normalization": "zscore"}, r"^normalization: 'zscore', expected one of \['dynamo', 'relative'\]"),
        ({"floor_decades": 0}, r"^floor_decades: 0, expected a positive finite number"),
        ({"floor_decades": math.inf}, r"^floor_decades: inf, expected a positive finite number"),
        ({"floor_decades": True}, r"^floor_decades: True, expected a number"),
    ],
    ids=[
        "an empty indicator",
        "no reduction",
        "alpha 0",
        "alpha 1",
        "alpha a string",
        "alpha nan",
        "budget 0",
        "budget a float",
        "an unknown rule",
        "floor 0",
        "floor infinite",
        "floor a bool",
    ],
)
def test_config_rejections(changes, pattern):
    with pytest.raises(ValueError, match=pattern):
        config(**changes)


def test_config_is_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        config().alpha = 0.5


# --- feature_names (S1) ----------------------------------------------------------

@pytest.mark.parametrize("ndim, count", [(1, 11), (2, 17), (3, 23)])
def test_feature_count(ndim, count):
    names = feature_names(ndim)
    assert len(names) == count == 6 * ndim + 5
    assert len(set(names)) == count


def test_feature_order_at_2d():
    assert feature_names(2) == NAMES_2D


def test_feature_order_at_1d():
    assert feature_names(1) == (
        "error",
        "error_x_min",
        "error_x_max",
        "error_family_max",
        "level",
        "level_x_min",
        "level_x_max",
        "wall_x_min",
        "wall_x_max",
        "resource",
        "round_progress",
    )


@pytest.mark.parametrize("ndim", [0, 4, "2"])
def test_feature_names_rejects_other_dimensions(ndim):
    with pytest.raises(ValueError, match="^ndim: "):
        feature_names(ndim)


# --- The instance, feature by feature (S2) ---------------------------------------

def test_cell_b_observed_with_five_committed():
    obs = actor_observation(instance(), at(5), 4, config())
    expected = [
        -3.0,  # own error, 1e-4
        1.0,  # x-min: two finer halves, positions 1 and 3; the larger, 1
        -1.0,  # x-max: halves 0 and 2 through the wrap; the larger, 1e-2
        -3.0,  # y-min: a wall, its own
        -3.0,  # y-max: a wall, its own
        -3.0,  # no family: its own
        0.0,  # level 0 of 2
        0.5,  # x-min: the halves are at level 1
        0.5,  # x-max
        0.0,  # y-min: a wall, its own
        0.0,  # y-max
        0.0,
        0.0,
        1.0,
        1.0,  # the two walls
        0.625,  # 5 of 8
        0.5,  # round 1 of 2
    ]
    assert obs.tolist() == pytest.approx(expected, abs=1e-6)


def test_position_0_observed_with_eight_committed():
    obs = actor_observation(instance(), at(8), 0, config())
    expected = [
        -2.0,  # own error, 1e-3
        -3.0,  # x-min: cell B, through the wrap
        0.0,  # x-max: position 1, on the threshold
        -2.0,  # y-min: a wall, its own
        -1.0,  # y-max: position 2
        1.0,  # the family's largest: position 3, which no face shows
        0.5,  # level 1 of 2
        0.0,  # x-min: cell B is coarser
        0.5,  # x-max
        0.5,  # y-min: a wall, its own
        0.5,  # y-max
        0.0,
        0.0,
        1.0,
        0.0,  # one wall
        1.0,  # 8 of 8
        0.5,
    ]
    assert obs.tolist() == pytest.approx(expected, abs=1e-6)


def test_the_family_maximum_shows_what_the_faces_hide():
    """Position 0's diagonal sibling carries the largest error on the mesh.
    None of its four face errors reaches it; the family feature does."""
    obs = actor_observation(instance(), at(), 0, config())
    assert obs[FAMILY_ERROR] == pytest.approx(1.0, abs=1e-6)
    assert obs[FACE_ERRORS].max() < obs[FAMILY_ERROR]


@pytest.mark.parametrize("position", range(4))
def test_siblings_share_the_family_maximum(position):
    obs = actor_observation(instance(), at(), position, config())
    assert obs[FAMILY_ERROR] == pytest.approx(1.0, abs=1e-6)
    assert obs[FAMILY_ERROR] >= obs[OWN_ERROR]


# --- The two normalizations (S3, S4) ---------------------------------------------

@pytest.mark.parametrize("position", range(5))
def test_dynamo_is_relative_minus_one_on_the_instance(position):
    """With e_max = 0.1 the two rules differ by the factor |log10(e_max)| = 1
    and the shift 1: relative = dynamo + 1."""
    relative = actor_observation(instance(), at(), position, config("relative"))
    dynamo = actor_observation(instance(), at(), position, config("dynamo"))
    assert dynamo[ERROR_FEATURES].tolist() == pytest.approx((relative[ERROR_FEATURES] - 1.0).tolist(), abs=1e-6)
    assert np.array_equal(dynamo[OWN_LEVEL:], relative[OWN_LEVEL:])


@pytest.mark.parametrize("scale", [1e-6, 10.0, 100.0, 1e6])
@pytest.mark.parametrize("position", [0, 4])
def test_relative_does_not_change_with_the_errors_scale(scale, position):
    base = actor_observation(instance(), at(), position, config())
    scaled = actor_observation(instance(ERRORS * scale), at(), position, config())
    assert scaled.tolist() == pytest.approx(base.tolist(), abs=1e-5)


@pytest.mark.parametrize("position", range(5))
def test_dynamo_is_flat_where_e_max_is_one(position):
    """Errors ten times the instance's: alpha * e_inf = 1, the denominator
    is zero, and the 1D rule returns 0 for every element."""
    obs = actor_observation(instance(ERRORS * 10.0), at(), position, config("dynamo"))
    assert obs[ERROR_FEATURES].tolist() == [0.0] * 6


def test_dynamo_reverses_where_e_max_is_above_one():
    """Errors a hundred times the instance's: the smallest error gets the
    largest value."""
    cache = instance(ERRORS * 100.0)
    smallest = actor_observation(cache, at(), 4, config("dynamo"))[OWN_ERROR]
    largest = actor_observation(cache, at(), 3, config("dynamo"))[OWN_ERROR]
    assert smallest == pytest.approx(2.0, abs=1e-5)
    assert largest == pytest.approx(-2.0, abs=1e-5)


def test_relative_is_bounded_by_the_floor_and_alpha():
    for position in range(5):
        errors = actor_observation(instance(), at(), position, config())[ERROR_FEATURES]
        assert (errors >= -FLOOR).all() and (errors <= -math.log10(ALPHA) + 1e-6).all()


# --- What moves within a round (S5) ----------------------------------------------

@pytest.mark.parametrize("position", range(5))
def test_only_the_resource_feature_follows_the_committed_count(position):
    before = actor_observation(instance(), at(5), position, config())
    after = actor_observation(instance(), at(8), position, config())
    assert np.flatnonzero(before != after).tolist() == [RESOURCE]
    assert (before[RESOURCE], after[RESOURCE]) == (0.625, 1.0)


def test_the_resource_feature_can_exceed_one():
    assert actor_observation(instance(), at(12), 0, config())[RESOURCE] == 1.5


@pytest.mark.parametrize("round_index, expected", [(1, 0.5), (2, 1.0)])
def test_round_progress(round_index, expected):
    obs = actor_observation(instance(), at(round_index=round_index), 0, config())
    assert obs[PROGRESS] == expected


def test_only_round_progress_follows_the_round():
    first = actor_observation(instance(), at(round_index=1), 2, config())
    last = actor_observation(instance(), at(round_index=2), 2, config())
    assert np.flatnonzero(first != last).tolist() == [PROGRESS]


# --- Walls and the wrap (S6) -----------------------------------------------------

@pytest.mark.parametrize("normalization", NORMALIZATIONS)
@pytest.mark.parametrize("position", range(5))
def test_a_wall_face_mirrors_the_element(position, normalization):
    cache = instance()
    obs = actor_observation(cache, at(), position, config(normalization))
    is_wall = (cache.state.neighbors[position] < 0).all(axis=1)
    assert obs[WALLS].tolist() == is_wall.astype(float).tolist()
    assert is_wall.any()
    for face in np.flatnonzero(is_wall):
        assert obs[FACE_ERRORS][face] == obs[OWN_ERROR]
        assert obs[FACE_LEVELS][face] == obs[OWN_LEVEL]


@pytest.mark.parametrize("position", range(5))
def test_the_wrap_is_not_a_feature(position):
    """Clearing neighbor_wrap changes no feature: a neighbor through the
    wrap is read as any other."""
    with_wrap = actor_observation(instance(wrap=True), at(), position, config())
    without = actor_observation(instance(wrap=False), at(), position, config())
    assert np.array_equal(with_wrap, without)


# --- Half-neighbors (S7) ---------------------------------------------------------

def test_a_face_with_two_halves_reads_the_larger_whichever_slot_holds_it():
    swapped = ERRORS.copy()
    swapped[[1, 3]] = swapped[[3, 1]]
    base = actor_observation(instance(), at(), 4, config())
    other = actor_observation(instance(swapped), at(), 4, config())
    assert base[FACE_ERRORS][0] == other[FACE_ERRORS][0] == pytest.approx(1.0, abs=1e-6)


def test_the_smaller_half_does_not_reach_the_face():
    """Cell B's x-max face holds positions 0 and 2. Lowering the smaller of
    the two changes nothing on that face; lowering the larger does. Neither
    is the round's largest error, so the normalization does not move."""
    base = actor_observation(instance(), at(), 4, config())[FACE_ERRORS][1]
    smaller = ERRORS.copy()
    smaller[0] = 1e-5
    assert actor_observation(instance(smaller), at(), 4, config())[FACE_ERRORS][1] == base
    larger = ERRORS.copy()
    larger[2] = 1e-3
    lowered = actor_observation(instance(larger), at(), 4, config())[FACE_ERRORS][1]
    assert lowered == pytest.approx(-2.0, abs=1e-6) and lowered != base


# --- Zero errors and the floor (S8) ----------------------------------------------

def test_a_zero_error_is_at_the_floor():
    zeroed = ERRORS.copy()
    zeroed[4] = 0.0
    obs = actor_observation(instance(zeroed), at(), 4, config())
    assert obs[OWN_ERROR] == -FLOOR
    # Its faces still read the neighbors.
    assert obs[FACE_ERRORS][:2].tolist() == pytest.approx([1.0, -1.0], abs=1e-6)


@pytest.mark.parametrize("position", range(5))
def test_an_all_zero_round_is_at_the_floor_everywhere(position):
    obs = actor_observation(instance(np.zeros(5)), at(), position, config())
    assert obs[ERROR_FEATURES].tolist() == [-FLOOR] * 6
    assert np.isfinite(obs).all()


def test_an_error_far_below_the_largest_is_at_the_floor():
    tiny = ERRORS.copy()
    tiny[0] = 1e-12
    assert actor_observation(instance(tiny), at(), 0, config())[OWN_ERROR] == -FLOOR


def test_the_floor_is_the_configurations():
    tiny = ERRORS.copy()
    tiny[0] = 1e-12
    obs = actor_observation(instance(tiny), at(), 0, config(floor_decades=20.0))
    assert obs[OWN_ERROR] == pytest.approx(-11.0, abs=1e-5)


def test_a_ratio_that_underflows_is_at_the_floor():
    extreme = np.array([1e-320, 1e299, 0.0, 0.0, 0.0])
    obs = actor_observation(instance(extreme), at(), 0, config())
    assert obs[OWN_ERROR] == -FLOOR
    assert np.isfinite(obs).all()


@pytest.mark.parametrize("normalization", NORMALIZATIONS)
@pytest.mark.parametrize("errors", [ERRORS, ERRORS * 1e8, np.zeros(5), np.array([0.0, 1e-300, 1e300, 1.0, 0.0])],
                         ids=["the instance", "large", "all zero", "extreme"])
def test_every_feature_is_finite(errors, normalization):
    for position in range(5):
        assert np.isfinite(actor_observation(instance(errors), at(), position, config(normalization))).all()


# --- The function itself (S9) ----------------------------------------------------

def test_the_signature_accepts_no_training_only_type():
    assert training_only_params(actor_observation) == []


def test_the_result_is_a_new_float32_vector():
    cache = instance()
    first = actor_observation(cache, at(), 0, config())
    second = actor_observation(cache, at(), 0, config())
    assert first.dtype == np.float32 and first.shape == (17,)
    assert first is not second and np.array_equal(first, second)
    first[0] = 99.0
    assert np.array_equal(actor_observation(cache, at(), 0, config()), second)


def test_the_vector_is_as_long_as_the_names():
    assert actor_observation(instance(), at(), 0, config()).shape == (len(feature_names(2)),)


def test_the_indicator_switches_every_error_feature_together():
    """Two indicators in one cache: the configuration's name picks which one
    all six error features read, and nothing else moves."""
    state = instance_state()
    cache = RoundCache(
        state=state,
        errors={"ind": ERRORS.reshape(5, 1).copy(), "other": ERRORS[::-1].reshape(5, 1).copy()},
        indicators=("ind", "other"),
        reduction="scalar",
    )
    first = actor_observation(cache, at(), 0, config(indicator="ind"))
    second = actor_observation(cache, at(), 0, config(indicator="other"))
    alone = actor_observation(instance(ERRORS[::-1]), at(), 0, config())
    assert np.array_equal(first, actor_observation(instance(), at(), 0, config()))
    assert np.array_equal(second, alone)
    assert (first[ERROR_FEATURES] != second[ERROR_FEATURES]).any()
    assert np.array_equal(first[OWN_LEVEL:], second[OWN_LEVEL:])


# --- What the builder rejects (S10) ----------------------------------------------

def test_an_indicator_the_cache_does_not_hold():
    with pytest.raises(ValueError, match=r"^indicator: 'zz_style' is not in the cache, which holds \['ind'\]"):
        actor_observation(instance(), at(), 0, config(indicator="zz_style"))


def test_a_reduction_the_cache_was_not_read_under():
    with pytest.raises(ValueError, match=r"^reduction: the cache was read under 'scalar'.*'per_variable'"):
        actor_observation(instance(), at(), 0, config(reduction="per_variable"))


@pytest.mark.parametrize("position", [-1, 5, 1.0, True, None], ids=["negative", "past the end", "a float", "a bool",
                                                                    "None"])
def test_a_position_that_names_no_element(position):
    with pytest.raises(ValueError, match=r"^position: .*expected an integer in \[0, 5\)"):
        actor_observation(instance(), at(), position, config())


def test_a_numpy_integer_position_is_accepted():
    cache = instance()
    assert np.array_equal(
        actor_observation(cache, at(), np.int32(4), config()), actor_observation(cache, at(), 4, config())
    )


def test_dynamo_takes_one_column():
    """The 1D rule reads an all-zero column as just above the threshold, so
    it would win the maximum over columns at every element (T5)."""
    with pytest.raises(
        ValueError, match=r"^normalization: 'dynamo' takes one column per indicator, and the cache has n_comp = 2"
    ):
        actor_observation(columns(ERRORS, SECOND), at(), 0, per_variable("dynamo"))


def test_a_mesh_that_cannot_refine_has_no_level_scale():
    with pytest.raises(ValueError, match="^max_level: 0, expected at least 1"):
        actor_observation(instance(max_level=0), at(), 0, config())


# --- More than one column per indicator (T2-T5) ----------------------------------

def test_two_columns_cell_b():
    obs = actor_observation(columns(ERRORS, SECOND), at(5), 4, per_variable())
    expected = [
        0.0,  # own: column 1, on its threshold (column 0 reads -3)
        1.0,  # x-min: column 0, position 3 (column 1 reads -2)
        1.0,  # x-max: column 1, position 0 (column 0 reads -1)
        0.0,  # y-min: a wall, its own
        0.0,  # y-max: a wall, its own
        0.0,  # no family: its own
        0.0,
        0.5,
        0.5,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        1.0,
        0.625,
        0.5,
    ]
    assert obs.tolist() == pytest.approx(expected, abs=1e-6)


def test_two_columns_position_0():
    obs = actor_observation(columns(ERRORS, SECOND), at(8), 0, per_variable())
    expected = [
        1.0,  # own: column 1, where it is the round's worst (column 0 reads -2)
        0.0,  # x-min: cell B through the wrap, column 1 (column 0 reads -3)
        0.0,  # x-max: position 1, column 0 (column 1 reads -2)
        1.0,  # y-min: a wall, its own
        -1.0,  # y-max: position 2, both columns
        1.0,  # the family's largest, both columns
        0.5,
        0.0,
        0.5,
        0.5,
        0.5,
        0.0,
        0.0,
        1.0,
        0.0,
        1.0,
        0.5,
    ]
    assert obs.tolist() == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize("position", range(5))
def test_an_error_feature_is_the_larger_of_the_columns_own_features(position):
    """Each column normalized alone, then the maximum, feature by feature:
    bit for bit, since it is the same arithmetic in the same order."""
    both = actor_observation(columns(ERRORS, SECOND), at(), position, per_variable())
    first = actor_observation(columns(ERRORS), at(), position, per_variable())
    second = actor_observation(columns(SECOND), at(), position, per_variable())
    assert np.array_equal(both[ERROR_FEATURES], np.maximum(first[ERROR_FEATURES], second[ERROR_FEATURES]))
    assert np.array_equal(both[OWN_LEVEL:], first[OWN_LEVEL:])


def test_the_deciding_column_changes_from_feature_to_feature():
    """Cell B's own error comes from column 1 and its x-min face from
    column 0."""
    first = actor_observation(columns(ERRORS), at(), 4, per_variable())
    second = actor_observation(columns(SECOND), at(), 4, per_variable())
    assert second[OWN_ERROR] > first[OWN_ERROR]
    assert first[FACE_ERRORS][0] > second[FACE_ERRORS][0]


def test_each_columns_worst_element_reads_minus_log10_alpha():
    cache = columns(ERRORS, SECOND)
    own = [actor_observation(cache, at(), p, per_variable())[OWN_ERROR] for p in range(5)]
    top = -math.log10(ALPHA)
    assert own[3] == pytest.approx(top, abs=1e-6)  # column 0's worst
    assert own[0] == pytest.approx(top, abs=1e-6)  # column 1's worst
    assert max(own[1], own[2], own[4]) < top - 0.5


@pytest.mark.parametrize("scale", [1e-6, 1e6])
@pytest.mark.parametrize("column", [0, 1])
def test_one_columns_scale_changes_no_feature(column, scale):
    """What tells this rule from a raw maximum over columns (T3). To a
    tolerance: a power of ten is not exact in binary."""
    variables = [ERRORS, SECOND]
    scaled = list(variables)
    scaled[column] = variables[column] * scale
    for position in range(5):
        base = actor_observation(columns(*variables), at(), position, per_variable())
        moved = actor_observation(columns(*scaled), at(), position, per_variable())
        assert moved.tolist() == pytest.approx(base.tolist(), abs=1e-6), f"position {position}"


def test_a_raw_maximum_over_columns_would_read_differently():
    """Under a raw maximum the larger variable decides everything: cell B's
    x-min face would read log10(1 / (alpha * 1e3)) = -2. It reads 1."""
    raw = np.maximum(ERRORS, SECOND)
    by_raw_maximum = actor_observation(instance(raw), at(), 4, config())[FACE_ERRORS][0]
    by_the_rule = actor_observation(columns(ERRORS, SECOND), at(), 4, per_variable())[FACE_ERRORS][0]
    assert by_raw_maximum == pytest.approx(-2.0, abs=1e-6)
    assert by_the_rule == pytest.approx(1.0, abs=1e-6)


def test_the_order_of_the_columns_changes_nothing():
    for position in range(5):
        forward = actor_observation(columns(ERRORS, SECOND), at(), position, per_variable())
        backward = actor_observation(columns(SECOND, ERRORS), at(), position, per_variable())
        assert np.array_equal(forward, backward)


@pytest.mark.parametrize("zero_first", [False, True], ids=["zero second", "zero first"])
def test_a_zero_column_drops_out(zero_first):
    """Bit for bit (T4): an all-zero column is at the floor everywhere, and
    the floor never wins a maximum."""
    variables = (np.zeros(5), ERRORS) if zero_first else (ERRORS, np.zeros(5))
    for position in range(5):
        with_zero = actor_observation(columns(*variables), at(), position, per_variable())
        alone = actor_observation(columns(ERRORS), at(), position, per_variable())
        assert np.array_equal(with_zero, alone)


def test_a_repeated_column_changes_nothing():
    for position in range(5):
        twice = actor_observation(columns(ERRORS, ERRORS), at(), position, per_variable())
        once = actor_observation(columns(ERRORS), at(), position, per_variable())
        assert np.array_equal(twice, once)


@pytest.mark.parametrize("normalization", NORMALIZATIONS)
def test_one_column_under_per_variable_is_the_scalar_vector(normalization):
    """A backend with one variable gives one column under either reduction,
    and dynamo accepts it (T5)."""
    for position in range(5):
        one = actor_observation(columns(ERRORS), at(), position, per_variable(normalization))
        scalar = actor_observation(instance(), at(), position, config(normalization))
        assert np.array_equal(one, scalar)


def test_three_equal_columns_put_every_element_at_the_top():
    """Every element is the worst of every column, so every error feature
    reads -log10(alpha)."""
    cache = RoundCache(
        state=instance_state(), errors={"ind": np.ones((5, 3))}, indicators=("ind",), reduction="per_variable"
    )
    for position in range(5):
        obs = actor_observation(cache, at(), position, per_variable())
        assert obs.shape == (17,)
        assert obs[ERROR_FEATURES].tolist() == pytest.approx([1.0] * 6, abs=1e-6)


# --- Every vector lies within its bounds (T7) ------------------------------------

@pytest.mark.parametrize("normalization", NORMALIZATIONS)
@pytest.mark.parametrize("errors", [ERRORS, ERRORS * 1e8, np.zeros(5), np.array([0.0, 1e-300, 1e300, 1.0, 0.0])],
                         ids=["the instance", "large", "all zero", "extreme"])
def test_every_instance_vector_is_inside_its_bounds(errors, normalization):
    cfg = config(normalization)
    for position in range(5):
        for committed in (1, 5, 40):
            assert inside(actor_observation(instance(errors), at(committed), position, cfg), cfg)


@pytest.mark.parametrize("alpha", [0.007, 0.1, 0.3, 0.9])
def test_every_two_column_vector_is_inside_its_bounds(alpha):
    cfg = per_variable(alpha=alpha)
    cache = columns(ERRORS, SECOND)
    for position in range(5):
        for round_index in (1, 2):
            assert inside(actor_observation(cache, at(round_index=round_index), position, cfg), cfg)


def test_the_worst_element_sits_on_the_upper_bound():
    cfg = config()
    _, high = feature_bounds(2, cfg)
    obs = actor_observation(instance(), at(), 3, cfg)
    assert obs[OWN_ERROR] == high[OWN_ERROR]


# --- On the mock: a reference built from the geometry alone (S11) ----------------

def _containing(state: SolverState, point: np.ndarray) -> int:
    """The position of the element a point lies strictly inside."""
    inside = (np.abs(state.centroid - point) < state.h / 2).all(axis=1)
    assert inside.sum() == 1
    return int(np.flatnonzero(inside)[0])


def _reference(cache: RoundCache, round_state: RoundState, position: int, cfg: ObservationConfig) -> list[float]:
    """The observation, found without reading neighbors or family.

    Each face is probed at two points just outside it, a quarter of the way
    in from either end, wrapped on a periodic axis; a probe that leaves a
    walled axis is a wall. The family is every element of the same level
    whose centroid lies in the same parent cell.
    """
    state = cache.state
    errors = cache.errors[cfg.indicator][:, 0]
    e_inf = float(errors.max())
    c, h = state.centroid[position], state.h[position]
    lo, extent = state.domain_lo, state.domain_hi - state.domain_lo

    def normalize(e: float) -> float:
        if cfg.normalization == "dynamo":
            return float(normalized_error(e, cfg.alpha, e_inf))
        if e <= 0.0 or e_inf <= 0.0:
            return -cfg.floor_decades
        return max(float(np.log10(e) - np.log10(cfg.alpha * e_inf)), -cfg.floor_decades)

    face_errors, face_levels, walls = [], [], []
    for axis in range(2):
        other = 1 - axis
        for s in (-1, 1):
            found = set()
            for t in (-1, 1):
                probe = c.copy()
                probe[axis] += s * (h[axis] / 2 + 1e-6 * extent[axis])
                probe[other] += t * h[other] / 4
                if not lo[axis] <= probe[axis] <= lo[axis] + extent[axis]:
                    if not state.periodic[axis]:
                        continue
                    probe[axis] = lo[axis] + (probe[axis] - lo[axis]) % extent[axis]
                found.add(_containing(state, probe))
            if found:
                face_errors.append(max(float(errors[q]) for q in found))
                levels = {int(state.level[q]) for q in found}
                assert len(levels) == 1
                face_levels.append(levels.pop())
                walls.append(0.0)
            else:
                face_errors.append(float(errors[position]))
                face_levels.append(int(state.level[position]))
                walls.append(1.0)

    level = int(state.level[position])
    parent_cell = np.floor((state.centroid - lo) / (2 * state.h))
    same = (state.level == level) & (parent_cell == parent_cell[position]).all(axis=1)
    siblings = np.flatnonzero(same & np.isclose(state.h, h).all(axis=1))
    complete = level > 0 and len(siblings) == 4
    family_error = float(errors[siblings].max()) if complete else float(errors[position])

    return [
        normalize(float(errors[position])),
        *(normalize(e) for e in face_errors),
        normalize(family_error),
        level / state.max_level,
        *(lv / state.max_level for lv in face_levels),
        *walls,
        round_state.committed / cfg.element_budget,
        round_state.round_index / round_state.n_rounds,
    ]


def _two_cell_mesh(nvar: int = 1) -> tuple[MockQuadBackend, SolverState]:
    """The mock's own five-element mesh: cells A and B, A refined."""
    mock = MockQuadBackend(
        nx=2, ny=1, periodic=(True, False), lo=(0.0, 0.0), hi=(2.0, 1.0), max_level=2, nvar=nvar, dt_base=0.25,
        rotate_cells=False,
    )
    state = mock.reset({"x0": 0.5, "y0": 0.5, "width": 0.25}, amr_freq=4, initial_level=0)
    state, _ = mock.apply_marks(np.where(state.centroid[:, 0] < 1.0, REFINE, HOLD).astype(np.int8))
    return mock, state


def _graded_mesh(periodic: tuple[bool, bool], nvar: int = 1) -> tuple[MockQuadBackend, SolverState]:
    """A 4x4 grid at level 1 with the cells near the feature refined once
    more, the rotation on: levels 1 and 2, with 2:1 faces between them."""
    mock = MockQuadBackend(nx=4, ny=4, periodic=periodic, max_level=2, nvar=nvar, dt_base=0.25, rotate_cells=True)
    ic = {"x0": 0.3, "y0": 0.6, "width": 0.2}
    state = mock.reset(ic, amr_freq=4, initial_level=1)
    near = np.hypot(state.centroid[:, 0] - ic["x0"], state.centroid[:, 1] - ic["y0"]) < 0.15
    assert near.any()
    state, _ = mock.apply_marks(np.where(near, REFINE, HOLD).astype(np.int8))
    return mock, state


MESHES = {
    "two cells": _two_cell_mesh,
    "4x4 periodic": lambda nvar=1: _graded_mesh((True, True), nvar),
    "4x4 walled": lambda nvar=1: _graded_mesh((False, False), nvar),
    "4x4 mixed": lambda nvar=1: _graded_mesh((True, False), nvar),
}


@pytest.mark.parametrize("normalization", NORMALIZATIONS)
@pytest.mark.parametrize("indicator", ["ind_a", "ind_b"])
@pytest.mark.parametrize("mesh", list(MESHES))
def test_the_builder_agrees_with_the_geometric_reference(mesh, indicator, normalization):
    mock, state = MESHES[mesh]()
    cache = read_round(mock, state, ("ind_a", "ind_b"), "scalar")
    cfg = config(normalization, indicator=indicator, element_budget=64)
    round_state = at(committed=state.n_active)
    for position in range(state.n_active):
        obs = actor_observation(cache, round_state, position, cfg)
        expected = _reference(cache, round_state, position, cfg)
        assert obs.tolist() == pytest.approx(expected, rel=1e-5, abs=1e-5), f"position {position}"


def test_the_mocks_five_elements_are_the_instances():
    """The hand-built instance and the mock's mesh agree on every feature
    that does not depend on the errors' values."""
    mock, state = _two_cell_mesh()
    cache = read_round(mock, state, ("ind_b",), "scalar")
    assert state.n_active == 5
    by_centroid = {tuple(c): p for p, c in enumerate(state.centroid.tolist())}
    hand = instance()
    for position in range(5):
        twin = by_centroid[tuple(hand.state.centroid[position].tolist())]
        ours = actor_observation(hand, at(), position, config())
        theirs = actor_observation(cache, at(), twin, config(indicator="ind_b"))
        assert np.array_equal(ours[OWN_LEVEL:], theirs[OWN_LEVEL:])


@pytest.mark.parametrize("mesh, expect_walls, expect_halves", [
    ("two cells", True, True),
    ("4x4 periodic", False, True),
    ("4x4 walled", True, True),
    ("4x4 mixed", True, True),
])
def test_the_meshes_exercise_what_they_are_for(mesh, expect_walls, expect_halves):
    """A periodic mesh has no wall bit set; every mesh has a face with two
    finer halves and both a complete and an absent family."""
    mock, state = MESHES[mesh]()
    cache = read_round(mock, state, ("ind_b",), "scalar")
    cfg = config(indicator="ind_b", element_budget=64)
    walls = np.array([actor_observation(cache, at(), p, cfg)[WALLS] for p in range(state.n_active)])
    assert bool(walls.any()) is expect_walls
    assert bool((state.neighbors[:, :, 1] >= 0).any()) is expect_halves
    assert (state.family[:, 0] >= 0).any() and (state.family[:, 0] < 0).any()


def test_the_mock_is_outside_the_1d_rules_regime():
    """Why relative exists. On the mock's level-0 instance alpha * e_inf is
    1.6: under dynamo the cell with the larger error gets the smaller value.
    Under relative it gets the larger."""
    mock = MockQuadBackend(
        nx=2, ny=1, periodic=(True, False), lo=(0.0, 0.0), hi=(2.0, 1.0), max_level=2, dt_base=0.25,
        rotate_cells=False,
    )
    state = mock.reset({"x0": 0.5, "y0": 0.5, "width": 0.25}, amr_freq=4, initial_level=0)
    cache = read_round(mock, state, ("ind_b",), "scalar")
    raw = cache.errors["ind_b"][:, 0]
    assert ALPHA * raw.max() == pytest.approx(1.6)
    big, small = int(raw.argmax()), int(raw.argmin())
    for normalization, larger_wins in (("dynamo", False), ("relative", True)):
        cfg = config(normalization, indicator="ind_b")
        own = [actor_observation(cache, at(committed=2), p, cfg)[OWN_ERROR] for p in (big, small)]
        assert bool(own[0] > own[1]) is larger_wins


# --- On the mock at nvar = 3 (T6, T7) --------------------------------------------

@pytest.mark.parametrize("indicator", ["ind_a", "ind_b"])
@pytest.mark.parametrize("mesh", list(MESHES))
def test_per_variable_at_nvar_3_gives_the_scalar_vector(mesh, indicator):
    """The mock's variables are one field at three scales, so per-column
    normalization reads all three alike and the vector is the scalar one.
    This cannot tell the rule from a raw maximum (the mock's scalar is its
    largest variable); test_one_columns_scale_changes_no_feature does."""
    mock3, state3 = MESHES[mesh](nvar=3)
    mock1, state1 = MESHES[mesh]()
    assert np.array_equal(state3.centroid, state1.centroid)
    three = read_round(mock3, state3, ("ind_a", "ind_b"), "per_variable")
    one = read_round(mock1, state1, ("ind_a", "ind_b"), "scalar")
    assert (three.n_comp, one.n_comp) == (3, 1)
    round_state = at(committed=state3.n_active)
    for position in range(state3.n_active):
        wide = actor_observation(three, round_state, position, per_variable(indicator=indicator, element_budget=64))
        narrow = actor_observation(one, round_state, position, config(indicator=indicator, element_budget=64))
        assert wide.tolist() == pytest.approx(narrow.tolist(), abs=1e-5), f"position {position}"


@pytest.mark.parametrize("normalization", NORMALIZATIONS)
@pytest.mark.parametrize("mesh", list(MESHES))
def test_on_the_mock_every_vector_is_inside_its_bounds(mesh, normalization):
    mock, state = MESHES[mesh]()
    cache = read_round(mock, state, ("ind_a", "ind_b"), "scalar")
    for indicator in ("ind_a", "ind_b"):
        cfg = config(normalization, indicator=indicator, element_budget=64)
        for position in range(state.n_active):
            assert inside(actor_observation(cache, at(committed=state.n_active), position, cfg), cfg)


# --- At ndim = 1, against the sequential core's observation (S12) -----------------

LEVELS_1D = [0, 1, 2, 3]
LEFT_1D = [3, 0, 1, 2]
RIGHT_1D = [1, 2, 3, 0]
ERRORS_1D = np.array([0.5, 0.1, 0.01, 0.001])


def _state_1d() -> SolverState:
    """Four elements on a periodic line, in the v2 schema. As in
    tests/agent/test_observation.py, only the arrays an observation reads
    are meaningful."""
    return SolverState(
        level=np.array(LEVELS_1D, dtype=np.int32),
        neighbors=np.array([[[left], [right]] for left, right in zip(LEFT_1D, RIGHT_1D, strict=True)], dtype=np.int32),
        neighbor_wrap=np.zeros((4, 2), dtype=bool),
        family=np.full((4, 2), -1, dtype=np.int32),
        can_refine=np.ones(4, dtype=bool),
        can_coarsen=np.zeros(4, dtype=bool),
        centroid=np.array([[0.125], [0.375], [0.625], [0.875]]),
        h=np.full((4, 1), 0.25),
        attributes=np.zeros((4, 0)),
        n_active=4,
        ndim=1,
        nvar=1,
        max_level=3,
        mesh_epoch=1,
        time=0.0,
        interval_T=1.0,
        dt_base=0.05,
        amr_freq=20,
        dt_stable=0.05,
        domain_lo=np.array([0.0]),
        domain_hi=np.array([1.0]),
        periodic=np.array([True]),
    )


def _sequential_state() -> SequentialState:
    """The same four elements in the 1D schema the sequential core reads."""
    level = np.asarray(LEVELS_1D)
    return SequentialState(
        element_id=np.arange(1, 5),
        level=level,
        left=np.asarray(LEFT_1D),
        right=np.asarray(RIGHT_1D),
        sibling=np.full(4, -1),
        can_refine=level < 3,
        can_coarsen=np.zeros(4, dtype=bool),
        n_active=4,
        max_level=3,
        stable_dt=0.01,
        wave_speed=1.0,
        domain_length=2.0,
    )


@pytest.mark.parametrize("round_number", [1, 2, 3])
@pytest.mark.parametrize("position", range(4))
def test_at_1d_under_dynamo_the_builder_reproduces_the_sequential_observation(position, round_number):
    """With the committed count at n_active, as at a round's start, every
    slot of the sequential 8-vector appears in the 11-vector, bit for bit.
    The three that are new at 1D: the family maximum (its own, with no
    family) and two wall bits (0 on a periodic line)."""
    cache = RoundCache(
        state=_state_1d(), errors={"ind": ERRORS_1D.reshape(4, 1).copy()}, indicators=("ind",), reduction="scalar"
    )
    new = actor_observation(
        cache, RoundState(round_index=round_number, n_rounds=3, committed=4), position, config("dynamo")
    )
    old = build_observation(
        _sequential_state(), ERRORS_1D, position, alpha=ALPHA, element_budget=BUDGET, round_number=round_number
    )
    names = feature_names(1)
    assert new.shape == (11,) and old.shape == (8,)
    picked = [names.index(n) for n in
              ("error", "error_x_min", "error_x_max", "level", "level_x_min", "level_x_max", "resource",
               "round_progress")]
    assert np.array_equal(new[picked], old)
    assert new[names.index("error_family_max")] == old[0]
    assert new[[names.index("wall_x_min"), names.index("wall_x_max")]].tolist() == [0.0, 0.0]
