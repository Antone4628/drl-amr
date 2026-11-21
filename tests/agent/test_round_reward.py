"""Tests for the reward of the 2D agent core: agent/round_reward.py, with
the signed distance it reads (zone_distance, agent/round_queue.py) and the
overshoot (agent/round_commit.py).

AGENT_2D_ROADMAP item 2.6, signals R2-R14. Eleven groups:

  - zone_distance on the five-element instance of the explanation file's
    section 2.6.2, its sign, and round_priority as its size (R2);
  - a family's distance is its largest member's (R3);
  - local_reward: the instance's table, cell by cell (R4), the three walks
    (R5), round 2 under the interval's thresholds (R6), a family below
    e_min (R7), two columns (R9), and at ndim = 1;
  - global_reward on the instance's interval, its guards and its
    normalization (R8);
  - against the sequential core's agent/reward.py, at one column under the
    1D rule (R10);
  - scale (R11);
  - purity and the vocabulary (R12);
  - overshoot;
  - on the mock quad backend: the instance's interval and whole intervals
    of random masked actions (R13);
  - what the functions reject (R14).

Records are written as text, one letter per position: C, H and R for the
marks and a dot for an undecided position.

Run: pytest tests/agent/test_round_reward.py -v
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from agent import reward as shared
from agent.round_cache import RoundCache, read_round
from agent.round_commit import commit, overshoot, start_round
from agent.round_mask import ACTION_COARSEN, ACTION_HOLD, ACTION_REFINE, COARSEN_RULES, action_mask
from agent.round_observation import ObservationConfig
from agent.round_queue import next_undecided, round_order, round_priority, round_thresholds, zone_distance
from agent.round_reward import global_reward, local_reward
from agent.round_state import UNDECIDED, RoundState
from backends.mock_quad.backend import MockQuadBackend
from contract.backend import BackendFailure
from contract.checks import training_only_params
from contract.solver_contract import COARSEN, HOLD, REFINE
from contract.state import SolverState

RULES = ("relative", "dynamo")
FAMILY_RULES = ("first_decider", "any_coarsen")
LETTER = {".": UNDECIDED, "C": COARSEN, "H": HOLD, "R": REFINE}
ACTIONS = (ACTION_COARSEN, ACTION_HOLD, ACTION_REFINE)
# Cell B's position on the instance; positions 0-3 are cell A's children.
B = 4
BETA = 2
# The instance's errors, section 2.6.2: round 1's column, and the second.
C0 = (2e-3, 3e-2, 3e-2, 1.0, 1e-4)
C1 = (1e3, 1.0, 10.0, 0.1, 30.0)
# A family all below e_min, with cell B holding the round's largest error.
LOW = (2e-4, 3e-3, 3e-3, 5e-3, 1.0)
# The interval's accumulated errors on the five-element mesh.
ACC = (2e-3, 3e-2, 0.4, 1.0, 1e-4)
# The interval's thresholds on the instance: alpha and alpha squared.
PAIR = (np.array([0.1]), np.array([0.1**2]))
PAIR2 = (np.array([0.1, 100.0]), np.array([0.1**2, 10.0]))
P_UR, P_OR, P_CR = 10.0, 5.0, 2.0
W = dict(p_ur=P_UR, p_or=P_OR, p_cr=P_CR)
G = dict(p_ur=P_UR, p_or=P_OR)
SCALES = (1e-6, 10.0, 100.0, 1e6)
# Round 1's distances on the instance.
D0 = (-math.log10(5.0), 0.0, 0.0, 1.0, -2.0)


def letters(decisions: np.ndarray) -> str:
    """A decision record as letters."""
    by_value = {value: letter for letter, value in LETTER.items()}
    return "".join(by_value[v] for v in decisions.tolist())


def instance_state() -> SolverState:
    """The five-element mesh, built by hand: cell A's children at level 1
    (positions 0-3 in child order) and cell B at level 0 (position 4), on
    [0,2]x[0,1], periodic in x and walled in y."""
    level = np.array([1, 1, 1, 1, 0], dtype=np.int32)
    return SolverState(
        level=level,
        neighbors=np.array(
            [
                [[4, -1], [1, -1], [-1, -1], [2, -1]],
                [[0, -1], [4, -1], [-1, -1], [3, -1]],
                [[4, -1], [3, -1], [0, -1], [-1, -1]],
                [[2, -1], [4, -1], [1, -1], [-1, -1]],
                [[1, 3], [0, 2], [-1, -1], [-1, -1]],
            ],
            dtype=np.int32,
        ),
        neighbor_wrap=np.zeros((5, 4), dtype=bool),
        family=np.array([[0, 1, 2, 3]] * 4 + [[-1, -1, -1, -1]], dtype=np.int32),
        can_refine=level < 2,
        can_coarsen=np.array([True, True, True, True, False]),
        centroid=np.array([[0.25, 0.25], [0.75, 0.25], [0.25, 0.75], [0.75, 0.75], [1.5, 0.5]]),
        h=np.array([[0.5, 0.5]] * 4 + [[1.0, 1.0]]),
        attributes=np.zeros((5, 0)),
        n_active=5,
        ndim=2,
        nvar=1,
        max_level=2,
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


def _state_1d() -> SolverState:
    """Four elements on a periodic line. Positions 1 and 2 are a sibling
    pair at the finest level; positions 0 and 3 have no sibling."""
    level = np.array([1, 2, 2, 1], dtype=np.int32)
    return SolverState(
        level=level,
        neighbors=np.array([[[3], [1]], [[0], [2]], [[1], [3]], [[2], [0]]], dtype=np.int32),
        neighbor_wrap=np.zeros((4, 2), dtype=bool),
        family=np.array([[-1, -1], [1, 2], [1, 2], [-1, -1]], dtype=np.int32),
        can_refine=level < 2,
        can_coarsen=np.array([False, True, True, False]),
        centroid=np.array([[0.125], [0.3125], [0.4375], [0.75]]),
        h=np.array([[0.25], [0.125], [0.125], [0.5]]),
        attributes=np.zeros((4, 0)),
        n_active=4,
        ndim=1,
        nvar=1,
        max_level=2,
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


def col(*columns) -> np.ndarray:
    """An errors array, (n, n_comp) float64, from its columns."""
    return np.stack([np.array(column, dtype=np.float64) for column in columns], axis=1)


def cache_of(*columns, state: SolverState | None = None) -> RoundCache:
    """A cache on the instance (or a given state) holding one indicator,
    "e", with the given columns."""
    reduction = "scalar" if len(columns) == 1 else "per_variable"
    return RoundCache(
        state=instance_state() if state is None else state, errors={"e": col(*columns)}, indicators=("e",),
        reduction=reduction,
    )


def config(normalization: str = "relative", reduction: str = "scalar", **changes) -> ObservationConfig:
    """The instance's configuration: alpha 0.1, a floor of 8 decades."""
    fields = dict(
        indicator="e", reduction=reduction, alpha=0.1, element_budget=8, normalization=normalization, floor_decades=8.0
    )
    fields.update(changes)
    return ObservationConfig(**fields)


def distance_of(*columns, pair=PAIR, normalization: str = "relative") -> np.ndarray:
    """zone_distance of the given columns under a threshold pair."""
    reduction = "scalar" if len(columns) == 1 else "per_variable"
    return zone_distance(col(*columns), config(normalization, reduction), *pair)


def scaled(column, factor: float) -> list[float]:
    return [factor * e for e in column]


# --- zone_distance (R2) ------------------------------------------------------------

@pytest.mark.parametrize("rule", RULES)
def test_the_instances_distances(rule):
    """Section 2.6.2's first table. Position 0 is a factor of 5 below
    e_min, positions 1 and 2 are in the zone, position 3 is one decade
    above e_max and B two decades below e_min."""
    distance = distance_of(C0, normalization=rule)
    assert distance.tolist() == pytest.approx(list(D0), abs=1e-12)
    assert distance[1] == 0.0 and distance[2] == 0.0
    assert distance.dtype == np.float64 and distance.shape == (5,)


def test_the_two_rules_give_the_same_distances_where_e_inf_is_one():
    assert np.array_equal(distance_of(C0), distance_of(C0, normalization="dynamo"))


def test_the_instances_distances_with_two_columns():
    """Positions 0 and 3 are each one decade above e_max in one column;
    nothing is below e_min in both, so nothing is negative."""
    distance = distance_of(C0, C1, pair=PAIR2)
    assert distance.tolist() == pytest.approx([1.0, 0.0, 0.0, 1.0, 0.0], abs=1e-12)
    assert (distance >= 0.0).all()


def test_the_sign_is_the_class():
    """Positive above e_max, negative below e_min, zero between, and zero
    exactly on either threshold."""
    distance = zone_distance(col([1.0, 0.25, 0.1, 0.0625, 0.01]), config(), np.array([0.25]), np.array([0.0625]))
    assert distance[0] > 0.0 and distance[4] < 0.0
    assert distance[1] == 0.0 and distance[2] == 0.0 and distance[3] == 0.0
    assert distance[0] == pytest.approx(math.log10(4.0)) and distance[4] == pytest.approx(-math.log10(6.25))


@pytest.mark.parametrize("rule", RULES)
@pytest.mark.parametrize("columns, pair", [((C0,), PAIR), ((scaled(C0, 0.1),), PAIR), ((LOW,), PAIR)])
def test_the_priority_is_the_size_of_the_distance(columns, pair, rule):
    """round_priority is zone_distance without its sign, bit for bit."""
    cache, cfg = cache_of(*columns), config(rule)
    distance = zone_distance(cache.errors["e"], cfg, *pair)
    assert np.array_equal(round_priority(cache, cfg, *pair), np.abs(distance))


def test_the_priority_is_the_size_of_the_distance_at_two_columns():
    cache, cfg = cache_of(C0, C1), config(reduction="per_variable")
    assert np.array_equal(round_priority(cache, cfg, *PAIR2), np.abs(zone_distance(cache.errors["e"], cfg, *PAIR2)))


def test_no_priority_is_a_negative_zero():
    """An over-refined element exactly on the floor has a distance of
    minus zero or zero; the priority's sign bit is never set."""
    cache, cfg = cache_of(C0), config()
    assert not np.signbit(round_priority(cache, cfg, *PAIR)).any()


def test_under_refined_wins_where_the_thresholds_cross():
    """With e_max 10 and e_min 100, an error of 50 is above the one and
    below the other and reads positive; an error of exactly 10 is not above
    e_max and reads one decade below e_min."""
    distance = zone_distance(col([50.0, 5.0, 500.0, 10.0]), config("dynamo"), np.array([10.0]), np.array([100.0]))
    assert distance.tolist() == pytest.approx([math.log10(5.0), -math.log10(20.0), math.log10(50.0), -1.0])


def test_a_zero_error_reads_the_floor():
    """7 decades below e_min under relative (the configuration's floor of
    8 below e_max) and 28 under dynamo (the sequential floor, 1e-30)."""
    zero = (0.0, 3e-2, 3e-2, 1.0, 1e-4)
    assert distance_of(zero)[0] == pytest.approx(-7.0)
    assert distance_of(zero, normalization="dynamo")[0] == pytest.approx(-28.0)


@pytest.mark.parametrize("rule", RULES)
def test_dead_thresholds_leave_everything_neutral(rule):
    distance = zone_distance(col(C0), config(rule), np.array([0.0]), np.array([0.0]))
    assert (distance == 0.0).all()


def test_a_dead_column_takes_no_part():
    """A second column of zeros has zero thresholds and changes nothing."""
    pair = (np.array([0.1, 0.0]), np.array([0.1**2, 0.0]))
    assert np.array_equal(distance_of(C0, [0.0] * 5, pair=pair), distance_of(C0))


def test_over_refined_takes_the_nearest_column_and_under_refined_the_farthest():
    pair = (np.array([0.1, 0.1]), np.array([0.01, 0.01]))
    below = distance_of([1e-3, 1.0], [1e-5, 1.0], pair=pair)
    above = distance_of([10.0, 0.05], [100.0, 0.05], pair=pair)
    mixed = distance_of([10.0, 0.05], [1e-6, 0.05], pair=pair)
    assert below[0] == pytest.approx(-1.0) and above[0] == pytest.approx(3.0) and mixed[0] == pytest.approx(2.0)


def test_the_rows_need_not_be_a_caches():
    """Any number of rows, a mesh's or not: the global reward hands in an
    interval's accumulated errors."""
    assert zone_distance(np.zeros((0, 1)), config(), *PAIR).shape == (0,)
    assert zone_distance(col([1.0]), config(), *PAIR).tolist() == [1.0]
    many = zone_distance(col(C0 * 3), config(), *PAIR)
    assert many.shape == (15,) and np.array_equal(many[:5], many[10:])


def test_the_distance_is_a_new_array_and_leaves_its_inputs_alone():
    errors = col(C0)
    kept = errors.copy()
    errors.flags.writeable = False
    first, second = zone_distance(errors, config(), *PAIR), zone_distance(errors, config(), *PAIR)
    assert first is not second and np.array_equal(first, second) and first.flags.writeable
    assert np.array_equal(errors, kept) and PAIR[0].tolist() == [0.1]


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("rule", RULES)
def test_every_distance_is_finite(rule, seed):
    rng = np.random.default_rng(seed)
    for _ in range(20):
        column = 10.0 ** rng.uniform(-12.0, 3.0, size=5)
        column[rng.random(5) < 0.3] = 0.0
        cache, cfg = cache_of(column), config(rule)
        assert np.isfinite(zone_distance(cache.errors["e"], cfg, *round_thresholds(cache, cfg, BETA))).all()


# --- A family's distance is its largest member's (R3) ------------------------------

@pytest.mark.parametrize("n_comp", [1, 2, 3])
@pytest.mark.parametrize("seed", range(5))
def test_a_familys_distance_is_its_largest_members(seed, n_comp):
    """The local reward classifies a family coarsen on the family's largest
    errors, column by column. That is exactly the largest of the members'
    distances, so one distance array per round serves every decision."""
    rng = np.random.default_rng(seed)
    cfg = config(reduction="scalar" if n_comp == 1 else "per_variable")
    for _ in range(40):
        errors = 10.0 ** rng.uniform(-9.0, 1.0, size=(4, n_comp))
        errors[rng.random((4, n_comp)) < 0.1] = 0.0
        e_inf = 10.0 ** rng.uniform(-2.0, 2.0, size=n_comp)
        pair = (0.1 * e_inf, 0.01 * e_inf)
        members = zone_distance(errors, cfg, *pair)
        family = zone_distance(errors.max(axis=0, keepdims=True), cfg, *pair)
        assert family[0] == members.max()


def test_the_three_classes_of_a_family():
    """One member above the zone makes the family under-refined; all below
    make it over-refined by its nearest member; anything else is neutral."""
    assert distance_of(C0)[:4].max() == 1.0
    assert distance_of(LOW)[:4].max() == pytest.approx(-math.log10(2.0))
    assert distance_of((2e-4, 3e-3, 3e-2, 5e-3, 1.0))[:4].max() == 0.0


# --- local_reward: the instance's table (R4) ---------------------------------------

def paid(distance, position, action, rule="first_decider", state=None, **weights) -> float:
    state = instance_state() if state is None else state
    return local_reward(state, distance, position, action, rule, **(weights or W))


@pytest.mark.parametrize("rule", FAMILY_RULES)
@pytest.mark.parametrize(
    "position, coarsen, refine",
    [(0, -10.0, -5.0 * math.log10(5.0)), (1, -10.0, 0.0), (2, -10.0, 0.0), (3, -10.0, 0.0)],
)
def test_the_table_under_a_family_rule(position, coarsen, refine, rule):
    """Section 2.6.2's first table. A coarsen is the family's and is paid
    on position 3, a decade above e_max, whoever decides: -p_ur. A refine
    is paid on the element's own distance."""
    distance = distance_of(C0)
    assert paid(distance, position, ACTION_COARSEN, rule) == pytest.approx(coarsen)
    assert paid(distance, position, ACTION_HOLD, rule) == 0.0
    assert paid(distance, position, ACTION_REFINE, rule) == pytest.approx(refine)


@pytest.mark.parametrize(
    "position, coarsen, refine",
    [(0, 2.0 * math.log10(5.0), -5.0 * math.log10(5.0)), (1, 0.0, 0.0), (2, 0.0, 0.0), (3, -10.0, 0.0)],
)
def test_the_table_under_unanimous(position, coarsen, refine):
    """A coarsen is one sibling's vote and is paid on its own distance:
    position 0, over-refined, earns p_cr where its family coarsen cost
    p_ur."""
    distance = distance_of(C0)
    assert paid(distance, position, ACTION_COARSEN, "unanimous") == pytest.approx(coarsen)
    assert paid(distance, position, ACTION_HOLD, "unanimous") == 0.0
    assert paid(distance, position, ACTION_REFINE, "unanimous") == pytest.approx(refine)


@pytest.mark.parametrize("rule", COARSEN_RULES)
def test_the_table_for_cell_b(rule):
    """Two decades below e_min: a refine costs 2 * p_or under every rule."""
    distance = distance_of(C0)
    assert paid(distance, B, ACTION_HOLD, rule) == 0.0
    assert paid(distance, B, ACTION_REFINE, rule) == pytest.approx(-10.0)


def test_a_decider_is_paid_for_the_sibling_it_cannot_see():
    """Position 0's own distance would earn +1.398; its family coarsen
    costs 10, because of position 3."""
    distance = distance_of(C0)
    assert paid(distance, 0, ACTION_COARSEN, "unanimous") == pytest.approx(1.39794, abs=1e-6)
    assert paid(distance, 0, ACTION_COARSEN, "first_decider") == pytest.approx(-10.0)


@pytest.mark.parametrize("rule", COARSEN_RULES)
@pytest.mark.parametrize("position", range(5))
def test_a_hold_is_never_paid(position, rule):
    for errors in (C0, LOW, scaled(C0, 0.1), (5.0, 5.0, 5.0, 5.0, 5.0)):
        assert paid(distance_of(errors), position, ACTION_HOLD, rule) == 0.0


@pytest.mark.parametrize("rule", COARSEN_RULES)
def test_a_refine_above_the_zone_is_not_rewarded(rule):
    assert paid(distance_of(C0), 3, ACTION_REFINE, rule) == 0.0


def test_each_weight_scales_its_own_cell():
    distance = distance_of(C0)
    weights = dict(p_ur=3.0, p_or=7.0, p_cr=11.0)
    assert paid(distance, 3, ACTION_COARSEN, "unanimous", **weights) == pytest.approx(-3.0)
    assert paid(distance, B, ACTION_REFINE, "unanimous", **weights) == pytest.approx(-14.0)
    assert paid(distance, 0, ACTION_COARSEN, "unanimous", **weights) == pytest.approx(11.0 * math.log10(5.0))
    assert paid(distance, 0, ACTION_COARSEN, "unanimous", p_ur=1.0, p_or=1.0, p_cr=0.0) == 0.0


def test_the_reward_is_a_python_float():
    distance = distance_of(C0)
    for action in ACTIONS:
        value = local_reward(
            instance_state(), distance, np.int64(0), np.int64(action), "first_decider", p_ur=10, p_or=5, p_cr=2
        )
        assert type(value) is float


# --- local_reward: the walks (R5) --------------------------------------------------

def walk(state, order, distance, rule, choices: dict[int, int]):
    """One round along an order, as next_undecided's docstring writes it,
    each asked position taking its choice (or holding) and paid at its
    commit. Returns the last round state, the positions asked, their
    rewards and the number skipped."""
    round_state = start_round(state, 1, 2)
    asked, rewards, skipped = [], [], 0
    position, cursor, passed = next_undecided(state, round_state, order, 0)
    while position is not None:
        skipped += passed
        action = choices.get(position, ACTION_HOLD)
        assert action_mask(state, round_state, position, rule)[action]
        rewards.append(local_reward(state, distance, position, action, rule, **W))
        round_state = commit(state, round_state, position, action, rule)
        asked.append(position)
        position, cursor, passed = next_undecided(state, round_state, order, cursor)
    return round_state, asked, rewards, skipped + passed


def instance_round():
    cache, cfg = cache_of(C0), config()
    return cache.state, round_order(cache, cfg, *PAIR), zone_distance(cache.errors["e"], cfg, *PAIR)


def test_walk_under_first_decider():
    """B holds; position 3 coarsens for the family and is paid -p_ur; the
    other three are skipped and paid nothing."""
    state, order, distance = instance_round()
    round_state, asked, rewards, skipped = walk(state, order, distance, "first_decider", {3: ACTION_COARSEN})
    assert asked == [4, 3] and skipped == 3
    assert rewards == pytest.approx([0.0, -10.0])
    assert letters(round_state.decisions) == "CCCCH" and round_state.committed == 2


def test_walk_under_any_coarsen_an_overwritten_hold_was_paid_nothing():
    """Position 3 holds and is paid 0; position 0's coarsen then takes the
    family, overwrites that hold and is paid once, on the family. Nothing
    paid earlier has to be taken back."""
    state, order, distance = instance_round()
    round_state, asked, rewards, skipped = walk(state, order, distance, "any_coarsen", {0: ACTION_COARSEN})
    assert asked == [4, 3, 0] and skipped == 2
    assert rewards == pytest.approx([0.0, 0.0, -10.0])
    assert letters(round_state.decisions) == "CCCCH"


def test_walk_under_unanimous_every_vote_is_paid_on_its_own():
    state, order, distance = instance_round()
    votes = {p: ACTION_COARSEN for p in range(4)}
    round_state, asked, rewards, skipped = walk(state, order, distance, "unanimous", votes)
    assert asked == [4, 3, 0, 1, 2] and skipped == 0
    assert rewards == pytest.approx([0.0, -10.0, 2.0 * math.log10(5.0), 0.0, 0.0])
    assert sum(rewards) == pytest.approx(-8.60206, abs=1e-6)
    assert letters(round_state.decisions) == "CCCCH" and round_state.committed == 2


def test_the_reward_does_not_depend_on_what_the_round_has_decided():
    """Nothing it reads moves within a round: the same call gives the same
    value before the commit, after it, and at the round's end."""
    state, order, distance = instance_round()
    before = local_reward(state, distance, 3, ACTION_COARSEN, "first_decider", **W)
    walk(state, order, distance, "first_decider", {3: ACTION_COARSEN})
    assert local_reward(state, distance, 3, ACTION_COARSEN, "first_decider", **W) == before


# --- Round 2 under the interval's thresholds (R6) ----------------------------------

def test_round_2_under_the_intervals_thresholds_the_family_coarsen_is_free():
    """Round 2's errors, a tenth of round 1's. Under the interval's pair
    position 3 has come down into the zone, so the family is neutral."""
    distance = distance_of(scaled(C0, 0.1))
    assert distance.tolist() == pytest.approx([-math.log10(50.0), -math.log10(10 / 3), -math.log10(10 / 3), 0.0, -3.0])
    assert distance[:4].max() == 0.0
    for position in range(4):
        assert paid(distance, position, ACTION_COARSEN) == 0.0


def test_round_2_under_its_own_thresholds_position_3_is_under_refined_again():
    """What the interval's thresholds avoid: a round's largest error is
    always one decade above its own e_max, so the family coarsen costs p_ur
    in every round, whatever the round before did."""
    tenth = cache_of(scaled(C0, 0.1))
    own = round_thresholds(tenth, config(), BETA)
    distance = zone_distance(tenth.errors["e"], config(), *own)
    assert distance.tolist() == pytest.approx(list(D0), abs=1e-9)
    assert paid(distance, 0, ACTION_COARSEN) == pytest.approx(-10.0)


@pytest.mark.parametrize("factor", [1.0, 0.3, 0.1, 1e-3])
def test_under_its_own_thresholds_a_rounds_largest_error_always_reads_one_decade(factor):
    cache = cache_of(scaled(C0, factor))
    distance = zone_distance(cache.errors["e"], config(), *round_thresholds(cache, config(), BETA))
    assert distance.max() == pytest.approx(1.0)


# --- A family below e_min (R7) -----------------------------------------------------

@pytest.mark.parametrize("rule", FAMILY_RULES)
@pytest.mark.parametrize("decider", range(4))
def test_a_family_below_e_min_is_paid_once_whoever_decides(decider, rule):
    """The nearest member is a factor of 2 below e_min: p_cr * log10(2)."""
    distance = distance_of(LOW)
    assert distance[:4].tolist() == pytest.approx(
        [-math.log10(50.0), -math.log10(10 / 3), -math.log10(10 / 3), -math.log10(2.0)]
    )
    assert paid(distance, decider, ACTION_COARSEN, rule) == pytest.approx(2.0 * math.log10(2.0))
    assert paid(distance, decider, ACTION_COARSEN, rule) == pytest.approx(0.60206, abs=1e-6)


def test_under_unanimous_a_merged_family_is_paid_per_vote():
    """Four votes, each on its own distance: ten times the family's one
    reward here. Local reward mass is not comparable across rules."""
    distance = distance_of(LOW)
    votes = [paid(distance, position, ACTION_COARSEN, "unanimous") for position in range(4)]
    assert votes == pytest.approx(
        [2 * math.log10(50.0), 2 * math.log10(10 / 3), 2 * math.log10(10 / 3), 2 * math.log10(2.0)]
    )
    assert sum(votes) == pytest.approx(6.091515, abs=1e-6)
    assert sum(votes) > 10 * paid(distance, 0, ACTION_COARSEN, "first_decider")


# --- Two columns (R9) --------------------------------------------------------------

def test_two_columns_over_refined_in_one_variable_is_not_over_refined():
    """Position 0 and cell B are below e_min in the first column and not
    in the second. Their refines, penalized at one column, cost nothing."""
    one, two = distance_of(C0), distance_of(C0, C1, pair=PAIR2)
    assert paid(one, 0, ACTION_REFINE) < 0.0 and paid(one, B, ACTION_REFINE) == pytest.approx(-10.0)
    assert paid(two, 0, ACTION_REFINE) == 0.0 and paid(two, B, ACTION_REFINE) == 0.0


def test_two_columns_the_family_coarsen_is_paid_on_the_worst_variable():
    two = distance_of(C0, C1, pair=PAIR2)
    assert two[:4].max() == pytest.approx(1.0)
    assert paid(two, 1, ACTION_COARSEN) == pytest.approx(-10.0)
    assert paid(two, 0, ACTION_COARSEN, "unanimous") == pytest.approx(-10.0)


# --- At ndim = 1 -------------------------------------------------------------------

def test_at_ndim_1_a_pair_coarsen_is_paid_on_the_pair():
    """A family is a pair. Position 1 is below e_min and its sibling above
    e_max: the pair coarsen costs p_ur, the lone vote earns p_cr."""
    state = _state_1d()
    distance = zone_distance(col([0.05, 1e-3, 1.0, 0.05]), config(), *PAIR)
    assert distance.tolist() == pytest.approx([0.0, -1.0, 1.0, 0.0])
    assert local_reward(state, distance, 1, ACTION_COARSEN, "first_decider", **W) == pytest.approx(-10.0)
    assert local_reward(state, distance, 1, ACTION_COARSEN, "unanimous", **W) == pytest.approx(2.0)
    assert local_reward(state, distance, 1, ACTION_REFINE, "first_decider", **W) == pytest.approx(-5.0)


# --- global_reward (R8) ------------------------------------------------------------

def g(errors, level, max_level=2, pair=PAIR, cfg=None, **weights) -> float:
    return global_reward(errors, level, max_level, cfg or config(), *pair, **(weights or G))


LEVEL = np.array([1, 1, 1, 1, 0], dtype=np.int32)


def test_the_intervals_global_reward():
    """Section 2.6.2: penalties 5 * log10(5) (position 0, below the zone at
    level 1), 10 * log10(4) and 10 (positions 2 and 3, above it below
    max_level), nothing for B (below the zone at level 0), over 5."""
    expected = -(5.0 * math.log10(5.0) + 10.0 * math.log10(4.0) + 10.0) / 5
    assert g(col(ACC), LEVEL) == pytest.approx(expected)
    assert g(col(ACC), LEVEL) == pytest.approx(-3.90309, abs=1e-6)


def test_over_refined_at_level_0_is_not_counted():
    """B could not be coarser. One level up it would cost 2 * p_or more."""
    assert g(col(ACC), np.array([1, 1, 1, 1, 1])) == pytest.approx(-5.90309, abs=1e-6)
    assert g(col(ACC), np.array([1, 1, 1, 1, 1])) == pytest.approx(g(col(ACC), LEVEL) - 10.0 / 5)


def test_under_refined_at_max_level_is_not_counted():
    """With max_level 1 the children could not refine: only position 0's
    over-refinement is left, 5 * log10(5) / 5."""
    assert g(col(ACC), LEVEL, max_level=1) == pytest.approx(-math.log10(5.0))
    assert g(col(ACC), LEVEL, max_level=1) == pytest.approx(-0.69897, abs=1e-6)


def test_a_mesh_that_fits_is_paid_zero():
    assert g(col((0.05,) * 5), LEVEL) == 0.0
    assert g(col(ACC), np.zeros(5, dtype=np.int32), max_level=0) == 0.0
    assert not math.copysign(1.0, g(col((0.05,) * 5), LEVEL)) < 0.0


def test_each_weight_scales_its_own_term():
    assert g(col(ACC), LEVEL, p_ur=10.0, p_or=0.0) == pytest.approx(-(10.0 * math.log10(4.0) + 10.0) / 5)
    assert g(col(ACC), LEVEL, p_ur=0.0, p_or=5.0) == pytest.approx(-math.log10(5.0))
    assert g(col(ACC), LEVEL, p_ur=0.0, p_or=0.0) == 0.0


def test_the_reward_is_the_mean_over_elements():
    """The same mesh twice is paid the same; five neutral elements added
    halve it. A smaller mesh is not better for being smaller."""
    acc, twice = col(ACC), np.concatenate([LEVEL, LEVEL])
    assert g(np.concatenate([acc, acc]), twice) == pytest.approx(g(acc, LEVEL))
    assert g(np.concatenate([acc, col((0.05,) * 5)]), twice) == pytest.approx(g(acc, LEVEL) / 2)


def test_the_global_reward_does_not_depend_on_the_order_of_the_elements():
    acc, order = col(ACC), np.array([3, 0, 4, 2, 1])
    assert g(acc[order], LEVEL[order]) == pytest.approx(g(acc, LEVEL))


def test_the_global_reward_at_two_columns():
    """Positions 0, 2 and 3 are above the zone in some column; B, below it
    in the first, is in it in the second and costs nothing."""
    two = col(ACC, C1)
    assert g(two, LEVEL, pair=PAIR2, cfg=config(reduction="per_variable")) == pytest.approx(
        -(10.0 + 10.0 * math.log10(4.0) + 10.0) / 5
    )


@pytest.mark.parametrize("rule", RULES)
@pytest.mark.parametrize("seed", range(6))
def test_the_global_reward_is_never_positive(seed, rule):
    rng = np.random.default_rng(seed)
    for _ in range(25):
        column = 10.0 ** rng.uniform(-10.0, 2.0, size=7)
        column[rng.random(7) < 0.2] = 0.0
        level = rng.integers(0, 3, size=7)
        e_inf = float(10.0 ** rng.uniform(-2.0, 0.0))
        value = g(col(column), level, pair=(np.array([0.1 * e_inf]), np.array([0.01 * e_inf])), cfg=config(rule))
        assert value <= 0.0 and math.isfinite(value)


def test_the_global_reward_is_a_python_float_and_zero_for_no_elements():
    assert type(global_reward(col(ACC), LEVEL, np.int64(2), config(), *PAIR, p_ur=10, p_or=5)) is float
    assert g(np.zeros((0, 1)), np.zeros(0, dtype=np.int32)) == 0.0


def test_levels_of_any_integer_dtype_are_taken():
    for dtype in (np.int32, np.int64, np.uint8):
        assert g(col(ACC), LEVEL.astype(dtype)) == pytest.approx(-3.90309, abs=1e-6)


# --- Against the sequential core's reward (R10) ------------------------------------

@pytest.mark.parametrize("action", ACTIONS)
@pytest.mark.parametrize("error", [1.0, 10.0, 0.05, 0.001, 1e-7, 0.0, 0.1, 0.01])
def test_at_one_column_under_dynamo_the_local_reward_is_the_shared_one(error, action):
    """The formula written twice, pinned: every class and action, on the
    thresholds of tests/agent/test_reward.py, with the element's own
    distance (the 1D reward has no family maximum)."""
    state = _state_1d()
    distance = zone_distance(col([error, 0.05, 0.05, 0.05]), config("dynamo"), np.array([0.1]), np.array([0.01]))
    new = local_reward(state, distance, 0, action, "unanimous", **W)
    assert new == pytest.approx(shared.local_reward(error, action, 0.1, 0.01, P_UR, P_OR, P_CR), abs=1e-12)


@pytest.mark.parametrize(
    "errors, level, expected",
    [
        ([0.05, 0.05, 0.05], [0, 1, 1], 0.0),
        ([1.0, 0.05], [0, 0], -5.0),
        ([1.0], [3], 0.0),
        ([0.001], [2], -5.0),
        ([0.001], [0], 0.0),
        ([1.0, 0.05, 0.05, 0.05], [0, 0, 0, 0], -2.5),
        ([1.0, 0.001, 0.05], [0, 2, 1], -5.0),
    ],
)
def test_at_one_column_under_dynamo_the_global_reward_is_the_shared_one(errors, level, expected):
    """The seven vectors of tests/agent/test_reward.py, max_level 3."""
    pair = (np.array([0.1]), np.array([0.01]))
    new = g(col(errors), np.array(level), max_level=3, pair=pair, cfg=config("dynamo"))
    assert new == pytest.approx(expected, abs=1e-12)
    old = shared.global_reward(np.array(errors), np.array(level), 0.1, 0.01, 3, P_UR, P_OR)
    assert new == pytest.approx(old, abs=1e-12)


@pytest.mark.parametrize("seed", range(5))
def test_at_one_column_under_dynamo_random_cases_agree_with_the_shared_reward(seed):
    """Zeros, dead thresholds and crossed thresholds among them."""
    rng = np.random.default_rng(seed)
    state = _state_1d()
    for _ in range(60):
        errors = 10.0 ** rng.uniform(-8.0, 0.5, size=4)
        errors[rng.random(4) < 0.15] = 0.0
        level = rng.integers(0, 3, size=4)
        e_max = float(10.0 ** rng.uniform(-3.0, 1.0)) if rng.random() > 0.05 else 0.0
        pair = (np.array([e_max]), np.array([e_max**2]))
        distance = zone_distance(col(errors), config("dynamo"), *pair)
        for position in range(4):
            for action in ACTIONS:
                new = local_reward(state, distance, position, action, "unanimous", **W)
                old = shared.local_reward(errors[position], action, e_max, e_max**2, P_UR, P_OR, P_CR)
                assert new == pytest.approx(old, abs=1e-12)
        new = g(col(errors), level, pair=pair, cfg=config("dynamo"))
        assert new == pytest.approx(shared.global_reward(errors, level, e_max, e_max**2, 2, P_UR, P_OR), abs=1e-12)


def test_where_the_two_part_a_zero_error_under_relative():
    """By design: under relative a zero error reads the configuration's
    floor, 7 decades below e_min, and its coarsen earns 14. The shared
    function floors at 1e-30 and pays 56 for the same vote."""
    distance = distance_of((0.0, 3e-2, 3e-2, 1.0, 1e-4))
    assert paid(distance, 0, ACTION_COARSEN, "unanimous") == pytest.approx(14.0)
    assert shared.local_reward(0.0, ACTION_COARSEN, 0.1, 0.1**2, P_UR, P_OR, P_CR) == pytest.approx(56.0)


# --- Scale (R11) -------------------------------------------------------------------

@pytest.mark.parametrize("factor", SCALES)
def test_relative_a_change_of_scale_changes_no_reward(factor):
    """Every error multiplied by one constant, the thresholds taken from
    the scaled round: the same distances, local rewards and global reward."""
    cfg = config()
    cache = cache_of(scaled(C0, factor))
    pair = round_thresholds(cache, cfg, BETA)
    distance = zone_distance(cache.errors["e"], cfg, *pair)
    assert distance.tolist() == pytest.approx(list(D0), abs=1e-9)
    reference = distance_of(C0)
    for position in range(4):
        for action in ACTIONS:
            assert paid(distance, position, action) == pytest.approx(paid(reference, position, action), abs=1e-8)
    assert g(col(scaled(ACC, factor)), LEVEL, pair=pair) == pytest.approx(-3.90309, abs=1e-6)


# --- Purity and the vocabulary (R12) -----------------------------------------------

def test_the_local_reward_changes_nothing_and_takes_read_only_arrays():
    state, distance = instance_state(), distance_of(C0)
    kept = distance.copy()
    distance.flags.writeable = False
    first = local_reward(state, distance, 0, ACTION_COARSEN, "first_decider", **W)
    assert local_reward(state, distance, 0, ACTION_COARSEN, "first_decider", **W) == first
    assert np.array_equal(distance, kept) and state.family.tolist()[0] == [0, 1, 2, 3]


def test_the_global_reward_changes_nothing_and_takes_read_only_arrays():
    acc, level = col(ACC), LEVEL.copy()
    kept_acc, kept_level = acc.copy(), level.copy()
    acc.flags.writeable = False
    level.flags.writeable = False
    first = g(acc, level)
    assert g(acc, level) == first
    assert np.array_equal(acc, kept_acc) and np.array_equal(level, kept_level)
    assert PAIR[0].tolist() == [0.1] and PAIR[1].tolist() == [0.1**2]


@pytest.mark.parametrize("fn", [zone_distance, local_reward, global_reward, overshoot], ids=lambda fn: fn.__name__)
def test_no_function_takes_a_training_only_type(fn):
    """The global reward is pure over arrays: it takes no IntervalReport,
    so what it is handed is the caller's to choose and to test."""
    assert training_only_params(fn) == []


# --- overshoot ---------------------------------------------------------------------

def test_overshoot_is_the_mesh_minus_the_count():
    state = instance_state()
    for committed, expected in ((5, 0), (2, 3), (8, -3)):
        round_state = RoundState(round_index=1, n_rounds=2, committed=committed)
        assert overshoot(round_state, state) == expected


def test_overshoot_is_a_python_int_and_checks_nothing():
    """It subtracts what it is handed: a count above the mesh gives a
    negative number, which on the mock no round produces."""
    state = instance_state()
    value = overshoot(RoundState(round_index=1, n_rounds=2, committed=np.int64(3)), state)
    assert type(value) is int and value == 2


# --- On the mock quad backend (R13) ------------------------------------------------

def _two_cell_mesh() -> tuple[MockQuadBackend, SolverState]:
    """The mock's own five-element mesh: cells A and B, A refined."""
    mock = MockQuadBackend(
        nx=2, ny=1, periodic=(True, False), lo=(0.0, 0.0), hi=(2.0, 1.0), max_level=2, dt_base=0.25,
        rotate_cells=False,
    )
    state = mock.reset({"x0": 0.5, "y0": 0.5, "width": 0.25}, amr_freq=4, initial_level=0)
    state, _ = mock.apply_marks(np.where(state.centroid[:, 0] < 1.0, REFINE, HOLD).astype(np.int8))
    return mock, state


def _graded_mesh(periodic: tuple[bool, bool]) -> tuple[MockQuadBackend, SolverState]:
    """A 4x4 grid at level 1 with the cells near the feature refined once
    more, the rotation on: levels 1 and 2, with 2:1 faces between them."""
    mock = MockQuadBackend(nx=4, ny=4, periodic=periodic, max_level=2, dt_base=0.25, rotate_cells=True)
    ic = {"x0": 0.3, "y0": 0.6, "width": 0.2}
    state = mock.reset(ic, amr_freq=4, initial_level=1)
    near = np.hypot(state.centroid[:, 0] - ic["x0"], state.centroid[:, 1] - ic["y0"]) < 0.15
    assert near.any()
    state, _ = mock.apply_marks(np.where(near, REFINE, HOLD).astype(np.int8))
    return mock, state


MESHES = {
    "two cells": _two_cell_mesh,
    "4x4 periodic": lambda: _graded_mesh((True, True)),
    "4x4 walled": lambda: _graded_mesh((False, False)),
    "4x4 mixed": lambda: _graded_mesh((True, False)),
}
INDICATORS = ("ind_a", "ind_b")


def _mock_config(indicator: str) -> ObservationConfig:
    return ObservationConfig(
        indicator=indicator, reduction="scalar", alpha=0.1, element_budget=64, normalization="relative",
        floor_decades=8.0,
    )


def _families(state: SolverState) -> list[tuple[int, ...]]:
    """Each complete family once, as its positions in child order."""
    return sorted({tuple(row) for row in state.family.tolist() if row[0] >= 0})


def test_the_mocks_two_cell_mesh_is_the_instance():
    _, state = _two_cell_mesh()
    literal = instance_state()
    assert np.array_equal(state.family, literal.family) and np.array_equal(state.level, literal.level)


@pytest.mark.parametrize("indicator", INDICATORS)
@pytest.mark.parametrize("mesh", list(MESHES))
def test_on_the_mock_the_priority_is_the_size_of_the_distance(mesh, indicator):
    mock, state = MESHES[mesh]()
    cfg = _mock_config(indicator)
    cache = read_round(mock, state, [indicator], "scalar")
    pair = round_thresholds(cache, cfg, BETA)
    distance = zone_distance(cache.errors[indicator], cfg, *pair)
    assert np.array_equal(round_priority(cache, cfg, *pair), np.abs(distance))
    assert np.isfinite(distance).all()


@pytest.mark.parametrize("indicator", INDICATORS)
@pytest.mark.parametrize("mesh", list(MESHES))
def test_on_the_mock_a_familys_distance_is_its_largest_members(mesh, indicator):
    """Signal R3 on the mock's own indicator values: for every complete
    family, the distance of its largest error is the largest distance."""
    mock, state = MESHES[mesh]()
    cfg = _mock_config(indicator)
    cache = read_round(mock, state, [indicator], "scalar")
    pair = round_thresholds(cache, cfg, BETA)
    errors = cache.errors[indicator]
    distance = zone_distance(errors, cfg, *pair)
    families = _families(state)
    assert families
    for family in families:
        rows = list(family)
        whole = zone_distance(errors[rows].max(axis=0, keepdims=True), cfg, *pair)
        assert whole[0] == distance[rows].max()
        assert local_reward(state, distance, rows[0], ACTION_COARSEN, "first_decider", **W) == local_reward(
            state, distance, rows[3], ACTION_COARSEN, "any_coarsen", **W
        )


def _hand_round(mock, state, errors, pair, choices, round_index):
    """One round on the mock's state with errors set by hand, in queue
    order under the interval's pair, first_decider. Returns the new state,
    the last round state, the positions asked, the rewards and the number
    skipped."""
    cache = RoundCache(state=state, errors={"e": col(errors)}, indicators=("e",), reduction="scalar")
    order = round_order(cache, config(), *pair)
    distance = zone_distance(cache.errors["e"], config(), *pair)
    round_state = start_round(state, round_index, 2)
    asked, rewards, skipped = [], [], 0
    position, cursor, passed = next_undecided(state, round_state, order, 0)
    while position is not None:
        skipped += passed
        action = choices.get(position, ACTION_HOLD)
        rewards.append(local_reward(state, distance, position, action, "first_decider", **W))
        round_state = commit(state, round_state, position, action, "first_decider")
        asked.append(position)
        position, cursor, passed = next_undecided(state, round_state, order, cursor)
    new_state, _ = mock.apply_marks(round_state.decisions)
    return new_state, round_state, asked, rewards, skipped + passed


def test_the_instances_interval_on_the_mock():
    """Section 2.6.2's interval, on the mock's meshes with the errors set
    by hand. Round 1: B holds, position 3 coarsens for the family (-10);
    count 2, mesh 2. Round 2, on the two-element mesh under the interval's
    thresholds: B, two decades below e_min, is asked first and holds; A
    refines for nothing; count 5, mesh 5. Then the advance, and the global
    reward on the mesh the rounds left."""
    mock, state = _two_cell_mesh()
    pair = round_thresholds(cache_of(C0, state=state), config(), BETA)
    assert pair[0].tolist() == [0.1] and pair[1].tolist() == [0.1**2]

    state, round_state, asked, rewards, skipped = _hand_round(mock, state, C0, pair, {3: ACTION_COARSEN}, 1)
    assert (asked, skipped, letters(round_state.decisions)) == ([4, 3], 3, "CCCCH")
    assert rewards == pytest.approx([0.0, -10.0])
    assert (round_state.committed, state.n_active, overshoot(round_state, state)) == (2, 2, 0)

    state, round_state, asked, rewards, skipped = _hand_round(mock, state, (0.5, 1e-4), pair, {0: ACTION_REFINE}, 2)
    assert (asked, skipped, letters(round_state.decisions)) == ([1, 0], 0, "RH")
    assert rewards == [0.0, 0.0]
    assert (round_state.committed, state.n_active, overshoot(round_state, state)) == (5, 5, 0)

    report = mock.step_interval(["ind_b"], "scalar", 1)
    assert report.acc["ind_b"].shape == (state.n_active, 1) and state.level.tolist() == [1, 1, 1, 1, 0]
    assert g(col(ACC), state.level, max_level=state.max_level, pair=pair) == pytest.approx(-3.90309, abs=1e-6)


def test_one_refine_that_balance_answers_overshoots_by_three():
    """Item 2.4's case: position 0 refines, its children touch B across the
    wrap two levels apart, and balance splits B. Count 8, mesh 11."""
    mock, state = _two_cell_mesh()
    state_after, round_state, asked, rewards, _ = _hand_round(mock, state, (0.05,) * 5, PAIR, {0: ACTION_REFINE}, 1)
    assert asked == [0, 1, 2, 3, 4] and rewards == [0.0] * 5
    assert (round_state.committed, state_after.n_active, overshoot(round_state, state_after)) == (8, 11, 3)


def _interval(mock, state, rule, indicator, rng, n_rounds=2):
    """One interval as a driver would run it: the thresholds from round
    1's cache and held; per round one read, the order and the distances
    under that pair, a walk of random masked actions each paid at its
    commit, apply_marks, the overshoot; then one step_interval and the
    global reward on its acc and the last state's levels.

    Returns one log per round and the global reward.
    """
    cfg = _mock_config(indicator)
    pair, logs = None, []
    for round_index in range(1, n_rounds + 1):
        cache = read_round(mock, state, [indicator], "scalar")
        if pair is None:
            pair = round_thresholds(cache, cfg, BETA)
        order = round_order(cache, cfg, *pair)
        distance = zone_distance(cache.errors[indicator], cfg, *pair)
        round_state = start_round(state, round_index, n_rounds)
        asked, rewards, skipped = 0, [], 0
        position, cursor, passed = next_undecided(state, round_state, order, 0)
        while position is not None:
            skipped += passed
            action = int(rng.choice(np.flatnonzero(action_mask(state, round_state, position, rule))))
            rewards.append(local_reward(state, distance, position, action, rule, **W))
            round_state = commit(state, round_state, position, action, rule)
            asked += 1
            position, cursor, passed = next_undecided(state, round_state, order, cursor)
        skipped += passed
        marks = round_state.decisions
        new_state, report = mock.apply_marks(marks)
        whole = sum(bool((marks[list(f)] == COARSEN).all()) for f in _families(state))
        kept = sum(bool((report.realized[list(f)] == COARSEN).all()) for f in _families(state))
        logs.append(
            dict(
                n=state.n_active, asked=asked, skipped=skipped, rewards=rewards, committed=round_state.committed,
                n_after=new_state.n_active, overshoot=overshoot(round_state, new_state),
                added=int((report.realized == REFINE).sum()) - int((marks == REFINE).sum()), undone=whole - kept,
            )
        )
        state = new_state
    acc = mock.step_interval([indicator], "scalar", 1).acc[indicator]
    return logs, global_reward(acc, state.level, state.max_level, cfg, *pair, **G)


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("indicator", INDICATORS)
@pytest.mark.parametrize("rule", COARSEN_RULES)
@pytest.mark.parametrize("mesh", list(MESHES))
def test_an_interval_of_random_masked_actions(mesh, rule, indicator, seed):
    """Two rounds and the advance. In every round: asked plus skipped is
    n_active; the overshoot is the mesh minus the count, never negative, a
    multiple of 3, and 3 for every refine balance added and every family
    coarsen it undid; every local reward is finite. The global reward is
    finite and not positive."""
    mock, state = MESHES[mesh]()
    logs, r_global = _interval(mock, state, rule, indicator, np.random.default_rng(seed))
    assert len(logs) == 2
    for log in logs:
        assert log["asked"] + log["skipped"] == log["n"] and len(log["rewards"]) == log["asked"]
        assert log["overshoot"] == log["n_after"] - log["committed"]
        assert log["overshoot"] >= 0 and log["overshoot"] % 3 == 0
        assert log["overshoot"] == 3 * (log["added"] + log["undone"])
        assert all(math.isfinite(r) for r in log["rewards"])
    assert logs[1]["n"] == logs[0]["n_after"]
    assert math.isfinite(r_global) and r_global <= 0.0


def test_the_random_intervals_overshoot_somewhere_and_nowhere_and_pay_both_ways():
    """So that the assertions above are not checked on zeros alone: across
    the random intervals some rounds overshoot and some do not, some local
    rewards are penalties and some are positive, and some global reward is
    below zero."""
    over = exact = penalties = earned = below = 0
    for mesh in MESHES:
        for rule in COARSEN_RULES:
            for seed in (0, 1, 2):
                mock, state = MESHES[mesh]()
                logs, r_global = _interval(mock, state, rule, "ind_b", np.random.default_rng(seed))
                for log in logs:
                    over += log["overshoot"] > 0
                    exact += log["overshoot"] == 0
                    penalties += sum(r < 0.0 for r in log["rewards"])
                    earned += sum(r > 0.0 for r in log["rewards"])
                below += r_global < 0.0
    assert over > 0 and exact > 0 and penalties > 0 and earned > 0 and below > 0


@pytest.mark.parametrize("rule", COARSEN_RULES)
@pytest.mark.parametrize("mesh", list(MESHES))
def test_an_interval_of_holds_pays_no_local_reward_and_overshoots_nothing(mesh, rule):
    """Every position asked, every reward 0, the mesh unchanged; the
    global reward is then the starting mesh's own."""
    mock, state = MESHES[mesh]()

    class Holds:
        def choice(self, open_actions):
            return ACTION_HOLD

    logs, r_global = _interval(mock, state, rule, "ind_b", Holds())
    for log in logs:
        assert log["skipped"] == 0 and log["overshoot"] == 0 and log["n_after"] == log["n"]
        assert log["rewards"] == [0.0] * log["n"]
    assert r_global <= 0.0


@pytest.mark.parametrize("rule", FAMILY_RULES)
@pytest.mark.parametrize("mesh", list(MESHES))
def test_on_the_mock_each_family_coarsen_is_paid_once(mesh, rule):
    """Coarsen wherever it is open, one round: exactly one reward that can
    be other than zero per complete family, the decider's, and it is the
    table's cell for the family's largest distance."""
    mock, state = MESHES[mesh]()
    cfg = _mock_config("ind_b")
    cache = read_round(mock, state, ["ind_b"], "scalar")
    pair = round_thresholds(cache, cfg, BETA)
    order, distance = round_order(cache, cfg, *pair), zone_distance(cache.errors["ind_b"], cfg, *pair)
    round_state = start_round(state, 1, 2)
    paid_to: dict[int, float] = {}
    position, cursor, _ = next_undecided(state, round_state, order, 0)
    while position is not None:
        mask = action_mask(state, round_state, position, rule)
        action = ACTION_COARSEN if mask[ACTION_COARSEN] else ACTION_HOLD
        paid_to[position] = local_reward(state, distance, position, action, rule, **W)
        round_state = commit(state, round_state, position, action, rule)
        position, cursor, _ = next_undecided(state, round_state, order, cursor)
    for family in _families(state):
        deciders = [p for p in family if p in paid_to]
        assert len(deciders) == 1
        worst = float(distance[list(family)].max())
        expected = -P_UR * worst if worst > 0.0 else P_CR * -worst
        assert paid_to[deciders[0]] == pytest.approx(expected)
    alone = [p for p in paid_to if state.family[p][0] < 0]
    assert all(paid_to[p] == 0.0 for p in alone)


@pytest.mark.parametrize("rotate", [False, True], ids=["pinned", "rotating"])
def test_an_interval_with_both_rewards_costs_2r_plus_1_calls(counting, rotate):
    """The rewards make no backend call: one compute_error and one
    apply_marks per round and one step_interval, after the reset."""
    mock = counting(
        MockQuadBackend(
            nx=2, ny=1, periodic=(True, False), lo=(0.0, 0.0), hi=(2.0, 1.0), max_level=2, dt_base=0.25,
            rotate_cells=rotate,
        )
    )
    state = mock.reset({"x0": 0.5, "y0": 0.5, "width": 0.25}, amr_freq=4, initial_level=1)
    logs, r_global = _interval(mock, state, "first_decider", "ind_b", np.random.default_rng(4))
    assert len(logs) == 2 and r_global <= 0.0
    assert dict(mock.calls) == {"reset": 1, "compute_error": 2, "apply_marks": 2, "step_interval": 1}


# --- What the functions reject (R14) -----------------------------------------------

@pytest.mark.parametrize(
    "errors",
    [[[1.0]], np.array([1.0, 2.0]), np.array([[1, 2]]), np.zeros((3, 0)), np.zeros((2, 1, 1)), None],
    ids=["a list", "1-D", "integers", "no column", "3-D", "None"],
)
def test_zone_distance_rejects_errors_of_the_wrong_form(errors):
    with pytest.raises(ValueError, match=r"^errors: expected a 2-D float64 numpy array with at least one column$"):
        zone_distance(errors, config(), *PAIR)


def test_zone_distance_rejects_dynamo_at_two_columns():
    with pytest.raises(ValueError, match=r"^normalization: 'dynamo' takes one column, and errors has 2$"):
        zone_distance(col(C0, C1), config("dynamo", "per_variable"), *PAIR2)


@pytest.mark.parametrize(
    "e_max, e_min, pattern",
    [
        (PAIR[0], PAIR[1], r"^e_max: expected a numpy array of shape \(2,\)$"),
        (PAIR2[0], PAIR[1], r"^e_min: expected a numpy array of shape \(2,\)$"),
        ([0.1, 100.0], PAIR2[1], r"^e_max: expected a numpy array of shape \(2,\)$"),
        (np.array([0.1, -1.0]), PAIR2[1], r"^e_max: \[0.1, -1.0\], expected finite values, none negative$"),
        (PAIR2[0], np.array([np.nan, 10.0]), r"^e_min: \[nan, 10.0\], expected finite values, none negative$"),
    ],
    ids=["one upper threshold", "one lower threshold", "a list", "a negative e_max", "a nan e_min"],
)
def test_zone_distance_rejects_thresholds_of_the_wrong_form(e_max, e_min, pattern):
    with pytest.raises(ValueError, match=pattern):
        zone_distance(col(C0, C1), config(reduction="per_variable"), e_max, e_min)


@pytest.mark.parametrize(
    "changes, pattern",
    [
        (dict(rule="majority"), r"^rule: 'majority', expected one of \['first_decider', 'any_coarsen', 'unanimous'\]$"),
        (dict(position=5), r"^position: 5, expected an integer in \[0, 5\)$"),
        (dict(position=-1), r"^position: -1, expected an integer in \[0, 5\)$"),
        (dict(position=1.0), r"^position: 1.0, expected an integer in \[0, 5\)$"),
        (dict(position=True), r"^position: True, expected an integer in \[0, 5\)$"),
        (dict(action=3), r"^action: 3, expected 0 \(coarsen\), 1 \(hold\) or 2 \(refine\)$"),
        (dict(action=-1), r"^action: -1, expected 0 \(coarsen\), 1 \(hold\) or 2 \(refine\)$"),
        (dict(action=True), r"^action: True, expected 0 \(coarsen\), 1 \(hold\) or 2 \(refine\)$"),
        (dict(action=None), r"^action: None, expected 0 \(coarsen\), 1 \(hold\) or 2 \(refine\)$"),
        (dict(distance=np.zeros(4)), r"^distance: expected a numpy array of shape \(5,\)$"),
        (dict(distance=np.zeros((5, 1))), r"^distance: expected a numpy array of shape \(5,\)$"),
        (dict(distance=[0.0] * 5), r"^distance: expected a numpy array of shape \(5,\)$"),
        (dict(p_ur=-1.0), r"^p_ur: -1.0, expected a finite number, not negative$"),
        (dict(p_or=math.inf), r"^p_or: inf, expected a finite number, not negative$"),
        (dict(p_cr=math.nan), r"^p_cr: nan, expected a finite number, not negative$"),
        (dict(p_cr="2"), r"^p_cr: '2', expected a finite number, not negative$"),
        (dict(p_ur=True), r"^p_ur: True, expected a finite number, not negative$"),
        (dict(position=B), r"^family: position 4 has no complete family, its row is \[-1, -1, -1, -1\]$"),
    ],
    ids=[
        "a rule",
        "a position past the end",
        "a negative position",
        "a float position",
        "a bool position",
        "an action past the end",
        "a negative action",
        "a bool action",
        "no action",
        "a short distance",
        "a 2-D distance",
        "a list",
        "a negative weight",
        "an infinite weight",
        "a nan weight",
        "a string weight",
        "a bool weight",
        "a family coarsen with no family",
    ],
)
def test_the_local_reward_rejects(changes, pattern):
    arguments = dict(position=0, action=ACTION_COARSEN, rule="first_decider", distance=distance_of(C0), **W)
    arguments.update(changes)
    distance, position, action, rule = (arguments.pop(k) for k in ("distance", "position", "action", "rule"))
    with pytest.raises(ValueError, match=pattern):
        local_reward(instance_state(), distance, position, action, rule, **arguments)


def test_a_vote_with_no_family_is_the_masks_to_refuse_not_the_rewards():
    """Under unanimous a coarsen reads the element's own distance, so cell
    B's vote is paid here; action_mask is what closes it, and commit raises."""
    state, distance = instance_state(), distance_of(C0)
    assert local_reward(state, distance, B, ACTION_COARSEN, "unanimous", **W) == pytest.approx(4.0)
    assert not action_mask(state, start_round(state, 1, 2), B, "unanimous")[ACTION_COARSEN]
    with pytest.raises(ValueError, match=r"^action: 0 is masked for position 4 under 'unanimous'$"):
        commit(state, start_round(state, 1, 2), B, ACTION_COARSEN, "unanimous")


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf], ids=["nan", "inf", "minus inf"])
def test_the_global_reward_refuses_an_error_that_is_not_finite(bad):
    """As RoundCache does for a round's read: IntervalReport does not check
    its acc, and the environment ends the episode on this one error."""
    errors = col(ACC)
    errors[2, 0] = bad
    with pytest.raises(BackendFailure, match=r"^errors: not finite$"):
        g(errors, LEVEL)


@pytest.mark.parametrize(
    "changes, pattern",
    [
        (dict(errors=[[1.0]] * 5), r"^errors: expected a 2-D float64 numpy array with at least one column$"),
        (dict(errors=np.array(ACC)), r"^errors: expected a 2-D float64 numpy array with at least one column$"),
        (dict(level=LEVEL[:4]), r"^level: expected an integer numpy array of shape \(5,\)$"),
        (dict(level=LEVEL.astype(float)), r"^level: expected an integer numpy array of shape \(5,\)$"),
        (dict(level=LEVEL.tolist()), r"^level: expected an integer numpy array of shape \(5,\)$"),
        (dict(level=LEVEL[:, None]), r"^level: expected an integer numpy array of shape \(5,\)$"),
        (dict(level=LEVEL > 0), r"^level: expected an integer numpy array of shape \(5,\)$"),
        (dict(max_level=2.0), r"^max_level: 2.0, expected an integer, not negative$"),
        (dict(max_level=-1), r"^max_level: -1, expected an integer, not negative$"),
        (dict(max_level=True), r"^max_level: True, expected an integer, not negative$"),
        (dict(p_ur=-1.0), r"^p_ur: -1.0, expected a finite number, not negative$"),
        (dict(p_or=math.nan), r"^p_or: nan, expected a finite number, not negative$"),
        (dict(e_max=np.array([0.1, 0.1])), r"^e_max: expected a numpy array of shape \(1,\)$"),
        (dict(e_min=np.array([-0.01])), r"^e_min: \[-0.01\], expected finite values, none negative$"),
    ],
    ids=[
        "errors as a list",
        "1-D errors",
        "a short level",
        "a float level",
        "a level list",
        "a 2-D level",
        "a bool level",
        "a float max_level",
        "a negative max_level",
        "a bool max_level",
        "a negative weight",
        "a nan weight",
        "two upper thresholds",
        "a negative e_min",
    ],
)
def test_the_global_reward_rejects(changes, pattern):
    arguments = dict(errors=col(ACC), level=LEVEL, max_level=2, e_max=PAIR[0], e_min=PAIR[1], **G)
    arguments.update(changes)
    names = ("errors", "level", "max_level", "e_max", "e_min")
    errors, level, max_level, e_max, e_min = (arguments.pop(name) for name in names)
    with pytest.raises(ValueError, match=pattern):
        global_reward(errors, level, max_level, config(), e_max, e_min, **arguments)


def test_the_global_reward_rejects_dynamo_at_two_columns():
    with pytest.raises(ValueError, match=r"^normalization: 'dynamo' takes one column, and errors has 2$"):
        g(col(ACC, C1), LEVEL, pair=PAIR2, cfg=config("dynamo", "per_variable"))
