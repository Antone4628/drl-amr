"""Tests for agent/round_queue.py: the queue of the 2D agent core. The
thresholds, each element's priority, the order a round's positions are
asked in, and the step that moves along it past decided positions.

AGENT_2D_ROADMAP item 2.5, signals Q2-Q13. Nine groups:

  - round_thresholds on the five-element instance of the explanation
    file's section 2.5.2, under both rules, and what it rejects (Q2);
  - round_priority and round_order on the instance (Q3), under a change of
    scale (Q4), with a second column (Q5), on ties (Q6) and on zeros (Q8);
  - thresholds handed in from another round;
  - against the sequential core's build_queue, at one column under the 1D
    rule (Q7);
  - what the two reject (Q12);
  - next_undecided: the instance's walks, step by step (Q9), and what it
    rejects (Q12);
  - the vocabulary: no function here takes a training-only type (Q13);
  - on the mock quad backend: the priority against a reference written as
    loops, and whole rounds walked in queue order (Q10);
  - on the mock quad backend: the queue makes no backend call (Q11).

Records are written as text, one letter per position: C, H and R for the
marks and a dot for an undecided position.

Run: pytest tests/agent/test_round_queue.py -v
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from agent.normalization import alpha_thresholds
from agent.queue import build_queue
from agent.round_cache import RoundCache, read_round
from agent.round_commit import commit, start_round
from agent.round_mask import ACTION_COARSEN, ACTION_HOLD, ACTION_REFINE, COARSEN_RULES, action_mask
from agent.round_observation import ObservationConfig
from agent.round_queue import next_undecided, round_order, round_priority, round_thresholds
from agent.round_state import UNDECIDED, RoundState
from backends.mock_quad.backend import MockQuadBackend
from contract import element_state
from contract.checks import training_only_params
from contract.solver_contract import COARSEN, HOLD, REFINE
from contract.state import SolverState

RULES = ("relative", "dynamo")
LETTER = {".": UNDECIDED, "C": COARSEN, "H": HOLD, "R": REFINE}
ACTION = {"coarsen": ACTION_COARSEN, "hold": ACTION_HOLD, "refine": ACTION_REFINE}
# Cell B's position on the instance; positions 0-3 are cell A's children.
B = 4
BETA = 2
# The instance's errors, section 2.5.2: its one column, and the second.
C0 = (2e-3, 3e-2, 3e-2, 1.0, 1e-4)
C1 = (1e3, 1.0, 10.0, 0.1, 30.0)
ORDER = (4, 3, 0, 1, 2)
SCALES = (1e-6, 10.0, 100.0, 1e6)


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


def cache_of(*columns, state: SolverState | None = None) -> RoundCache:
    """A cache on the instance (or a given state) holding one indicator,
    "e", with the given columns: read under scalar at one column and under
    per_variable at more."""
    errors = np.stack([np.array(column, dtype=np.float64) for column in columns], axis=1)
    reduction = "scalar" if len(columns) == 1 else "per_variable"
    return RoundCache(
        state=instance_state() if state is None else state, errors={"e": errors}, indicators=("e",), reduction=reduction
    )


def config(normalization: str = "relative", reduction: str = "scalar", **changes) -> ObservationConfig:
    """The instance's configuration: alpha 0.1, a floor of 8 decades."""
    fields = dict(
        indicator="e", reduction=reduction, alpha=0.1, element_budget=8, normalization=normalization, floor_decades=8.0
    )
    fields.update(changes)
    return ObservationConfig(**fields)


def queue_of(cache: RoundCache, cfg: ObservationConfig, beta: float = BETA) -> tuple[np.ndarray, np.ndarray]:
    """The priority and the order of a cache under its own thresholds."""
    e_max, e_min = round_thresholds(cache, cfg, beta)
    return round_priority(cache, cfg, e_max, e_min), round_order(cache, cfg, e_max, e_min)


def scaled(column, factor: float) -> list[float]:
    return [factor * e for e in column]


# --- round_thresholds (Q2) ---------------------------------------------------------

@pytest.mark.parametrize("rule", RULES)
def test_thresholds_on_the_instance(rule):
    """e_inf is 1, so both rules give alpha and alpha squared. As floats
    0.1 ** 2 is one unit in the last place above 0.01."""
    e_max, e_min = round_thresholds(cache_of(C0), config(rule), BETA)
    assert e_max.tolist() == [0.1] and e_min.tolist() == [0.1**2]
    assert e_min[0] == pytest.approx(0.01, rel=1e-15)
    for array in (e_max, e_min):
        assert array.dtype == np.float64 and array.shape == (1,) and not array.flags.writeable


def test_the_two_rules_agree_bit_for_bit_where_e_inf_is_one():
    relative = round_thresholds(cache_of(C0), config("relative"), BETA)
    dynamo = round_thresholds(cache_of(C0), config("dynamo"), BETA)
    assert relative[0].tolist() == dynamo[0].tolist() and relative[1].tolist() == dynamo[1].tolist()


def test_thresholds_with_two_columns():
    """One pair per column, each against its own largest error."""
    e_max, e_min = round_thresholds(cache_of(C0, C1), config(reduction="per_variable"), BETA)
    assert e_max.shape == e_min.shape == (2,)
    assert e_max.tolist() == pytest.approx([0.1, 100.0], rel=1e-15)
    assert e_min.tolist() == pytest.approx([0.01, 10.0], rel=1e-15)


def test_dynamo_is_the_shared_1d_function():
    """Under dynamo the pair is agent.normalization.alpha_thresholds',
    unchanged, on the one column."""
    for factor in (1.0, 10.0, 100.0, 1e-6):
        column = np.array(scaled(C0, factor))
        e_max, e_min = round_thresholds(cache_of(column), config("dynamo"), BETA)
        expected = alpha_thresholds(column, 0.1, BETA)
        assert (e_max[0], e_min[0]) == (expected[0], expected[1])


@pytest.mark.parametrize("factor", SCALES)
def test_relative_thresholds_scale_with_the_errors(factor):
    """The zone keeps its width, (beta - 1) * -log10(alpha) = 1 decade."""
    e_max, e_min = round_thresholds(cache_of(scaled(C0, factor)), config("relative"), BETA)
    assert e_max[0] == pytest.approx(0.1 * factor) and e_min[0] == pytest.approx(0.01 * factor)
    assert math.log10(e_max[0] / e_min[0]) == pytest.approx(1.0)


def test_dynamo_thresholds_cross_above_its_regime():
    """Section 2.2.1's finding, as numbers: with every error multiplied by
    100, e_max is 10 and e_min is 100, above it."""
    e_max, e_min = round_thresholds(cache_of(scaled(C0, 100.0)), config("dynamo"), BETA)
    assert (e_max[0], e_min[0]) == (10.0, 100.0)


@pytest.mark.parametrize("beta, width", [(1.5, 0.5), (2, 1.0), (3, 2.0), (np.float64(2.5), 1.5)])
def test_relative_the_zone_is_beta_minus_one_times_the_decades_of_alpha(beta, width):
    e_max, e_min = round_thresholds(cache_of(scaled(C0, 37.0)), config("relative"), beta)
    assert math.log10(e_max[0] / e_min[0]) == pytest.approx(width)


@pytest.mark.parametrize("rule", RULES)
def test_thresholds_of_an_all_zero_round_are_zero(rule):
    e_max, e_min = round_thresholds(cache_of([0.0] * 5), config(rule), BETA)
    assert e_max.tolist() == [0.0] and e_min.tolist() == [0.0]


def test_thresholds_of_a_zero_column_are_zero():
    e_max, e_min = round_thresholds(cache_of(C0, [0.0] * 5), config(reduction="per_variable"), BETA)
    assert e_max[1] == 0.0 and e_min[1] == 0.0 and e_max[0] == 0.1


def test_thresholds_are_new_arrays_and_leave_the_cache_alone():
    cache = cache_of(C0)
    kept = cache.errors["e"].copy()
    first, second = round_thresholds(cache, config(), BETA), round_thresholds(cache, config(), BETA)
    assert first[0] is not second[0] and first[1] is not second[1]
    assert np.array_equal(cache.errors["e"], kept)


@pytest.mark.parametrize(
    "beta",
    [1, 1.0, 0.5, 0, -2, True, "2", None, math.inf, math.nan],
    ids=["1", "1.0", "0.5", "0", "negative", "a bool", "a string", "None", "infinite", "nan"],
)
@pytest.mark.parametrize("rule", RULES)
def test_thresholds_reject_a_beta_that_is_not_a_finite_number_above_one(rule, beta):
    with pytest.raises(ValueError, match=r"^beta: .*, expected a finite number above 1$"):
        round_thresholds(cache_of(C0), config(rule), beta)


# --- The instance, ordered (Q3) ----------------------------------------------------

@pytest.mark.parametrize("rule", RULES)
def test_the_instances_priorities_and_order(rule):
    """Section 2.5.2's first table. B is two decades below e_min, position
    3 one decade above e_max, position 0 a factor of 5 below e_min, and
    positions 1 and 2 are neutral: 4, 3, 0, 1, 2."""
    priority, order = queue_of(cache_of(C0), config(rule))
    assert priority.tolist() == pytest.approx([math.log10(5.0), 0.0, 0.0, 1.0, 2.0], abs=1e-12)
    assert priority[1] == 0.0 and priority[2] == 0.0
    assert tuple(order.tolist()) == ORDER


def test_what_the_two_functions_return():
    cache, cfg = cache_of(C0), config()
    e_max, e_min = round_thresholds(cache, cfg, BETA)
    priority, order = round_priority(cache, cfg, e_max, e_min), round_order(cache, cfg, e_max, e_min)
    assert priority.dtype == np.float64 and priority.shape == (5,)
    assert order.dtype.kind == "i" and order.shape == (5,) and not order.flags.writeable
    assert round_priority(cache, cfg, e_max, e_min) is not priority
    assert round_order(cache, cfg, e_max, e_min) is not order
    assert np.array_equal(round_priority(cache, cfg, e_max, e_min), priority)


def test_the_order_is_the_stable_descending_sort_of_the_priority():
    cache, cfg = cache_of(C0), config()
    priority, order = queue_of(cache, cfg)
    assert order.tolist() == sorted(range(5), key=lambda p: (-priority[p], p))


@pytest.mark.parametrize("rule", RULES)
def test_an_error_on_a_threshold_is_neutral(rule):
    """Above e_max and below e_min are strict, as in the sequential queue."""
    cache, cfg = cache_of([1.0, 0.25, 0.0625, 0.5, 0.5]), config(rule)
    priority = round_priority(cache, cfg, np.array([0.25]), np.array([0.0625]))
    assert priority.tolist() == pytest.approx([math.log10(4.0), 0.0, 0.0, math.log10(2.0), math.log10(2.0)])
    assert priority[1] == 0.0 and priority[2] == 0.0


def test_under_refined_is_tested_first_where_the_thresholds_cross():
    """Dynamo above its regime: with e_max 10 and e_min 100, an error of 50
    is above the one and below the other, and counts as under-refined. An
    error of exactly 10 is not above e_max, so it is over-refined, one
    decade below e_min, as the sequential queue has it."""
    cache, cfg = cache_of([50.0, 5.0, 500.0, 10.0, 50.0]), config("dynamo")
    priority = round_priority(cache, cfg, np.array([10.0]), np.array([100.0]))
    assert priority.tolist() == pytest.approx(
        [math.log10(5.0), math.log10(20.0), math.log10(50.0), 1.0, math.log10(5.0)]
    )


# --- Scale (Q4) --------------------------------------------------------------------

@pytest.mark.parametrize("factor", SCALES)
def test_relative_a_change_of_scale_changes_nothing(factor):
    reference, _ = queue_of(cache_of(C0), config("relative"))
    priority, order = queue_of(cache_of(scaled(C0, factor)), config("relative"))
    assert tuple(order.tolist()) == ORDER
    assert priority.tolist() == pytest.approx(reference.tolist(), abs=1e-9)
    assert priority[1] == 0.0 and priority[2] == 0.0


@pytest.mark.parametrize(
    "factor, expected_order, neutral",
    [(1e-6, [3, 0, 1, 2, 4], 4), (10.0, [4, 0, 3, 1, 2], 0), (100.0, [4, 0, 1, 2, 3], 0), (1e6, [4, 0, 1, 2, 3], 0)],
)
def test_dynamo_a_change_of_scale_changes_the_order(factor, expected_order, neutral):
    """The 1D rule outside its regime. Below it (e_min is e_max squared,
    far below every error) only the largest error is off zero. At x10 the
    thresholds are both 1 and nothing is neutral."""
    priority, order = queue_of(cache_of(scaled(C0, factor)), config("dynamo"))
    assert order.tolist() == expected_order
    assert int((priority == 0.0).sum()) == neutral


# --- Two columns (Q5) --------------------------------------------------------------

def test_two_columns_the_instances_priorities_and_order():
    """Section 2.5.2's third table. Positions 0 and 3 are each one decade
    above e_max in one column; nothing is below e_min in both."""
    priority, order = queue_of(cache_of(C0, C1), config(reduction="per_variable"))
    assert priority.tolist() == pytest.approx([1.0, 0.0, 0.0, 1.0, 0.0], abs=1e-12)
    assert priority[0] == priority[3]
    assert order.tolist() == [0, 3, 1, 2, 4]


def test_over_refined_in_one_variable_is_not_over_refined():
    """Cell B leads the one-column queue and is last in the two-column one."""
    _, alone = queue_of(cache_of(C0), config())
    _, beside = queue_of(cache_of(C0, C1), config(reduction="per_variable"))
    assert alone[0] == B and beside[-1] == B


@pytest.mark.parametrize("column", [0, 1])
@pytest.mark.parametrize("factor", SCALES)
def test_two_columns_a_columns_scale_changes_nothing(factor, column):
    columns = [list(C0), list(C1)]
    columns[column] = scaled(columns[column], factor)
    priority, order = queue_of(cache_of(*columns), config(reduction="per_variable"))
    assert order.tolist() == [0, 3, 1, 2, 4]
    assert priority.tolist() == pytest.approx([1.0, 0.0, 0.0, 1.0, 0.0], abs=1e-9)


def test_two_columns_swapped_give_the_same_queue():
    first, order = queue_of(cache_of(C0, C1), config(reduction="per_variable"))
    second, swapped = queue_of(cache_of(C1, C0), config(reduction="per_variable"))
    assert np.array_equal(first, second) and np.array_equal(order, swapped)


def test_a_zero_second_column_drops_out():
    """Bit for bit the one-column queue: a column with nothing in it takes
    no part in either test."""
    priority, order = queue_of(cache_of(C0), config())
    beside, order_beside = queue_of(cache_of(C0, [0.0] * 5), config(reduction="per_variable"))
    assert np.array_equal(priority, beside) and np.array_equal(order, order_beside)


def test_a_copy_of_the_column_changes_nothing():
    priority, order = queue_of(cache_of(C0), config())
    beside, order_beside = queue_of(cache_of(C0, C0), config(reduction="per_variable"))
    assert np.array_equal(priority, beside) and np.array_equal(order, order_beside)


def test_over_refined_takes_the_margin_of_the_worst_column():
    """Every column below its e_min: the priority is the smallest margin."""
    cache = cache_of([1e-3, 1.0, 1.0, 1.0, 1.0], [1e-5, 1.0, 1.0, 1.0, 1.0])
    priority = round_priority(cache, config(reduction="per_variable"), np.array([0.1, 0.1]), np.array([0.01, 0.01]))
    assert priority[0] == pytest.approx(1.0)


def test_under_refined_takes_the_largest_excess():
    cache = cache_of([10.0, 0.05, 0.05, 0.05, 0.05], [100.0, 0.05, 0.05, 0.05, 0.05])
    priority = round_priority(cache, config(reduction="per_variable"), np.array([0.1, 0.1]), np.array([0.01, 0.01]))
    assert priority[0] == pytest.approx(3.0) and (priority[1:] == 0.0).all()


def test_one_column_above_outweighs_one_below():
    cache = cache_of([10.0, 0.05, 0.05, 0.05, 0.05], [1e-6, 0.05, 0.05, 0.05, 0.05])
    priority = round_priority(cache, config(reduction="per_variable"), np.array([0.1, 0.1]), np.array([0.01, 0.01]))
    assert priority[0] == pytest.approx(2.0)


# --- Ties (Q6) ---------------------------------------------------------------------

@pytest.mark.parametrize("rule", RULES)
def test_equal_errors_come_in_position_order(rule):
    """Every element is the round's largest error: one decade above e_max
    each (alpha is 0.1 and the errors are 1), and the order is the
    identity."""
    priority, order = queue_of(cache_of([1.0] * 5), config(rule))
    assert priority.tolist() == pytest.approx([1.0] * 5) and len(set(priority.tolist())) == 1
    assert order.tolist() == [0, 1, 2, 3, 4]


@pytest.mark.parametrize("rule", RULES)
def test_an_all_zero_round_is_all_neutral_in_position_order(rule):
    priority, order = queue_of(cache_of([0.0] * 5), config(rule))
    assert (priority == 0.0).all() and order.tolist() == [0, 1, 2, 3, 4]


def test_tied_elements_keep_their_position_order_between_others():
    """Two pairs of ties around a single element: position 2 leads, two
    decades below e_min; then 1 and 4, tied one decade above e_max; then 0
    and 3, neutral."""
    priority, order = queue_of(cache_of([3e-2, 1.0, 1e-4, 3e-2, 1.0]), config())
    assert priority[1] == priority[4] and priority[0] == priority[3] == 0.0
    assert order.tolist() == [2, 1, 4, 0, 3]


@pytest.mark.parametrize("seed", range(8))
def test_every_order_is_a_permutation_sorted_by_priority(seed):
    rng = np.random.default_rng(seed)
    column = 10.0 ** rng.uniform(-6.0, 0.0, size=5)
    column[rng.integers(5)] = 0.0
    priority, order = queue_of(cache_of(column), config())
    assert sorted(order.tolist()) == [0, 1, 2, 3, 4]
    assert (np.diff(priority[order]) <= 0.0).all()
    assert order.tolist() == sorted(range(5), key=lambda p: (-priority[p], p))


# --- Zeros and the floor (Q8) ------------------------------------------------------

def test_relative_a_zero_error_has_the_floors_priority():
    """Floored at e_max * 1e-8, which is 7 decades below e_min here: the
    most over-refined an element can read, and finite."""
    priority, order = queue_of(cache_of([0.0, 3e-2, 3e-2, 1.0, 1e-4]), config("relative"))
    assert priority[0] == pytest.approx(7.0)
    assert np.isfinite(priority).all()
    assert order.tolist() == [0, 4, 3, 1, 2]


def test_dynamo_a_zero_error_has_the_sequential_floors_priority():
    """Floored at 1e-30, as agent/queue.py floors it: log10(0.01 / 1e-30)."""
    priority, _ = queue_of(cache_of([0.0, 3e-2, 3e-2, 1.0, 1e-4]), config("dynamo"))
    assert priority[0] == pytest.approx(28.0)
    assert np.isfinite(priority).all()


def test_relative_errors_below_the_floor_tie_with_zero():
    """The queue cannot separate what the observation cannot: 1e-12 and 0
    are both below e_max * 1e-8 and read the same."""
    priority, order = queue_of(cache_of([0.0, 1e-12, 3e-2, 1.0, 1e-4]), config("relative"))
    assert priority[0] == priority[1] == pytest.approx(7.0)
    assert order.tolist() == [0, 1, 4, 3, 2]


@pytest.mark.parametrize("floor_decades, expected", [(8.0, 7.0), (4.0, 3.0), (2.0, 1.0)])
def test_relative_the_floor_is_the_configurations(floor_decades, expected):
    cfg = config("relative", floor_decades=floor_decades)
    priority, _ = queue_of(cache_of([0.0, 3e-2, 3e-2, 1.0, 1e-4]), cfg)
    assert priority[0] == pytest.approx(expected)


def test_relative_a_floor_shallower_than_the_zone_leaves_nothing_over_refined():
    """The limit section 2.5.3 states: with the zone 1 decade wide and a
    floor half a decade below e_max, no error can read below e_min."""
    cfg = config("relative", floor_decades=0.5)
    priority, _ = queue_of(cache_of([0.0, 1e-6, 3e-2, 1.0, 1e-4]), cfg)
    assert priority.tolist() == pytest.approx([0.0, 0.0, 0.0, 1.0, 0.0])


@pytest.mark.parametrize("rule", RULES)
def test_no_priority_is_negative_or_not_finite(rule):
    rng = np.random.default_rng(3)
    for _ in range(20):
        column = 10.0 ** rng.uniform(-12.0, 3.0, size=5)
        column[rng.random(5) < 0.3] = 0.0
        priority, _ = queue_of(cache_of(column), config(rule))
        assert np.isfinite(priority).all() and (priority >= 0.0).all()


@pytest.mark.parametrize("rule", RULES)
def test_a_zero_upper_threshold_leaves_nothing_under_refined(rule):
    cache = cache_of([1.0, 1e-3, 1e-3, 1e-3, 1e-3])
    priority = round_priority(cache, config(rule), np.array([0.0]), np.array([0.01]))
    assert priority[0] == 0.0 and (priority[1:] > 0.0).all()


@pytest.mark.parametrize("rule", RULES)
def test_a_zero_lower_threshold_leaves_nothing_over_refined(rule):
    cache = cache_of([1.0, 1e-3, 1e-3, 1e-3, 0.0])
    priority = round_priority(cache, config(rule), np.array([0.1]), np.array([0.0]))
    assert priority.tolist() == pytest.approx([1.0, 0.0, 0.0, 0.0, 0.0])


def test_dynamo_a_threshold_at_the_sequential_floor_is_dead():
    """The sequential queue's guard: a threshold must be above 1e-30."""
    cache = cache_of([1.0, 1e-3, 1e-3, 1e-3, 1e-3])
    priority = round_priority(cache, config("dynamo"), np.array([1e-30]), np.array([1e-30]))
    assert (priority == 0.0).all()


# --- Thresholds handed in from another round ---------------------------------------

def test_the_order_follows_the_thresholds_it_is_handed():
    """Round 2's errors, a tenth of round 1's. Under its own thresholds
    the queue is round 1's (scale changes nothing). Under round 1's, held
    for the interval, position 3 has come down into the zone and goes last.
    Which of the two a driver uses is its choice (D-021); the queue takes
    either."""
    first, second, cfg = cache_of(C0), cache_of(scaled(C0, 0.1)), config()
    held = round_thresholds(first, cfg, BETA)
    own = round_thresholds(second, cfg, BETA)
    assert tuple(round_order(second, cfg, *own).tolist()) == ORDER
    assert round_order(second, cfg, *held).tolist() == [4, 0, 1, 2, 3]
    priority = round_priority(second, cfg, *held)
    assert priority.tolist() == pytest.approx([math.log10(50.0), math.log10(10 / 3), math.log10(10 / 3), 0.0, 3.0])


# --- Against the sequential core's queue (Q7) --------------------------------------

def _old_state(element_id) -> element_state.SolverState:
    """A 1D-contract SolverState, as tests/agent/test_queue.py builds it."""
    n = len(element_id)
    return element_state.SolverState(
        element_id=np.asarray(element_id),
        level=np.zeros(n, dtype=int),
        left=np.asarray([(i - 1) % n for i in range(n)]),
        right=np.asarray([(i + 1) % n for i in range(n)]),
        sibling=np.full(n, -1),
        can_refine=np.ones(n, dtype=bool),
        can_coarsen=np.zeros(n, dtype=bool),
        n_active=n,
        max_level=3,
        stable_dt=0.01,
        wave_speed=1.0,
        domain_length=2.0,
    )


@pytest.mark.parametrize(
    "errors, e_max, e_min",
    [
        ([100.0, 1.0, 0.05, 0.0001], 0.1, 0.01),
        ([10.0, 0.05, 1.0, 0.002], 0.1, 0.01),
        ([1e-9, 3.0, 0.7, 1e-5], 0.1, 0.01),
        ([0.0, 2.0, 0.05, 1e-4], 0.1, 0.01),
        ([50.0, 5.0, 500.0, 2000.0], 10.0, 100.0),
        ([0.3, 0.02, 7.0, 1e-6], 0.5, 0.25),
        ([4e-7, 9e-8, 2e-3, 6e-5], 1e-4, 1e-6),
    ],
    ids=["test_queue's", "one neutral", "wide", "a zero", "crossed thresholds", "beta 2", "small"],
)
def test_at_one_column_under_dynamo_the_order_is_build_queues(errors, e_max, e_min):
    """The formula written twice, pinned: with the same thresholds, on
    errors whose priorities are all different, the 2D queue's positions are
    the sequential queue's element ids. (Where priorities tie, build_queue
    promises no order, so only distinct ones are compared.)"""
    ids = [10, 20, 30, 40]
    cache, cfg = cache_of(errors, state=_state_1d()), config("dynamo")
    thresholds = np.array([e_max]), np.array([e_min])
    priority = round_priority(cache, cfg, *thresholds)
    assert len(set(priority.tolist())) == 4
    order = round_order(cache, cfg, *thresholds)
    assert [ids[p] for p in order.tolist()] == build_queue(_old_state(ids), np.array(errors), e_max, e_min)


def test_at_one_column_under_dynamo_with_its_own_thresholds():
    """The whole 1D path: alpha_thresholds, then the queue, both ways."""
    errors, ids = [0.5, 0.004, 0.9, 1e-7], [7, 5, 3, 1]
    cache, cfg = cache_of(errors, state=_state_1d()), config("dynamo")
    e_max, e_min = alpha_thresholds(np.array(errors), 0.1, BETA)
    order = round_order(cache, cfg, *round_thresholds(cache, cfg, BETA))
    assert [ids[p] for p in order.tolist()] == build_queue(_old_state(ids), np.array(errors), e_max, e_min)


# --- What the priority and the order reject (Q12) ----------------------------------

FUNCTIONS = {
    "round_thresholds": lambda cache, cfg: round_thresholds(cache, cfg, BETA),
    "round_priority": lambda cache, cfg: round_priority(cache, cfg, np.array([0.1]), np.array([0.01])),
    "round_order": lambda cache, cfg: round_order(cache, cfg, np.array([0.1]), np.array([0.01])),
}


@pytest.mark.parametrize("name", list(FUNCTIONS))
def test_an_indicator_the_cache_does_not_hold_is_rejected(name):
    with pytest.raises(ValueError, match=r"^indicator: 'other' is not in the cache, which holds \['e'\]$"):
        FUNCTIONS[name](cache_of(C0), config(indicator="other"))


@pytest.mark.parametrize("name", list(FUNCTIONS))
def test_a_reduction_that_is_not_the_caches_is_rejected(name):
    with pytest.raises(ValueError, match=r"^reduction: the cache was read under 'scalar', the configuration says"):
        FUNCTIONS[name](cache_of(C0), config(reduction="per_variable"))


@pytest.mark.parametrize("name", list(FUNCTIONS))
def test_dynamo_at_two_columns_is_rejected(name):
    with pytest.raises(ValueError, match=r"^normalization: 'dynamo' takes one column per indicator.*n_comp = 2$"):
        FUNCTIONS[name](cache_of(C0, C1), config("dynamo", "per_variable"))


@pytest.mark.parametrize("function", [round_priority, round_order], ids=lambda fn: fn.__name__)
@pytest.mark.parametrize(
    "e_max, e_min, pattern",
    [
        ([0.1], np.array([0.01]), r"^e_max: expected a numpy array of shape \(1,\)$"),
        (0.1, np.array([0.01]), r"^e_max: expected a numpy array of shape \(1,\)$"),
        (np.array([0.1, 0.1]), np.array([0.01]), r"^e_max: expected a numpy array of shape \(1,\)$"),
        (np.array([[0.1]]), np.array([0.01]), r"^e_max: expected a numpy array of shape \(1,\)$"),
        (np.array([0.1]), np.array([0.01, 0.01]), r"^e_min: expected a numpy array of shape \(1,\)$"),
        (np.array([0.1]), None, r"^e_min: expected a numpy array of shape \(1,\)$"),
        (np.array([-0.1]), np.array([0.01]), r"^e_max: \[-0.1\], expected finite values, none negative$"),
        (np.array([0.1]), np.array([-0.01]), r"^e_min: \[-0.01\], expected finite values, none negative$"),
        (np.array([np.inf]), np.array([0.01]), r"^e_max: \[inf\], expected finite values, none negative$"),
        (np.array([0.1]), np.array([np.nan]), r"^e_min: \[nan\], expected finite values, none negative$"),
    ],
    ids=[
        "a list",
        "a float",
        "two upper thresholds",
        "a 2-D array",
        "two lower thresholds",
        "None",
        "a negative e_max",
        "a negative e_min",
        "an infinite e_max",
        "a nan e_min",
    ],
)
def test_thresholds_of_the_wrong_form_are_rejected(function, e_max, e_min, pattern):
    with pytest.raises(ValueError, match=pattern):
        function(cache_of(C0), config(), e_max, e_min)


def test_one_threshold_pair_per_column_is_required():
    cache, cfg = cache_of(C0, C1), config(reduction="per_variable")
    with pytest.raises(ValueError, match=r"^e_max: expected a numpy array of shape \(2,\)$"):
        round_order(cache, cfg, np.array([0.1]), np.array([0.01]))


# --- next_undecided: the instance's walks (Q9) -------------------------------------

def run_walk(state: SolverState, order: np.ndarray, rule: str, choices: dict[int, str]):
    """One round along an order, as next_undecided's docstring writes it:
    each asked position takes its choice, or holds. Returns the last round
    state, the steps as (position, cursor, passed, letters, count), the
    last call's result, and the total skipped."""
    round_state = start_round(state, 1, 2)
    steps, skipped = [], 0
    position, cursor, passed = next_undecided(state, round_state, order, 0)
    while position is not None:
        skipped += passed
        action = ACTION[choices.get(position, "hold")]
        assert action_mask(state, round_state, position, rule)[action]
        round_state = commit(state, round_state, position, action, rule)
        steps.append((position, cursor, passed, letters(round_state.decisions), round_state.committed))
        position, cursor, passed = next_undecided(state, round_state, order, cursor)
    return round_state, steps, (position, cursor, passed), skipped + passed


def instance_order() -> np.ndarray:
    return queue_of(cache_of(C0), config())[1]


def test_walk_1_a_refine_closes_coarsen_and_nothing_is_skipped():
    """Section 2.5.2's first walk: every position is asked, in queue order."""
    state = instance_state()
    _, steps, last, skipped = run_walk(state, instance_order(), "first_decider", {3: "refine"})
    assert steps == [
        (4, 1, 0, "....H", 5),
        (3, 2, 0, "...RH", 8),
        (0, 3, 0, "H..RH", 8),
        (1, 4, 0, "HH.RH", 8),
        (2, 5, 0, "HHHRH", 8),
    ]
    assert last == (None, 5, 0) and skipped == 0


def test_walk_1_the_refine_closed_coarsen_to_the_over_refined_sibling():
    """The queue chose the family's decider: position 3, the under-refined
    one. Position 0, over-refined, is asked after it with coarsen closed."""
    state, order = instance_state(), instance_order()
    round_state = commit(state, start_round(state, 1, 2), B, ACTION_HOLD, "first_decider")
    assert next_undecided(state, round_state, order, 1) == (3, 2, 0)
    assert action_mask(state, round_state, 3, "first_decider").tolist() == [True, True, True]
    round_state = commit(state, round_state, 3, ACTION_REFINE, "first_decider")
    assert next_undecided(state, round_state, order, 2) == (0, 3, 0)
    assert action_mask(state, round_state, 0, "first_decider").tolist() == [False, True, True]


def test_walk_2_a_family_coarsen_skips_the_three_siblings_in_one_call():
    """Section 2.5.2's second walk: two asked, three passed over."""
    state = instance_state()
    round_state, steps, last, skipped = run_walk(state, instance_order(), "first_decider", {3: "coarsen"})
    assert steps == [(4, 1, 0, "....H", 5), (3, 2, 0, "CCCCH", 2)]
    assert last == (None, 5, 3) and skipped == 3
    assert len(steps) + skipped == state.n_active
    assert (round_state.decisions != UNDECIDED).all()


def test_under_unanimous_nothing_is_skipped():
    state = instance_state()
    _, steps, last, skipped = run_walk(state, instance_order(), "unanimous", {3: "coarsen", 0: "coarsen"})
    assert [step[0] for step in steps] == list(ORDER)
    assert steps[-1][3:] == ("CHHCH", 5)
    assert last == (None, 5, 0) and skipped == 0


def test_under_any_coarsen_an_overwritten_hold_was_asked_not_skipped():
    """Position 3 held before position 0's coarsen took the family, so the
    round is three asked and two skipped: a family coarsen skips at most
    three."""
    state = instance_state()
    round_state, steps, last, skipped = run_walk(state, instance_order(), "any_coarsen", {0: "coarsen"})
    assert steps == [(4, 1, 0, "....H", 5), (3, 2, 0, "...HH", 5), (0, 3, 0, "CCCCH", 2)]
    assert last == (None, 5, 2) and skipped == 2
    assert letters(round_state.decisions) == "CCCCH"


def test_skipped_positions_between_asked_ones_are_counted_where_they_fall():
    """An order that puts a sibling before cell B: after position 0's
    family coarsen the next call passes 1, 2 and 3 and returns B with 3."""
    state = instance_state()
    order = np.array([0, 1, 2, 3, 4])
    round_state = commit(state, start_round(state, 1, 2), 0, ACTION_COARSEN, "first_decider")
    assert next_undecided(state, round_state, order, 1) == (4, 5, 3)
    order = np.array([0, 1, 4, 2, 3])
    assert next_undecided(state, round_state, order, 1) == (4, 3, 1)
    round_state = commit(state, round_state, 4, ACTION_HOLD, "first_decider")
    assert next_undecided(state, round_state, order, 3) == (None, 5, 2)


def test_an_exhausted_queue_asked_again():
    state, order = instance_state(), instance_order()
    round_state, _, _, _ = run_walk(state, order, "first_decider", {3: "coarsen"})
    assert next_undecided(state, round_state, order, 5) == (None, 5, 0)
    assert next_undecided(state, round_state, order, 0) == (None, 5, 5)


def test_the_step_holds_no_state():
    """The same arguments give the same answer, and nothing handed in is
    changed: the cursor is the caller's to keep."""
    state, order = instance_state(), instance_order()
    round_state = start_round(state, 1, 2)
    kept = round_state.decisions.copy()
    assert next_undecided(state, round_state, order, 1) == (3, 2, 0)
    assert next_undecided(state, round_state, order, 1) == (3, 2, 0)
    assert next_undecided(state, round_state, order, 0) == (4, 1, 0)
    assert np.array_equal(round_state.decisions, kept) and tuple(order.tolist()) == ORDER


def test_the_step_returns_python_integers():
    state, order = instance_state(), instance_order()
    position, cursor, skipped = next_undecided(state, start_round(state, 1, 2), order, np.int64(0))
    assert (type(position), type(cursor), type(skipped)) == (int, int, int)


def test_a_round_state_with_no_record_is_a_rounds_start():
    state, order = instance_state(), instance_order()
    bare = RoundState(round_index=1, n_rounds=2, committed=5)
    assert next_undecided(state, bare, order, 0) == (4, 1, 0)


def test_at_ndim_1_a_pair_coarsen_skips_the_sibling():
    state = _state_1d()
    order = np.array([1, 0, 2, 3])
    round_state = commit(state, start_round(state, 1, 2), 1, ACTION_COARSEN, "first_decider")
    assert next_undecided(state, round_state, order, 1) == (0, 2, 0)
    round_state = commit(state, round_state, 0, ACTION_HOLD, "first_decider")
    assert next_undecided(state, round_state, order, 2) == (3, 4, 1)


# --- What next_undecided rejects (Q12) ---------------------------------------------

@pytest.mark.parametrize(
    "order, cursor, pattern",
    [
        (np.array([4, 3, 0, 1]), 0, r"^order: expected a numpy array of shape \(5,\)$"),
        (np.array([4, 3, 0, 1, 2, 2]), 0, r"^order: expected a numpy array of shape \(5,\)$"),
        ([4, 3, 0, 1, 2], 0, r"^order: expected a numpy array of shape \(5,\)$"),
        (None, 0, r"^order: expected a numpy array of shape \(5,\)$"),
        (np.array(ORDER), 6, r"^cursor: 6, expected an integer in \[0, 5\]$"),
        (np.array(ORDER), -1, r"^cursor: -1, expected an integer in \[0, 5\]$"),
        (np.array(ORDER), 1.0, r"^cursor: 1.0, expected an integer in \[0, 5\]$"),
        (np.array(ORDER), True, r"^cursor: True, expected an integer in \[0, 5\]$"),
        (np.array(ORDER), None, r"^cursor: None, expected an integer in \[0, 5\]$"),
        (np.array([5, 3, 0, 1, 2]), 0, r"^position: 5, expected an integer in \[0, 5\)$"),
        (np.array([-1, 3, 0, 1, 2]), 0, r"^position: -1, expected an integer in \[0, 5\)$"),
    ],
    ids=[
        "an order one short",
        "an order one long",
        "a list",
        "None",
        "a cursor past the end",
        "a negative cursor",
        "a float cursor",
        "a bool cursor",
        "no cursor",
        "an entry past the end",
        "a negative entry",
    ],
)
def test_the_step_rejects(order, cursor, pattern):
    state = instance_state()
    with pytest.raises(ValueError, match=pattern):
        next_undecided(state, start_round(state, 1, 2), order, cursor)


def test_the_step_rejects_a_record_of_the_wrong_length():
    state = instance_state()
    short = RoundState(round_index=1, n_rounds=2, committed=5, decisions=np.full(4, UNDECIDED, dtype=np.int8))
    with pytest.raises(ValueError, match=r"^decisions: length 4, expected n_active = 5$"):
        next_undecided(state, short, np.array(ORDER), 0)


def test_a_repeated_position_is_not_caught_by_the_step():
    """The limit section 2.5.3 states: only the order's length is checked.
    A position that appears twice is returned twice while it is undecided,
    and the mask refuses the second ask once it has decided."""
    state = instance_state()
    order = np.array([4, 4, 0, 1, 2])
    round_state = start_round(state, 1, 2)
    assert next_undecided(state, round_state, order, 0) == (4, 1, 0)
    assert next_undecided(state, round_state, order, 1) == (4, 2, 0)
    round_state = commit(state, round_state, 4, ACTION_HOLD, "first_decider")
    assert next_undecided(state, round_state, order, 1) == (0, 3, 1)


# --- The vocabulary (Q13) ----------------------------------------------------------

@pytest.mark.parametrize(
    "fn", [round_thresholds, round_priority, round_order, next_undecided], ids=lambda fn: fn.__name__
)
def test_no_function_takes_a_training_only_type(fn):
    """All four belong to the within-round logic that is implemented twice."""
    assert training_only_params(fn) == []


# --- On the mock quad backend ------------------------------------------------------

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


def _mock_config(indicator: str, normalization: str = "relative") -> ObservationConfig:
    return ObservationConfig(
        indicator=indicator, reduction="scalar", alpha=0.1, element_budget=64, normalization=normalization,
        floor_decades=8.0,
    )


def _families(state: SolverState) -> list[tuple[int, ...]]:
    """Each complete family once, as its positions in child order."""
    return sorted({tuple(row) for row in state.family.tolist() if row[0] >= 0})


def _whole_families(state: SolverState, marks: np.ndarray) -> int:
    """How many complete families carry COARSEN on every member."""
    return sum(bool((marks[list(family)] == COARSEN).all()) for family in _families(state))


def _recount(state: SolverState, decisions: np.ndarray) -> int:
    """The committed count from scratch, as tests/agent/test_round_commit.py
    writes it: n_active, plus 3 per REFINE entry, minus 3 per family whose
    entries are all COARSEN."""
    arity = 2**state.ndim - 1
    return state.n_active + arity * int((decisions == REFINE).sum()) - arity * _whole_families(state, decisions)


def _slow_priority(errors: np.ndarray, e_max: float, e_min: float, floor: float) -> list[float]:
    """The priority at one column, as a loop: the reference. The floor is
    the rule's, handed in."""
    out = []
    for e in errors.tolist():
        e = max(e, floor)
        if e > e_max > 0.0:
            out.append(math.log10(e / e_max))
        elif e < e_min and e_min > 0.0:
            out.append(math.log10(e_min / e))
        else:
            out.append(0.0)
    return out


def _queue_round(mock, state, rule, indicator, choose, round_index=1):
    """One round as a driver would run it: one read, the thresholds and the
    order from it, then start_round and next_undecided until the queue is
    exhausted, each asked position committing choose(mask).

    Returns the last round state, the positions asked in order, the number
    skipped, and the cache and the order used.
    """
    cfg = _mock_config(indicator)
    cache = read_round(mock, state, [indicator], "scalar")
    order = round_order(cache, cfg, *round_thresholds(cache, cfg, BETA))
    round_state = start_round(state, round_index, 2)
    asked, skipped = [], 0
    position, cursor, passed = next_undecided(state, round_state, order, 0)
    while position is not None:
        skipped += passed
        mask = action_mask(state, round_state, position, rule)
        round_state = commit(state, round_state, position, choose(mask), rule)
        assert round_state.committed == _recount(state, round_state.decisions)
        asked.append(position)
        position, cursor, passed = next_undecided(state, round_state, order, cursor)
    assert cursor == state.n_active
    return round_state, asked, skipped + passed, cache, order


def _prefer(action: int):
    """A policy: the given action wherever it is open, else hold."""
    return lambda mask: action if mask[action] else ACTION_HOLD


def test_the_mocks_two_cell_mesh_is_the_instance():
    """With the rotation off, positions 0-3 are cell A's children and
    position 4 is cell B, as the hand-built state has them."""
    _, state = _two_cell_mesh()
    literal = instance_state()
    assert np.array_equal(state.family, literal.family)
    assert np.array_equal(state.can_coarsen, literal.can_coarsen)
    assert np.array_equal(state.centroid, literal.centroid)


@pytest.mark.parametrize("normalization", RULES)
@pytest.mark.parametrize("indicator", INDICATORS)
@pytest.mark.parametrize("mesh", list(MESHES))
def test_on_the_mock_the_priority_is_the_loops(mesh, indicator, normalization):
    """Against a reference written as a loop with math.log10, at every
    position, on the mock's own indicator values, zeros included."""
    mock, state = MESHES[mesh]()
    cfg = _mock_config(indicator, normalization)
    cache = read_round(mock, state, [indicator], "scalar")
    e_max, e_min = round_thresholds(cache, cfg, BETA)
    floor = 1e-30 if normalization == "dynamo" else e_max[0] * 1e-8
    expected = _slow_priority(cache.errors[indicator][:, 0], e_max[0], e_min[0], floor)
    priority = round_priority(cache, cfg, e_max, e_min)
    assert priority.tolist() == pytest.approx(expected, abs=1e-12)
    assert np.isfinite(priority).all() and (priority >= 0.0).all()


@pytest.mark.parametrize("normalization", RULES)
@pytest.mark.parametrize("indicator", INDICATORS)
@pytest.mark.parametrize("mesh", list(MESHES))
def test_on_the_mock_the_order_is_a_stable_sort(mesh, indicator, normalization):
    mock, state = MESHES[mesh]()
    cfg = _mock_config(indicator, normalization)
    cache = read_round(mock, state, [indicator], "scalar")
    thresholds = round_thresholds(cache, cfg, BETA)
    priority, order = round_priority(cache, cfg, *thresholds), round_order(cache, cfg, *thresholds)
    assert sorted(order.tolist()) == list(range(state.n_active))
    assert order.tolist() == sorted(range(state.n_active), key=lambda p: (-priority[p], p))
    assert not order.flags.writeable


@pytest.mark.parametrize("mesh", ["4x4 periodic", "4x4 walled", "4x4 mixed"])
def test_on_the_mock_the_relative_queue_holds_all_three_classes(mesh):
    """So that the rounds below are not walked over one class alone: under
    relative, on ind_b, every graded mesh has under-refined, neutral and
    over-refined elements, and the round's largest error reads exactly one
    decade above e_max (alpha is 0.1)."""
    indicator = "ind_b"
    mock, state = MESHES[mesh]()
    cfg = _mock_config(indicator)
    cache = read_round(mock, state, [indicator], "scalar")
    e_max, e_min = round_thresholds(cache, cfg, BETA)
    errors = cache.errors[indicator][:, 0]
    assert (errors > e_max[0]).any() and (errors < e_min[0]).any()
    assert ((errors <= e_max[0]) & (errors >= e_min[0])).any()
    priority = round_priority(cache, cfg, e_max, e_min)
    assert priority[int(errors.argmax())] == pytest.approx(1.0)


# --- Whole rounds in queue order (Q10) ---------------------------------------------

@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("indicator", INDICATORS)
@pytest.mark.parametrize("rule", COARSEN_RULES)
@pytest.mark.parametrize("mesh", list(MESHES))
def test_a_round_of_random_masked_actions_in_queue_order(mesh, rule, indicator, seed):
    """Actions drawn uniformly from the mask, positions asked in queue
    order. Asked plus skipped is n_active; the asked positions are the
    queue with the skipped ones taken out; every position ends decided; the
    count is the recount (checked after every commit in _queue_round); and
    the backend takes the record as its marks."""
    mock, state = MESHES[mesh]()
    rng = np.random.default_rng(seed)
    round_state, asked, skipped, _, order = _queue_round(
        mock, state, rule, indicator, lambda mask: int(rng.choice(np.flatnonzero(mask)))
    )
    assert len(asked) + skipped == state.n_active
    assert len(set(asked)) == len(asked)
    assert asked == [p for p in order.tolist() if p in set(asked)]
    assert (round_state.decisions != UNDECIDED).all()

    families = _whole_families(state, round_state.decisions)
    if rule == "unanimous":
        assert skipped == 0
    elif rule == "first_decider":
        assert skipped == 3 * families
    else:
        assert skipped <= 3 * families

    new_state, _ = mock.apply_marks(round_state.decisions)
    assert new_state.n_active >= round_state.committed


@pytest.mark.parametrize("indicator", INDICATORS)
@pytest.mark.parametrize("rule", COARSEN_RULES)
@pytest.mark.parametrize("mesh", list(MESHES))
def test_a_round_of_holds_asks_every_position_in_queue_order(mesh, rule, indicator):
    mock, state = MESHES[mesh]()
    round_state, asked, skipped, _, order = _queue_round(mock, state, rule, indicator, _prefer(ACTION_HOLD))
    assert asked == order.tolist() and skipped == 0
    assert round_state.committed == state.n_active


@pytest.mark.parametrize("indicator", INDICATORS)
@pytest.mark.parametrize("rule", ["first_decider", "any_coarsen"])
@pytest.mark.parametrize("mesh", list(MESHES))
def test_coarsen_wherever_open_each_family_is_decided_by_its_first_in_the_queue(mesh, rule, indicator):
    """The queue chooses each family's decider. With coarsen taken wherever
    it is open, exactly one member of every complete family is asked: the
    one the queue reaches first, its member of highest priority (the lowest
    position among equals). Three are skipped per family."""
    mock, state = MESHES[mesh]()
    families = _families(state)
    round_state, asked, skipped, cache, order = _queue_round(mock, state, rule, indicator, _prefer(ACTION_COARSEN))
    cfg = _mock_config(indicator)
    priority = round_priority(cache, cfg, *round_thresholds(cache, cfg, BETA))
    rank = {p: i for i, p in enumerate(order.tolist())}
    for family in families:
        deciders = [p for p in family if p in set(asked)]
        assert deciders == [min(family, key=lambda p: rank[p])]
        assert deciders == [min(family, key=lambda p: (-priority[p], p))]
    assert skipped == 3 * len(families)
    assert len(asked) == state.n_active - 3 * len(families)
    assert round_state.committed == state.n_active - 3 * len(families)


@pytest.mark.parametrize("indicator", INDICATORS)
@pytest.mark.parametrize("mesh", list(MESHES))
def test_coarsen_wherever_open_under_unanimous_asks_everyone(mesh, indicator):
    mock, state = MESHES[mesh]()
    round_state, asked, skipped, _, order = _queue_round(mock, state, "unanimous", indicator, _prefer(ACTION_COARSEN))
    assert asked == order.tolist() and skipped == 0
    assert round_state.committed == state.n_active - 3 * len(_families(state))


@pytest.mark.parametrize("rule", COARSEN_RULES)
@pytest.mark.parametrize("mesh", list(MESHES))
def test_two_rounds_each_with_its_own_queue(mesh, rule):
    """Two rounds as a driver would run them: a new read, new thresholds
    and a new order per round, on the mesh the round before realized."""
    mock, state = MESHES[mesh]()
    rng = np.random.default_rng(7)
    for round_index in (1, 2):
        round_state, asked, skipped, cache, order = _queue_round(
            mock, state, rule, "ind_b", lambda mask: int(rng.choice(np.flatnonzero(mask))), round_index=round_index
        )
        assert cache.state is state and order.shape == (state.n_active,)
        assert len(asked) + skipped == state.n_active
        assert round_state.round_index == round_index
        state, _ = mock.apply_marks(round_state.decisions)


def test_the_random_rounds_skip_somewhere_and_nowhere():
    """So that the identity above is not checked on zeros alone."""
    some = none = 0
    for mesh in MESHES:
        for rule in ("first_decider", "any_coarsen"):
            for seed in (0, 1, 2):
                mock, state = MESHES[mesh]()
                rng = np.random.default_rng(seed)
                _, _, skipped, _, _ = _queue_round(
                    mock, state, rule, "ind_b", lambda mask, rng=rng: int(rng.choice(np.flatnonzero(mask)))
                )
                some += skipped > 0
                none += skipped == 0
    assert some > 0 and none > 0


# --- The queue makes no backend call (Q11) -----------------------------------------

@pytest.mark.parametrize("rotate", [False, True], ids=["pinned", "rotating"])
def test_two_rounds_with_a_queue_cost_one_read_each(counting, rotate):
    """Section 2.1.2's instance with the queue built and walked in each
    round: one compute_error per round, 2R + 1 calls after reset, and no
    get_state."""
    mock = counting(
        MockQuadBackend(
            nx=2, ny=1, periodic=(True, False), lo=(0.0, 0.0), hi=(2.0, 1.0), max_level=2, dt_base=0.25,
            rotate_cells=rotate,
        )
    )
    state = mock.reset({"x0": 0.5, "y0": 0.5, "width": 0.25}, amr_freq=4, initial_level=0)
    for round_index in (1, 2):
        choose = _prefer(ACTION_REFINE) if round_index == 1 else _prefer(ACTION_HOLD)
        round_state, asked, skipped, _, _ = _queue_round(mock, state, "first_decider", "ind_b", choose, round_index)
        assert len(asked) + skipped == state.n_active
        state, _ = mock.apply_marks(round_state.decisions)
    mock.step_interval(["ind_b"], "scalar", 1)
    assert dict(mock.calls) == {"reset": 1, "compute_error": 2, "apply_marks": 2, "step_interval": 1}
