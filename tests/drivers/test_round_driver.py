"""Tests for the interval driver of the 2D stack, drivers/round_driver.py:
one remesh interval held in one place, on the mock quad backend.

AGENT_2D_ROADMAP item 4.1, signals D1-D9. No RL library, no environment and
no reward is imported: the driver is walked by scripted and by random
masked actions. Nine groups:

  - construction: no backend call, what is refused, the threshold timing;
  - the reset: one reset and one read, what reaches the backend;
  - the instance of the explanation file's section 4.1.2, step by step
    (D1, D2, D4);
  - the calls: 2R + 1 per interval, none between an interval's end and
    begin_interval (D3);
  - the driver against a reference loop: the interval written as a plain
    loop over the core's functions, given the same actions (D5);
  - run_interval, the loop with a policy and no rewards (D6);
  - a backend that fails (D8);
  - what is refused, and when (D9);
  - what the module imports, and where the thresholds are assigned (D7).

Run: pytest tests/drivers/test_round_driver.py -v
"""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

import agent.round_spec as round_spec
import drivers.round_driver as round_driver
from agent.round_cache import read_round
from agent.round_commit import commit, overshoot, start_round
from agent.round_mask import ACTION_COARSEN, ACTION_HOLD, ACTION_REFINE, COARSEN_RULES, action_mask
from agent.round_observation import ObservationConfig, actor_observation
from agent.round_queue import next_undecided, round_order, round_thresholds
from agent.round_spec import ObservationSpec
from backends.mock_quad.backend import MockQuadBackend
from contract.backend import BackendFailure
from contract.reports import IntervalReport
from contract.state import SolverState
from drivers.round_driver import THRESHOLD_TIMING, RoundDriver, RoundRecord

C, H, R = ACTION_COARSEN, ACTION_HOLD, ACTION_REFINE
# The feature: off every centroid, so no two of the instance's errors tie.
IC = {"x0": 0.3, "y0": 0.6, "width": 0.2}
BETA = 2
ALPHA = 0.1
# Literal values are the explanation file's, printed to six decimals.
TOL = 2e-6


# --- The backends, the spec and the driver -----------------------------------------

def _two_cells(max_level: int = 2, nvar: int = 1) -> MockQuadBackend:
    """Cells A and B on [0,2]x[0,1], periodic in x and walled in y, the
    rotation off: position 0 is A and position 1 is B at level 0."""
    return MockQuadBackend(
        nx=2, ny=1, periodic=(True, False), lo=(0.0, 0.0), hi=(2.0, 1.0), max_level=max_level, nvar=nvar,
        dt_base=0.25, rotate_cells=False,
    )


def _grid(periodic: tuple[bool, bool], nvar: int = 1, max_level: int = 2) -> MockQuadBackend:
    """A 4x4 grid on the unit square, the rotation on."""
    return MockQuadBackend(
        nx=4, ny=4, periodic=periodic, max_level=max_level, nvar=nvar, dt_base=0.25, rotate_cells=True
    )


BACKENDS = {
    "two cells": _two_cells,
    "4x4 periodic": lambda nvar=1: _grid((True, True), nvar),
    "4x4 walled": lambda nvar=1: _grid((False, False), nvar),
    "4x4 mixed": lambda nvar=1: _grid((True, False), nvar),
}


def _config(
    indicator: str = "ind_b", reduction: str = "scalar", normalization: str = "relative", budget: int = 8
) -> ObservationConfig:
    return ObservationConfig(
        indicator=indicator, reduction=reduction, alpha=ALPHA, element_budget=budget, normalization=normalization,
        floor_decades=8.0,
    )


def _spec(
    config: ObservationConfig | None = None, rule: str = "first_decider", max_level: int = 2, nvar: int = 1
) -> ObservationSpec:
    config = _config() if config is None else config
    return ObservationSpec(
        observation=config, beta=BETA, coarsen_rule=rule, ndim=2, max_level=max_level, n_rounds=max_level,
        n_comp=1 if config.reduction == "scalar" else nvar,
    )


def _driver(backend=None, spec: ObservationSpec | None = None, **keywords) -> RoundDriver:
    return RoundDriver(_two_cells() if backend is None else backend, _spec() if spec is None else spec, **keywords)


def _counts(record: RoundRecord | None) -> tuple[int, ...] | None:
    if record is None:
        return None
    return (record.interval, record.round_index, record.committed, record.overshoot, record.skipped, record.n_active)


class _Recording:
    """Forward every verb, and keep the arguments of one."""

    def __init__(self, inner, verb):
        self.inner, self.verb, self.seen = inner, verb, []

    def __getattr__(self, name):
        attr = getattr(self.inner, name)
        if name != self.verb:
            return attr

        def recorded(*args, **kwargs):
            self.seen.append((args, kwargs))
            return attr(*args, **kwargs)

        return recorded


# --- Two ways to run intervals: through the driver, and as a plain loop ------------

def _listed(actions):
    """Actions read off a list, by step."""
    return lambda step, mask: actions[step]


def _drawn(seed: int):
    """Random masked actions fixed by the step's number alone, so the driver
    and the reference loop are given the same ones for as long as they
    agree."""

    def choose(step, mask):
        open_actions = np.flatnonzero(mask)
        return int(open_actions[int(np.random.default_rng([seed, step]).random() * len(open_actions))])

    return choose


def _holds(step, mask):
    return H


def _walk(driver: RoundDriver, n_intervals: int, choose, *, ic=IC, amr_freq=2, initial_level=0):
    """n_intervals through reset, commit, advance and begin_interval. One
    dict per decision: what was asked and seen, the action, and what the
    advance ended; and the interval reports, in order."""
    driver.reset(ic, amr_freq, initial_level)
    steps, reports = [], []
    while True:
        e_max, e_min = driver.thresholds
        step = dict(
            position=driver.position, obs=driver.actor_observation(), mask=driver.action_mask(),
            interval=driver.interval + 1, round_index=driver.round_state.round_index, e_max=e_max, e_min=e_min,
        )
        step["action"] = choose(len(steps), step["mask"])
        assert step["mask"][step["action"]]
        driver.commit(step["action"])
        step["ended"] = driver.advance()
        step["record"] = driver.round_record
        steps.append(step)
        assert len(steps) < 5000
        if step["ended"] == "interval":
            reports.append(driver.interval_report)
            if driver.interval == n_intervals:
                return steps, reports
            driver.begin_interval()


def _reference(
    backend, spec: ObservationSpec, n_intervals: int, choose, *, ic=IC, amr_freq=2, initial_level=0, cadence=1,
    accumulate=(),
):
    """The same intervals as a plain loop over the core's functions, the
    way tests/agent/test_round_nvar3.py walks one: the pair from each
    interval's first cache and held; per round one read, the order under
    that pair, the walk, apply_marks; after the last round one
    step_interval. The same dicts and reports as _walk."""
    config, rule = spec.observation, spec.coarsen_rule
    state = backend.reset(ic, amr_freq, initial_level)
    steps, reports = [], []
    for interval in range(1, n_intervals + 1):
        pair = None
        for round_index in range(1, spec.n_rounds + 1):
            cache = read_round(backend, state, [config.indicator], config.reduction)
            if pair is None:
                pair = round_thresholds(cache, config, spec.beta)
            order = round_order(cache, config, *pair)
            round_state = start_round(state, round_index, spec.n_rounds)
            position, cursor, skipped = next_undecided(state, round_state, order, 0)
            while position is not None:
                mask = action_mask(state, round_state, position, rule)
                steps.append(
                    dict(
                        position=position, obs=actor_observation(cache, round_state, position, config), mask=mask,
                        interval=interval, round_index=round_index, e_max=pair[0], e_min=pair[1],
                        action=choose(len(steps), mask), ended="element", record=None,
                    )
                )
                round_state = commit(state, round_state, position, steps[-1]["action"], rule)
                position, cursor, passed = next_undecided(state, round_state, order, cursor)
                skipped += passed
            new_state, _ = backend.apply_marks(round_state.decisions)
            steps[-1].update(
                ended="round",
                record=RoundRecord(
                    interval=interval, round_index=round_index, committed=round_state.committed,
                    overshoot=overshoot(round_state, new_state), skipped=skipped, n_active=new_state.n_active,
                ),
            )
            state = new_state
        reports.append(backend.step_interval(accumulate, config.reduction, cadence))
        steps[-1]["ended"] = "interval"
    return steps, reports


def _same(ours: list[dict], theirs: list[dict]) -> None:
    """The two runs agree exactly: the same functions on the same arrays."""
    assert len(ours) == len(theirs)
    for one, other in zip(ours, theirs, strict=True):
        assert set(one) == set(other)
        for key in other:
            if key in ("obs", "mask", "e_max", "e_min"):
                assert np.array_equal(one[key], other[key]), key
            else:
                assert one[key] == other[key], key


def _same_reports(ours: list[IntervalReport], theirs: list[IntervalReport]) -> None:
    assert len(ours) == len(theirs)
    for one, other in zip(ours, theirs, strict=True):
        assert (one.time, one.n_substeps) == (other.time, other.n_substeps)
        assert list(one.acc) == list(other.acc)
        assert all(np.array_equal(one.acc[name], other.acc[name]) for name in other.acc)
        assert np.array_equal(one.mass, other.mass)


# --- Construction ------------------------------------------------------------------

def test_construction_calls_no_verb_of_the_backend(counting):
    backend = counting(_two_cells())
    _driver(backend)
    _driver(backend, cadence=3, accumulate=["ind_b"])
    assert dict(backend.calls) == {}


def test_the_driver_keeps_what_it_was_given():
    backend, spec = _two_cells(), _spec()
    driver = RoundDriver(backend, spec)
    assert driver.backend is backend and driver.spec is spec
    assert (driver.cadence, driver.accumulate) == (1, ())
    other = RoundDriver(backend, spec, cadence=np.int64(3), accumulate=["ind_b", "ind_a"])
    assert (other.cadence, other.accumulate) == (3, ("ind_b", "ind_a"))
    assert type(other.cadence) is int


REFUSED = [
    (dict(spec=None), "spec: None, expected an ObservationSpec"),
    (dict(spec="ind_b"), "spec: 'ind_b', expected an ObservationSpec"),
    (dict(cadence=0), "cadence: 0, expected an integer of at least 1"),
    (dict(cadence=True), "cadence: True, expected an integer of at least 1"),
    (dict(cadence=2.0), "cadence: 2.0, expected an integer of at least 1"),
    (dict(cadence="1"), "cadence: '1', expected an integer of at least 1"),
    (dict(accumulate="ind_b"), "accumulate: expected a sequence of names, got the string 'ind_b'"),
]


@pytest.mark.parametrize("keywords, message", REFUSED, ids=[message.split(",")[0] for _, message in REFUSED])
def test_a_bad_argument_is_refused_by_name(keywords, message):
    arguments = dict(spec=_spec())
    arguments.update(keywords)
    spec = arguments.pop("spec")
    with pytest.raises(ValueError) as caught:
        RoundDriver(_two_cells(), spec, **arguments)
    assert str(caught.value) == message


def test_the_arguments_after_the_spec_are_keyword_only():
    with pytest.raises(TypeError):
        RoundDriver(_two_cells(), _spec(), 3)


def test_the_driver_implements_the_threshold_timing_the_spec_file_names():
    assert round_spec.FIXED_RULES["thresholds"] == THRESHOLD_TIMING == "interval_first"


def test_another_threshold_timing_is_refused_at_construction(monkeypatch):
    monkeypatch.setitem(round_spec.FIXED_RULES, "thresholds", "per_round")
    with pytest.raises(ValueError) as caught:
        _driver()
    assert str(caught.value) == "thresholds: the fixed rule is 'per_round', this driver holds 'interval_first'"


def test_nothing_is_held_before_a_reset():
    driver = _driver()
    held = (driver.state, driver.cache, driver.thresholds, driver.round_state, driver.position)
    assert held == (None, None, None, None, None)
    assert (driver.interval, driver.round_record, driver.interval_report) == (0, None, None)


NOTHING_RUNS = [
    ("action_mask", lambda driver: driver.action_mask(), "action_mask: nothing is asked"),
    ("actor_observation", lambda driver: driver.actor_observation(), "actor_observation: nothing is asked"),
    ("commit", lambda driver: driver.commit(H), "commit: nothing is asked"),
    ("advance", lambda driver: driver.advance(), "advance: no decision has been committed"),
    ("begin_interval", lambda driver: driver.begin_interval(), "begin_interval: no interval has just ended"),
    ("run_interval", lambda driver: driver.run_interval(_holds), "run_interval: nothing is asked"),
]


@pytest.mark.parametrize("name, call, message", NOTHING_RUNS, ids=[name for name, _, _ in NOTHING_RUNS])
def test_nothing_runs_before_a_reset(counting, name, call, message):
    backend = counting(_two_cells())
    driver = _driver(backend)
    with pytest.raises(RuntimeError) as caught:
        call(driver)
    assert str(caught.value) == message
    assert dict(backend.calls) == {}


@pytest.mark.parametrize(
    "name", ["state", "cache", "thresholds", "round_state", "position", "interval", "round_record", "interval_report"]
)
def test_what_the_driver_holds_is_read_only(name):
    driver = _driver()
    driver.reset(IC, 2, 0)
    with pytest.raises(AttributeError):
        setattr(driver, name, None)


# --- The reset ---------------------------------------------------------------------

def test_the_reset_costs_one_reset_and_one_read(counting):
    backend = counting(_two_cells())
    _driver(backend).reset(IC, 2, 0)
    assert dict(backend.calls) == {"reset": 1, "compute_error": 1}


def test_the_reset_returns_the_state_it_holds_and_asks_the_first_position():
    driver = _driver()
    state = driver.reset(IC, 2, 0)
    assert isinstance(state, SolverState)
    assert state is driver.state is driver.cache.state
    assert (state.n_active, driver.position, driver.interval) == (2, 1, 0)
    assert (driver.round_state.round_index, driver.round_state.n_rounds, driver.round_state.committed) == (1, 2, 2)
    assert type(driver.position) is int
    assert (driver.round_record, driver.interval_report) == (None, None)


def test_the_episodes_values_reach_the_backends_reset():
    backend = _Recording(_grid((True, True)), "reset")
    driver = _driver(backend)
    state = driver.reset(IC, 3, 1)
    assert backend.seen == [((IC, 3, 1), {})]
    assert (state.n_active, state.amr_freq) == (64, 3)


def test_the_default_initial_level_is_one():
    assert _driver(_grid((True, True))).reset(IC, 2).n_active == 64


def test_the_indicator_and_the_reduction_reach_compute_error():
    backend = _Recording(_two_cells(nvar=3), "compute_error")
    config = _config("ind_a", "per_variable")
    _walk(_driver(backend, _spec(config, nvar=3)), 2, _holds)
    assert [(list(args[0]), args[1]) for args, _ in backend.seen] == [(["ind_a"], "per_variable")] * 4


@pytest.mark.parametrize(
    "accumulate, named", [((), []), (("ind_b",), ["ind_b"]), (["ind_b", "ind_a"], ["ind_b", "ind_a"])]
)
def test_the_accumulator_the_reduction_and_the_cadence_reach_step_interval(accumulate, named):
    backend = _Recording(_two_cells(), "step_interval")
    _, reports = _walk(_driver(backend, cadence=3, accumulate=accumulate), 2, _holds)
    assert [(list(args[0]), args[1], args[2]) for args, _ in backend.seen] == [(named, "scalar", 3)] * 2
    assert all(kwargs == {} and len(args) == 3 for args, kwargs in backend.seen)
    assert [list(report.acc) for report in reports] == [named] * 2


def test_a_bad_initial_condition_is_the_backends_to_refuse_and_leaves_nothing_asked():
    driver = _driver()
    driver.reset(IC, 2, 0)
    with pytest.raises(ValueError, match=r"^ic: unknown names \[\], missing names \['width'\]$"):
        driver.reset({"x0": 0.3, "y0": 0.6}, 2, 0)
    assert driver.position is None
    with pytest.raises(RuntimeError):
        driver.commit(H)
    assert driver.reset(IC, 2, 0).n_active == 2 and driver.position == 1


def test_a_reset_in_the_middle_of_an_interval_starts_over(instance):
    driver = _driver()
    driver.reset(IC, 2, 0)
    for action in SCRIPT[:5]:
        driver.commit(action)
        driver.advance()
    again, _ = _walk(driver, 2, _listed(SCRIPT))
    _same(again, instance.steps)
    assert driver.interval == 2


# --- The instance, step by step (D1, D2, D4) ---------------------------------------

# Section 4.1.2's walk: refine A and hold B; hold everything; coarsen A's
# family from its first-asked child and refine B; hold everything.
SCRIPT = [H, R, H, H, H, H, H, C, R, H, H, H, H, H]
# (interval, round, position, mask, what the advance ended)
TABLE = [
    (1, 1, 1, [0, 1, 1], "element"),
    (1, 1, 0, [0, 1, 1], "round"),
    (1, 2, 4, [0, 1, 1], "element"),
    (1, 2, 2, [1, 1, 1], "element"),
    (1, 2, 1, [0, 1, 1], "element"),
    (1, 2, 0, [0, 1, 1], "element"),
    (1, 2, 3, [0, 1, 1], "interval"),
    (2, 1, 3, [1, 1, 1], "element"),
    (2, 1, 4, [0, 1, 1], "round"),
    (2, 2, 2, [1, 1, 1], "element"),
    (2, 2, 4, [0, 1, 1], "element"),
    (2, 2, 0, [0, 1, 1], "element"),
    (2, 2, 1, [0, 1, 1], "element"),
    (2, 2, 3, [0, 1, 1], "interval"),
]
# The records, by the step whose advance made them.
RECORDS = {2: (1, 1, 5, 0, 0, 5), 7: (1, 2, 5, 0, 0, 5), 9: (2, 1, 5, 0, 3, 5), 14: (2, 2, 5, 0, 0, 5)}


class _Instance:
    def __init__(self) -> None:
        self.steps, self.reports = _walk(_driver(), 2, _listed(SCRIPT))


@pytest.fixture(scope="module")
def instance() -> _Instance:
    return _Instance()


def test_the_instance_takes_fourteen_decisions(instance):
    assert len(instance.steps) == 14 and len(instance.reports) == 2


@pytest.mark.parametrize("number", range(14))
def test_the_instances_step(instance, number):
    interval, round_index, position, mask, ended = TABLE[number]
    step = instance.steps[number]
    assert (step["interval"], step["round_index"], step["position"]) == (interval, round_index, position)
    assert step["mask"].astype(int).tolist() == mask
    assert step["ended"] == ended


def test_the_instances_records(instance):
    made = {number + 1: _counts(step["record"]) for number, step in enumerate(instance.steps) if step["record"]}
    assert made == RECORDS
    assert all((step["record"] is None) == (step["ended"] == "element") for step in instance.steps)


def test_a_record_is_six_python_integers_and_cannot_be_changed(instance):
    record = instance.steps[8]["record"]
    assert isinstance(record, RoundRecord)
    assert all(type(value) is int for value in _counts(record))
    assert record.committed + record.overshoot == record.n_active
    with pytest.raises(AttributeError):
        record.skipped = 0
    with pytest.raises(TypeError):
        RoundRecord(1, 1, 5, 0, 0, 5)


def test_the_first_intervals_thresholds(instance):
    step = instance.steps[0]
    assert step["e_max"] == pytest.approx([1.338154], abs=TOL)
    assert step["e_min"] == pytest.approx([0.133815], abs=TOL)
    assert not step["e_max"].flags.writeable and not step["e_min"].flags.writeable


def test_the_pair_is_held_through_an_interval_and_taken_again_at_the_next(instance):
    """The same two arrays at every decision of an interval, rounds 1 and 2
    alike, and another pair in interval 2 (D-021)."""
    first, second = instance.steps[0], instance.steps[7]
    for step in instance.steps[:7]:
        assert step["e_max"] is first["e_max"] and step["e_min"] is first["e_min"]
    for step in instance.steps[7:]:
        assert step["e_max"] is second["e_max"] and step["e_min"] is second["e_min"]
    assert second["e_max"] is not first["e_max"]
    assert second["e_max"][0] != first["e_max"][0]


def test_the_pair_is_not_the_one_a_later_rounds_cache_would_give():
    """After the refine of step 2 the mesh's largest error is a child's, so
    round 2's own cache would give another pair than the one held."""
    driver = _driver()
    driver.reset(IC, 2, 0)
    pair = driver.thresholds
    for action in SCRIPT[:2]:
        driver.commit(action)
        driver.advance()
    assert driver.round_state.round_index == 2 and driver.thresholds is pair
    own = round_thresholds(driver.cache, driver.spec.observation, driver.spec.beta)
    assert own[0][0] != pair[0][0]
    assert np.array_equal(driver._order, round_order(driver.cache, driver.spec.observation, *pair))


def test_after_an_interval_the_pair_is_still_that_intervals_until_begin_interval():
    driver = _driver()
    driver.reset(IC, 2, 0)
    pair = driver.thresholds
    for action in SCRIPT[:7]:
        driver.commit(action)
        ended = driver.advance()
    assert ended == "interval" and driver.thresholds is pair
    driver.begin_interval()
    assert driver.thresholds is not pair
    assert np.array_equal(
        driver.thresholds[0], round_thresholds(driver.cache, driver.spec.observation, driver.spec.beta)[0]
    )


def test_nothing_is_asked_between_a_commit_and_its_advance_or_after_an_interval():
    driver = _driver()
    driver.reset(IC, 2, 0)
    asked = []
    for action in SCRIPT[:7]:
        asked.append(driver.position)
        driver.commit(action)
        assert driver.position is None
        ended = driver.advance()
    assert asked == [step[2] for step in TABLE[:7]]
    assert ended == "interval" and driver.position is None and driver.interval == 1
    driver.begin_interval()
    assert driver.position == 3 and driver.interval == 1


def test_the_interval_count_rises_at_an_intervals_end_only(instance):
    driver = _driver()
    driver.reset(IC, 2, 0)
    seen = []
    for action in SCRIPT:
        driver.commit(action)
        if driver.advance() == "interval" and driver.interval < 2:
            driver.begin_interval()
        seen.append(driver.interval)
    assert seen == [0] * 6 + [1] * 7 + [2]
    assert [step["interval"] for step in instance.steps] == [1] * 7 + [2] * 7


def test_the_observation_and_the_mask_are_the_cores_on_what_the_driver_holds(instance):
    driver = _driver()
    driver.reset(IC, 2, 0)
    for number, action in enumerate(SCRIPT):
        obs, mask = driver.actor_observation(), driver.action_mask()
        assert obs.dtype == np.float32 and obs.shape == (17,)
        assert np.array_equal(
            obs, actor_observation(driver.cache, driver.round_state, driver.position, driver.spec.observation)
        )
        assert np.array_equal(
            mask, action_mask(driver.state, driver.round_state, driver.position, driver.spec.coarsen_rule)
        )
        assert np.array_equal(obs, instance.steps[number]["obs"])
        assert obs is not driver.actor_observation() and mask is not driver.action_mask()
        driver.commit(action)
        if driver.advance() == "interval" and driver.interval < 2:
            driver.begin_interval()


def test_the_instances_first_observation(instance):
    expected = [-2.2572, 1.0, 1.0, -2.2572, -2.2572, -2.2572, 0, 0, 0, 0, 0, 0, 0, 1.0, 1.0, 0.25, 0.5]
    assert instance.steps[0]["obs"] == pytest.approx(expected, abs=1e-4)


def test_the_resource_feature_moves_within_a_round_and_the_progress_between(instance):
    """Features 15 and 16 are the resource and the round progress."""
    seen = [[float(step["obs"][15]), float(step["obs"][16])] for step in instance.steps]
    assert seen[0] == [0.25, 0.5] and seen[1] == [0.25, 0.5]
    assert seen[2] == [0.625, 1.0]
    # Step 8 coarsens a family of four: the count falls from 5 to 2.
    assert seen[7] == [0.625, 0.5] and seen[8] == [0.25, 0.5]


def test_the_cache_is_the_rounds_and_the_state_is_the_one_the_adapt_returned():
    driver = _driver()
    first = driver.reset(IC, 2, 0)
    cache = driver.cache
    driver.commit(H)
    assert driver.advance() == "element"
    assert driver.cache is cache and driver.state is first
    driver.commit(R)
    assert driver.advance() == "round"
    assert driver.cache is not cache and driver.state is not first
    assert driver.cache.state is driver.state
    assert (driver.state.n_active, driver.state.mesh_epoch) == (5, first.mesh_epoch + 1)


# --- The calls (D3) ----------------------------------------------------------------

def test_the_instance_costs_2r_plus_1_calls_per_interval(counting):
    backend = counting(_two_cells())
    _walk(_driver(backend), 2, _listed(SCRIPT))
    assert dict(backend.calls) == {"reset": 1, "compute_error": 4, "apply_marks": 4, "step_interval": 2}


@pytest.mark.parametrize("max_level", [1, 2, 3])
def test_an_interval_costs_2r_plus_1_calls(counting, max_level):
    backend = counting(_two_cells(max_level))
    steps, _ = _walk(_driver(backend, _spec(max_level=max_level)), 3, _drawn(2))
    assert dict(backend.calls) == {
        "reset": 1, "compute_error": 3 * max_level, "apply_marks": 3 * max_level, "step_interval": 3,
    }
    assert max(step["round_index"] for step in steps) == max_level
    assert [step["interval"] for step in steps if step["ended"] == "interval"] == [1, 2, 3]
    assert sum(step["ended"] == "round" for step in steps) == 3 * (max_level - 1)


def test_a_commit_calls_no_verb_and_an_advance_within_a_round_calls_none(counting):
    backend = counting(_two_cells())
    driver = _driver(backend)
    driver.reset(IC, 2, 0)
    before = dict(backend.calls)
    driver.commit(H)
    assert dict(backend.calls) == before
    assert driver.advance() == "element"
    assert dict(backend.calls) == before
    driver.commit(R)
    assert dict(backend.calls) == before
    assert driver.advance() == "round"
    assert dict(backend.calls) == {"reset": 1, "compute_error": 2, "apply_marks": 1}


def test_nothing_is_read_after_an_intervals_end_until_begin_interval(counting):
    backend = counting(_two_cells())
    driver = _driver(backend)
    driver.reset(IC, 2, 0)
    for action in SCRIPT[:7]:
        driver.commit(action)
        driver.advance()
    ended = {"reset": 1, "compute_error": 2, "apply_marks": 2, "step_interval": 1}
    assert dict(backend.calls) == ended
    assert driver.state.n_active == 5 and driver.interval_report.n_substeps == 4
    assert dict(backend.calls) == ended
    driver.begin_interval()
    assert dict(backend.calls) == dict(ended, compute_error=3)


def test_a_round_of_all_holds_still_applies_its_marks(counting):
    backend = counting(_two_cells())
    steps, _ = _walk(_driver(backend), 2, _holds)
    assert len(steps) == 2 * 2 * 2
    assert backend.calls["apply_marks"] == 4
    assert [_counts(step["record"]) for step in steps if step["record"]] == [
        (1, 1, 2, 0, 0, 2), (1, 2, 2, 0, 0, 2), (2, 1, 2, 0, 0, 2), (2, 2, 2, 0, 0, 2),
    ]


# --- The driver against the reference loop (D5) ------------------------------------

def test_the_instance_is_the_reference_loop(instance):
    steps, reports = _reference(_two_cells(), _spec(), 2, _listed(SCRIPT))
    _same(instance.steps, steps)
    _same_reports(instance.reports, reports)


CELLS = [
    (1, "scalar", "relative"), (1, "scalar", "dynamo"), (3, "scalar", "relative"), (3, "per_variable", "relative"),
]
CELL_IDS = ["one variable", "one variable, dynamo", "three, scalar", "three, per_variable"]


@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("indicator", ["ind_a", "ind_b"])
@pytest.mark.parametrize("nvar, reduction, normalization", CELLS, ids=CELL_IDS)
@pytest.mark.parametrize("rule", COARSEN_RULES)
@pytest.mark.parametrize("mesh", list(BACKENDS))
def test_random_masked_walks_are_the_reference_loop(mesh, rule, nvar, reduction, normalization, indicator, seed):
    """Whatever the mesh, the rule, the reduction and the indicator, two
    intervals through the driver are the plain loop, decision for decision:
    the same positions asked, the same observations, masks and thresholds,
    the same record at every round's end, the same interval reports."""
    spec = _spec(_config(indicator, reduction, normalization, budget=64), rule, nvar=nvar)
    driver = RoundDriver(BACKENDS[mesh](nvar=nvar), spec, accumulate=(indicator,))
    ours, our_reports = _walk(driver, 2, _drawn(seed), initial_level=1)
    theirs, their_reports = _reference(
        BACKENDS[mesh](nvar=nvar), spec, 2, _drawn(seed), initial_level=1, accumulate=(indicator,)
    )
    _same(ours, theirs)
    _same_reports(our_reports, their_reports)
    assert ours[-1]["ended"] == "interval" and driver.interval == 2


@pytest.mark.parametrize("rule", COARSEN_RULES)
@pytest.mark.parametrize("mesh", list(BACKENDS))
def test_asked_and_skipped_add_up_to_the_elements_of_every_round(mesh, rule):
    driver = RoundDriver(BACKENDS[mesh](), _spec(_config(budget=64), rule))
    steps, _ = _walk(driver, 2, _drawn(11), initial_level=1)
    n_before, asked, rounds = BACKENDS[mesh]().reset(IC, 2, 1).n_active, 0, 0
    for step in steps:
        asked += 1
        record = step["record"]
        if record is not None:
            assert asked + record.skipped == n_before
            assert record.committed + record.overshoot == record.n_active
            assert record.overshoot >= 0 and record.overshoot % 3 == 0
            n_before, asked, rounds = record.n_active, 0, rounds + 1
    assert rounds == 2 * 2


def test_the_random_walks_are_not_all_holds():
    steps, _ = _walk(_driver(_grid((True, True)), _spec(_config(budget=64))), 2, _drawn(0), initial_level=1)
    assert {step["action"] for step in steps} == {C, H, R}
    records = [step["record"] for step in steps if step["record"] is not None]
    assert any(record.skipped > 0 for record in records) and any(record.overshoot > 0 for record in records)


def test_a_refine_that_balance_spreads_is_reported_as_overshoot():
    """Refine A, then its child beside B: B must refine. The count stood at
    8 and the mesh is 11."""
    steps, _ = _walk(_driver(), 1, _listed([H, R, H, H, H, H, R]))
    assert _counts(steps[1]["record"]) == (1, 1, 5, 0, 0, 5)
    assert _counts(steps[-1]["record"]) == (1, 2, 8, 3, 0, 11)
    assert steps[-1]["position"] == 3 and steps[-1]["ended"] == "interval"


@pytest.mark.parametrize(
    "rule, skipped, n_after", [("first_decider", 3, 5), ("any_coarsen", 3, 5), ("unanimous", 0, 8)]
)
def test_the_coarsen_rule_reaches_the_mask_and_the_commit(rule, skipped, n_after):
    """One coarsen, then holds: a family commit skips three siblings and
    merges four elements into one; one vote under unanimous does neither.
    After a hold, coarsen is closed to the next sibling under first_decider
    alone."""
    driver = _driver(spec=_spec(rule=rule))
    driver.reset(IC, 2, 1)
    assert driver.action_mask()[C]
    driver.commit(C)
    while driver.advance() == "element":
        driver.commit(H)
    assert (driver.round_record.skipped, driver.round_record.n_active, driver.round_record.overshoot) == (
        skipped, n_after, 0,
    )
    driver.reset(IC, 2, 1)
    driver.commit(H)
    driver.advance()
    assert bool(driver.action_mask()[C]) is (rule != "first_decider")


# --- run_interval: the loop with a policy and no rewards (D6) ----------------------

def _numbered(choose):
    """A decide for run_interval that numbers its calls and keeps what it
    was handed."""
    seen = []

    def decide(features, mask):
        seen.append((features, mask))
        return choose(len(seen) - 1, mask)

    return decide, seen


@pytest.mark.parametrize("rule", COARSEN_RULES)
@pytest.mark.parametrize("mesh", list(BACKENDS))
def test_run_interval_is_the_walk_by_hand(mesh, rule):
    spec = _spec(_config(budget=64), rule)
    by_hand, hand_reports = _walk(RoundDriver(BACKENDS[mesh](), spec), 3, _drawn(4), initial_level=1)
    driver = RoundDriver(BACKENDS[mesh](), spec)
    driver.reset(IC, 2, 1)
    decide, seen = _numbered(_drawn(4))
    records, reports = [], []
    for interval in range(3):
        made, report = driver.run_interval(decide)
        records += made
        reports.append(report)
        assert len(made) == 2 and report is driver.interval_report
        assert driver.position is None and driver.interval == interval + 1
        if interval < 2:
            driver.begin_interval()
    assert records == [step["record"] for step in by_hand if step["record"] is not None]
    _same_reports(reports, hand_reports)
    assert len(seen) == len(by_hand)
    for (features, mask), step in zip(seen, by_hand, strict=True):
        assert np.array_equal(features, step["obs"]) and np.array_equal(mask, step["mask"])


def test_run_interval_costs_2r_plus_1_calls_per_interval(counting):
    backend = counting(_two_cells())
    driver = _driver(backend)
    driver.reset(IC, 2, 0)
    for _ in range(3):
        driver.run_interval(lambda features, mask: H)
        if driver.interval < 3:
            driver.begin_interval()
    assert dict(backend.calls) == {"reset": 1, "compute_error": 6, "apply_marks": 6, "step_interval": 3}


def test_with_no_accumulator_the_report_holds_no_array_and_with_one_the_indicators():
    driver = _driver()
    driver.reset(IC, 2, 0)
    decide, _ = _numbered(_listed(SCRIPT))
    records, report = driver.run_interval(decide)
    assert [_counts(record) for record in records] == [RECORDS[2], RECORDS[7]]
    assert isinstance(report, IntervalReport) and dict(report.acc) == {} and report.aux is None
    named = _driver(accumulate=("ind_b",))
    named.reset(IC, 2, 0)
    decide, _ = _numbered(_listed(SCRIPT))
    _, with_array = named.run_interval(decide)
    assert list(with_array.acc) == ["ind_b"] and with_array.acc["ind_b"].shape == (5, 1)
    assert (with_array.time, with_array.n_substeps) == (report.time, report.n_substeps)
    _, theirs = _reference(_two_cells(), _spec(), 1, _listed(SCRIPT), accumulate=("ind_b",))
    assert np.array_equal(with_array.acc["ind_b"], theirs[0].acc["ind_b"])


def test_decide_is_handed_the_actors_features_and_the_mask():
    driver = _driver()
    driver.reset(IC, 2, 0)
    decide, seen = _numbered(_listed(SCRIPT))
    driver.run_interval(decide)
    assert len(seen) == 7
    for features, mask in seen:
        assert features.dtype == np.float32 and features.shape == (17,)
        assert mask.dtype == bool and mask.shape == (3,)
    assert seen[0][0] == pytest.approx(
        [-2.2572, 1.0, 1.0, -2.2572, -2.2572, -2.2572, 0, 0, 0, 0, 0, 0, 0, 1.0, 1.0, 0.25, 0.5], abs=1e-4
    )


@pytest.mark.parametrize(
    "hold", [np.int64(H), np.array(H), np.array([H])[0], True], ids=["int64", "0-d", "item", "bool"]
)
def test_run_interval_takes_what_a_policy_returns(hold):
    """A policy's action may be a numpy scalar or a zero-dimensional array;
    run_interval makes it an integer before the commit."""
    driver = _driver()
    driver.reset(IC, 2, 0)
    records, _ = driver.run_interval(lambda features, mask: hold)
    assert [_counts(record) for record in records] == [(1, 1, 2, 0, 0, 2), (1, 2, 2, 0, 0, 2)]


def test_run_interval_runs_from_wherever_the_interval_stands():
    driver = _driver()
    driver.reset(IC, 2, 0)
    for action in SCRIPT[:3]:
        driver.commit(action)
        driver.advance()
    decide, seen = _numbered(_listed(SCRIPT[3:]))
    records, _ = driver.run_interval(decide)
    assert len(seen) == 4 and [_counts(record) for record in records] == [RECORDS[7]]


def test_a_masked_action_from_decide_is_refused_and_the_position_is_still_asked(counting):
    backend = counting(_two_cells())
    driver = _driver(backend)
    driver.reset(IC, 2, 0)
    with pytest.raises(ValueError, match=r"^action: 0 is masked for position 1 under 'first_decider'$"):
        driver.run_interval(lambda features, mask: C)
    assert driver.position == 1 and dict(backend.calls) == {"reset": 1, "compute_error": 1}
    records, _ = driver.run_interval(lambda features, mask: H)
    assert len(records) == 2


def test_run_interval_after_an_interval_needs_begin_interval():
    driver = _driver()
    driver.reset(IC, 2, 0)
    driver.run_interval(lambda features, mask: H)
    with pytest.raises(RuntimeError, match=r"^run_interval: nothing is asked$"):
        driver.run_interval(lambda features, mask: H)


# --- A backend that fails (D8) -----------------------------------------------------

# (verb, which call fails, the decision whose advance or begin_interval it
# fails in, the record left): the instance's five points.
FAILURES = [
    ("compute_error", 2, 2, (1, 1, 5, 0, 0, 5)),
    ("compute_error", 3, 7, (1, 2, 5, 0, 0, 5)),
    ("step_interval", 1, 7, (1, 2, 5, 0, 0, 5)),
    ("apply_marks", 3, 9, None),
    ("step_interval", 2, 14, (2, 2, 5, 0, 0, 5)),
]
FAILURE_IDS = [f"{verb} call {at}" for verb, at, _, _ in FAILURES]


def _until_failure(driver: RoundDriver) -> tuple[int, BackendFailure]:
    """Walk the instance until the backend fails: the number of the
    decision it failed after, and the failure."""
    driver.reset(IC, 2, 0)
    for number, action in enumerate(SCRIPT, start=1):
        driver.commit(action)
        try:
            if driver.advance() == "interval" and driver.interval < 2:
                driver.begin_interval()
        except BackendFailure as failure:
            return number, failure
    raise AssertionError("the backend did not fail")


@pytest.mark.parametrize("verb, at, last, record", FAILURES, ids=FAILURE_IDS)
def test_a_failure_comes_out_and_leaves_nothing_asked(failing, verb, at, last, record):
    driver = _driver(failing(_two_cells(), verb, at))
    number, failure = _until_failure(driver)
    assert number == last
    assert str(failure) == f"{verb}: failed on demand at call {at}"
    assert driver.position is None


@pytest.mark.parametrize("verb, at, last, record", FAILURES, ids=FAILURE_IDS)
def test_a_round_whose_marks_were_applied_keeps_its_record(failing, verb, at, last, record):
    """A failed adapt leaves no record. A failure after the adapt, in the
    next read or in the advance, leaves the round's."""
    driver = _driver(failing(_two_cells(), verb, at))
    _until_failure(driver)
    assert _counts(driver.round_record) == record
    assert driver.interval_report is None or verb == "compute_error"


@pytest.mark.parametrize("name, call, message", NOTHING_RUNS, ids=[name for name, _, _ in NOTHING_RUNS])
@pytest.mark.parametrize("verb, at, last, record", FAILURES, ids=FAILURE_IDS)
def test_after_a_failure_only_a_reset_runs(failing, instance, verb, at, last, record, name, call, message):
    driver = _driver(failing(_two_cells(), verb, at))
    _until_failure(driver)
    with pytest.raises(RuntimeError) as caught:
        call(driver)
    assert str(caught.value) == message
    again, _ = _walk(driver, 2, _listed(SCRIPT))
    _same(again, instance.steps)


@pytest.mark.parametrize("verb", ["reset", "compute_error"])
def test_a_failure_inside_reset_is_raised_and_leaves_nothing_asked(failing, verb):
    driver = _driver(failing(_two_cells(), verb, 1))
    with pytest.raises(BackendFailure, match=rf"^{verb}: failed on demand at call 1$"):
        driver.reset(IC, 2, 0)
    assert driver.position is None
    with pytest.raises(RuntimeError):
        driver.commit(H)
    with pytest.raises(RuntimeError):
        driver.begin_interval()
    assert driver.reset(IC, 2, 0).n_active == 2 and driver.position == 1


def test_a_failure_that_never_comes_changes_nothing(failing, instance):
    steps, _ = _walk(_driver(failing(_two_cells(), "step_interval", 3)), 2, _listed(SCRIPT))
    _same(steps, instance.steps)


class _NotFinite:
    """Forward every verb, but put a NaN in one read."""

    def __init__(self, inner, at):
        self.inner, self.at, self.seen = inner, at, 0

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def compute_error(self, indicators, reduction):
        errors = self.inner.compute_error(indicators, reduction)
        self.seen += 1
        if self.seen == self.at:
            errors[indicators[0]][0, 0] = np.nan
        return errors


def test_a_read_that_is_not_finite_is_a_failure():
    driver = _driver(_NotFinite(_two_cells(), 2))
    number, failure = _until_failure(driver)
    assert (number, str(failure)) == (2, "errors['ind_b']: not finite")
    assert driver.position is None and _counts(driver.round_record) == (1, 1, 5, 0, 0, 5)


def test_any_error_from_a_verb_leaves_nothing_asked():
    class Broken:
        def __init__(self, inner):
            self.inner = inner

        def __getattr__(self, name):
            return getattr(self.inner, name)

        def apply_marks(self, marks):
            raise KeyError("not a backend failure")

    driver = _driver(Broken(_two_cells()))
    driver.reset(IC, 2, 0)
    driver.commit(H)
    driver.advance()
    driver.commit(R)
    with pytest.raises(KeyError):
        driver.advance()
    assert driver.position is None and driver.round_record is None
    with pytest.raises(RuntimeError):
        driver.advance()


# --- What is refused, and when (D9) ------------------------------------------------

def test_a_masked_action_is_refused_and_nothing_moves(counting, instance):
    backend = counting(_two_cells())
    driver = _driver(backend)
    driver.reset(IC, 2, 0)
    calls, round_state = dict(backend.calls), driver.round_state
    with pytest.raises(ValueError, match=r"^action: 0 is masked for position 1 under 'first_decider'$"):
        driver.commit(C)
    assert dict(backend.calls) == calls
    assert driver.position == 1 and driver.round_state is round_state
    assert driver.action_mask().tolist() == [False, True, True]
    steps = []
    for action in SCRIPT:
        steps.append(driver.position)
        driver.commit(action)
        if driver.advance() == "interval" and driver.interval < 2:
            driver.begin_interval()
    assert steps == [step["position"] for step in instance.steps]


@pytest.mark.parametrize("action", [3, -1, 1.0, None, np.array(H)], ids=["3", "-1", "1.0", "None", "0-d array"])
def test_what_is_not_an_action_index_is_refused_by_commit(action):
    driver = _driver()
    driver.reset(IC, 2, 0)
    with pytest.raises(ValueError, match=r"^action: .*, expected 0 \(coarsen\), 1 \(hold\) or 2 \(refine\)$"):
        driver.commit(action)
    assert driver.position == 1
    driver.commit(np.int64(H))
    assert driver.advance() == "element"


def test_each_call_has_its_moment():
    """A commit is followed by an advance and by nothing else; an interval
    is begun only when one has just ended."""
    driver = _driver()
    driver.reset(IC, 2, 0)
    with pytest.raises(RuntimeError, match=r"^advance: no decision has been committed$"):
        driver.advance()
    with pytest.raises(RuntimeError, match=r"^begin_interval: no interval has just ended$"):
        driver.begin_interval()
    assert driver.position == 1
    driver.commit(H)
    for call, message in (
        (lambda: driver.commit(H), "commit: nothing is asked"),
        (driver.action_mask, "action_mask: nothing is asked"),
        (driver.actor_observation, "actor_observation: nothing is asked"),
        (driver.begin_interval, "begin_interval: no interval has just ended"),
    ):
        with pytest.raises(RuntimeError) as caught:
            call()
        assert str(caught.value) == message
    assert driver.advance() == "element" and driver.position == 0
    for action in SCRIPT[1:7]:
        driver.commit(action)
        driver.advance()
    with pytest.raises(RuntimeError, match=r"^commit: nothing is asked$"):
        driver.commit(H)
    with pytest.raises(RuntimeError, match=r"^advance: no decision has been committed$"):
        driver.advance()
    driver.begin_interval()
    with pytest.raises(RuntimeError, match=r"^begin_interval: no interval has just ended$"):
        driver.begin_interval()
    assert driver.position == 3


def test_a_record_and_a_report_are_the_last_advances_only():
    driver = _driver()
    driver.reset(IC, 2, 0)
    seen = []
    for action in SCRIPT[:8]:
        driver.commit(action)
        ended = driver.advance()
        seen.append((ended, driver.round_record is not None, driver.interval_report is not None))
        if ended == "interval":
            report = driver.interval_report
            driver.begin_interval()
            assert driver.interval_report is report
    assert seen == [
        ("element", False, False), ("round", True, False), ("element", False, False), ("element", False, False),
        ("element", False, False), ("element", False, False), ("interval", True, True), ("element", False, False),
    ]
    for action in SCRIPT[8:]:
        driver.commit(action)
        ended = driver.advance()
    assert ended == "interval" and driver.round_record is not None and driver.interval_report is not None
    driver.reset(IC, 2, 0)
    assert (driver.round_record, driver.interval_report) == (None, None)


# --- What the module imports, and where the thresholds are assigned (D7) -----------

def _tree() -> ast.Module:
    return ast.parse(Path(round_driver.__file__).read_text(encoding="utf-8"))


def test_the_driver_imports_no_reward_no_environment_and_nothing_sequential():
    names = set()
    for node in ast.walk(_tree()):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    ours = {name for name in names if name.split(".")[0] in {
        "agent", "analysis", "backends", "contract", "deployment", "drivers", "envs", "tools", "training",
    }}
    assert ours == {
        "agent.round_cache", "agent.round_commit", "agent.round_mask", "agent.round_observation",
        "agent.round_queue", "agent.round_spec", "agent.round_state", "contract.backend", "contract.reports",
        "contract.state",
    }
    assert {name.split(".")[0] for name in names} - {"agent", "contract"} == {
        "__future__", "collections", "dataclasses", "numbers", "numpy",
    }


def test_the_thresholds_are_assigned_in_begin_interval_alone():
    """The constructor sets them to None; one method assigns them."""
    where = []
    for function in ast.walk(_tree()):
        if not isinstance(function, ast.FunctionDef):
            continue
        for node in ast.walk(function):
            targets = node.targets if isinstance(node, ast.Assign) else [getattr(node, "target", None)]
            for target in targets:
                for leaf in ast.walk(target) if target is not None else ():
                    stored = isinstance(leaf, ast.Attribute) and isinstance(leaf.ctx, ast.Store)
                    if stored and leaf.attr == "_thresholds":
                        where.append(function.name)
    assert where == ["__init__", "begin_interval"]


def test_the_driver_calls_four_verbs_and_never_describe_or_get_state():
    called = {
        node.func.attr
        for node in ast.walk(_tree())
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Attribute) and node.func.value.attr == "backend"
    }
    assert called == {"reset", "apply_marks", "step_interval"}
    # The fourth, compute_error, is read_round's one call.
    assert "read_round(self.backend" in Path(round_driver.__file__).read_text(encoding="utf-8")
