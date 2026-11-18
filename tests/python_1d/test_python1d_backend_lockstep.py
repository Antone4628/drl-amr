"""Lockstep tests for the 1D adapter: Python1DBackend against the contract it
wraps, bit for bit (backends/python_1d/backend.py).

AGENT_2D_ROADMAP item 1.4, signals A3-A7. Two solvers are built alike. One is
driven through Python1DBackend; its twin through Python1DSolverContract and,
for the advance, the sequential driver's own advance(). Every comparison is
exact equality: the adapter translates and repeats, it computes nothing new.

Run: pytest tests/python_1d/test_python1d_backend_lockstep.py -v
"""
from __future__ import annotations

import numpy as np
import pytest

from agent.core import AgentCore
from backends.python_1d.backend import Python1DBackend
from backends.python_1d.contract_impl import Python1DSolverContract
from backends.python_1d.solvers.dg_advection_solver_multiround import DGAdvectionSolver
from contract.checks import topology_violations
from contract.reports import KIND_CHILD, KIND_MERGED, KIND_UNCHANGED
from contract.solver_contract import COARSEN, HOLD, REFINE
from drivers.multiround_driver import MultiroundDriver

BASE_XELEM = np.array([-1.0, -0.4, 0.0, 0.4, 1.0])
INDICATORS = ("zz_style", "raw_jump")


def make_solver(max_level):
    return DGAdvectionSolver(
        nop=4, xelem=BASE_XELEM.copy(), max_elements=120, max_level=max_level, icase=1, balance=False
    )


def make_pair(*, icase=1, initial_level=1, amr_freq=20, max_level=3):
    """A backend and its twin contract, both reset alike."""
    backend = Python1DBackend(make_solver(max_level))
    backend.reset({"icase": icase}, amr_freq=amr_freq, initial_level=initial_level)
    twin = Python1DSolverContract(make_solver(max_level))
    if initial_level > 0:
        twin.reset(icase=icase, refinement_mode="fixed", refinement_level=initial_level)
    else:
        twin.reset(icase=icase)
    return backend, twin


def assert_states_match(new, old):
    """The v2 state against the 1D state of the twin: A4."""
    n = old.n_active
    assert new.n_active == n
    assert new.max_level == old.max_level
    assert new.dt_base == old.stable_dt
    assert np.array_equal(new.level, old.level)
    assert np.array_equal(new.element_id, old.element_id)
    assert np.array_equal(new.can_refine, old.can_refine)
    assert np.array_equal(new.can_coarsen, old.can_coarsen)
    assert np.array_equal(new.neighbors[:, 0, 0], old.left)
    assert np.array_equal(new.neighbors[:, 1, 0], old.right)

    position = np.arange(n)
    has = old.sibling >= 0
    expected = np.full((n, 2), -1)
    expected[has, 0] = np.minimum(position, old.sibling)[has]
    expected[has, 1] = np.maximum(position, old.sibling)[has]
    assert np.array_equal(new.family, expected)


def assert_solutions_match(backend, twin):
    assert np.array_equal(backend._solver.q, twin._solver.q)
    assert np.array_equal(backend._solver.xelem, twin._solver.xelem)
    assert backend._solver.time == twin._solver.time


def assert_errors_match(backend, twin):
    """A5, on whatever mesh the pair is on."""
    out = backend.compute_error(INDICATORS, "scalar")
    for name in INDICATORS:
        assert out[name].shape == (twin.get_state().n_active, 1)
        assert np.array_equal(out[name][:, 0], twin.compute_error(name))


def assert_report_fits(report, before, after):
    """The report against the two states it connects, kind by kind."""
    assert (report.epoch_old, report.epoch_new) == (before.mesh_epoch, after.mesh_epoch)
    assert after.mesh_epoch == before.mesh_epoch + 1
    assert (report.n_old, report.n_new) == (before.n_active, after.n_active)
    used = []
    for q in range(report.n_new):
        first, second = (int(p) for p in report.new_to_old[q])
        kind = report.new_kind[q]
        if kind == KIND_UNCHANGED:
            assert second == -1
            assert after.element_id[q] == before.element_id[first]
            assert after.level[q] == before.level[first]
            assert report.realized[first] == HOLD
            used.append(first)
        elif kind == KIND_CHILD:
            assert second == -1
            assert after.level[q] == before.level[first] + 1
            assert report.realized[first] == REFINE
            assert abs(after.centroid[q, 0] - before.centroid[first, 0]) < before.h[first, 0] / 2
            used.append(first)
        else:
            assert kind == KIND_MERGED
            assert before.family[first].tolist() == [first, second]
            assert after.level[q] == before.level[first] - 1
            assert report.realized[first] == COARSEN
            assert report.realized[second] == COARSEN
            assert after.centroid[q, 0] == pytest.approx(before.centroid[[first, second], 0].mean(), abs=1e-12)
            used += [first, second]
    # Every old element is the source of something.
    assert set(used) == set(range(report.n_old))


def twin_advance(twin, duration, sample_steps=None):
    """Advance the twin with the sequential driver's advance(), accumulating
    both indicators' maxima at the start and after the chosen steps.

    Args:
        sample_steps: Step counts (1-based) to sample after; None for all.

    Returns:
        The maxima by name, and the number of steps taken.
    """
    driver = MultiroundDriver(AgentCore(twin, error_indicator="zz_style"))
    n = twin.get_state().n_active
    peak = {name: np.zeros(n) for name in INDICATORS}
    calls = [0]

    def accumulate():
        step = calls[0]
        calls[0] += 1
        if step == 0 or sample_steps is None or step in sample_steps:
            for name in INDICATORS:
                peak[name] = np.maximum(peak[name], twin.compute_error(name))

    driver.advance(duration, substep_callback=accumulate)
    return peak, calls[0] - 1


# --- A4: the state ------------------------------------------------------------

@pytest.mark.parametrize("initial_level", [0, 1, 2])
def test_states_match_after_reset(initial_level):
    backend, twin = make_pair(initial_level=initial_level)
    assert_states_match(backend.get_state(), twin.get_state())
    assert_solutions_match(backend, twin)
    assert_errors_match(backend, twin)


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_states_and_reports_match_under_random_batch_rounds(seed):
    rng = np.random.default_rng(seed)
    backend, twin = make_pair(icase=int(rng.choice([1, 12, 16])), initial_level=1)
    before = backend.get_state()
    for _ in range(5):
        marks = rng.integers(-1, 2, size=before.n_active).astype(np.int8)
        after, report = backend.apply_marks(marks)
        old = twin.apply_marks(marks.astype(int))

        assert_states_match(after, old)
        assert_solutions_match(backend, twin)
        assert_errors_match(backend, twin)
        assert_report_fits(report, before, after)
        assert topology_violations(after, uniform_base=False) == []
        before = after


@pytest.mark.parametrize("seed", [0, 1])
def test_states_match_under_sequential_actions(seed):
    """The states the sequential triad reaches, cascades included. The triad
    is the wrapped contract's, so the backend's is driven through its private
    handle: the adapter has no sequential verbs of its own."""
    rng = np.random.default_rng(seed)
    backend, twin = make_pair(initial_level=1)
    for _ in range(12):
        old = twin.get_state()
        idx = int(rng.integers(old.n_active))
        mark = int(rng.choice([COARSEN, REFINE]))
        if (mark == REFINE and not old.can_refine[idx]) or (mark == COARSEN and not old.can_coarsen[idx]):
            continue
        for contract in (backend._contract, twin):
            contract.adapt_element(idx, mark)
            contract.balance()
            contract.rebuild()
        state = backend.get_state()
        assert_states_match(state, twin.get_state())
        assert topology_violations(state, uniform_base=False) == []


# --- A5: the indicators ---------------------------------------------------------

@pytest.mark.parametrize("indicator", INDICATORS)
@pytest.mark.parametrize("reduction", ["scalar", "per_variable"])
def test_compute_error_is_the_old_verb_bit_for_bit(indicator, reduction):
    backend, twin = make_pair(icase=13, initial_level=2)
    out = backend.compute_error([indicator], reduction)[indicator]
    assert out.shape == (16, 1)
    assert np.array_equal(out[:, 0], twin.compute_error(indicator))


# --- A6: the interval -----------------------------------------------------------

@pytest.mark.parametrize("icase, initial_level", [(1, 1), (13, 0), (16, 2)])
def test_interval_matches_the_drivers_advance(icase, initial_level):
    backend, twin = make_pair(icase=icase, initial_level=initial_level)
    duration = backend.get_state().interval_T
    report = backend.step_interval(INDICATORS, "scalar", 1)
    peak, steps = twin_advance(twin, duration)

    assert report.n_substeps == steps
    assert report.time == twin._solver.time
    for name in INDICATORS:
        assert np.array_equal(report.acc[name][:, 0], peak[name])
    assert_solutions_match(backend, twin)
    assert_states_match(backend.get_state(), twin.get_state())


def test_interval_matches_on_an_adapted_mesh():
    backend, twin = make_pair(initial_level=0)
    for marks in ([HOLD, REFINE, HOLD, HOLD], [HOLD, REFINE, HOLD, HOLD, HOLD]):
        backend.apply_marks(np.array(marks, dtype=np.int8))
        twin.apply_marks(np.array(marks))
    state = backend.get_state()
    assert state.level.tolist() == [1, 1, 2, 2, 1, 0, 0]

    report = backend.step_interval(INDICATORS, "scalar", 1)
    peak, steps = twin_advance(twin, state.interval_T)
    assert report.n_substeps == steps
    for name in INDICATORS:
        assert np.array_equal(report.acc[name][:, 0], peak[name])
    assert_solutions_match(backend, twin)


def test_cadence_3_samples_the_start_every_third_step_and_the_last():
    backend, twin = make_pair(icase=12, initial_level=1)
    duration = backend.get_state().interval_T
    report = backend.step_interval(INDICATORS, "scalar", 3)
    assert report.n_substeps == 20
    peak, _ = twin_advance(twin, duration, sample_steps={3, 6, 9, 12, 15, 18, 20})
    for name in INDICATORS:
        assert np.array_equal(report.acc[name][:, 0], peak[name])
    assert_solutions_match(backend, twin)


def test_cadence_1_is_at_least_any_other_cadence():
    fine, _ = make_pair(icase=12, initial_level=1)
    coarse, _ = make_pair(icase=12, initial_level=1)
    every = fine.step_interval(INDICATORS, "scalar", 1)
    third = coarse.step_interval(INDICATORS, "scalar", 3)
    for name in INDICATORS:
        assert (third.acc[name] <= every.acc[name]).all()


def test_a_cadence_past_the_interval_samples_the_two_ends():
    backend, twin = make_pair(icase=16, initial_level=1)
    duration = backend.get_state().interval_T
    report = backend.step_interval(INDICATORS, "scalar", 100)
    peak, steps = twin_advance(twin, duration, sample_steps={20})
    assert steps == 20
    for name in INDICATORS:
        assert np.array_equal(report.acc[name][:, 0], peak[name])


def test_the_clock_at_max_level_2():
    backend, twin = make_pair(initial_level=1, amr_freq=10, max_level=2)
    state = backend.get_state()
    assert state.interval_T == 10 * state.dt_base
    report = backend.step_interval(INDICATORS, "scalar", 1)
    peak, steps = twin_advance(twin, state.interval_T)
    assert report.n_substeps == steps == 10
    assert np.array_equal(report.acc["zz_style"][:, 0], peak["zz_style"])
    assert_solutions_match(backend, twin)
