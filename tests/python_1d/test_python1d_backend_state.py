"""Tests for the 1D adapter, Python1DBackend: the state, the reports and the
verbs' surfaces (backends/python_1d/backend.py).

AGENT_2D_ROADMAP item 1.4, signals A1-A3 and A7-A9. Every literal here was
read off the real solver before the test was written; the comparison against
the old contract, bit for bit, is test_python1d_backend_lockstep.py.

Run: pytest tests/python_1d/test_python1d_backend_state.py -v
"""
from __future__ import annotations

import numpy as np
import pytest

from backends.python_1d.backend import Python1DBackend
from backends.python_1d.contract_impl import Python1DSolverContract
from backends.python_1d.solvers.dg_advection_solver_multiround import DGAdvectionSolver
from contract.backend import SolverBackend
from contract.checks import topology_violations
from contract.reports import AdaptReport, IntervalReport
from contract.solver_contract import COARSEN, HOLD, REFINE
from contract.solver_snapshot import SolverSnapshot
from contract.state import SolverState

# The base mesh: four elements A, B, C, D of widths 0.6, 0.4, 0.4, 0.6.
BASE_XELEM = np.array([-1.0, -0.4, 0.0, 0.4, 1.0])
MAX_LEVEL = 3
ZZ = "zz_style"
RAW = "raw_jump"


def make_backend(*, max_level=MAX_LEVEL, **kwargs):
    solver = DGAdvectionSolver(
        nop=4, xelem=BASE_XELEM.copy(), max_elements=120, max_level=max_level, icase=1, balance=False
    )
    return Python1DBackend(solver, **kwargs)


def marks_of(*values):
    return np.array(values, dtype=np.int8)


@pytest.fixture
def backend():
    """A backend one reset in, on the four-element base mesh."""
    b = make_backend()
    b.reset({"icase": 1}, amr_freq=20, initial_level=0)
    return b


def to_mesh1(b):
    """Refine B: five elements A, B1, B2, C, D."""
    return b.apply_marks(marks_of(HOLD, REFINE, HOLD, HOLD))


def to_mesh2(b):
    """Refine B1 again; balance refines A: seven elements."""
    to_mesh1(b)
    return b.apply_marks(marks_of(HOLD, REFINE, HOLD, HOLD, HOLD))


# --- A8, A9: the protocol and describe ---------------------------------------

def test_is_a_solver_backend():
    assert isinstance(make_backend(), SolverBackend)


def test_the_old_class_is_still_not_a_solver_backend():
    assert not issubclass(Python1DSolverContract, SolverBackend)


def test_describe():
    info = make_backend().describe()
    assert info.ndim == 1
    assert info.nvar == 1
    assert info.indicators == (RAW, ZZ)
    assert info.reductions == ("scalar", "per_variable")
    assert dict(info.ic_params) == {"icase": (1.0, 16.0)}
    assert info.optional_verbs == {"get_snapshot", "reinitialize_ic"}


def test_describe_bounds_follow_ic_cases():
    info = make_backend(ic_cases=(12, 10)).describe()
    assert dict(info.ic_params) == {"icase": (10.0, 12.0)}


def test_empty_ic_cases_rejected():
    with pytest.raises(ValueError, match=r"^ic_cases: "):
        make_backend(ic_cases=())


@pytest.mark.parametrize(
    "call",
    [
        lambda b: b.get_state(),
        lambda b: b.compute_error([ZZ], "scalar"),
        lambda b: b.apply_marks(marks_of(HOLD, HOLD, HOLD, HOLD)),
        lambda b: b.step_interval([ZZ], "scalar", 1),
        lambda b: b.get_snapshot(),
        lambda b: b.reinitialize_ic(),
    ],
    ids=["get_state", "compute_error", "apply_marks", "step_interval", "get_snapshot", "reinitialize_ic"],
)
def test_a_verb_before_reset_raises(call):
    with pytest.raises(RuntimeError, match="call reset first"):
        call(make_backend())


def test_dry_run_is_absent(backend):
    with pytest.raises(NotImplementedError):
        make_backend().dry_run(marks_of(HOLD, HOLD, HOLD, HOLD))
    with pytest.raises(NotImplementedError):
        backend.dry_run(marks_of(HOLD, HOLD, HOLD, HOLD))


# --- reset --------------------------------------------------------------------

@pytest.mark.parametrize("initial_level", [0, 1, 2])
def test_reset_gives_the_uniform_mesh(initial_level):
    b = make_backend()
    state = b.reset({"icase": 1}, amr_freq=20, initial_level=initial_level)
    assert isinstance(state, SolverState)
    assert state.n_active == 4 * 2**initial_level
    assert (state.level == initial_level).all()
    assert state.mesh_epoch == 1
    assert state.time == 0.0
    assert state.can_coarsen.all() == (initial_level > 0)
    assert topology_violations(state, uniform_base=False) == []


def test_reset_globals(backend):
    state = backend.get_state()
    solver = backend._solver
    assert (state.ndim, state.nvar, state.max_level) == (1, 1, MAX_LEVEL)
    assert state.amr_freq == 20
    assert state.dt_base == solver.courant_max * solver.dx_min / solver.wave_speed
    assert state.dt_stable == state.dt_base
    assert state.interval_T == 20 * state.dt_base
    assert state.domain_lo.tolist() == [-1.0]
    assert state.domain_hi.tolist() == [1.0]
    assert state.periodic.tolist() == [True]
    assert state.attributes.shape == (4, 0)
    assert state.state_mean is None


def test_mesh_epoch_rises_once_per_reset(backend):
    assert backend.get_state().mesh_epoch == 1
    assert backend.reset({"icase": 10}, amr_freq=5, initial_level=2).mesh_epoch == 2


def test_a_float_icase_that_is_a_member_is_accepted():
    b = make_backend()
    b.reset({"icase": 13.0}, amr_freq=20, initial_level=0)
    assert b._solver.icase == 13


@pytest.mark.parametrize(
    "ic, amr_freq, initial_level, starts",
    [
        ({}, 20, 0, r"^ic: "),
        ({"icase": 1.0, "x0": 0.0}, 20, 0, r"^ic: "),
        ({"icase": 2}, 20, 0, r"^ic\['icase'\]: "),
        ({"icase": 1.5}, 20, 0, r"^ic\['icase'\]: "),
        ({"icase": float("nan")}, 20, 0, r"^ic\['icase'\]: "),
        ({"icase": 1}, 0, 0, r"^amr_freq: "),
        ({"icase": 1}, -3, 0, r"^amr_freq: "),
        ({"icase": 1}, 20, -1, r"^initial_level: "),
        ({"icase": 1}, 20, MAX_LEVEL + 1, r"^initial_level: "),
    ],
)
def test_reset_rejections(ic, amr_freq, initial_level, starts):
    with pytest.raises(ValueError, match=starts):
        make_backend().reset(ic, amr_freq, initial_level)


def test_a_rejected_reset_leaves_the_episode_intact(backend):
    to_mesh1(backend)
    with pytest.raises(ValueError):
        backend.reset({"icase": 2}, amr_freq=20, initial_level=0)
    state = backend.get_state()
    assert state.n_active == 5
    assert state.mesh_epoch == 2
    assert state.amr_freq == 20


# --- A1, A2, A3: the state ----------------------------------------------------

def test_mesh_1_is_the_table(backend):
    state, _ = to_mesh1(backend)
    assert state.level.tolist() == [0, 1, 1, 0, 0]
    assert state.neighbors.shape == (5, 2, 1)
    assert state.neighbors[:, :, 0].tolist() == [[4, 1], [0, 2], [1, 3], [2, 4], [3, 0]]
    assert np.argwhere(state.neighbor_wrap).tolist() == [[0, 0], [4, 1]]
    assert state.family.tolist() == [[-1, -1], [1, 2], [1, 2], [-1, -1], [-1, -1]]
    assert state.can_coarsen.tolist() == [False, True, True, False, False]
    assert state.can_refine.all()
    assert np.allclose(state.centroid[:, 0], [-0.7, -0.3, -0.1, 0.2, 0.7], rtol=0, atol=1e-12)
    assert np.allclose(state.h[:, 0], [0.6, 0.2, 0.2, 0.4, 0.6], rtol=0, atol=1e-12)
    assert state.element_id.tolist() == [1, 7, 8, 3, 4]
    assert topology_violations(state, uniform_base=False) == []


def test_mesh_2_a_complete_family_that_cannot_coarsen(backend):
    state, _ = to_mesh2(backend)
    assert state.level.tolist() == [1, 1, 2, 2, 1, 0, 0]
    assert state.family.tolist() == [[0, 1], [0, 1], [2, 3], [2, 3], [-1, -1], [-1, -1], [-1, -1]]
    # A1 and A2 are a complete family; merging them would put level 0 beside
    # level 2, so the 1D balance check keeps can_coarsen false.
    assert state.can_coarsen.tolist() == [False, False, True, True, False, False, False]
    assert state.element_id.tolist() == [5, 6, 17, 18, 8, 3, 4]
    assert topology_violations(state, uniform_base=False) == []


def test_the_default_check_reports_the_non_uniform_base(backend):
    found = topology_violations(backend.get_state())
    assert found
    assert all(line.startswith("uniform_base.") for line in found)
    assert found[0].startswith("uniform_base.h:")


def test_state_arrays_are_read_only_and_the_solver_is_not_frozen(backend):
    state = backend.get_state()
    with pytest.raises(ValueError):
        state.level[0] = 3
    with pytest.raises(ValueError):
        state.domain_lo[0] = 0.0
    assert backend._solver.xelem.flags.writeable


def test_get_state_builds_fresh_equal_arrays(backend):
    first, second = backend.get_state(), backend.get_state()
    for name in ("level", "neighbors", "neighbor_wrap", "family", "can_refine", "can_coarsen", "centroid", "h"):
        assert getattr(first, name) is not getattr(second, name)
        assert np.array_equal(getattr(first, name), getattr(second, name))


# --- A7: the reports ----------------------------------------------------------

def test_refine_report(backend):
    state, report = to_mesh1(backend)
    assert isinstance(report, AdaptReport)
    assert report.realized.tolist() == [HOLD, REFINE, HOLD, HOLD]
    assert report.new_kind.tolist() == [0, 1, 1, 0, 0]
    assert report.new_to_old.tolist() == [[0, -1], [1, -1], [1, -1], [2, -1], [3, -1]]
    assert (report.epoch_old, report.epoch_new) == (1, 2)
    assert state.mesh_epoch == 2
    assert state.time == 0.0


def test_reverse_report(backend):
    to_mesh1(backend)
    state, report = backend.apply_marks(marks_of(HOLD, COARSEN, COARSEN, HOLD, HOLD))
    assert state.n_active == 4
    assert report.realized.tolist() == [HOLD, COARSEN, COARSEN, HOLD, HOLD]
    assert report.new_kind.tolist() == [0, 2, 0, 0]
    assert report.new_to_old.tolist() == [[0, -1], [1, 2], [3, -1], [4, -1]]
    assert (report.epoch_old, report.epoch_new) == (2, 3)


def test_a_balance_added_refine_is_reported(backend):
    state, report = to_mesh2(backend)
    assert state.n_active == 7
    # A was marked HOLD and came back REFINE.
    assert report.realized.tolist() == [REFINE, REFINE, HOLD, HOLD, HOLD]
    assert report.new_kind.tolist() == [1, 1, 1, 1, 0, 0, 0]
    assert report.new_to_old[:, 0].tolist() == [0, 0, 1, 1, 2, 3, 4]
    assert (report.new_to_old[:, 1] == -1).all()


def test_a_coarsen_the_balance_check_refuses_does_nothing(backend):
    to_mesh2(backend)
    state, report = backend.apply_marks(marks_of(COARSEN, COARSEN, HOLD, HOLD, HOLD, HOLD, HOLD))
    assert state.n_active == 7
    assert (report.realized == HOLD).all()
    assert (report.new_kind == 0).all()
    assert report.new_to_old[:, 0].tolist() == list(range(7))


def test_an_all_hold_round_still_raises_the_epoch(backend):
    state, report = backend.apply_marks(marks_of(HOLD, HOLD, HOLD, HOLD))
    assert state.n_active == 4
    assert (report.epoch_old, report.epoch_new) == (1, 2)
    assert state.element_id.tolist() == [1, 2, 3, 4]


@pytest.mark.parametrize(
    "bad",
    [
        np.zeros(3, dtype=int),
        np.zeros(4),
        np.full(4, 2),
        [0, 0, 0, 0],
        np.zeros(4, dtype=bool),
    ],
    ids=["length", "float", "value", "list", "bool"],
)
def test_bad_marks_rejected_and_nothing_changes(backend, bad):
    with pytest.raises(ValueError, match=r"^marks: "):
        backend.apply_marks(bad)
    state = backend.get_state()
    assert state.mesh_epoch == 1
    assert state.n_active == 4


# --- compute_error ------------------------------------------------------------

@pytest.mark.parametrize("reduction", ["scalar", "per_variable"])
def test_compute_error_shapes(backend, reduction):
    out = backend.compute_error([ZZ, RAW], reduction)
    assert list(out) == [ZZ, RAW]
    for arr in out.values():
        assert arr.shape == (4, 1)
        assert arr.dtype == np.float64
        assert np.isfinite(arr).all()


def test_compute_error_drops_a_repeated_name(backend):
    assert list(backend.compute_error([ZZ, RAW, ZZ], "scalar")) == [ZZ, RAW]


def test_compute_error_arrays_are_the_callers(backend):
    first = backend.compute_error([ZZ], "scalar")[ZZ]
    kept = first.copy()
    first[:] = 0.0
    assert np.array_equal(backend.compute_error([ZZ], "scalar")[ZZ], kept)
    assert (kept > 0).all()


@pytest.mark.parametrize(
    "indicators, reduction, starts",
    [
        (ZZ, "scalar", r"^indicators: "),
        (["true_error"], "scalar", r"^indicators: "),
        ([ZZ], "max", r"^reduction: "),
    ],
)
def test_compute_error_rejections(backend, indicators, reduction, starts):
    with pytest.raises(ValueError, match=starts):
        backend.compute_error(indicators, reduction)


# --- the snapshot and reinitialize_ic ------------------------------------------

def test_get_snapshot_is_the_1d_array_snapshot(backend):
    snap = backend.get_snapshot()
    assert isinstance(snap, SolverSnapshot)
    assert snap.n_active == 4


def test_get_snapshot_cannot_write(backend):
    with pytest.raises(NotImplementedError):
        backend.get_snapshot(write=True)


def test_reinitialize_ic_zeroes_time_and_keeps_the_mesh(backend):
    to_mesh1(backend)
    backend.step_interval([], "scalar", 1)
    assert backend.get_state().time > 0.0
    backend.reinitialize_ic()
    state = backend.get_state()
    assert state.time == 0.0
    assert state.n_active == 5
    assert state.mesh_epoch == 2


# --- step_interval: the surface -------------------------------------------------

def test_interval_report(backend):
    before = backend.get_state()
    report = backend.step_interval([ZZ], "scalar", 1)
    after = backend.get_state()
    assert isinstance(report, IntervalReport)
    assert report.n_substeps == 20
    assert report.time == after.time
    assert 0.0 < report.time
    assert list(report.acc) == [ZZ]
    assert report.acc[ZZ].shape == (4, 1)
    assert report.mass.shape == (1,)
    assert report.aux is None
    assert report.snapshot is None
    assert after.mesh_epoch == before.mesh_epoch
    assert np.array_equal(after.element_id, before.element_id)
    with pytest.raises(ValueError):
        report.acc[ZZ][0, 0] = 0.0


def test_acc_is_at_least_the_start_and_the_end(backend):
    start = backend.compute_error([ZZ, RAW], "scalar")
    report = backend.step_interval([ZZ, RAW], "scalar", 1)
    end = backend.compute_error([ZZ, RAW], "scalar")
    for name in (ZZ, RAW):
        assert (report.acc[name] >= start[name]).all()
        assert (report.acc[name] >= end[name]).all()


def test_an_indicator_in_aux_equals_its_acc(backend):
    report = backend.step_interval([ZZ], "scalar", 1, aux=[ZZ, RAW])
    assert list(report.aux.fields) == [ZZ, RAW]
    assert np.array_equal(report.aux.fields[ZZ], report.acc[ZZ])
    assert report.aux.fields[ZZ] is not report.acc[ZZ]
    assert report.aux.fields[RAW].shape == (4, 1)


def test_an_interval_that_accumulates_nothing(backend):
    report = backend.step_interval([], "scalar", 1)
    assert dict(report.acc) == {}
    assert report.aux is None
    assert report.n_substeps == 20


def test_two_intervals_advance_time_twice(backend):
    first = backend.step_interval([], "scalar", 1)
    second = backend.step_interval([], "scalar", 1)
    assert second.time > first.time
    assert backend.get_state().time == second.time
    assert second.time == pytest.approx(2 * backend.get_state().interval_T, rel=1e-12)


@pytest.mark.parametrize(
    "accumulate, reduction, cadence, aux, starts",
    [
        ([ZZ], "scalar", 0, (), r"^cadence: "),
        (["true_error"], "scalar", 1, (), r"^accumulate: "),
        ([ZZ], "scalar", 1, ["true_error"], r"^aux: "),
        ([ZZ], "max", 1, (), r"^reduction: "),
    ],
)
def test_step_interval_rejections_leave_time_alone(backend, accumulate, reduction, cadence, aux, starts):
    with pytest.raises(ValueError, match=starts):
        backend.step_interval(accumulate, reduction, cadence, aux)
    assert backend.get_state().time == 0.0


def test_mass_is_the_integral_of_the_solution():
    b = make_backend()
    b.reset({"icase": 1}, amr_freq=20, initial_level=1)
    before = float(b._mass()[0])
    after = float(b.step_interval([], "scalar", 1).mass[0])
    # The Gaussian exp(-256 x^2) integrates to sqrt(pi / 256) = 0.11078; the
    # level-1 mesh's quadrature gave 0.11133 when this was written.
    assert before == pytest.approx(0.11078, abs=2e-3)
    assert after == pytest.approx(before, abs=1e-12)
