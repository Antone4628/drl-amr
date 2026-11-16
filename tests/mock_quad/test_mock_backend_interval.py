"""Tests for the mock backend's compute_error and step_interval
(backends/mock_quad/backend.py).

AGENT_2D_ROADMAP item 1.2, sitting 3, signals F2-F11; SOLVER_CONTRACT_V2
section 7.2, behaviors 9 and 10. The field functions themselves are tested
in test_mock_fields.py.

The instance is the explanation doc's (section 1.2.14): two base cells A and
B on [0,2]x[0,1], periodic in x and walled in y, at level 0. The feature
starts on A's centroid with width 0.25 and velocity (1, 0); dt_base = 0.25
and amr_freq = 4, so one interval is T = 1 and carries the feature to B's
centroid in four substeps. Along it:

    sample   A ind_a   A ind_b   B ind_a   B ind_b
    0        0         16        0.0054    0.0054
    1        2.426     9.70      0.1333    0.178
    2        1.083     2.165     1.083     2.165
    3        0.1333    0.178     2.426     9.70
    4        0.0054    0.0054    0         16

Elements are addressed by centroid wherever a mesh has been adapted.

Run: pytest tests/mock_quad/test_mock_backend_interval.py -v
"""
import numpy as np
import pytest

from backends.mock_quad.backend import MockQuadBackend
from contract.backend import SolverBackend
from contract.reports import KIND_UNCHANGED, AuxFields, IntervalReport
from contract.solver_contract import COARSEN, HOLD, REFINE

C, H, R = COARSEN, HOLD, REFINE
E = np.exp

IC = {"x0": 0.5, "y0": 0.5, "width": 0.25}
ARRAY_FIELDS = ("level", "neighbors", "neighbor_wrap", "family", "can_refine", "can_coarsen", "centroid", "h")
BOTH = ["ind_a", "ind_b"]


def instance(initial_level=0, **overrides):
    """The two-cell backend of the explanation doc, after reset."""
    fields = dict(nx=2, ny=1, periodic=(True, False), hi=(2.0, 1.0), dt_base=0.25, rotate_cells=False)
    fields.update(overrides)
    backend = MockQuadBackend(**fields)
    backend.reset(IC, amr_freq=4, initial_level=initial_level)
    return backend


def position_at(state, x, y):
    """The position of the element whose centroid is (x, y)."""
    matches = np.flatnonzero(np.isclose(state.centroid, (x, y)).all(axis=1))
    assert len(matches) == 1, f"no single element at ({x}, {y})"
    return int(matches[0])


def marks_at(state, where):
    """A marks array: HOLD everywhere but at the centroids in where."""
    marks = np.zeros(state.n_active, dtype=np.int8)
    for (x, y), mark in where.items():
        marks[position_at(state, x, y)] = mark
    return marks


def column(array):
    """An (n, 1) array as a list of n floats."""
    assert array.shape[1] == 1
    return array[:, 0].tolist()


# --- F2: compute_error -----------------------------------------------------

def test_compute_error_at_the_start():
    out = instance().compute_error(BOTH, "scalar")
    assert list(out) == BOTH
    assert all(arr.shape == (2, 1) and arr.dtype == np.float64 for arr in out.values())
    assert column(out["ind_a"]) == pytest.approx([0.0, 16 * E(-8)])
    assert column(out["ind_b"]) == pytest.approx([16.0, 16 * E(-8)])
    assert out["ind_a"][0, 0] == 0.0 and out["ind_b"][0, 0] == 16.0


def test_compute_error_returns_new_arrays_the_caller_owns():
    backend = instance()
    first, second = backend.compute_error(BOTH, "scalar"), backend.compute_error(BOTH, "scalar")
    for name in BOTH:
        assert np.array_equal(first[name], second[name])
        assert first[name] is not second[name]
    first["ind_b"][...] = -1.0
    assert column(backend.compute_error(["ind_b"], "scalar")["ind_b"]) == pytest.approx([16.0, 16 * E(-8)])


def test_compute_error_changes_nothing():
    backend = instance()
    before = backend.get_state()
    backend.compute_error(BOTH, "per_variable")
    after = backend.get_state()
    assert (after.mesh_epoch, after.time) == (before.mesh_epoch, before.time) == (1, 0.0)
    assert all(getattr(after, field) is getattr(before, field) for field in ARRAY_FIELDS)


def test_compute_error_returns_only_the_names_asked_for():
    backend = instance()
    assert list(backend.compute_error(["ind_b"], "scalar")) == ["ind_b"]
    assert backend.compute_error([], "scalar") == {}
    assert list(backend.compute_error(("ind_b", "ind_a"), "scalar")) == ["ind_b", "ind_a"]


def test_the_backend_is_still_a_solver_backend():
    assert isinstance(instance(), SolverBackend)


# --- F3: one interval on the instance --------------------------------------

def test_one_interval_on_the_instance():
    backend = instance()
    report = backend.step_interval(BOTH, "scalar", 1)

    assert isinstance(report, IntervalReport)
    assert report.time == 1.0
    assert report.n_substeps == 4
    assert list(report.acc) == BOTH
    # The ring passed over each centroid at d = w: once leaving A, once
    # arriving at B. The bump sat on A at the start and on B at the end.
    assert column(report.acc["ind_a"]) == pytest.approx([4 * E(-0.5)] * 2)
    assert column(report.acc["ind_b"]) == [16.0, 16.0]
    assert report.mass.tolist() == pytest.approx([2 * np.pi * 0.0625])
    assert report.aux is None
    assert report.snapshot is None


def test_the_interval_moves_the_time_and_not_the_mesh():
    backend = instance()
    before = backend.get_state()
    backend.step_interval(BOTH, "scalar", 1)
    after = backend.get_state()

    assert (before.time, after.time) == (0.0, 1.0)
    assert after.mesh_epoch == before.mesh_epoch == 1
    assert all(getattr(after, field) is getattr(before, field) for field in ARRAY_FIELDS)
    assert (after.interval_T, after.amr_freq, after.dt_stable) == (1.0, 4, 0.25)


def test_after_the_interval_the_feature_sits_on_b():
    backend = instance()
    backend.step_interval([], "scalar", 1)
    out = backend.compute_error(BOTH, "scalar")
    # The end state alone says A is quiet; acc is what remembers the feature
    # leaving it.
    assert column(out["ind_b"]) == pytest.approx([16 * E(-8), 16.0])
    assert column(out["ind_a"]) == pytest.approx([16 * E(-8), 0.0])


def test_the_reports_arrays_are_read_only():
    report = instance().step_interval(BOTH, "scalar", 1, aux=["true_error"])
    arrays = [*report.acc.values(), *report.aux.fields.values(), report.mass]
    assert len(arrays) == 4
    for array in arrays:
        with pytest.raises(ValueError, match="read-only"):
            array[...] = 0


# --- F4: the cadence -------------------------------------------------------

@pytest.mark.parametrize(
    "cadence, ring",
    [
        (1, [4 * E(-0.5), 4 * E(-0.5)]),  # samples 0 1 2 3 4
        (2, [8 * E(-2), 8 * E(-2)]),  # samples 0 2 4
        (3, [12 * E(-4.5), 4 * E(-0.5)]),  # samples 0 3 4
        (4, [16 * E(-8), 16 * E(-8)]),  # samples 0 4: the ring is missed
        (100, [16 * E(-8), 16 * E(-8)]),  # the start and the last, always
    ],
)
def test_the_cadence_decides_what_the_accumulator_sees(cadence, ring):
    report = instance().step_interval(BOTH, "scalar", cadence)
    assert column(report.acc["ind_a"]) == pytest.approx(ring)
    # The start and the end are always sampled, and the bump peaks there.
    assert column(report.acc["ind_b"]) == [16.0, 16.0]
    assert report.n_substeps == 4


# --- F5: the clock ---------------------------------------------------------

def five_elements(**overrides):
    backend = instance(**overrides)
    state, _ = backend.apply_marks(marks_at(backend.get_state(), {(0.5, 0.5): R}))
    return backend, state


def eleven_elements(**overrides):
    backend, state = five_elements(**overrides)
    state, _ = backend.apply_marks(marks_at(state, {(0.25, 0.25): R}))
    return backend, state


@pytest.mark.parametrize(
    "build, n_active, n_substeps",
    [(five_elements, 5, 8), (eleven_elements, 11, 16)],
    ids=["five_elements", "eleven_elements"],
)
def test_the_substep_count_follows_the_realized_mesh(build, n_active, n_substeps):
    backend, state = build()
    assert state.n_active == n_active
    report = backend.step_interval(BOTH, "scalar", 1)
    assert report.n_substeps == n_substeps == round(state.interval_T / state.dt_stable)
    assert report.time == 1.0
    assert all(arr.shape == (n_active, 1) for arr in report.acc.values())


@pytest.mark.parametrize("initial_level, n_substeps", [(0, 3), (1, 6), (2, 12), (3, 24)])
def test_substeps_are_amr_freq_times_two_to_the_finest_level(initial_level, n_substeps):
    backend = MockQuadBackend(nx=2, ny=2, max_level=3)
    backend.reset({"x0": 0.5, "y0": 0.5, "width": 0.1}, amr_freq=3, initial_level=initial_level)
    assert backend.step_interval([], "scalar", 1).n_substeps == n_substeps


def test_time_is_a_whole_number_of_intervals():
    backend = MockQuadBackend(nx=2, ny=2, dt_base=0.01)
    state = backend.reset({"x0": 0.5, "y0": 0.5, "width": 0.1}, amr_freq=10)
    for k in range(1, 8):
        report = backend.step_interval([], "scalar", 5)
        assert report.time == backend.get_state().time == k * state.interval_T


def test_a_lap_of_the_periodic_axis_brings_the_fields_back():
    backend = instance()
    start = backend.compute_error(BOTH, "scalar")
    assert backend.step_interval(BOTH, "scalar", 1).time == 1.0
    assert backend.step_interval(BOTH, "scalar", 1).time == 2.0
    later = backend.compute_error(BOTH, "scalar")
    for name in BOTH:
        assert column(later[name]) == pytest.approx(column(start[name]))


def test_every_interval_starts_where_the_last_one_ended():
    backend = instance()
    for _ in range(3):
        before = backend.compute_error(["ind_b"], "scalar")["ind_b"]
        report = backend.step_interval(["ind_b"], "scalar", 1)
        after = backend.compute_error(["ind_b"], "scalar")["ind_b"]
        assert (report.acc["ind_b"] >= before).all()
        assert (report.acc["ind_b"] >= after).all()
        assert column(report.acc["ind_b"]) == [16.0, 16.0]


def test_reset_puts_the_clock_and_the_feature_back():
    backend = instance()
    start = backend.compute_error(BOTH, "scalar")
    backend.step_interval(BOTH, "scalar", 1)
    state = backend.reset(IC, amr_freq=4, initial_level=0)
    assert state.time == 0.0
    again = backend.compute_error(BOTH, "scalar")
    assert all(np.array_equal(again[name], start[name]) for name in BOTH)
    assert backend.step_interval(BOTH, "scalar", 1).time == 1.0


def test_a_state_held_across_an_interval_is_unchanged():
    backend = instance()
    old = backend.get_state()
    kept = {field: getattr(old, field).copy() for field in ARRAY_FIELDS}
    backend.step_interval(BOTH, "scalar", 1)
    assert old.time == 0.0
    assert all(np.array_equal(getattr(old, field), kept[field]) for field in ARRAY_FIELDS)


# --- F6: an adapt transfers nothing ----------------------------------------

def test_compute_error_after_an_adapt_is_the_fields_on_the_new_mesh():
    backend, state = five_elements(rotate_cells=True)
    out = backend.compute_error(BOTH, "scalar")
    children = [position_at(state, x, y) for x in (0.25, 0.75) for y in (0.25, 0.75)]
    b = position_at(state, 1.5, 0.5)
    # A had ind_b = 16 and ind_a = 0. Its children: rho = 2, d / w = sqrt(2).
    assert out["ind_b"][children, 0].tolist() == pytest.approx([4 * E(-1)] * 4)
    assert out["ind_a"][children, 0].tolist() == pytest.approx([2 * np.sqrt(2) * E(-1)] * 4)
    assert out["ind_b"][b, 0] == pytest.approx(16 * E(-8))
    assert out["ind_a"][b, 0] == pytest.approx(16 * E(-8))


def test_an_adapt_does_not_move_the_time_and_an_interval_does_not_move_the_mesh():
    backend, state = five_elements()
    assert (state.time, state.mesh_epoch) == (0.0, 2)
    backend.step_interval(BOTH, "scalar", 1)
    state, _ = backend.apply_marks(np.zeros(5, dtype=np.int8))
    assert (state.time, state.mesh_epoch) == (1.0, 3)


@pytest.mark.parametrize("seed", range(4))
def test_unchanged_elements_keep_their_values_across_an_adapt(seed):
    rng = np.random.default_rng(200 + seed)
    backend = MockQuadBackend(nx=3, ny=2, hi=(3.0, 2.0), max_level=3, nvar=2, dt_base=0.05)
    old = backend.reset({"x0": 1.2, "y0": 0.7, "width": 0.3}, amr_freq=4, initial_level=1)

    for _ in range(5):
        before = backend.compute_error(BOTH, "per_variable")
        new, report = backend.apply_marks(rng.choice([C, C, H, R], size=old.n_active).astype(np.int8))
        after = backend.compute_error(BOTH, "per_variable")
        unchanged = report.new_kind == KIND_UNCHANGED
        assert unchanged.any()
        for name in BOTH:
            assert after[name].shape == (new.n_active, 2)
            assert np.array_equal(after[name][unchanged], before[name][report.new_to_old[unchanged, 0]])
        backend.step_interval([], "scalar", 4)
        old = new


# --- F7: nvar = 3 (behavior 9) ---------------------------------------------

def test_every_shape_at_nvar_three():
    backend, state = five_elements(nvar=3)
    assert state.nvar == 3 and backend.describe().nvar == 3
    assert state.n_active == 5

    per_variable = backend.compute_error(BOTH, "per_variable")
    scalar = backend.compute_error(BOTH, "scalar")
    for name in BOTH:
        assert per_variable[name].shape == (5, 3)
        assert scalar[name].shape == (5, 1)
        assert per_variable[name][:, 1].tolist() == pytest.approx((per_variable[name][:, 0] / 2).tolist())
        assert per_variable[name][:, 2].tolist() == pytest.approx((per_variable[name][:, 0] / 3).tolist())
        assert np.array_equal(scalar[name], per_variable[name].max(axis=1, keepdims=True))

    report = backend.step_interval(BOTH, "per_variable", 2, aux=["true_error"])
    assert all(arr.shape == (5, 3) and arr.dtype == np.float64 for arr in report.acc.values())
    assert report.aux.fields["true_error"].shape == (5, 3)
    assert report.mass.shape == (3,) and report.mass.dtype == np.float64
    assert report.mass.tolist() == pytest.approx([2 * np.pi * 0.0625 / v for v in (1, 2, 3)])
    assert backend.get_state().nvar == 3


def test_the_scalar_interval_at_nvar_three_is_one_column():
    backend, _ = five_elements(nvar=3)
    report = backend.step_interval(BOTH, "scalar", 1, aux=["true_error"])
    assert all(arr.shape == (5, 1) for arr in report.acc.values())
    assert report.aux.fields["true_error"].shape == (5, 1)
    assert report.mass.shape == (3,)


def test_the_accumulated_scalar_is_the_accumulated_first_variable():
    scalar = instance(nvar=3).step_interval(BOTH, "scalar", 1)
    per_variable = instance(nvar=3).step_interval(BOTH, "per_variable", 1)
    for name in BOTH:
        assert np.array_equal(scalar.acc[name][:, 0], per_variable.acc[name][:, 0])


# --- F8: the auxiliary channel ---------------------------------------------

def test_true_error_comes_back_as_its_interval_maximum():
    report = instance().step_interval(BOTH, "scalar", 1, aux=["true_error"])
    assert isinstance(report.aux, AuxFields)
    assert list(report.aux.fields) == ["true_error"]
    # rho**3 = 64 on A at the start and on B at the end.
    assert column(report.aux.fields["true_error"]) == [64.0, 64.0]
    assert report.aux.fields["true_error"].shape == report.acc["ind_a"].shape


def test_an_indicator_named_in_aux_equals_its_accumulator():
    report = instance().step_interval(BOTH, "scalar", 3, aux=["ind_a", "true_error"])
    assert list(report.aux.fields) == ["ind_a", "true_error"]
    assert np.array_equal(report.aux.fields["ind_a"], report.acc["ind_a"])
    assert column(report.aux.fields["ind_a"]) == pytest.approx([12 * E(-4.5), 4 * E(-0.5)])


def test_aux_follows_the_cadence_like_acc():
    backend, _ = five_elements()
    every = backend.step_interval([], "scalar", 1, aux=["ind_a"]).aux.fields["ind_a"]
    backend.reset(IC, amr_freq=4, initial_level=0)
    backend.apply_marks(np.array([R, H], dtype=np.int8))
    ends = backend.step_interval([], "scalar", 8, aux=["ind_a"]).aux.fields["ind_a"]
    assert (every >= ends).all() and (every > ends).any()


@pytest.mark.parametrize("aux", [(), []], ids=["tuple", "list"])
def test_no_aux_names_gives_no_aux(aux):
    assert instance().step_interval(BOTH, "scalar", 1, aux=aux).aux is None
    assert instance().step_interval(BOTH, "scalar", 1).aux is None


def test_aux_alone_and_nothing_at_all():
    report = instance().step_interval([], "scalar", 1, aux=["true_error"])
    assert report.acc == {}
    assert column(report.aux.fields["true_error"]) == [64.0, 64.0]

    empty = instance().step_interval([], "scalar", 1)
    assert empty.acc == {} and empty.aux is None
    assert (empty.time, empty.n_substeps) == (1.0, 4)


def test_acc_holds_exactly_the_names_asked_for():
    assert list(instance().step_interval(["ind_b"], "scalar", 1).acc) == ["ind_b"]
    assert list(instance().step_interval(["ind_b", "ind_a", "ind_b"], "scalar", 1).acc) == ["ind_b", "ind_a"]


def test_mass_is_constant_across_intervals_and_adapts():
    backend = instance()
    first = backend.step_interval([], "scalar", 1).mass
    backend.apply_marks(np.array([R, R], dtype=np.int8))
    second = backend.step_interval([], "scalar", 1).mass
    assert np.array_equal(first, second)
    assert first is not second


# --- F9: the accumulator under random meshes -------------------------------

def random_backend(seed):
    rng = np.random.default_rng(300 + seed)
    periodic = (bool(rng.integers(2)), bool(rng.integers(2)))
    velocity = tuple(float(v) for v in rng.uniform(-1.5, 1.5, size=2))
    nvar = int(rng.integers(1, 4))
    backend = MockQuadBackend(
        nx=3, ny=2, periodic=periodic, hi=(3.0, 2.0), max_level=3, nvar=nvar, dt_base=0.05, velocity=velocity
    )
    ic = {"x0": float(rng.uniform(0.0, 3.0)), "y0": float(rng.uniform(0.0, 2.0)), "width": float(rng.uniform(0.1, 0.5))}
    backend.reset(ic, amr_freq=int(rng.integers(1, 5)), initial_level=1)
    return backend, nvar


@pytest.mark.parametrize("seed", range(6))
def test_the_accumulator_bounds_the_start_the_end_and_any_coarser_cadence(seed):
    # Two backends built alike and driven alike: one samples every substep,
    # the other at a random cadence.
    fine, nvar = random_backend(seed)
    coarse, _ = random_backend(seed)
    rng = np.random.default_rng(400 + seed)
    names = BOTH + ["true_error"]

    for _ in range(5):
        marks = rng.choice([C, C, H, R], size=fine.get_state().n_active).astype(np.int8)
        state, _ = fine.apply_marks(marks)
        coarse.apply_marks(marks)
        reduction = str(rng.choice(["scalar", "per_variable"]))
        cadence = int(rng.integers(2, 12))
        shape = (state.n_active, 1 if reduction == "scalar" else nvar)

        start = fine.compute_error(BOTH, reduction)
        every = fine.step_interval(BOTH, reduction, 1, aux=names)
        some = coarse.step_interval(BOTH, reduction, cadence, aux=names)
        end = fine.compute_error(BOTH, reduction)

        assert every.n_substeps == some.n_substeps == state.amr_freq * 2 ** int(state.level.max())
        assert every.time == some.time == fine.get_state().time
        assert every.mass.shape == (nvar,)
        for name in BOTH:
            assert every.acc[name].shape == some.acc[name].shape == shape
            assert (every.acc[name] >= start[name]).all() and (every.acc[name] >= end[name]).all()
            assert (some.acc[name] >= start[name]).all() and (some.acc[name] >= end[name]).all()
        for name in names:
            assert every.aux.fields[name].shape == shape
            assert (every.aux.fields[name] >= some.aux.fields[name]).all()
            assert np.isfinite(every.aux.fields[name]).all()


# --- F10: a walled axis ----------------------------------------------------

def test_through_a_wall_the_feature_does_not_come_back():
    backend = MockQuadBackend(nx=2, ny=2, periodic=(False, False), dt_base=0.25, rotate_cells=False)
    backend.reset({"x0": 0.375, "y0": 0.375, "width": 0.05}, amr_freq=4, initial_level=1)
    names = BOTH + ["true_error"]

    first = backend.step_interval(BOTH, "scalar", 1, aux=names)
    assert first.acc["ind_b"].max() == pytest.approx(25.0)
    backend.step_interval(BOTH, "scalar", 1)
    backend.step_interval(BOTH, "scalar", 1)

    # Three units out through the x-max wall, sixty widths from any centroid.
    fourth = backend.step_interval(BOTH, "scalar", 1, aux=names)
    assert all((arr < 1e-12).all() for arr in fourth.aux.fields.values())
    assert all((arr < 1e-12).all() for arr in backend.compute_error(BOTH, "scalar").values())
    # The mock's mass is the feature's integral over the plane: unchanged.
    assert np.array_equal(fourth.mass, first.mass)


def test_on_a_periodic_grid_the_same_feature_comes_back():
    backend = MockQuadBackend(nx=2, ny=2, periodic=(True, True), dt_base=0.25, rotate_cells=False)
    backend.reset({"x0": 0.375, "y0": 0.375, "width": 0.05}, amr_freq=4, initial_level=1)
    for _ in range(3):
        backend.step_interval([], "scalar", 1)
    assert backend.compute_error(["ind_b"], "scalar")["ind_b"].max() == pytest.approx(25.0)


# --- F11: what the two verbs reject ----------------------------------------

BAD_CALLS = {
    "compute_error_unknown": (
        lambda b: b.compute_error(["ind_a", "nope"], "scalar"),
        r"^indicators: unknown names \['nope'\], expected names from \['ind_a', 'ind_b'\]",
    ),
    "compute_error_true_error": (
        lambda b: b.compute_error(["true_error"], "scalar"),
        r"^indicators: unknown names \['true_error'\]",
    ),
    "compute_error_reduction": (
        lambda b: b.compute_error(["ind_a"], "mean"),
        r"^reduction: 'mean', expected one of \['scalar', 'per_variable'\]",
    ),
    "accumulate_unknown": (
        lambda b: b.step_interval(["nope"], "scalar", 1),
        r"^accumulate: unknown names \['nope'\]",
    ),
    "accumulate_true_error": (
        lambda b: b.step_interval(["true_error"], "scalar", 1),
        r"^accumulate: unknown names \['true_error'\]",
    ),
    "aux_unknown": (
        lambda b: b.step_interval(["ind_a"], "scalar", 1, aux=["exact"]),
        r"^aux: unknown names \['exact'\], expected names from \['ind_a', 'ind_b', 'true_error'\]",
    ),
    "interval_reduction": (
        lambda b: b.step_interval(["ind_a"], "mean", 1),
        r"^reduction: 'mean', expected one of \['scalar', 'per_variable'\]",
    ),
    "interval_reduction_with_no_names": (
        lambda b: b.step_interval([], "mean", 1),
        r"^reduction: 'mean'",
    ),
    "cadence_zero": (lambda b: b.step_interval(["ind_a"], "scalar", 0), r"^cadence: 0, expected at least 1"),
    "cadence_negative": (lambda b: b.step_interval(["ind_a"], "scalar", -3), r"^cadence: -3, expected at least 1"),
}


@pytest.mark.parametrize("case", BAD_CALLS)
def test_a_rejected_call_changes_nothing(case):
    call, message = BAD_CALLS[case]
    backend = instance()
    with pytest.raises(ValueError, match=message):
        call(backend)
    assert backend.get_state().time == 0.0
    # The interval counter did not move either: the next interval is the first.
    report = backend.step_interval(BOTH, "scalar", 1)
    assert report.time == 1.0
    assert column(report.acc["ind_b"]) == [16.0, 16.0]


@pytest.mark.parametrize(
    "call",
    [lambda b: b.compute_error(["ind_a"], "scalar"), lambda b: b.step_interval(["ind_a"], "scalar", 1)],
    ids=["compute_error", "step_interval"],
)
def test_field_verbs_before_reset_raise(call):
    backend = MockQuadBackend(nx=2, ny=2)
    with pytest.raises(RuntimeError, match=r"^no mesh yet: call reset first"):
        call(backend)
