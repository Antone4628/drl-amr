"""Tests for the mock backend's apply_marks and dry_run
(backends/mock_quad/backend.py).

AGENT_2D_ROADMAP item 1.2, sitting 2, signals R2, R8-R11 and R3-R6 through
the backend; SOLVER_CONTRACT_V2 section 7.2, behaviors 1 through 7. The
realization itself is tested on the pure functions in test_mock_realize.py.

The backend can only reach a mesh by adapting a uniform one, so each test
builds its mesh with apply_marks. Elements are addressed by centroid, as the
spec asks of any script (section 5.2): a position is good for one mesh_epoch
only. Literal positions are asserted only with rotate_cells=False.

Run: pytest tests/mock_quad/test_mock_backend_adapt.py -v
"""
import numpy as np
import pytest

from backends.mock_quad.backend import MockQuadBackend
from contract.reports import KIND_CHILD, KIND_MERGED, KIND_UNCHANGED, AdaptReport
from contract.solver_contract import COARSEN, HOLD, REFINE
from contract.state import SolverState

C, H, R = COARSEN, HOLD, REFINE

ARRAY_FIELDS = ("level", "neighbors", "neighbor_wrap", "family", "can_refine", "can_coarsen", "centroid", "h")
TWO_CELL_IC = {"x0": 1.0, "y0": 0.5, "width": 0.1}
SQUARE_IC = {"x0": 0.5, "y0": 0.5, "width": 0.1}


def two_cell_backend(**overrides):
    """Two base cells A and B on [0,2]x[0,1]: periodic in x, walled in y."""
    fields = dict(nx=2, ny=1, periodic=(True, False), hi=(2.0, 1.0), rotate_cells=False)
    fields.update(overrides)
    return MockQuadBackend(**fields)


def square_backend(**overrides):
    """A 2x2 grid on the unit square."""
    fields = dict(nx=2, ny=2, rotate_cells=False)
    fields.update(overrides)
    return MockQuadBackend(**fields)


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


def leaf_set(state):
    """The mesh as a set of (x, y, level), whatever its order."""
    return {(x, y, int(level)) for (x, y), level in zip(state.centroid.tolist(), state.level, strict=True)}


def five_elements(**overrides):
    """A two-cell backend on the five-element mesh: A refined once."""
    backend = two_cell_backend(**overrides)
    state = backend.reset(TWO_CELL_IC, amr_freq=10, initial_level=0)
    state, _ = backend.apply_marks(marks_at(state, {(0.5, 0.5): R}))
    return backend, state


# Each scenario builds a backend on some mesh and names one round of marks by
# centroid. They are the variations of the explanation doc's section 1.2.8.

def scenario_main(rotate):
    backend, state = five_elements(rotate_cells=rotate)
    return backend, state, {(0.25, 0.25): R}


def scenario_reverse(rotate):
    backend, state = five_elements(rotate_cells=rotate)
    return backend, state, {(0.25, 0.25): C, (0.75, 0.25): C, (0.25, 0.75): C, (0.75, 0.75): C}


def scenario_partial(rotate):
    backend = two_cell_backend(rotate_cells=rotate)
    state = backend.reset(TWO_CELL_IC, amr_freq=10, initial_level=1)
    return backend, state, {(0.25, 0.25): C, (0.75, 0.25): C, (0.25, 0.75): C}


def scenario_undone(rotate):
    backend = two_cell_backend(rotate_cells=rotate)
    state = backend.reset(TWO_CELL_IC, amr_freq=10, initial_level=1)
    where = {(0.25, 0.25): C, (0.75, 0.25): C, (0.25, 0.75): C, (0.75, 0.75): C, (1.25, 0.25): R}
    return backend, state, where


def scenario_corner_walled(rotate):
    backend = square_backend(periodic=(False, False), rotate_cells=rotate)
    state = backend.reset(SQUARE_IC, amr_freq=10, initial_level=0)
    state, _ = backend.apply_marks(marks_at(state, {(0.25, 0.25): R}))
    return backend, state, {(0.375, 0.375): R}


def scenario_corner_periodic(rotate):
    backend = square_backend(periodic=(True, True), rotate_cells=rotate)
    state = backend.reset(SQUARE_IC, amr_freq=10, initial_level=0)
    state, _ = backend.apply_marks(marks_at(state, {(0.25, 0.25): R}))
    return backend, state, {(0.125, 0.125): R}


def scenario_coarsen_to_refine(rotate):
    backend = two_cell_backend(max_level=3, rotate_cells=rotate)
    state = backend.reset(TWO_CELL_IC, amr_freq=10, initial_level=1)
    state, _ = backend.apply_marks(marks_at(state, {(1.25, 0.25): R}))
    where = {(0.25, 0.25): C, (0.75, 0.25): C, (0.25, 0.75): C, (0.75, 0.75): C, (1.125, 0.125): R}
    return backend, state, where


SCENARIOS = {
    "main": (scenario_main, 5, 11),
    "reverse": (scenario_reverse, 5, 2),
    "partial": (scenario_partial, 8, 8),
    "undone": (scenario_undone, 8, 11),
    "corner_walled": (scenario_corner_walled, 7, 19),
    "corner_periodic": (scenario_corner_periodic, 7, 19),
    "coarsen_to_refine": (scenario_coarsen_to_refine, 11, 17),
}


# --- R2: the reports of the explanation doc --------------------------------

def test_first_adapt_reproduces_the_five_element_instance():
    backend = two_cell_backend()
    state = backend.reset(TWO_CELL_IC, amr_freq=10, initial_level=0)
    state, report = backend.apply_marks(np.array([R, H], dtype=np.int8))

    assert isinstance(state, SolverState) and isinstance(report, AdaptReport)
    assert report.realized.tolist() == [R, H]
    assert report.new_to_old.tolist() == [[0, -1, -1, -1]] * 4 + [[1, -1, -1, -1]]
    assert report.new_kind.tolist() == [KIND_CHILD] * 4 + [KIND_UNCHANGED]
    assert (report.epoch_old, report.epoch_new) == (1, 2)
    assert (report.n_old, report.n_new) == (2, 5)

    # The state is section 1.1.2's table.
    assert state.mesh_epoch == 2
    assert state.level.tolist() == [1, 1, 1, 1, 0]
    assert state.neighbors.tolist() == [
        [[4, -1], [1, -1], [-1, -1], [2, -1]],
        [[0, -1], [4, -1], [-1, -1], [3, -1]],
        [[4, -1], [3, -1], [0, -1], [-1, -1]],
        [[2, -1], [4, -1], [1, -1], [-1, -1]],
        [[1, 3], [0, 2], [-1, -1], [-1, -1]],
    ]
    assert state.family.tolist() == [[0, 1, 2, 3]] * 4 + [[-1, -1, -1, -1]]
    assert state.can_coarsen.tolist() == [True, True, True, True, False]


def test_second_refine_is_the_main_instance():
    backend, state = five_elements(dt_base=0.01)
    assert state.dt_stable == 0.01 / 2
    state, report = backend.apply_marks(np.array([R, H, H, H, H], dtype=np.int8))

    assert state.n_active == 11
    assert state.level.tolist() == [2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1]
    assert report.realized.tolist() == [R, H, H, H, R]
    assert report.new_to_old[:, 0].tolist() == [0, 0, 0, 0, 1, 2, 3, 4, 4, 4, 4]
    assert (report.new_to_old[:, 1:] == -1).all()
    assert report.new_kind.tolist() == [KIND_CHILD] * 4 + [KIND_UNCHANGED] * 3 + [KIND_CHILD] * 4
    assert (report.epoch_old, report.epoch_new) == (2, 3)
    assert state.mesh_epoch == 3
    assert state.dt_stable == 0.01 / 4

    # One refine was asked for: the agent's counter says 5 + 3. Balance
    # added B's, so the realized mesh overshoots it by 3.
    assert state.n_active - (5 + 3) == 3


def test_family_coarsen_reverses_the_first_adapt():
    backend, state = five_elements()
    state, report = backend.apply_marks(np.array([C, C, C, C, H], dtype=np.int8))

    assert state.n_active == 2
    assert state.level.tolist() == [0, 0]
    assert report.realized.tolist() == [C, C, C, C, H]
    assert report.new_to_old.tolist() == [[0, 1, 2, 3], [4, -1, -1, -1]]
    assert report.new_kind.tolist() == [KIND_MERGED, KIND_UNCHANGED]
    assert not state.can_coarsen.any()
    assert state.dt_stable == state.dt_base


# --- R3-R6 through the backend ---------------------------------------------

@pytest.mark.parametrize("name", SCENARIOS)
@pytest.mark.parametrize("rotate", [False, True], ids=["pinned", "rotating"])
def test_scenario_counts(name, rotate):
    build, n_before, n_after = SCENARIOS[name]
    backend, state, where = build(rotate)
    assert state.n_active == n_before
    new_state, report = backend.apply_marks(marks_at(state, where))
    assert new_state.n_active == n_after
    assert (report.n_old, report.n_new) == (n_before, n_after)
    assert new_state.mesh_epoch == state.mesh_epoch + 1


def test_partly_marked_family_is_dropped_and_the_epoch_still_rises():
    backend, state, where = scenario_partial(rotate=False)
    new_state, report = backend.apply_marks(marks_at(state, where))
    assert leaf_set(new_state) == leaf_set(state)
    assert report.realized.tolist() == [H] * 8
    assert report.new_kind.tolist() == [KIND_UNCHANGED] * 8
    assert new_state.mesh_epoch == state.mesh_epoch + 1


def test_family_coarsen_undone_by_balance_comes_back_as_hold():
    backend, state, where = scenario_undone(rotate=True)
    marks = marks_at(state, where)
    new_state, report = backend.apply_marks(marks)
    asked = np.flatnonzero(marks == C)
    assert len(asked) == 4
    assert (report.realized[asked] == H).all()
    assert report.realized[position_at(state, 1.25, 0.25)] == R
    assert int((report.new_kind == KIND_MERGED).sum()) == 0
    assert new_state.n_active == 11


@pytest.mark.parametrize(
    "build, diagonal",
    [(scenario_corner_walled, (0.75, 0.75)), (scenario_corner_periodic, (0.75, 0.75))],
    ids=["walled", "through_both_wraps"],
)
def test_corner_balance_refines_the_diagonal_cell(build, diagonal):
    backend, state, where = build(rotate=True)
    marks = marks_at(state, where)
    diagonal_position = position_at(state, *diagonal)
    assert marks[diagonal_position] == H
    new_state, report = backend.apply_marks(marks)
    assert report.realized[diagonal_position] == R
    assert int((report.realized == R).sum()) == 4
    assert new_state.n_active == 19


def test_coarsen_mark_realized_as_refine():
    backend, state, where = scenario_coarsen_to_refine(rotate=False)
    marks = marks_at(state, where)
    assert marks[1] == C
    new_state, report = backend.apply_marks(marks)
    assert report.realized[1] == R
    assert report.realized.tolist() == [H, R, H, H, R, H, H, H, H, H, H]
    assert new_state.n_active == 17
    assert new_state.level.max() == 3


def test_refine_at_max_level_is_ignored_and_the_adapt_still_runs():
    backend = square_backend()
    state = backend.reset(SQUARE_IC, amr_freq=10, initial_level=2)
    new_state, report = backend.apply_marks(np.full(state.n_active, R, dtype=np.int8))
    assert new_state.n_active == state.n_active == 64
    assert not report.realized.any()
    assert new_state.mesh_epoch == state.mesh_epoch + 1


def test_coarsening_everything_merges_every_family_in_child_order():
    backend = square_backend(rotate_cells=True)
    state = backend.reset(SQUARE_IC, amr_freq=10, initial_level=2)
    new_state, report = backend.apply_marks(np.full(64, C, dtype=np.int8))
    assert new_state.n_active == 16
    assert (new_state.level == 1).all()
    assert (report.realized == C).all()
    assert (report.new_kind == KIND_MERGED).all()
    assert sorted(report.new_to_old.ravel().tolist()) == list(range(64))
    # Child c = ix + 2*iy sits on the (ix, iy) side of its parent's center.
    offsets = state.centroid[report.new_to_old] - new_state.centroid[:, None, :]
    assert (np.sign(offsets[:, :, 0]) == [-1, 1, -1, 1]).all()
    assert (np.sign(offsets[:, :, 1]) == [-1, -1, 1, 1]).all()


# --- R8: dry_run (behavior 7) ----------------------------------------------

@pytest.mark.parametrize("name", SCENARIOS)
@pytest.mark.parametrize("rotate", [False, True], ids=["pinned", "rotating"])
def test_dry_run_agrees_with_apply_marks_and_changes_nothing(name, rotate):
    build, _, n_after = SCENARIOS[name]
    backend, state, where = build(rotate)
    marks = marks_at(state, where)

    count, predicted = backend.dry_run(marks)
    after_dry_run = backend.get_state()
    assert after_dry_run.mesh_epoch == state.mesh_epoch
    assert all(getattr(after_dry_run, field) is getattr(state, field) for field in ARRAY_FIELDS)
    assert predicted.dtype == np.int8

    new_state, report = backend.apply_marks(marks)
    assert count == new_state.n_active == n_after
    assert np.array_equal(predicted, report.realized)


def test_dry_run_twice_gives_the_same_answer():
    backend, state, where = scenario_main(rotate=True)
    marks = marks_at(state, where)
    first, second = backend.dry_run(marks), backend.dry_run(marks)
    assert first[0] == second[0] == 11
    assert np.array_equal(first[1], second[1])
    assert first[1] is not second[1]


# --- R9: the frozen state and the changing order (behaviors 1 and 6) -------

def test_a_state_held_across_an_adapt_is_unchanged():
    backend, old = five_elements()
    kept = {field: getattr(old, field).copy() for field in ARRAY_FIELDS}
    new, _ = backend.apply_marks(np.array([R, H, H, H, H], dtype=np.int8))

    assert (old.n_active, old.mesh_epoch) == (5, 2)
    assert (new.n_active, new.mesh_epoch) == (11, 3)
    for field in ARRAY_FIELDS:
        assert np.array_equal(getattr(old, field), kept[field])
        assert getattr(old, field) is not getattr(new, field)
        with pytest.raises(ValueError, match="read-only"):
            getattr(old, field)[...] = 0
        with pytest.raises(ValueError, match="read-only"):
            getattr(new, field)[...] = 0


def test_apply_marks_returns_the_current_state_and_a_frozen_report():
    backend, _ = five_elements()
    returned, report = backend.apply_marks(np.array([R, H, H, H, H], dtype=np.int8))
    current = backend.get_state()
    assert all(getattr(returned, field) is getattr(current, field) for field in ARRAY_FIELDS)
    assert returned.mesh_epoch == current.mesh_epoch == report.epoch_new
    for array in (report.realized, report.new_to_old, report.new_kind):
        with pytest.raises(ValueError, match="read-only"):
            array[...] = 0


def test_time_does_not_advance_across_an_adapt():
    backend, state = five_elements()
    new_state, _ = backend.apply_marks(np.array([R, H, H, H, H], dtype=np.int8))
    assert new_state.time == state.time == 0.0
    assert new_state.interval_T == state.interval_T
    assert new_state.amr_freq == state.amr_freq


def test_all_hold_changes_the_order_when_the_rotation_is_on():
    backend = square_backend(rotate_cells=True)
    state = backend.reset(SQUARE_IC, amr_freq=10, initial_level=1)
    new_state, report = backend.apply_marks(np.zeros(16, dtype=np.int8))

    assert leaf_set(new_state) == leaf_set(state)
    assert new_state.centroid[0].tolist() != state.centroid[0].tolist()
    assert new_state.mesh_epoch == state.mesh_epoch + 1
    assert (report.realized == H).all()
    assert (report.new_kind == KIND_UNCHANGED).all()
    old_positions = report.new_to_old[:, 0]
    assert sorted(old_positions.tolist()) == list(range(16))
    assert old_positions.tolist() != list(range(16))
    # The report is the link across the adapt: it finds each element again.
    assert np.array_equal(new_state.centroid, state.centroid[old_positions])


def test_all_hold_repeats_the_order_when_the_rotation_is_off():
    backend = square_backend(rotate_cells=False)
    state = backend.reset(SQUARE_IC, amr_freq=10, initial_level=1)
    new_state, report = backend.apply_marks(np.zeros(16, dtype=np.int8))
    assert np.array_equal(new_state.centroid, state.centroid)
    assert report.new_to_old[:, 0].tolist() == list(range(16))
    assert new_state.mesh_epoch == state.mesh_epoch + 1
    assert new_state.level is not state.level


def test_epoch_rises_by_one_at_every_adapt_and_every_reset():
    backend = square_backend()
    epochs = [backend.reset(SQUARE_IC, amr_freq=10, initial_level=0).mesh_epoch]
    for _ in range(2):
        state, report = backend.apply_marks(np.zeros(4, dtype=np.int8))
        assert report.epoch_new == state.mesh_epoch
        epochs.append(state.mesh_epoch)
    epochs.append(backend.reset(SQUARE_IC, amr_freq=10, initial_level=0).mesh_epoch)
    assert epochs == [1, 2, 3, 4]


# --- R10: reset at level k is k all-refine adapts --------------------------

@pytest.mark.parametrize("k", [1, 2])
@pytest.mark.parametrize(
    "make",
    [
        lambda: square_backend(rotate_cells=True),
        lambda: MockQuadBackend(nx=3, ny=2, periodic=(False, False), hi=(3.0, 1.0)),
    ],
    ids=["periodic_2x2", "walled_3x2"],
)
def test_reset_at_level_k_equals_k_all_refine_adapts(make, k):
    ic = {"x0": 0.5, "y0": 0.5, "width": 0.1}
    direct = make().reset(ic, amr_freq=10, initial_level=k)

    backend = make()
    state = backend.reset(ic, amr_freq=10, initial_level=0)
    for _ in range(k):
        state, report = backend.apply_marks(np.full(state.n_active, R, dtype=np.int8))
        assert (report.realized == R).all()

    assert leaf_set(state) == leaf_set(direct)
    assert state.n_active == direct.n_active
    assert (direct.mesh_epoch, state.mesh_epoch) == (1, 1 + k)


# --- R8 and behavior 5 under random marks ----------------------------------

@pytest.mark.parametrize("seed", range(6))
def test_random_rounds_through_the_backend(seed):
    rng = np.random.default_rng(100 + seed)
    periodic = (bool(rng.integers(2)), bool(rng.integers(2)))
    backend = MockQuadBackend(nx=3, ny=2, periodic=periodic, hi=(3.0, 2.0), max_level=3)
    old = backend.reset({"x0": 1.5, "y0": 1.0, "width": 0.1}, amr_freq=10, initial_level=1)

    for _ in range(6):
        marks = rng.choice([C, C, H, R], size=old.n_active).astype(np.int8)
        count, predicted = backend.dry_run(marks)
        new, report = backend.apply_marks(marks)
        assert count == new.n_active == report.n_new
        assert np.array_equal(predicted, report.realized)
        assert (report.epoch_old, report.epoch_new) == (old.mesh_epoch, new.mesh_epoch)

        # The report against the two states, kind by kind.
        first = report.new_to_old[:, 0]
        unchanged = report.new_kind == KIND_UNCHANGED
        child = report.new_kind == KIND_CHILD
        merged = report.new_kind == KIND_MERGED
        assert np.array_equal(new.centroid[unchanged], old.centroid[first[unchanged]])
        assert np.array_equal(new.level[unchanged], old.level[first[unchanged]])
        assert np.array_equal(new.level[child], old.level[first[child]] + 1)
        assert (np.abs(new.centroid[child] - old.centroid[first[child]]) < old.h[first[child]] / 2).all()
        assert (old.level[report.new_to_old[merged]] == new.level[merged][:, None] + 1).all()
        assert np.allclose(old.centroid[report.new_to_old[merged]].mean(axis=1), new.centroid[merged])

        # realized against the report: a refined element has four children,
        # a coarsened one is in a merged row, a held one is unchanged.
        assert (report.realized[first[child]] == R).all()
        assert (report.realized[report.new_to_old[merged]] == C).all()
        assert (report.realized[first[unchanged]] == H).all()
        assert (new.level <= 3).all()

        old = new


# --- R11: what the two verbs reject ----------------------------------------

BAD_MARKS = {
    "wrong_length": (np.zeros(4, dtype=np.int8), r"^marks: shape \(4,\), expected \(5,\)"),
    "float_dtype": (np.zeros(5), r"^marks: dtype float64, expected an integer dtype"),
    "bad_value": (np.array([2, 0, 0, 0, 0]), r"^marks: values must be COARSEN, HOLD or REFINE"),
}


@pytest.mark.parametrize("verb", ["dry_run", "apply_marks"])
@pytest.mark.parametrize("case", BAD_MARKS)
def test_bad_marks_rejected_and_nothing_changes(verb, case):
    marks, message = BAD_MARKS[case]
    backend, before = five_elements()
    with pytest.raises(ValueError, match=message):
        getattr(backend, verb)(marks)
    after = backend.get_state()
    assert after.mesh_epoch == before.mesh_epoch == 2
    assert after.level is before.level


@pytest.mark.parametrize("verb", ["dry_run", "apply_marks"])
def test_adapt_verbs_before_reset_raise(verb):
    with pytest.raises(RuntimeError, match=r"^no mesh yet: call reset first"):
        getattr(two_cell_backend(), verb)(np.zeros(2, dtype=np.int8))


@pytest.mark.parametrize(
    "marks",
    [[R, H, H, H, H], np.array([R, H, H, H, H]), np.array([R, H, H, H, H], dtype=np.int32)],
    ids=["list", "int64", "int32"],
)
def test_marks_in_any_integer_form_are_accepted(marks):
    backend, _ = five_elements()
    state, report = backend.apply_marks(marks)
    assert state.n_active == 11
    assert report.realized.dtype == np.int8
