"""Tests for the mock backend's describe, reset and get_state
(backends/mock_quad/backend.py).

AGENT_2D_ROADMAP item 1.2, sitting 1, signals Q3-Q9. The meshes here are
uniform: reset is the only verb built that makes one. The adapt verbs and the
interval verbs are sittings 2 and 3; until then they raise.

Positions are pinned with rotate_cells=False unless a test is about the
rotation.

Run: pytest tests/mock_quad/test_mock_backend_state.py -v
"""
import numpy as np
import pytest

from backends.mock_quad.backend import INDICATORS, REDUCTIONS, MockQuadBackend
from contract.backend import BackendInfo, SolverBackend
from contract.state import SolverState

IC = {"x0": 0.5, "y0": 0.5, "width": 0.1}
ARRAY_FIELDS = ("level", "neighbors", "neighbor_wrap", "family", "can_refine", "can_coarsen", "centroid", "h")


def periodic_backend(**overrides):
    """A 4x4 grid on the unit square, periodic on both axes."""
    fields = dict(nx=4, ny=4, rotate_cells=False)
    fields.update(overrides)
    return MockQuadBackend(**fields)


def walled_backend(**overrides):
    """A walled 3x2 grid on [0,3]x[0,1]: base cells of size (1.0, 0.5)."""
    fields = dict(nx=3, ny=2, periodic=(False, False), hi=(3.0, 1.0), rotate_cells=False)
    fields.update(overrides)
    return MockQuadBackend(**fields)


def centroid_set(state):
    """The centroids of a state as a set: the mesh, whatever its order."""
    return {tuple(c) for c in state.centroid.tolist()}


# --- Q8: the protocol, and what is not built yet ---------------------------

def test_the_mock_is_a_solver_backend():
    assert isinstance(periodic_backend(), SolverBackend)


@pytest.mark.parametrize(
    "call",
    [
        lambda b: b.compute_error(["ind_a"], "scalar"),
        lambda b: b.dry_run(np.zeros(16, dtype=np.int8)),
        lambda b: b.apply_marks(np.zeros(16, dtype=np.int8)),
        lambda b: b.step_interval(["ind_a"], 1),
        lambda b: b.get_snapshot(),
        lambda b: b.reinitialize_ic(),
    ],
    ids=["compute_error", "dry_run", "apply_marks", "step_interval", "get_snapshot", "reinitialize_ic"],
)
def test_unbuilt_and_absent_verbs_raise(call):
    backend = periodic_backend()
    backend.reset(IC, amr_freq=10, initial_level=0)
    with pytest.raises(NotImplementedError):
        call(backend)


def test_get_state_before_reset_raises():
    with pytest.raises(RuntimeError, match=r"^no mesh yet: call reset first"):
        periodic_backend().get_state()


# --- describe --------------------------------------------------------------

def test_describe_needs_no_reset_and_declares_what_is_built():
    info = periodic_backend(nvar=3).describe()
    assert isinstance(info, BackendInfo)
    assert (info.ndim, info.nvar) == (2, 3)
    assert info.indicators == INDICATORS == ("ind_a", "ind_b")
    assert info.reductions == REDUCTIONS == ("scalar", "per_variable")
    assert info.optional_verbs == frozenset()


def test_ic_bounds_come_from_the_domain():
    params = walled_backend().describe().ic_params
    assert set(params) == {"x0", "y0", "width"}
    assert params["x0"] == (0.0, 3.0)
    assert params["y0"] == (0.0, 1.0)
    assert params["width"] == (0.02, 0.25)


# --- the constructor -------------------------------------------------------

@pytest.mark.parametrize(
    "overrides, message",
    [
        (dict(nvar=0), r"^nvar: 0"),
        (dict(dt_base=0.0), r"^dt_base: 0.0"),
        (dict(dt_base=float("nan")), r"^dt_base: nan"),
        (dict(nx=1), r"^nx: 1, a periodic axis needs at least 2"),
        (dict(max_level=-1), r"^max_level: -1"),
    ],
)
def test_bad_constructor_arguments_rejected(overrides, message):
    with pytest.raises(ValueError, match=message):
        periodic_backend(**overrides)


# --- Q3: a uniform periodic mesh through reset -----------------------------

def test_periodic_reset_at_level_one():
    backend = periodic_backend(dt_base=0.01)
    state = backend.reset(IC, amr_freq=10, initial_level=1)
    assert isinstance(state, SolverState)
    assert state.n_active == 64
    assert (state.level == 1).all()
    assert (state.neighbors[:, :, 0] >= 0).all()
    assert (state.neighbors[:, :, 1] == -1).all()
    assert (state.family >= 0).all()
    assert state.can_coarsen.all() and state.can_refine.all()
    assert state.neighbor_wrap.sum() == 32
    assert state.dt_stable == 0.01 / 2


def test_globals_of_the_first_state():
    state = periodic_backend(nvar=3, dt_base=0.01).reset(IC, amr_freq=10, initial_level=1)
    assert (state.ndim, state.nvar, state.max_level) == (2, 3, 2)
    assert state.time == 0.0
    assert (state.dt_base, state.amr_freq) == (0.01, 10)
    assert state.interval_T == pytest.approx(0.1)
    assert state.domain_lo.tolist() == [0.0, 0.0]
    assert state.domain_hi.tolist() == [1.0, 1.0]
    assert state.periodic.tolist() == [True, True]
    assert state.attributes.shape == (64, 0)
    assert state.state_mean is None
    assert state.element_id is None


@pytest.mark.parametrize("level, count", [(0, 16), (1, 64), (2, 256)])
def test_element_count_and_stable_step_follow_the_level(level, count):
    state = periodic_backend(dt_base=0.01).reset(IC, amr_freq=10, initial_level=level)
    assert state.n_active == count
    assert state.dt_stable == 0.01 / 2**level
    assert state.can_refine.all() == (level < 2)
    assert state.can_coarsen.all() == (level > 0)


def test_default_initial_level_is_one():
    assert (periodic_backend().reset(IC, amr_freq=10).level == 1).all()


# --- Q4: a walled mesh through reset ---------------------------------------

def test_walled_reset_at_level_zero():
    state = walled_backend().reset({"x0": 1.5, "y0": 0.5, "width": 0.1}, amr_freq=10, initial_level=0)
    assert state.n_active == 6
    walls = (state.neighbors == -1).all(axis=2)
    assert walls.sum() == 10
    assert not state.neighbor_wrap.any()
    assert state.h.tolist() == [[1.0, 0.5]] * 6
    assert state.periodic.tolist() == [False, False]
    assert state.domain_hi.tolist() == [3.0, 1.0]


def test_two_periodic_cells_meet_on_both_sides():
    # Two base cells, periodic in x and walled in y: each is the other's
    # neighbor across both x faces, once directly and once through the wrap.
    backend = MockQuadBackend(nx=2, ny=1, periodic=(True, False), hi=(2.0, 1.0), rotate_cells=False)
    state = backend.reset({"x0": 1.0, "y0": 0.5, "width": 0.1}, amr_freq=10, initial_level=0)
    assert state.neighbors.tolist() == [
        [[1, -1], [1, -1], [-1, -1], [-1, -1]],
        [[0, -1], [0, -1], [-1, -1], [-1, -1]],
    ]
    assert state.neighbor_wrap.tolist() == [[True, False, False, False], [False, True, False, False]]


# --- Q5: mesh_epoch --------------------------------------------------------

def test_epoch_rises_by_one_per_reset_whatever_the_level():
    backend = periodic_backend()
    assert backend.reset(IC, amr_freq=10, initial_level=0).mesh_epoch == 1
    assert backend.reset(IC, amr_freq=10, initial_level=2).mesh_epoch == 2
    assert backend.reset(IC, amr_freq=10, initial_level=1).mesh_epoch == 3


def test_get_state_does_not_raise_the_epoch():
    backend = periodic_backend()
    backend.reset(IC, amr_freq=10)
    assert [backend.get_state().mesh_epoch for _ in range(3)] == [1, 1, 1]


# --- Q6: the rotation ------------------------------------------------------

def test_rotation_moves_the_first_base_cell_with_the_epoch():
    backend = MockQuadBackend(nx=2, ny=2, rotate_cells=True)
    first = backend.reset(IC, amr_freq=10, initial_level=0)
    second = backend.reset(IC, amr_freq=10, initial_level=0)
    assert first.centroid[0].tolist() == [0.75, 0.25]  # cell 1 at epoch 1
    assert second.centroid[0].tolist() == [0.25, 0.75]  # cell 2 at epoch 2
    assert centroid_set(first) == centroid_set(second)
    assert len(centroid_set(first)) == 4


def test_without_rotation_the_order_repeats():
    backend = MockQuadBackend(nx=2, ny=2, rotate_cells=False)
    first = backend.reset(IC, amr_freq=10, initial_level=1)
    second = backend.reset(IC, amr_freq=10, initial_level=1)
    assert first.centroid[0].tolist() == [0.125, 0.125]
    assert np.array_equal(first.centroid, second.centroid)


def test_rotation_is_on_by_default():
    state = MockQuadBackend(nx=2, ny=2).reset(IC, amr_freq=10, initial_level=0)
    assert state.centroid[0].tolist() == [0.75, 0.25]


# --- Q7: one epoch, one set of read-only arrays ----------------------------

def test_reset_returns_the_current_state():
    backend = periodic_backend()
    returned = backend.reset(IC, amr_freq=10)
    current = backend.get_state()
    assert all(np.array_equal(getattr(returned, name), getattr(current, name)) for name in ARRAY_FIELDS)
    assert returned.mesh_epoch == current.mesh_epoch


@pytest.mark.parametrize("name", ARRAY_FIELDS)
def test_states_of_one_epoch_share_read_only_arrays(name):
    backend = periodic_backend()
    backend.reset(IC, amr_freq=10)
    first, second = backend.get_state(), backend.get_state()
    assert getattr(first, name) is getattr(second, name)
    with pytest.raises(ValueError, match="read-only"):
        getattr(second, name)[...] = 0


def test_an_old_state_is_untouched_by_a_new_mesh():
    backend = periodic_backend()
    old = backend.reset(IC, amr_freq=10, initial_level=0)
    kept = {name: getattr(old, name).copy() for name in ARRAY_FIELDS}
    new = backend.reset(IC, amr_freq=10, initial_level=2)
    assert new.n_active == 256
    assert old.n_active == 16
    assert all(np.array_equal(getattr(old, name), kept[name]) for name in ARRAY_FIELDS)
    assert old.level is not new.level


# --- Q9: what reset rejects ------------------------------------------------

@pytest.mark.parametrize(
    "ic, message",
    [
        ({**IC, "amplitude": 1.0}, r"^ic: unknown names \['amplitude'\], missing names \[\]"),
        ({"x0": 0.5, "y0": 0.5}, r"^ic: unknown names \[\], missing names \['width'\]"),
        ({**IC, "x0": 1.5}, r"^ic\['x0'\]: 1.5, expected within \[0.0, 1.0\]"),
        ({**IC, "width": 0.5}, r"^ic\['width'\]: 0.5"),
        ({**IC, "y0": float("nan")}, r"^ic\['y0'\]: nan"),
    ],
)
def test_bad_ic_rejected(ic, message):
    with pytest.raises(ValueError, match=message):
        periodic_backend().reset(ic, amr_freq=10)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        (dict(amr_freq=0), r"^amr_freq: 0"),
        (dict(amr_freq=10, initial_level=3), r"^initial_level: 3, expected 0 to max_level = 2"),
        (dict(amr_freq=10, initial_level=-1), r"^initial_level: -1"),
    ],
)
def test_bad_reset_arguments_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        periodic_backend().reset(IC, **kwargs)


def test_a_rejected_reset_leaves_the_episode_as_it_was():
    backend = periodic_backend()
    before = backend.reset(IC, amr_freq=10, initial_level=1)
    with pytest.raises(ValueError):
        backend.reset(IC, amr_freq=10, initial_level=3)
    after = backend.get_state()
    assert after.mesh_epoch == before.mesh_epoch == 1
    assert after.level is before.level
    assert after.amr_freq == 10
