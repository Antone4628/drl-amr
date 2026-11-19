"""Tests for agent/round_cache.py — the round cache and its one read.

AGENT_2D_ROADMAP item 2.1, signals C1-C8. Three groups:

  - on a literal state and a stub backend: what the cache checks and rejects
    (C2, C6), with no real backend involved;
  - on the mock quad backend: the two-round instance of the explanation
    file's section 2.1.2, counted by a wrapper (C1, C3, C4, C5, C7);
  - on Python1DBackend: two intervals of three rounds, counted, each cache
    compared with a twin backend's direct read, bit for bit (C8).

The counting wrapper is the counting fixture of tests/conftest.py: it
forwards every verb to the real backend and counts the calls by name, so
"one compute_error call per round" is read off a Counter.

Run: pytest tests/agent/test_round_cache.py -v
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from agent.round_cache import RoundCache, read_round
from backends.mock_quad.backend import MockQuadBackend
from backends.python_1d.backend import Python1DBackend
from backends.python_1d.solvers.dg_advection_solver_multiround import DGAdvectionSolver
from contract.backend import BackendFailure
from contract.solver_contract import HOLD, REFINE
from contract.state import SolverState

MOCK_NAMES = ("ind_a", "ind_b")
ONE_D_NAMES = ("zz_style", "raw_jump")
# The instance's feature: on cell A's centroid, a quarter of the domain's
# shorter side wide.
IC = {"x0": 0.5, "y0": 0.5, "width": 0.25}
# Cell B's value of both indicators at t = 0: four widths from the feature.
FAR = 16 * np.exp(-8.0)
BASE_XELEM = np.array([-1.0, -0.4, 0.0, 0.4, 1.0])


class StubBackend:
    """A backend whose compute_error returns whatever it was given."""

    def __init__(self, returns):
        self.returns = returns
        self.calls = 0

    def compute_error(self, indicators, reduction):
        self.calls += 1
        return self.returns


def literal_state() -> SolverState:
    """The instance's two-cell mesh at level 0, built by hand: periodic in x,
    walled in y. The cache reads n_active and mesh_epoch from it."""
    return SolverState(
        level=np.zeros(2, dtype=np.int32),
        neighbors=np.array(
            [[[1, -1], [1, -1], [-1, -1], [-1, -1]], [[0, -1], [0, -1], [-1, -1], [-1, -1]]], dtype=np.int32
        ),
        neighbor_wrap=np.array([[True, False, False, False], [False, True, False, False]]),
        family=np.full((2, 4), -1, dtype=np.int32),
        can_refine=np.array([True, True]),
        can_coarsen=np.array([False, False]),
        centroid=np.array([[0.5, 0.5], [1.5, 0.5]]),
        h=np.ones((2, 2)),
        attributes=np.zeros((2, 0)),
        n_active=2,
        ndim=2,
        nvar=1,
        max_level=2,
        mesh_epoch=1,
        time=0.0,
        interval_T=1.0,
        dt_base=0.25,
        amr_freq=4,
        dt_stable=0.25,
        domain_lo=np.array([0.0, 0.0]),
        domain_hi=np.array([2.0, 1.0]),
        periodic=np.array([True, False]),
    )


def good_errors() -> dict[str, np.ndarray]:
    """Fresh, writable arrays of the right form for literal_state."""
    return {"ind_a": np.array([[0.0], [FAR]]), "ind_b": np.array([[16.0], [FAR]])}


def make_mock(*, rotate_cells=False, nvar=1) -> MockQuadBackend:
    """The instance's grid: cells A and B on [0,2]x[0,1]."""
    return MockQuadBackend(
        nx=2,
        ny=1,
        periodic=(True, False),
        lo=(0.0, 0.0),
        hi=(2.0, 1.0),
        max_level=2,
        nvar=nvar,
        dt_base=0.25,
        rotate_cells=rotate_cells,
    )


def refine_cell_a(state: SolverState) -> np.ndarray:
    """Marks that refine whatever lies in cell A, addressed by centroid."""
    return np.where(state.centroid[:, 0] < 1.0, REFINE, HOLD).astype(np.int8)


def make_1d(*, icase=1, initial_level=1):
    """A Python1DBackend on the real base mesh, reset; and its first state."""
    solver = DGAdvectionSolver(
        nop=4, xelem=BASE_XELEM.copy(), max_elements=120, max_level=3, icase=icase, balance=False
    )
    backend = Python1DBackend(solver)
    return backend, backend.reset({"icase": icase}, amr_freq=20, initial_level=initial_level)


# --- The cache itself, on a literal state (C2) -----------------------------------

def test_cache_holds_what_it_was_given():
    state = literal_state()
    cache = RoundCache(state=state, errors=good_errors(), indicators=MOCK_NAMES, reduction="scalar")
    assert cache.state is state
    assert cache.indicators == MOCK_NAMES
    assert cache.reduction == "scalar"
    assert (cache.n_active, cache.mesh_epoch, cache.n_comp) == (2, 1, 1)
    assert list(cache.errors) == list(MOCK_NAMES)
    assert cache.errors["ind_b"].tolist() == [[16.0], [FAR]]


def test_indicators_are_stored_as_a_tuple():
    cache = RoundCache(state=literal_state(), errors=good_errors(), indicators=list(MOCK_NAMES), reduction="scalar")
    assert cache.indicators == MOCK_NAMES
    assert isinstance(cache.indicators, tuple)


@pytest.mark.parametrize("name", MOCK_NAMES)
def test_cached_arrays_are_read_only(name):
    errors = good_errors()
    cache = RoundCache(state=literal_state(), errors=errors, indicators=MOCK_NAMES, reduction="scalar")
    with pytest.raises(ValueError, match="read-only"):
        cache.errors[name][0, 0] = 1.0
    # Frozen in place: the caller's own reference is read-only too.
    assert not errors[name].flags.writeable


def test_the_mapping_is_sealed():
    errors = good_errors()
    cache = RoundCache(state=literal_state(), errors=errors, indicators=MOCK_NAMES, reduction="scalar")
    with pytest.raises(TypeError):
        cache.errors["ind_c"] = cache.errors["ind_a"]
    with pytest.raises(TypeError):
        del cache.errors["ind_a"]
    # The cache holds its own mapping: a key added to the caller's dict
    # afterward does not appear in it.
    errors["ind_c"] = np.zeros((2, 1))
    assert list(cache.errors) == list(MOCK_NAMES)


@pytest.mark.parametrize("field", ["state", "errors", "indicators", "reduction"])
def test_fields_cannot_be_rebound(field):
    cache = RoundCache(state=literal_state(), errors=good_errors(), indicators=MOCK_NAMES, reduction="scalar")
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(cache, field, None)


def test_the_cache_holds_no_backend():
    backend = StubBackend(good_errors())
    cache = read_round(backend, literal_state(), MOCK_NAMES, "scalar")
    assert backend.calls == 1
    assert [f.name for f in dataclasses.fields(cache)] == ["state", "errors", "indicators", "reduction"]
    assert all(getattr(cache, f.name) is not backend for f in dataclasses.fields(cache))


# --- What read_round rejects before it calls the backend (C6) --------------------

@pytest.mark.parametrize(
    "indicators, fragment",
    [("ind_b", "got the string"), ([], "at least one name"), (["ind_a", "ind_a"], "repeated names")],
    ids=["a bare string", "no name", "a repeated name"],
)
def test_bad_names_are_rejected_before_any_call(indicators, fragment):
    backend = StubBackend(good_errors())
    with pytest.raises(ValueError, match="^indicators: ") as excinfo:
        read_round(backend, literal_state(), indicators, "scalar")
    assert fragment in str(excinfo.value)
    assert backend.calls == 0


# --- What the cache rejects of what a backend returns (C6) -----------------------

def _missing():
    return {"ind_a": np.zeros((2, 1))}


def _extra():
    return {**good_errors(), "ind_c": np.zeros((2, 1))}


def _wrong_rows():
    return {**good_errors(), "ind_a": np.zeros((3, 1))}


def _two_widths():
    return {**good_errors(), "ind_b": np.zeros((2, 2))}


def _float32():
    return {**good_errors(), "ind_a": np.zeros((2, 1), dtype=np.float32)}


def _one_axis():
    return {**good_errors(), "ind_a": np.zeros(2)}


def _no_columns():
    return {"ind_a": np.zeros((2, 0)), "ind_b": np.zeros((2, 0))}


def _a_list():
    return {**good_errors(), "ind_a": [[0.0], [1.0]]}


@pytest.mark.parametrize(
    "build, pattern",
    [
        (_missing, r"^errors: names \['ind_a'\], expected \['ind_a', 'ind_b'\]"),
        (_extra, r"^errors: names \['ind_a', 'ind_b', 'ind_c'\]"),
        (_wrong_rows, r"^errors\['ind_a'\]: shape \(3, 1\), expected \(2, 1\)"),
        (_two_widths, r"^errors\['ind_b'\]: shape \(2, 2\), expected \(2, 1\)"),
        (_float32, r"^errors\['ind_a'\]: dtype float32"),
        (_one_axis, r"^errors\['ind_a'\]: expected a 2-D numpy array"),
        (_no_columns, r"^errors\['ind_a'\]: expected a 2-D numpy array with at least one column"),
        (_a_list, r"^errors\['ind_a'\]: expected a 2-D numpy array"),
    ],
    ids=["a missing name", "an extra name", "a wrong row count", "two widths", "float32", "one axis", "no columns",
         "a list"],
)
def test_a_malformed_return_is_rejected_by_name(build, pattern):
    backend = StubBackend(build())
    with pytest.raises(ValueError, match=pattern):
        read_round(backend, literal_state(), MOCK_NAMES, "scalar")
    assert backend.calls == 1


@pytest.mark.parametrize("bad", [np.nan, np.inf], ids=["nan", "inf"])
def test_a_non_finite_value_is_a_backend_failure(bad):
    errors = good_errors()
    errors["ind_b"][1, 0] = bad
    with pytest.raises(BackendFailure, match=r"^errors\['ind_b'\]: not finite"):
        read_round(StubBackend(errors), literal_state(), MOCK_NAMES, "scalar")


def test_wrong_names_freeze_nothing():
    """The names are checked before any array is touched."""
    stray = np.zeros((2, 1))
    with pytest.raises(ValueError, match="^errors: names"):
        RoundCache(state=literal_state(), errors={"ind_x": stray}, indicators=("ind_a",), reduction="scalar")
    assert stray.flags.writeable


# --- On the mock: the instance (C1, C3, C4, C5, C7) ------------------------------

def test_cache_1_of_the_instance(counting):
    mock = make_mock()
    state = mock.reset(IC, amr_freq=4, initial_level=0)
    backend = counting(mock)
    cache = read_round(backend, state, MOCK_NAMES, "scalar")

    assert dict(backend.calls) == {"compute_error": 1}
    assert cache.state is state
    assert (cache.n_active, cache.mesh_epoch, cache.n_comp) == (2, 1, 1)
    assert cache.errors["ind_a"].shape == cache.errors["ind_b"].shape == (2, 1)
    assert np.allclose(cache.errors["ind_a"][:, 0], [0.0, FAR], rtol=1e-12, atol=0.0)
    assert np.allclose(cache.errors["ind_b"][:, 0], [16.0, FAR], rtol=1e-12, atol=0.0)
    with pytest.raises(ValueError, match="read-only"):
        cache.errors["ind_b"][0, 0] = 1.0


@pytest.mark.parametrize("rotate_cells", [False, True], ids=["pinned", "rotating"])
def test_one_interval_of_two_rounds_is_five_calls(rotate_cells, counting):
    mock = make_mock(rotate_cells=rotate_cells)
    state = mock.reset(IC, amr_freq=4, initial_level=0)
    backend = counting(mock)

    cache1 = read_round(backend, state, MOCK_NAMES, "scalar")
    state, _ = backend.apply_marks(refine_cell_a(cache1.state))
    cache2 = read_round(backend, state, MOCK_NAMES, "scalar")
    state, _ = backend.apply_marks(np.full(cache2.n_active, HOLD, dtype=np.int8))
    backend.step_interval(["ind_b"], "scalar", 1)

    # 2R + 1 for R = 2, and no get_state: reset and apply_marks return the state.
    assert dict(backend.calls) == {"compute_error": 2, "apply_marks": 2, "step_interval": 1}

    assert (cache2.n_active, cache2.mesh_epoch, cache2.n_comp) == (5, 2, 1)
    fine = cache2.state.level == 1
    assert fine.sum() == 4
    # Rows are read against the cache's own state, never by a pinned position.
    assert np.allclose(cache2.errors["ind_b"][fine, 0], 4 / np.e, rtol=1e-12, atol=0.0)
    assert np.allclose(cache2.errors["ind_a"][fine, 0], 2 * np.sqrt(2) / np.e, rtol=1e-12, atol=0.0)
    assert np.allclose(cache2.errors["ind_b"][~fine, 0], FAR, rtol=1e-12, atol=0.0)
    assert np.allclose(cache2.errors["ind_a"][~fine, 0], FAR, rtol=1e-12, atol=0.0)
    assert (state.n_active, state.mesh_epoch) == (5, 3)


def test_cache_1_survives_the_adapt():
    mock = make_mock()
    state = mock.reset(IC, amr_freq=4, initial_level=0)
    cache1 = read_round(mock, state, MOCK_NAMES, "scalar")
    before = {name: cache1.errors[name].copy() for name in MOCK_NAMES}

    new_state, _ = mock.apply_marks(refine_cell_a(state))
    read_round(mock, new_state, MOCK_NAMES, "scalar")
    mock.step_interval(["ind_b"], "scalar", 1)

    assert (cache1.n_active, cache1.mesh_epoch) == (2, 1)
    assert cache1.state is state
    for name in MOCK_NAMES:
        assert np.array_equal(cache1.errors[name], before[name])
        assert not cache1.errors[name].flags.writeable


def test_reading_the_cache_calls_nothing(counting):
    mock = make_mock()
    backend = counting(mock)
    state, _ = backend.apply_marks(refine_cell_a(mock.reset(IC, amr_freq=4, initial_level=0)))
    cache = read_round(backend, state, MOCK_NAMES, "scalar")
    before = dict(backend.calls)

    total = 0.0
    for p in range(cache.n_active):
        for name in cache.indicators:
            total += float(cache.errors[name][p, 0])
        total += int(cache.state.level[p]) + int(cache.state.neighbors[p].max()) + int(cache.state.family[p].max())
        total += float(cache.state.centroid[p].sum()) + float(cache.state.h[p].sum())
        total += int(cache.state.can_refine[p]) + int(cache.state.can_coarsen[p])
    total += cache.n_active + cache.mesh_epoch + cache.n_comp

    assert np.isfinite(total)
    assert dict(backend.calls) == before == {"apply_marks": 1, "compute_error": 1}


def test_rows_follow_the_centroids_through_an_all_hold_adapt():
    """With the rotation on, an all-HOLD adapt keeps the mesh and changes the
    order. A cache read against the state apply_marks returned is aligned;
    one paired with the earlier state is not, and its shape cannot tell."""
    mock = MockQuadBackend(nx=2, ny=2, max_level=2, dt_base=0.25, rotate_cells=True)
    state = mock.reset({"x0": 0.3, "y0": 0.6, "width": 0.2}, amr_freq=4, initial_level=1)
    before = read_round(mock, state, MOCK_NAMES, "scalar")
    assert before.n_active == 16

    def by_centroid(cache):
        keys = [tuple(c) for c in cache.state.centroid.tolist()]
        return dict(zip(keys, cache.errors["ind_b"][:, 0].tolist(), strict=True))

    new_state, _ = mock.apply_marks(np.full(16, HOLD, dtype=np.int8))
    after = read_round(mock, new_state, MOCK_NAMES, "scalar")

    assert after.mesh_epoch == before.mesh_epoch + 1
    assert after.state.centroid[0].tolist() != before.state.centroid[0].tolist()
    assert by_centroid(after) == pytest.approx(by_centroid(before), rel=1e-12)

    # The stale pairing: the new order's arrays against the old state. It
    # constructs, since the count matches, and its rows are misplaced.
    stale = RoundCache(
        state=before.state, errors=mock.compute_error(MOCK_NAMES, "scalar"), indicators=MOCK_NAMES, reduction="scalar"
    )
    assert by_centroid(stale) != pytest.approx(by_centroid(before), rel=1e-12)


def test_per_variable_at_nvar_3():
    mock = make_mock(nvar=3)
    state, _ = mock.apply_marks(refine_cell_a(mock.reset(IC, amr_freq=4, initial_level=0)))
    cache = read_round(mock, state, MOCK_NAMES, "per_variable")

    assert cache.n_comp == 3
    assert cache.reduction == "per_variable"
    for name in MOCK_NAMES:
        assert cache.errors[name].shape == (5, 3)
        for v in range(3):
            assert np.allclose(cache.errors[name][:, v], cache.errors[name][:, 0] / (v + 1), rtol=1e-12, atol=0.0)


def test_scalar_at_nvar_3_is_one_column():
    mock = make_mock(nvar=3)
    state = mock.reset(IC, amr_freq=4, initial_level=1)
    cache = read_round(mock, state, MOCK_NAMES, "scalar")
    assert cache.n_comp == 1
    assert cache.errors["ind_b"].shape == (8, 1)


def test_an_unknown_indicator_is_the_backends_error():
    mock = make_mock()
    state = mock.reset(IC, amr_freq=4, initial_level=0)
    with pytest.raises(ValueError, match="^indicators: unknown names"):
        read_round(mock, state, ["ind_z"], "scalar")


def test_an_unknown_reduction_is_the_backends_error():
    mock = make_mock()
    state = mock.reset(IC, amr_freq=4, initial_level=0)
    with pytest.raises(ValueError, match="^reduction: "):
        read_round(mock, state, MOCK_NAMES, "median")


# --- On Python1DBackend (C8) -----------------------------------------------------

def test_1d_first_cache(counting):
    inner, state = make_1d()
    twin, _ = make_1d()
    backend = counting(inner)
    cache = read_round(backend, state, ONE_D_NAMES, "scalar")

    assert dict(backend.calls) == {"compute_error": 1}
    assert (cache.n_active, cache.mesh_epoch, cache.n_comp) == (8, 1, 1)
    direct = twin.compute_error(ONE_D_NAMES, "scalar")
    for name in ONE_D_NAMES:
        assert cache.errors[name].shape == (8, 1)
        assert np.array_equal(cache.errors[name], direct[name])
        assert not cache.errors[name].flags.writeable


@pytest.mark.parametrize("seed", [0, 1])
def test_1d_two_intervals_of_three_rounds(seed, counting):
    """One compute_error per round on real numerics. The twin is a second
    backend driven through the same marks and intervals; it is read directly
    and never through a cache."""
    rng = np.random.default_rng(seed)
    icase = int(rng.choice([1, 12, 16]))
    inner, state = make_1d(icase=icase)
    twin, _ = make_1d(icase=icase)
    backend = counting(inner)
    assert state.max_level == 3

    for _ in range(2):
        for _ in range(state.max_level):
            cache = read_round(backend, state, ONE_D_NAMES, "scalar")
            direct = twin.compute_error(ONE_D_NAMES, "scalar")
            assert cache.mesh_epoch == state.mesh_epoch
            for name in ONE_D_NAMES:
                assert cache.errors[name].shape == (state.n_active, 1)
                assert np.array_equal(cache.errors[name], direct[name])

            marks = rng.integers(-1, 2, size=state.n_active).astype(np.int8)
            state, _ = backend.apply_marks(marks)
            twin.apply_marks(marks)
        backend.step_interval(ONE_D_NAMES[:1], "scalar", 1)
        twin.step_interval(ONE_D_NAMES[:1], "scalar", 1)

    assert dict(backend.calls) == {"compute_error": 6, "apply_marks": 6, "step_interval": 2}
