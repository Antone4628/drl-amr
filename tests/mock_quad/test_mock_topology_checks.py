"""The backend-independent topology checks (contract/checks.py) on every
kind of state the mock quad backend makes.

AGENT_2D_ROADMAP item 1.3, signals T3 and T4. The checks were written from
the spec's conventions and not from the mock's quadtree, so a clean result
here is two implementations agreeing. The checks themselves, and what each
one catches, are tested on literal states in
tests/contract/test_contract_topology.py.

Run: pytest tests/mock_quad/test_mock_topology_checks.py -v
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from backends.mock_quad.backend import MockQuadBackend
from contract.checks import topology_violations
from contract.solver_contract import COARSEN, HOLD, REFINE

C, H, R = COARSEN, HOLD, REFINE

GRIDS = {
    "periodic_4x4": dict(nx=4, ny=4),
    "walled_3x2": dict(nx=3, ny=2, periodic=(False, False), hi=(3.0, 1.0)),
    "mixed_2x1": dict(nx=2, ny=1, periodic=(True, False), hi=(2.0, 1.0)),
    "offset_3x2": dict(nx=3, ny=2, periodic=(False, True), lo=(-1.0, 0.5), hi=(1.0, 3.5)),
}


def middle_ic(backend):
    """An IC at the middle of every parameter's bounds."""
    return {name: (low + high) / 2 for name, (low, high) in backend.describe().ic_params.items()}


def start(backend, level):
    return backend.reset(middle_ic(backend), amr_freq=10, initial_level=level)


def marks_at(state, where):
    """A marks array: HOLD everywhere but at the centroids in where."""
    marks = np.zeros(state.n_active, dtype=np.int8)
    for (x, y), mark in where.items():
        matches = np.flatnonzero(np.isclose(state.centroid, (x, y)).all(axis=1))
        assert len(matches) == 1, f"no single element at ({x}, {y})"
        marks[matches[0]] = mark
    return marks


def two_cell(rotate, **overrides):
    """Two base cells on [0,2]x[0,1]: periodic in x, walled in y."""
    return MockQuadBackend(nx=2, ny=1, periodic=(True, False), hi=(2.0, 1.0), rotate_cells=rotate, **overrides)


# Each scenario is a backend, a starting level, and rounds of marks by
# centroid: the meshes of the explanation doc's section 1.2.8, with the
# element count expected after the last round.
FAMILY_A = {(0.25, 0.25): C, (0.75, 0.25): C, (0.25, 0.75): C, (0.75, 0.75): C}
SCENARIOS = {
    "main": (lambda r: two_cell(r), 0, [{(0.5, 0.5): R}, {(0.25, 0.25): R}], 11),
    "reverse": (lambda r: two_cell(r), 0, [{(0.5, 0.5): R}, FAMILY_A], 2),
    "undone": (lambda r: two_cell(r), 1, [{**FAMILY_A, (1.25, 0.25): R}], 11),
    "corner_walled": (
        lambda r: MockQuadBackend(nx=2, ny=2, periodic=(False, False), rotate_cells=r),
        0,
        [{(0.25, 0.25): R}, {(0.375, 0.375): R}],
        19,
    ),
    "corner_periodic": (
        lambda r: MockQuadBackend(nx=2, ny=2, rotate_cells=r),
        0,
        [{(0.25, 0.25): R}, {(0.125, 0.125): R}],
        19,
    ),
    "coarsen_to_refine": (
        lambda r: two_cell(r, max_level=3),
        1,
        [{(1.25, 0.25): R}, {**FAMILY_A, (1.125, 0.125): R}],
        17,
    ),
}


# --- T3: uniform meshes ----------------------------------------------------

@pytest.mark.parametrize("level", [0, 1, 2])
@pytest.mark.parametrize("grid", GRIDS)
def test_reset_is_clean(grid, level):
    state = start(MockQuadBackend(**GRIDS[grid]), level)
    assert topology_violations(state) == []


def test_reset_is_clean_with_and_without_the_rotation():
    for rotate in (False, True):
        backend = MockQuadBackend(nx=3, ny=2, rotate_cells=rotate)
        for _ in range(4):
            assert topology_violations(start(backend, 1)) == []


# --- T3: the adapted meshes of section 1.2.8 -------------------------------

@pytest.mark.parametrize("name", SCENARIOS)
@pytest.mark.parametrize("rotate", [False, True], ids=["pinned", "rotating"])
def test_scenario_is_clean_after_every_adapt(name, rotate):
    make, level, rounds, n_after = SCENARIOS[name]
    backend = make(rotate)
    state = start(backend, level)
    assert topology_violations(state) == []
    for where in rounds:
        state, _ = backend.apply_marks(marks_at(state, where))
        assert topology_violations(state) == []
    assert state.n_active == n_after


def test_the_scenarios_reach_half_neighbors_walls_and_wraps():
    # What the clean results above were clean on.
    make, level, rounds, _ = SCENARIOS["main"]
    backend = make(False)
    state = start(backend, level)
    for where in rounds:
        state, _ = backend.apply_marks(marks_at(state, where))
    assert (state.neighbors[:, :, 1] >= 0).any()
    assert (state.neighbors < 0).all(axis=2).any()
    assert state.neighbor_wrap.any()
    assert (state.family[:, 0] >= 0).any() and (state.family[:, 0] < 0).any()


# --- T4: random rounds -----------------------------------------------------

@pytest.mark.parametrize("seed", range(8))
def test_random_rounds_are_clean(seed):
    rng = np.random.default_rng(300 + seed)
    periodic = (bool(rng.integers(2)), bool(rng.integers(2)))
    backend = MockQuadBackend(
        nx=int(rng.integers(2, 5)),
        ny=int(rng.integers(2, 4)),
        periodic=periodic,
        lo=(-0.5, 1.0),
        hi=(2.5, 3.0),
        max_level=3,
    )
    state = start(backend, 1)
    assert topology_violations(state) == []
    for _ in range(6):
        marks = rng.choice([C, C, H, R], size=state.n_active).astype(np.int8)
        state, _ = backend.apply_marks(marks)
        assert topology_violations(state) == []


# --- The checks see a mock state that has been tampered with ---------------

def test_a_tampered_mock_state_is_caught():
    backend = MockQuadBackend(nx=2, ny=2, rotate_cells=False)
    state = start(backend, 0)
    state, _ = backend.apply_marks(marks_at(state, {(0.25, 0.25): R}))
    assert topology_violations(state) == []

    family = state.family.copy()
    family[family[:, 0] >= 0] = family[family[:, 0] >= 0][:, [1, 0, 2, 3]]
    swapped = dataclasses.replace(state, family=family)
    assert [f.split(":")[0] for f in topology_violations(swapped)] == ["family.quadrants"]

    wrap = ~state.neighbor_wrap
    flipped = dataclasses.replace(state, neighbor_wrap=wrap)
    assert [f.split(":")[0] for f in topology_violations(flipped)] == ["faces.wrap"]
