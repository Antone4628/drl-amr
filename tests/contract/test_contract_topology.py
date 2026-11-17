"""Tests for the backend-independent topology checks (contract/checks.py).

AGENT_2D_ROADMAP item 1.3, signals T1, T2 and T5-T7. No backend is involved:
every state here is built from literal arrays, the main one from the table
of the explanation doc's section 1.1.2 (two base cells on [0,2]x[0,1],
periodic in x, walled in y, the first refined once). A corrupted state is a
copy with one array changed, rebuilt through SolverState so it is checked
and frozen like any other.

The checks on states a backend makes are in
tests/mock_quad/test_mock_topology_checks.py.

Run: pytest tests/contract/test_contract_topology.py -v
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from contract.checks import (
    face_violations,
    family_violations,
    neighbor_violations,
    topology_violations,
    uniform_base_violations,
)
from contract.state import SolverState

CHECKS = {
    "family": family_violations,
    "neighbors": neighbor_violations,
    "faces": face_violations,
    "uniform_base": uniform_base_violations,
}
ARRAY_FIELDS = ("level", "neighbors", "neighbor_wrap", "family", "centroid", "h")


def make_state(*, level, neighbors, neighbor_wrap, family, centroid, h, lo, hi, periodic):
    """A SolverState from the arrays the checks read; the rest is filler."""
    level = np.array(level, dtype=np.int32)
    family = np.array(family, dtype=np.int32)
    n, ndim = len(level), len(lo)
    return SolverState(
        level=level,
        neighbors=np.array(neighbors, dtype=np.int32),
        neighbor_wrap=np.array(neighbor_wrap, dtype=bool),
        family=family,
        can_refine=level < 2,
        can_coarsen=family[:, 0] >= 0,
        centroid=np.array(centroid, dtype=np.float64),
        h=np.array(h, dtype=np.float64),
        attributes=np.zeros((n, 0)),
        n_active=n,
        ndim=ndim,
        nvar=1,
        max_level=2,
        mesh_epoch=1,
        time=0.0,
        interval_T=0.1,
        dt_base=0.01,
        amr_freq=10,
        dt_stable=0.005,
        domain_lo=np.array(lo, dtype=np.float64),
        domain_hi=np.array(hi, dtype=np.float64),
        periodic=np.array(periodic, dtype=bool),
    )


def five_elements():
    """Section 1.1.2's table: positions 0-3 are A's children, 4 is B."""
    return make_state(
        level=[1, 1, 1, 1, 0],
        neighbors=[
            [[4, -1], [1, -1], [-1, -1], [2, -1]],
            [[0, -1], [4, -1], [-1, -1], [3, -1]],
            [[4, -1], [3, -1], [0, -1], [-1, -1]],
            [[2, -1], [4, -1], [1, -1], [-1, -1]],
            [[1, 3], [0, 2], [-1, -1], [-1, -1]],
        ],
        neighbor_wrap=[
            [True, False, False, False],
            [False, False, False, False],
            [True, False, False, False],
            [False, False, False, False],
            [False, True, False, False],
        ],
        family=[[0, 1, 2, 3]] * 4 + [[-1, -1, -1, -1]],
        centroid=[[0.25, 0.25], [0.75, 0.25], [0.25, 0.75], [0.75, 0.75], [1.5, 0.5]],
        h=[[0.5, 0.5]] * 4 + [[1.0, 1.0]],
        lo=[0.0, 0.0],
        hi=[2.0, 1.0],
        periodic=[True, False],
    )


def three_elements_1d():
    """A periodic line [0,2]: one coarse element, then its neighbor's two children."""
    return make_state(
        level=[0, 1, 1],
        neighbors=[[[2], [1]], [[0], [2]], [[1], [0]]],
        neighbor_wrap=[[True, False], [False, False], [False, True]],
        family=[[-1, -1], [1, 2], [1, 2]],
        centroid=[[0.5], [1.25], [1.75]],
        h=[[1.0], [0.5], [0.5]],
        lo=[0.0],
        hi=[2.0],
        periodic=[True],
    )


def cube_3d():
    """One walled base cell refined once: eight elements, child p at bits of p."""
    neighbors = np.full((8, 6, 4), -1, dtype=np.int32)
    centroid = np.zeros((8, 3))
    for p in range(8):
        for axis in range(3):
            bit = (p >> axis) & 1
            centroid[p, axis] = 0.25 + 0.5 * bit
            # The sibling across this axis lies on the side facing the center.
            neighbors[p, 2 * axis + (1 - bit), 0] = p ^ (1 << axis)
    return make_state(
        level=[1] * 8,
        neighbors=neighbors,
        neighbor_wrap=np.zeros((8, 6), dtype=bool),
        family=[list(range(8))] * 8,
        centroid=centroid,
        h=np.full((8, 3), 0.5),
        lo=[0.0, 0.0, 0.0],
        hi=[1.0, 1.0, 1.0],
        periodic=[False, False, False],
    )


def unequal_cells():
    """Two base cells of widths 1 and 2 on [0,3]x[0,1], periodic in x."""
    return make_state(
        level=[0, 0],
        neighbors=[[[1, -1], [1, -1], [-1, -1], [-1, -1]], [[0, -1], [0, -1], [-1, -1], [-1, -1]]],
        neighbor_wrap=[[True, False, False, False], [False, True, False, False]],
        family=[[-1] * 4] * 2,
        centroid=[[0.5, 0.5], [2.0, 0.5]],
        h=[[1.0, 1.0], [2.0, 1.0]],
        lo=[0.0, 0.0],
        hi=[3.0, 1.0],
        periodic=[True, False],
    )


def changed(state, *edits):
    """A copy of state with (field, index, value) edits applied in order."""
    arrays = {}
    for field, index, value in edits:
        if field not in arrays:
            arrays[field] = getattr(state, field).copy()
        arrays[field][index] = value
    if "family" in arrays:
        arrays["can_coarsen"] = arrays["family"][:, 0] >= 0
    return dataclasses.replace(state, **arrays)


def rules(findings):
    """The rule names of a list of findings, in order."""
    return [finding.split(":")[0] for finding in findings]


F0, F1, F2, F3 = 0, 1, 2, 3
ROWS = slice(0, 4)

# Each corruption of the five-element state, and every rule it trips, in the
# order topology_violations reports them. The first seven are the table of
# the explanation doc's section 1.3.2.
CORRUPTIONS = {
    "family_rows_swapped_together": ([("family", ROWS, [0, 2, 1, 3])], ["family.quadrants"]),
    "one_family_row_changed": ([("family", 0, [0, 2, 1, 3])], ["family.rows_agree", "family.quadrants"]),
    "half_neighbors_out_of_order": ([("neighbors", (4, F0), [3, 1])], ["faces.tangential"]),
    "interior_face_emptied": ([("neighbors", (1, F1), [-1, -1])], ["neighbors.symmetry", "faces.wall"]),
    "wrap_flag_cleared": ([("neighbor_wrap", (0, F0), False)], ["faces.wrap"]),
    "coarse_level_raised": (
        [("level", 4, 1)],
        ["neighbors.two_to_one", "faces.tangential", "uniform_base.h"],
    ),
    "families_not_reported": ([("family", ROWS, -1)], ["uniform_base.family_complete"]),
    "family_row_partly_filled": (
        [("family", (0, 3), -1)],
        ["family.partial_row", "family.rows_agree", "uniform_base.family_complete"],
    ),
    "family_at_two_levels": (
        [("level", 0, 2)],
        ["family.level", "neighbors.two_to_one", "uniform_base.h"],
    ),
    "family_row_without_its_owner": ([("family", 0, [1, 1, 2, 3])], ["family.rows_agree", "family.quadrants"]),
    "slot_one_alone": (
        [("neighbors", (4, F0), [-1, 3])],
        ["neighbors.slots", "neighbors.two_to_one", "neighbors.symmetry"],
    ),
    "neighbor_on_a_wall_face": (
        [("neighbors", (0, F2), [1, -1])],
        ["neighbors.symmetry", "faces.wall", "faces.normal", "faces.tangential"],
    ),
    "wrap_faces_emptied_on_both_sides": (
        [
            ("neighbors", (0, F0), -1),
            ("neighbors", (2, F0), -1),
            ("neighbors", (4, F1), -1),
            ("neighbor_wrap", (0, F0), False),
            ("neighbor_wrap", (2, F0), False),
            ("neighbor_wrap", (4, F1), False),
        ],
        ["faces.wall"],
    ),
    "wrap_flag_set_on_an_interior_face": ([("neighbor_wrap", (1, F0), True)], ["faces.wrap"]),
    "wrap_flag_set_on_a_wall_face": ([("neighbor_wrap", (0, F2), True)], ["faces.wrap"]),
    "wrong_neighbor_same_level": (
        [("neighbors", (0, F1), [3, -1])],
        ["neighbors.symmetry", "faces.tangential"],
    ),
    "centroid_moved": (
        [("centroid", (1, 0), 0.8)],
        ["family.quadrants", "faces.normal", "faces.tangential", "uniform_base.centroid"],
    ),
    "coarse_h_doubled": (
        [("h", 4, [2.0, 2.0])],
        [
            "faces.wall",
            "faces.normal",
            "faces.tangential",
            "uniform_base.h",
            "uniform_base.centroid",
            "uniform_base.area",
        ],
    ),
}


# --- T1: the instance is clean ---------------------------------------------

def test_five_element_instance_is_clean():
    assert topology_violations(five_elements()) == []


@pytest.mark.parametrize("name", CHECKS)
def test_each_check_alone_is_clean_on_the_instance(name):
    assert CHECKS[name](five_elements()) == []


# --- T2: each corruption trips its rule ------------------------------------

@pytest.mark.parametrize("name", CORRUPTIONS)
def test_corruption_trips_its_rules(name):
    edits, expected = CORRUPTIONS[name]
    assert rules(topology_violations(changed(five_elements(), *edits))) == expected


def test_a_finding_names_the_count_and_the_first_offender():
    state = changed(five_elements(), ("level", 4, 1))
    assert topology_violations(state) == [
        "neighbors.two_to_one: 2, first at position 4, face 0",
        "faces.tangential: 2, first at position 4, face 0",
        "uniform_base.h: 1, first at position 4",
    ]


def test_the_area_finding_carries_both_areas():
    state = changed(five_elements(), ("h", 4, [2.0, 2.0]))
    (finding,) = [f for f in topology_violations(state) if f.startswith("uniform_base.area")]
    assert finding == "uniform_base.area: the elements cover 5.0, the domain is 2.0"


def test_unreported_families_pass_everything_but_the_completeness_rule():
    # The blind side of a soundness check: nothing reported, nothing wrong.
    state = changed(five_elements(), ("family", ROWS, -1))
    assert family_violations(state) == []
    assert neighbor_violations(state) == []
    assert face_violations(state) == []
    assert uniform_base_violations(state) == ["uniform_base.family_complete: 4, first at position 0"]
    assert topology_violations(state, uniform_base=False) == []


# --- The index range comes first -------------------------------------------

BAD_INDICES = {
    "neighbor_too_large": ("neighbors", (0, F1, 0), 5),
    "neighbor_below_absent": ("neighbors", (4, F2, 1), -2),
    "family_too_large": ("family", (2, 1), 5),
    "family_below_absent": ("family", (4, 0), -3),
}


@pytest.mark.parametrize("case", BAD_INDICES)
def test_an_index_that_names_no_element_is_reported_alone(case):
    state = changed(five_elements(), BAD_INDICES[case])
    findings = topology_violations(state)
    assert rules(findings) == ["indices.range"]
    assert findings[0].startswith("indices.range: 1, first at position ")


@pytest.mark.parametrize("name", CHECKS)
def test_each_check_stops_at_a_bad_index(name):
    state = changed(five_elements(), ("neighbors", (0, F1, 0), 5))
    assert CHECKS[name](state) == ["indices.range: 1, first at position 0"]


# --- T5: one dimension and three -------------------------------------------

def test_1d_state_is_clean():
    state = three_elements_1d()
    assert state.neighbors.shape == (3, 2, 1) and state.family.shape == (3, 2)
    assert topology_violations(state) == []


def test_1d_family_in_the_wrong_order_is_caught():
    state = changed(three_elements_1d(), ("family", slice(1, 3), [2, 1]))
    assert rules(topology_violations(state)) == ["family.quadrants"]


def test_1d_neighbor_two_levels_finer_is_caught():
    state = changed(three_elements_1d(), ("level", slice(1, 3), 2))
    assert rules(topology_violations(state)) == ["neighbors.two_to_one", "uniform_base.h"]


def test_1d_wrap_flag_is_checked():
    state = changed(three_elements_1d(), ("neighbor_wrap", (0, 0), False))
    assert rules(topology_violations(state)) == ["faces.wrap"]


def test_3d_cube_is_clean():
    state = cube_3d()
    assert state.neighbors.shape == (8, 6, 4) and state.family.shape == (8, 8)
    assert topology_violations(state) == []


def test_3d_family_in_the_wrong_order_is_caught():
    state = changed(cube_3d(), ("family", slice(None), [0, 1, 2, 3, 5, 4, 6, 7]))
    assert rules(topology_violations(state)) == ["family.quadrants"]


def test_3d_missing_neighbor_is_caught():
    state = changed(cube_3d(), ("neighbors", (0, 1, 0), -1))
    assert rules(topology_violations(state)) == ["neighbors.symmetry", "faces.wall"]


# --- T6: a base grid that is not uniform -----------------------------------

def test_unequal_base_cells_fail_only_the_uniform_base_rules():
    state = unequal_cells()
    assert family_violations(state) == []
    assert neighbor_violations(state) == []
    assert face_violations(state) == []
    assert "uniform_base.h" in rules(topology_violations(state))
    assert topology_violations(state, uniform_base=False) == []


# --- The tolerance ---------------------------------------------------------

def test_a_perturbation_below_the_tolerance_is_clean():
    state = changed(five_elements(), ("centroid", (1, 0), 0.75 + 1e-12))
    assert topology_violations(state) == []


def test_a_perturbation_above_the_tolerance_is_caught_unless_rtol_allows_it():
    state = changed(five_elements(), ("centroid", (1, 0), 0.75 + 1e-6))
    assert "faces.normal" in rules(topology_violations(state))
    assert topology_violations(state, rtol=1e-5) == []


# --- T7: the checks read and do not write ----------------------------------

@pytest.mark.parametrize("build", [five_elements, three_elements_1d, cube_3d], ids=["2d", "1d", "3d"])
def test_checks_leave_the_state_unchanged_and_repeat(build):
    state = build()
    kept = {field: getattr(state, field).copy() for field in ARRAY_FIELDS}
    first, second = topology_violations(state), topology_violations(state)
    assert first == second == []
    assert first is not second
    for field in ARRAY_FIELDS:
        assert np.array_equal(getattr(state, field), kept[field])
        assert not getattr(state, field).flags.writeable


def test_findings_repeat_on_a_broken_state():
    state = changed(five_elements(), ("neighbors", (1, F1), [-1, -1]))
    assert topology_violations(state) == topology_violations(state) != []
