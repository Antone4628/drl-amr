"""Tests for the mock backend's quadtree (backends/mock_quad/quadtree.py).

AGENT_2D_ROADMAP item 1.2, sitting 1, signals Q1-Q4 at the function level.
The main instance is the five-element mesh of the phase's explanation doc:
two base cells A and B side by side, periodic in x and walled in y, with A
refined once. A leaf is (level, X, Y).

    y=1  +-----+-----+-----------+
         |  2  |  3  |           |
         +-----+-----+     4     |
         |  0  |  1  |           |
    y=0  +-----+-----+-----------+
        x=0         x=1         x=2

These tests cover the leaf arithmetic, the element order and the topology
arrays. Refinement, coarsening and balance are sitting 2's.

Run: pytest tests/mock_quad/test_mock_quadtree.py -v
"""
import numpy as np
import pytest

from backends.mock_quad.quadtree import (
    QuadGrid,
    children,
    face_neighbors,
    order_leaves,
    parent,
    topology,
    uniform_leaves,
)
from contract.state import SolverState

# Faces: f = 2*axis + side.
X_MIN, X_MAX, Y_MIN, Y_MAX = 0, 1, 2, 3

A_CHILDREN = [(1, 0, 0), (1, 1, 0), (1, 0, 1), (1, 1, 1)]
B = (0, 1, 0)
FIVE = A_CHILDREN + [B]


def grid_fields(**overrides):
    """Keyword arguments for the two-cell grid: periodic in x, walled in y."""
    fields = dict(nx=2, ny=1, periodic=(True, False), lo=(0.0, 0.0), hi=(2.0, 1.0), max_level=2)
    fields.update(overrides)
    return fields


def two_cell_grid():
    return QuadGrid(**grid_fields())


def stacked_grid():
    """The same instance turned on its side: periodic in y, walled in x."""
    return QuadGrid(nx=1, ny=2, periodic=(False, True), lo=(0.0, 0.0), hi=(1.0, 2.0), max_level=2)


def index_of(leaves):
    return {leaf: p for p, leaf in enumerate(leaves)}


# --- QuadGrid --------------------------------------------------------------

def test_cells_double_with_each_level():
    grid = QuadGrid(nx=3, ny=2, periodic=(False, False), lo=(0.0, 0.0), hi=(3.0, 1.0), max_level=2)
    assert [grid.cells(0, level) for level in range(3)] == [3, 6, 12]
    assert [grid.cells(1, level) for level in range(3)] == [2, 4, 8]


@pytest.mark.parametrize(
    "overrides, message",
    [
        (dict(nx=0), r"^nx: 0, expected at least 1"),
        (dict(ny=0), r"^ny: 0, expected at least 1"),
        (dict(nx=1), r"^nx: 1, a periodic axis needs at least 2"),
        (dict(ny=1, periodic=(True, True)), r"^ny: 1, a periodic axis needs at least 2"),
        (dict(hi=(0.0, 1.0)), r"^lo, hi: axis 0"),
        (dict(lo=(0.0, 1.0)), r"^lo, hi: axis 1"),
        (dict(hi=(float("nan"), 1.0)), r"^lo, hi: axis 0"),
        (dict(max_level=-1), r"^max_level: -1"),
    ],
)
def test_bad_grid_rejected(overrides, message):
    with pytest.raises(ValueError, match=message):
        QuadGrid(**grid_fields(**overrides))


def test_one_cell_is_allowed_on_a_walled_axis():
    assert two_cell_grid().ny == 1


# --- parent and children ---------------------------------------------------

def test_children_are_in_child_order():
    # c = ix + 2*iy: lower-left, lower-right, upper-left, upper-right.
    assert children((0, 0, 0)) == ((1, 0, 0), (1, 1, 0), (1, 0, 1), (1, 1, 1))
    assert children((1, 3, 2)) == ((2, 6, 4), (2, 7, 4), (2, 6, 5), (2, 7, 5))


@pytest.mark.parametrize("leaf", [(0, 1, 0), (1, 3, 1), (2, 5, 2)])
def test_every_child_has_its_parent(leaf):
    assert all(parent(child) == leaf for child in children(leaf))


# --- uniform meshes and the order ------------------------------------------

@pytest.mark.parametrize("level, count", [(0, 2), (1, 8), (2, 32)])
def test_uniform_leaves_count(level, count):
    leaves = uniform_leaves(two_cell_grid(), level)
    assert len(leaves) == count
    assert len(set(leaves)) == count
    assert all(leaf[0] == level for leaf in leaves)


def test_order_of_the_five_element_instance():
    shuffled = [FIVE[i] for i in (4, 2, 0, 3, 1)]
    assert order_leaves(two_cell_grid(), shuffled) == FIVE


def test_start_cell_rotates_the_base_cells():
    assert order_leaves(two_cell_grid(), FIVE, start_cell=1) == [B] + A_CHILDREN


def test_morton_order_across_levels():
    # A's first child refined again: its four children come before A's second.
    grandchildren = list(children((1, 0, 0)))
    leaves = grandchildren + A_CHILDREN[1:] + [B]
    assert order_leaves(two_cell_grid(), leaves[::-1]) == leaves


def test_uniform_level_one_is_morton_within_each_cell():
    grid = two_cell_grid()
    ordered = order_leaves(grid, uniform_leaves(grid, 1))
    assert ordered == list(children((0, 0, 0))) + list(children((0, 1, 0)))


# --- face_neighbors: the four cases ----------------------------------------

def test_same_level_neighbor():
    assert face_neighbors(two_cell_grid(), index_of(FIVE), (1, 0, 0), X_MAX) == (1, -1, False)
    assert face_neighbors(two_cell_grid(), index_of(FIVE), (1, 0, 0), Y_MAX) == (2, -1, False)


def test_coarser_neighbor_sits_in_slot_zero():
    assert face_neighbors(two_cell_grid(), index_of(FIVE), (1, 1, 0), X_MAX) == (4, -1, False)


def test_finer_neighbors_ascend_along_the_face():
    assert face_neighbors(two_cell_grid(), index_of(FIVE), B, X_MIN) == (1, 3, False)


def test_wrap_to_finer_neighbors():
    assert face_neighbors(two_cell_grid(), index_of(FIVE), B, X_MAX) == (0, 2, True)


def test_wrap_to_a_coarser_neighbor():
    assert face_neighbors(two_cell_grid(), index_of(FIVE), (1, 0, 0), X_MIN) == (4, -1, True)


@pytest.mark.parametrize("leaf, face", [((1, 0, 0), Y_MIN), ((1, 1, 1), Y_MAX), (B, Y_MIN), (B, Y_MAX)])
def test_wall_is_all_absent(leaf, face):
    assert face_neighbors(two_cell_grid(), index_of(FIVE), leaf, face) == (-1, -1, False)


def test_unbalanced_mesh_is_named():
    # A refined twice in one corner while B stays at level 0: a 4:1 face.
    grid = two_cell_grid()
    leaves = [(2, 2, 0), (2, 3, 0), (2, 2, 1), (2, 3, 1), (1, 0, 0), (1, 0, 1), (1, 1, 1), B]
    with pytest.raises(ValueError, match=r"^leaf \(0, 1, 0\), face 0: the mesh is not 2:1 balanced"):
        face_neighbors(grid, index_of(leaves), B, X_MIN)


# --- Q1: topology on the five-element instance -----------------------------

def test_five_element_topology_matches_the_explanation_table():
    arrays = topology(two_cell_grid(), FIVE)

    assert arrays["level"].tolist() == [1, 1, 1, 1, 0]
    assert arrays["neighbors"].tolist() == [
        [[4, -1], [1, -1], [-1, -1], [2, -1]],
        [[0, -1], [4, -1], [-1, -1], [3, -1]],
        [[4, -1], [3, -1], [0, -1], [-1, -1]],
        [[2, -1], [4, -1], [1, -1], [-1, -1]],
        [[1, 3], [0, 2], [-1, -1], [-1, -1]],
    ]
    wrap = np.zeros((5, 4), dtype=bool)
    wrap[0, X_MIN] = wrap[2, X_MIN] = wrap[4, X_MAX] = True
    assert np.array_equal(arrays["neighbor_wrap"], wrap)
    assert arrays["family"].tolist() == [[0, 1, 2, 3]] * 4 + [[-1, -1, -1, -1]]
    assert arrays["can_refine"].tolist() == [True] * 5
    assert arrays["can_coarsen"].tolist() == [True, True, True, True, False]
    assert arrays["centroid"].tolist() == [[0.25, 0.25], [0.75, 0.25], [0.25, 0.75], [0.75, 0.75], [1.5, 0.5]]
    assert arrays["h"].tolist() == [[0.5, 0.5]] * 4 + [[1.0, 1.0]]


def test_topology_keys_and_dtypes():
    arrays = topology(two_cell_grid(), FIVE)
    assert {name: arr.dtype for name, arr in arrays.items()} == {
        "level": np.int32,
        "neighbors": np.int32,
        "neighbor_wrap": np.bool_,
        "family": np.int32,
        "can_refine": np.bool_,
        "can_coarsen": np.bool_,
        "centroid": np.float64,
        "h": np.float64,
    }


def test_topology_arrays_build_a_solver_state():
    # The real noun checks every dtype and shape at construction.
    arrays = topology(two_cell_grid(), FIVE)
    state = SolverState(
        **arrays,
        attributes=np.zeros((5, 0)),
        n_active=5,
        ndim=2,
        nvar=1,
        max_level=2,
        mesh_epoch=1,
        time=0.0,
        interval_T=0.1,
        dt_base=0.01,
        amr_freq=10,
        dt_stable=0.005,
        domain_lo=np.array([0.0, 0.0]),
        domain_hi=np.array([2.0, 1.0]),
        periodic=np.array([True, False]),
    )
    assert state.neighbors.shape == (5, 4, 2)


def test_can_refine_is_false_at_max_level():
    grid = QuadGrid(**grid_fields(max_level=1))
    assert topology(grid, FIVE)["can_refine"].tolist() == [False, False, False, False, True]


def test_family_is_absent_when_a_sibling_is_not_a_leaf():
    # Both cells at level 1, then A's first child refined again: A's other
    # three children lose their family; B's children keep theirs.
    grid = two_cell_grid()
    leaves = order_leaves(grid, list(children((1, 0, 0))) + A_CHILDREN[1:] + list(children(B)))
    arrays = topology(grid, leaves)
    assert arrays["family"].tolist() == [[0, 1, 2, 3]] * 4 + [[-1, -1, -1, -1]] * 3 + [[7, 8, 9, 10]] * 4
    assert arrays["can_coarsen"].tolist() == [True] * 4 + [False] * 3 + [True] * 4


# --- Q2: the instance on its side ------------------------------------------

def test_stacked_instance_half_neighbors_ascend_in_x():
    # Two cells stacked, the lower one refined, periodic in y. The upper cell
    # is position 4; its y faces each hold two finer neighbors.
    leaves = A_CHILDREN + [(0, 0, 1)]
    arrays = topology(stacked_grid(), leaves)
    assert arrays["neighbors"][4].tolist() == [[-1, -1], [-1, -1], [2, 3], [0, 1]]
    assert arrays["neighbor_wrap"][4].tolist() == [False, False, False, True]
    assert arrays["neighbor_wrap"].sum() == 3
    assert arrays["centroid"][4].tolist() == [0.5, 1.5]


# --- Q3 and Q4 at the function level ---------------------------------------

def test_uniform_periodic_mesh():
    grid = QuadGrid(nx=4, ny=4, periodic=(True, True), lo=(0.0, 0.0), hi=(1.0, 1.0), max_level=2)
    arrays = topology(grid, order_leaves(grid, uniform_leaves(grid, 1)))
    assert arrays["level"].shape == (64,)
    assert (arrays["neighbors"][:, :, 0] >= 0).all()
    assert (arrays["neighbors"][:, :, 1] == -1).all()
    assert (arrays["family"] >= 0).all()
    assert arrays["can_coarsen"].all() and arrays["can_refine"].all()
    assert arrays["neighbor_wrap"].sum() == 32


def test_walled_mesh_with_non_square_cells():
    grid = QuadGrid(nx=3, ny=2, periodic=(False, False), lo=(0.0, 0.0), hi=(3.0, 1.0), max_level=2)
    arrays = topology(grid, order_leaves(grid, uniform_leaves(grid, 0)))
    walls = (arrays["neighbors"] == -1).all(axis=2)
    assert walls.sum() == 10
    assert not arrays["neighbor_wrap"].any()
    assert arrays["h"].tolist() == [[1.0, 0.5]] * 6
    assert not arrays["can_coarsen"].any()


def test_every_family_row_lists_its_own_position():
    grid = QuadGrid(nx=4, ny=4, periodic=(True, True), lo=(0.0, 0.0), hi=(1.0, 1.0), max_level=2)
    family = topology(grid, order_leaves(grid, uniform_leaves(grid, 2)))["family"]
    own = np.arange(len(family))
    assert ((family == own[:, None]).sum(axis=1) == 1).all()
