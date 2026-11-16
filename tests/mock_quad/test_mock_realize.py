"""Tests for the mock backend's realization (backends/mock_quad/realize.py).

AGENT_2D_ROADMAP item 1.2, sitting 2, signals R1, R3-R7 and R11 at the
function level. realize takes a mesh and one round of marks and returns the
mesh p4est's rules would give: refine, then coarsen complete families, then
corner balance. correspondence maps the new positions to the old ones.

The main instance is the five-element mesh of the phase's explanation doc
(two base cells A and B, periodic in x, walled in y, A refined once), with
one more refine in A's lower-left child. That child's children reach B
through the wrap, two levels finer than it, so balance splits B.

    before                           after
    +-----+-----+-----------+        +-----+-----+-----+-----+
    |  2  |  3  |           |        |     |     |     |     |
    +-----+-----+     4     |        +--+--+-----+-----+-----+
    |  0  |  1  |           |        |--+--|     |     |     |
    +-----+-----+-----------+        +--+--+-----+-----+-----+

A leaf is (level, X, Y). The verbs built on these functions are tested in
test_mock_backend_adapt.py.

Run: pytest tests/mock_quad/test_mock_realize.py -v
"""
import numpy as np
import pytest

from backends.mock_quad.quadtree import QuadGrid, children, order_leaves, parent, topology
from backends.mock_quad.realize import balance, correspondence, realize
from contract.reports import KIND_CHILD, KIND_MERGED, KIND_UNCHANGED
from contract.solver_contract import COARSEN, HOLD, REFINE

C, H, R = COARSEN, HOLD, REFINE

A_CHILDREN = [(1, 0, 0), (1, 1, 0), (1, 0, 1), (1, 1, 1)]
B = (0, 1, 0)
B_CHILDREN = [(1, 2, 0), (1, 3, 0), (1, 2, 1), (1, 3, 1)]
FIVE = A_CHILDREN + [B]
LEVEL_ONE = A_CHILDREN + B_CHILDREN

# The 2x2 grid's base cells; SW refined is the corner-balance instance.
SW, SE, NW, NE = (0, 0, 0), (0, 1, 0), (0, 0, 1), (0, 1, 1)
SW_REFINED = A_CHILDREN + [SE, NW, NE]


def two_cell_grid(max_level=2):
    """Two base cells side by side: periodic in x, walled in y."""
    return QuadGrid(nx=2, ny=1, periodic=(True, False), lo=(0.0, 0.0), hi=(2.0, 1.0), max_level=max_level)


def square_grid(periodic, max_level=2):
    """A 2x2 grid on the unit square."""
    return QuadGrid(nx=2, ny=2, periodic=periodic, lo=(0.0, 0.0), hi=(1.0, 1.0), max_level=max_level)


# --- R1: the main instance -------------------------------------------------

def test_refine_reaches_b_through_the_wrap():
    new, realized = realize(two_cell_grid(), FIVE, np.array([R, H, H, H, H], dtype=np.int8))
    assert new == sorted(list(children((1, 0, 0))) + A_CHILDREN[1:] + B_CHILDREN)
    assert len(new) == 11
    # B was marked HOLD and comes back REFINE: the balance-added refine.
    assert realized.tolist() == [R, H, H, H, R]


def test_realized_is_int8_and_the_leaves_are_a_sorted_list():
    new, realized = realize(two_cell_grid(), FIVE, [R, H, H, H, H])
    assert realized.dtype == np.int8
    assert realized.shape == (5,)
    assert isinstance(new, list)
    assert new == sorted(new)


def test_realize_changes_neither_the_leaves_nor_the_marks():
    leaves = list(FIVE)
    marks = np.array([R, H, H, H, H], dtype=np.int8)
    realize(two_cell_grid(), leaves, marks)
    assert leaves == FIVE
    assert marks.tolist() == [R, H, H, H, H]


def test_all_hold_changes_nothing():
    new, realized = realize(two_cell_grid(), FIVE, [H] * 5)
    assert new == sorted(FIVE)
    assert realized.tolist() == [H] * 5


# --- Coarsen: complete families only (behavior 3) --------------------------

def test_a_complete_family_merges():
    new, realized = realize(two_cell_grid(), FIVE, [C, C, C, C, H])
    assert new == [(0, 0, 0), B]
    assert realized.tolist() == [C, C, C, C, H]


@pytest.mark.parametrize("left_out", range(4))
def test_a_partly_marked_family_is_dropped(left_out):
    marks = [C] * 4 + [H] * 4
    marks[left_out] = H
    new, realized = realize(two_cell_grid(), LEVEL_ONE, marks)
    assert new == sorted(LEVEL_ONE)
    assert realized.tolist() == [H] * 8


def test_a_family_with_a_refined_member_is_dropped():
    new, realized = realize(two_cell_grid(), LEVEL_ONE, [R, C, C, C, H, H, H, H])
    assert new == sorted(list(children((1, 0, 0))) + LEVEL_ONE[1:])
    assert realized.tolist() == [R, H, H, H, H, H, H, H]


def test_two_families_merge_in_one_adapt():
    new, realized = realize(two_cell_grid(), LEVEL_ONE, [C] * 8)
    assert new == [(0, 0, 0), B]
    assert realized.tolist() == [C] * 8


# --- Ineligible marks are ignored ------------------------------------------

def test_coarsen_at_level_zero_is_ignored():
    new, realized = realize(two_cell_grid(), FIVE, [H, H, H, H, C])
    assert new == sorted(FIVE)
    assert realized.tolist() == [H] * 5


def test_refine_at_max_level_is_ignored():
    new, realized = realize(two_cell_grid(max_level=1), FIVE, [R, R, R, R, H])
    assert new == sorted(FIVE)
    assert realized.tolist() == [H] * 5


# --- R4: a coarsen undone by balance (behavior 2's order) ------------------

def test_a_family_coarsen_is_undone_by_balance():
    # A's family is merged at the coarsen step; B's child next to it went to
    # level 2 at the refine step, so balance splits A again.
    new, realized = realize(two_cell_grid(), LEVEL_ONE, [C, C, C, C, R, H, H, H])
    assert new == sorted(A_CHILDREN + list(children((1, 2, 0))) + B_CHILDREN[1:])
    assert len(new) == 11
    assert realized.tolist() == [H, H, H, H, R, H, H, H]


# --- R5: corner balance ----------------------------------------------------

def test_a_corner_touch_forces_a_refine():
    # SW's child at the domain's center is refined. Its children touch SE and
    # NW across faces and NE at one corner only; all three are split.
    new, realized = realize(square_grid((False, False)), SW_REFINED, [H, H, H, R, H, H, H])
    assert len(new) == 19
    assert NE not in new
    assert all(child in new for child in children(NE))
    assert realized.tolist() == [H, H, H, R, R, R, R]


def test_a_corner_touch_through_both_wraps_forces_a_refine():
    # Periodic on both axes: the children of SW's lower-left child reach NE
    # diagonally, across the x wrap and the y wrap at once.
    new, realized = realize(square_grid((True, True)), SW_REFINED, [R, H, H, H, H, H, H])
    assert len(new) == 19
    assert all(child in new for child in children(NE))
    assert realized.tolist() == [R, H, H, H, R, R, R]


def test_a_corner_across_a_wall_asks_nothing():
    # Periodic in x only. The same refine reaches SE through the x wrap; the
    # diagonal step leaves the domain through the y wall, so NE is untouched,
    # and so is NW.
    new, realized = realize(square_grid((True, False)), SW_REFINED, [R, H, H, H, H, H, H])
    assert len(new) == 13
    assert NW in new and NE in new
    assert realized.tolist() == [R, H, H, H, R, H, H]


# --- R6: a COARSEN mark realized as REFINE ---------------------------------

def test_a_coarsen_mark_can_come_back_as_refine():
    # max_level 3. B's first child is already refined; its lower-left child
    # (2,4,0) is refined again, to level 3, beside A's child (1,1,0). A's
    # family, all marked COARSEN, is merged, split again by balance, and
    # (1,1,0) is then split once more.
    leaves = A_CHILDREN + list(children((1, 2, 0))) + B_CHILDREN[1:]
    marks = [C, C, C, C, R, H, H, H, H, H, H]
    new, realized = realize(two_cell_grid(max_level=3), leaves, marks)
    assert len(new) == 17
    assert realized.tolist() == [H, R, H, H, R, H, H, H, H, H, H]
    assert all(child in new for child in children((1, 1, 0)))


# --- balance ---------------------------------------------------------------

def test_balance_leaves_a_balanced_mesh_alone_and_returns_a_new_set():
    mesh = set(FIVE)
    balanced = balance(two_cell_grid(), mesh)
    assert balanced == mesh
    assert balanced is not mesh


def test_balance_splits_the_coarse_side_and_keeps_its_input():
    # The 4:1 mesh of sitting 1's unbalanced-mesh test: B at level 0 beside
    # level-2 leaves.
    mesh = {(2, 2, 0), (2, 3, 0), (2, 2, 1), (2, 3, 1), (1, 0, 0), (1, 0, 1), (1, 1, 1), B}
    kept = set(mesh)
    balanced = balance(two_cell_grid(), mesh)
    assert balanced == (mesh - {B}) | set(B_CHILDREN)
    assert mesh == kept


def test_balance_can_split_one_leaf_twice():
    # max_level 3, a level-3 leaf beside the level-0 B: B must reach level 2
    # where they touch, so it is split and one of its children split again.
    grid = two_cell_grid(max_level=3)
    fine = list(children((2, 3, 0)))
    mesh = set(fine) | {(2, 2, 0), (2, 2, 1), (2, 3, 1), (1, 0, 0), (1, 0, 1), (1, 1, 1), B}
    balanced = balance(grid, mesh)
    assert B not in balanced
    assert (1, 2, 0) not in balanced
    assert all(child in balanced for child in children((1, 2, 0)))


# --- R11: what realize rejects ---------------------------------------------

@pytest.mark.parametrize(
    "marks, message",
    [
        ([R, H], r"^marks: shape \(2,\), expected \(5,\)"),
        (np.zeros((5, 1), dtype=np.int8), r"^marks: shape \(5, 1\), expected \(5,\)"),
        (np.zeros(5), r"^marks: dtype float64, expected an integer dtype"),
        (np.zeros(5, dtype=bool), r"^marks: dtype bool, expected an integer dtype"),
        ([2, H, H, H, H], r"^marks: values must be COARSEN, HOLD or REFINE"),
        ([-2, H, H, H, H], r"^marks: values must be COARSEN, HOLD or REFINE"),
    ],
)
def test_bad_marks_rejected(marks, message):
    with pytest.raises(ValueError, match=message):
        realize(two_cell_grid(), FIVE, marks)


@pytest.mark.parametrize("dtype", [np.int8, np.int32, np.int64])
def test_any_integer_dtype_is_accepted(dtype):
    new, realized = realize(two_cell_grid(), FIVE, np.array([R, H, H, H, H], dtype=dtype))
    assert len(new) == 11
    assert realized.dtype == np.int8


def test_a_mesh_that_is_not_corner_balanced_can_break_the_one_level_bound():
    # The 4:1 mesh again, at max_level 3, with a refine beside B: B would
    # have to move two levels in one adapt.
    leaves = [(2, 2, 0), (2, 3, 0), (2, 2, 1), (2, 3, 1), (1, 0, 0), (1, 0, 1), (1, 1, 1), B]
    with pytest.raises(RuntimeError, match=r"^leaf \(0, 1, 0\): moved more than one level"):
        realize(two_cell_grid(max_level=3), leaves, [H, R, H, H, H, H, H, H])


# --- correspondence --------------------------------------------------------

def test_correspondence_of_a_refine():
    new_to_old, new_kind = correspondence([(0, 0, 0), B], FIVE)
    assert new_to_old.tolist() == [[0, -1, -1, -1]] * 4 + [[1, -1, -1, -1]]
    assert new_kind.tolist() == [KIND_CHILD] * 4 + [KIND_UNCHANGED]


def test_correspondence_of_a_merge_lists_the_children_in_child_order():
    # The old order is scrambled, so child order is not position order.
    old = [A_CHILDREN[3], B, A_CHILDREN[0], A_CHILDREN[2], A_CHILDREN[1]]
    new_to_old, new_kind = correspondence(old, [(0, 0, 0), B])
    assert new_to_old.tolist() == [[2, 4, 3, 0], [1, -1, -1, -1]]
    assert new_kind.tolist() == [KIND_MERGED, KIND_UNCHANGED]


def test_correspondence_dtypes_and_shapes():
    new_to_old, new_kind = correspondence(FIVE, FIVE)
    assert new_to_old.dtype == np.int32 and new_to_old.shape == (5, 4)
    assert new_kind.dtype == np.int8 and new_kind.shape == (5,)
    assert new_kind.tolist() == [KIND_UNCHANGED] * 5
    assert new_to_old[:, 0].tolist() == [0, 1, 2, 3, 4]


def test_correspondence_of_the_main_instance():
    grid = two_cell_grid()
    new, _ = realize(grid, FIVE, [R, H, H, H, H])
    new_to_old, new_kind = correspondence(FIVE, order_leaves(grid, new))
    assert new_to_old[:, 0].tolist() == [0, 0, 0, 0, 1, 2, 3, 4, 4, 4, 4]
    assert (new_to_old[:, 1:] == -1).all()
    assert new_kind.tolist() == [KIND_CHILD] * 4 + [KIND_UNCHANGED] * 3 + [KIND_CHILD] * 4


def test_an_undone_coarsen_is_unchanged_in_the_correspondence():
    grid = two_cell_grid()
    new, _ = realize(grid, LEVEL_ONE, [C, C, C, C, R, H, H, H])
    _, new_kind = correspondence(LEVEL_ONE, order_leaves(grid, new))
    assert new_kind.tolist() == [KIND_UNCHANGED] * 4 + [KIND_CHILD] * 4 + [KIND_UNCHANGED] * 3


def test_correspondence_rejects_a_mesh_two_levels_away():
    with pytest.raises(RuntimeError, match=r"^leaf \(2, 0, 0\): not within one level of the old mesh"):
        correspondence([(0, 0, 0), B], [(2, 0, 0)])


# --- R7: random marks on random grids --------------------------------------

@pytest.mark.parametrize("seed", range(8))
def test_random_adapts_keep_every_invariant(seed):
    rng = np.random.default_rng(seed)
    nx, ny = (int(n) for n in rng.integers(2, 4, size=2))
    periodic = (bool(rng.integers(2)), bool(rng.integers(2)))
    grid = QuadGrid(nx=nx, ny=ny, periodic=periodic, lo=(0.0, 0.0), hi=(float(nx), float(ny)), max_level=3)
    leaves = order_leaves(grid, [(0, x, y) for y in range(ny) for x in range(nx)])

    for round_index in range(6):
        marks = rng.choice([C, C, H, R], size=len(leaves)).astype(np.int8)
        new, realized = realize(grid, leaves, marks)
        new = order_leaves(grid, new, start_cell=round_index % (nx * ny))

        # The mesh is a 2:1 mesh of the whole domain, within max_level.
        arrays = topology(grid, new)
        assert arrays["level"].max() <= grid.max_level
        assert np.isclose(arrays["h"].prod(axis=1).sum(), nx * ny)

        # It is already corner balanced: a second adapt of all HOLD is a no-op.
        again, held = realize(grid, new, np.zeros(len(new), dtype=np.int8))
        assert again == sorted(new)
        assert not held.any()

        # The count follows from realized: +3 per refine, -3 per merged family.
        n_refined, n_coarsened = int((realized == R).sum()), int((realized == C).sum())
        assert n_coarsened % 4 == 0
        assert len(new) == len(leaves) + 3 * n_refined - 3 * (n_coarsened // 4)

        # A REFINE mark below max_level is always realized; a realized
        # COARSEN was always asked for.
        levels = np.array([leaf[0] for leaf in leaves])
        assert (realized[(marks == R) & (levels < grid.max_level)] == R).all()
        assert (marks[realized == C] == C).all()

        # Every old position is named once, or four times if it was refined.
        new_to_old, new_kind = correspondence(leaves, new)
        named = np.bincount(new_to_old[new_to_old >= 0], minlength=len(leaves))
        assert (named == np.where(realized == R, 4, 1)).all()
        assert int((new_kind == KIND_MERGED).sum()) == n_coarsened // 4
        assert int((new_kind == KIND_CHILD).sum()) == 4 * n_refined

        # A merged row's children are the new leaf's four children, in order.
        for q in np.flatnonzero(new_kind == KIND_MERGED):
            assert [leaves[p] for p in new_to_old[q]] == list(children(new[q]))
        for q in np.flatnonzero(new_kind == KIND_CHILD):
            assert leaves[new_to_old[q, 0]] == parent(new[q])

        leaves = new
