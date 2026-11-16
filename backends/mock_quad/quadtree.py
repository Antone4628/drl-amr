"""Quadtree mesh for the mock 2D backend: leaves, their order, and the
topology arrays of the v2 SolverState.

A leaf is (level, X, Y): its refinement level, and its column and row on the
uniform grid of that level, counted across the whole domain. At level l that
grid is (nx * 2**l) by (ny * 2**l) cells. A mesh is a list of leaves, and
everything here is a pure function of one: nothing is stored, and there is
no PDE.

Conventions are the contract's (SOLVER_CONTRACT_V2 section 5.2): faces
f = 2*axis + side, children c = ix + 2*iy, -1 for absent.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Leaf = tuple[int, int, int]


@dataclass(frozen=True, slots=True, kw_only=True)
class QuadGrid:
    """The base grid a quadtree mesh refines.

    Fields:
        nx: Base cells along x.
        ny: Base cells along y.
        periodic: Whether each axis wraps, (x, y). An axis that does not
            wrap ends in walls.
        lo: The domain's lower corner, (x, y).
        hi: The domain's upper corner, (x, y).
        max_level: The finest level a leaf may reach; 0 is the base grid.
    """

    nx: int
    ny: int
    periodic: tuple[bool, bool]
    lo: tuple[float, float]
    hi: tuple[float, float]
    max_level: int

    def __post_init__(self) -> None:
        """Check the cell counts, the domain and the level."""
        for axis, name in enumerate(("nx", "ny")):
            n = self.base_cells(axis)
            if n < 1:
                raise ValueError(f"{name}: {n}, expected at least 1")
            if self.periodic[axis] and n < 2:
                raise ValueError(f"{name}: {n}, a periodic axis needs at least 2 base cells")
            if not self.lo[axis] < self.hi[axis]:
                raise ValueError(f"lo, hi: axis {axis} has lo {self.lo[axis]} not below hi {self.hi[axis]}")
        if self.max_level < 0:
            raise ValueError(f"max_level: {self.max_level}, expected at least 0")

    def base_cells(self, axis: int) -> int:
        """Base cells along an axis (0 is x, 1 is y)."""
        return (self.nx, self.ny)[axis]

    def cells(self, axis: int, level: int) -> int:
        """Cells along an axis on the uniform grid of a level."""
        return self.base_cells(axis) << level


def parent(leaf: Leaf) -> Leaf:
    """The cell one level up that contains a leaf. Not defined at level 0."""
    level, x, y = leaf
    return (level - 1, x >> 1, y >> 1)


def children(leaf: Leaf) -> tuple[Leaf, ...]:
    """The four cells one level down, in child order c = ix + 2*iy."""
    level, x, y = leaf
    return tuple((level + 1, 2 * x + ix, 2 * y + iy) for iy in (0, 1) for ix in (0, 1))


def uniform_leaves(grid: QuadGrid, level: int) -> list[Leaf]:
    """Every cell of the uniform grid of a level, in no particular order."""
    return [(level, x, y) for y in range(grid.cells(1, level)) for x in range(grid.cells(0, level))]


def _morton(i: int, j: int, bits: int) -> int:
    """Interleave the low bits of i and j, j the more significant of each pair."""
    key = 0
    for b in range(bits):
        key |= ((i >> b) & 1) << (2 * b)
        key |= ((j >> b) & 1) << (2 * b + 1)
    return key


def order_leaves(grid: QuadGrid, leaves: list[Leaf], start_cell: int = 0) -> list[Leaf]:
    """Sort leaves into the mock's element order.

    Base cells come in sequence starting at start_cell and wrapping around;
    inside a base cell the leaves are in Morton order. A position in the
    returned list is an element's position in every state array.

    Args:
        grid: The base grid.
        leaves: The leaves of one mesh, in any order.
        start_cell: The base cell (bx + nx*by) that comes first.
    """
    n_cells = grid.nx * grid.ny
    depth = grid.max_level

    def key(leaf: Leaf) -> tuple[int, int]:
        level, x, y = leaf
        bx, by = x >> level, y >> level
        cell = bx + grid.nx * by
        shift = depth - level
        i = (x - (bx << level)) << shift
        j = (y - (by << level)) << shift
        return ((cell - start_cell) % n_cells, _morton(i, j, depth))

    return sorted(leaves, key=key)


def face_neighbors(grid: QuadGrid, index: dict[Leaf, int], leaf: Leaf, face: int) -> tuple[int, int, bool]:
    """Positions of a leaf's neighbors across one face.

    Step one cell at the leaf's own level, then ask in order: is that cell a
    leaf, is its parent a leaf, are its two children on the shared face
    leaves.

    Args:
        grid: The base grid.
        index: Position of every leaf of the mesh.
        leaf: The leaf whose neighbors are wanted.
        face: f = 2*axis + side, side 0 the minimum.

    Returns:
        (slot0, slot1, wrap). A neighbor at the same level or coarser is in
        slot0 with slot1 = -1; two finer neighbors fill both, ascending along
        the face. A wall gives (-1, -1, False). wrap is True when the step
        crossed a periodic boundary.

    Raises:
        ValueError: If none of the three is a leaf, so the mesh is not 2:1
            balanced across this face.
    """
    level, x, y = leaf
    axis, side = divmod(face, 2)
    pos = [x, y]
    pos[axis] += 2 * side - 1
    n = grid.cells(axis, level)
    wrap = False
    if not 0 <= pos[axis] < n:
        if not grid.periodic[axis]:
            return (-1, -1, False)
        pos[axis] %= n
        wrap = True

    cell = (level, pos[0], pos[1])
    if cell in index:
        return (index[cell], -1, wrap)
    if level > 0 and parent(cell) in index:
        return (index[parent(cell)], -1, wrap)

    # Finer neighbors: the two children of cell that touch the shared face.
    # Along the face's axis they are on the side facing the leaf; along the
    # other axis they take both values, in ascending order.
    halves = []
    for t in (0, 1):
        offset = [0, 0]
        offset[axis] = 1 - side
        offset[1 - axis] = t
        halves.append((level + 1, 2 * pos[0] + offset[0], 2 * pos[1] + offset[1]))
    if halves[0] not in index or halves[1] not in index:
        raise ValueError(f"leaf {leaf}, face {face}: the mesh is not 2:1 balanced")
    return (index[halves[0]], index[halves[1]], wrap)


def topology(grid: QuadGrid, leaves: list[Leaf]) -> dict[str, np.ndarray]:
    """The per-element arrays of a SolverState, for leaves in element order.

    Args:
        grid: The base grid.
        leaves: The leaves of one 2:1-balanced mesh, already ordered; a
            leaf's place in the list is its position.

    Returns:
        level, neighbors, neighbor_wrap, family, can_refine, can_coarsen,
        centroid and h, keyed by SolverState's field names and in its dtypes
        and shapes. The arrays are new and writable.
    """
    n = len(leaves)
    index = {leaf: p for p, leaf in enumerate(leaves)}

    level = np.array([leaf[0] for leaf in leaves], dtype=np.int32)
    neighbors = np.full((n, 4, 2), -1, dtype=np.int32)
    neighbor_wrap = np.zeros((n, 4), dtype=bool)
    family = np.full((n, 4), -1, dtype=np.int32)
    centroid = np.empty((n, 2), dtype=np.float64)
    h = np.empty((n, 2), dtype=np.float64)

    for p, leaf in enumerate(leaves):
        lvl, x, y = leaf
        for face in range(4):
            slot0, slot1, wrap = face_neighbors(grid, index, leaf, face)
            neighbors[p, face] = (slot0, slot1)
            neighbor_wrap[p, face] = wrap
        if lvl > 0:
            siblings = children(parent(leaf))
            if all(s in index for s in siblings):
                family[p] = [index[s] for s in siblings]
        for axis, coord in enumerate((x, y)):
            h[p, axis] = (grid.hi[axis] - grid.lo[axis]) / grid.cells(axis, lvl)
            centroid[p, axis] = grid.lo[axis] + (coord + 0.5) * h[p, axis]

    return {
        "level": level,
        "neighbors": neighbors,
        "neighbor_wrap": neighbor_wrap,
        "family": family,
        "can_refine": level < grid.max_level,
        "can_coarsen": family[:, 0] >= 0,
        "centroid": centroid,
        "h": h,
    }