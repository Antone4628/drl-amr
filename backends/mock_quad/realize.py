"""How a quadtree mesh changes: the realization of one round of marks.

The marks are a request. What the mesh becomes follows p4est's order
(SOLVER_CONTRACT_V2 section 7): refine, then coarsen complete families, then
corner balance, one level per call. realize does that to a list of leaves and
reports what happened to each one; correspondence maps the positions of the
new mesh to those of the old. Both are pure functions: nothing is stored.
"""
from __future__ import annotations

from collections.abc import Iterator, Sequence

import numpy as np

from backends.mock_quad.quadtree import Leaf, QuadGrid, children, parent
from contract.reports import KIND_CHILD, KIND_MERGED, KIND_UNCHANGED
from contract.solver_contract import COARSEN, HOLD, REFINE


def _check_marks(marks: np.ndarray, n: int) -> np.ndarray:
    """Return the marks as an int8 array, or raise naming what is wrong."""
    marks = np.asarray(marks)
    if marks.shape != (n,):
        raise ValueError(f"marks: shape {marks.shape}, expected ({n},)")
    if not np.issubdtype(marks.dtype, np.integer):
        raise ValueError(f"marks: dtype {marks.dtype}, expected an integer dtype")
    if not np.isin(marks, (COARSEN, HOLD, REFINE)).all():
        raise ValueError("marks: values must be COARSEN, HOLD or REFINE")
    return marks.astype(np.int8)


def _refine(grid: QuadGrid, leaves: Sequence[Leaf], marks: np.ndarray) -> set[Leaf]:
    """Step 1: split every leaf marked REFINE that is below max_level."""
    mesh = set(leaves)
    for leaf, mark in zip(leaves, marks, strict=True):
        if mark == REFINE and leaf[0] < grid.max_level:
            mesh.remove(leaf)
            mesh.update(children(leaf))
    return mesh


def _coarsen(mesh: set[Leaf], leaves: Sequence[Leaf], marks: np.ndarray) -> None:
    """Step 2: merge every family whose four members are all marked COARSEN.

    Only leaves of the pre-adapt mesh carry a mark, so a child made by step 1
    never merges, and a family with a member refined or unmarked is dropped.
    Changes mesh in place.
    """
    marked = {leaf for leaf, mark in zip(leaves, marks, strict=True) if mark == COARSEN and leaf[0] > 0}
    for leaf in marked:
        # Visit each family once, at its first child (ix = iy = 0).
        if leaf[1] & 1 or leaf[2] & 1:
            continue
        family = children(parent(leaf))
        if all(sibling in marked for sibling in family):
            mesh.difference_update(family)
            mesh.add(parent(leaf))


def _adjacent(grid: QuadGrid, leaf: Leaf) -> Iterator[Leaf]:
    """The same-level cells that touch a leaf across a face or a corner.

    Up to eight. A step off a walled axis has no cell; a step off a periodic
    axis wraps, each axis on its own.
    """
    level, x, y = leaf
    nx, ny = grid.cells(0, level), grid.cells(1, level)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            ax, ay = x + dx, y + dy
            if not 0 <= ax < nx:
                if not grid.periodic[0]:
                    continue
                ax %= nx
            if not 0 <= ay < ny:
                if not grid.periodic[1]:
                    continue
                ay %= ny
            yield (level, ax, ay)


def _covering(mesh: set[Leaf], cell: Leaf) -> Leaf | None:
    """The leaf that is a cell or one of its ancestors, or None.

    None means finer leaves cover the cell, and those ask nothing of the
    leaf that touches them.
    """
    while cell not in mesh:
        if cell[0] == 0:
            return None
        cell = parent(cell)
    return cell


def balance(grid: QuadGrid, mesh: set[Leaf]) -> set[Leaf]:
    """Step 3: the coarsest corner-balanced refinement of a mesh.

    A leaf may not touch, across a face or a corner, a leaf more than one
    level coarser. Wherever one does, the coarser leaf is split and its
    children are checked in turn. The answer is unique, so the order in
    which leaves are visited does not matter. Returns a new set.
    """
    mesh = set(mesh)
    work = list(mesh)
    while work:
        leaf = work.pop()
        if leaf not in mesh:
            continue
        for cell in _adjacent(grid, leaf):
            cover = _covering(mesh, cell)
            while cover is not None and cover[0] < leaf[0] - 1:
                mesh.remove(cover)
                mesh.update(children(cover))
                work.extend(children(cover))
                cover = _covering(mesh, cell)
    return mesh


def realize(grid: QuadGrid, leaves: Sequence[Leaf], marks: np.ndarray) -> tuple[list[Leaf], np.ndarray]:
    """Realize one round of marks on a mesh. Pure: nothing is changed.

    Args:
        grid: The base grid.
        leaves: The leaves of one corner-balanced mesh, in element order.
        marks: COARSEN, HOLD or REFINE for each leaf; (n,), any integer
            dtype. A REFINE at max_level and a COARSEN outside a fully
            marked family are ignored.

    Returns:
        The leaves of the realized mesh, sorted as tuples (not element
        order), and realized: what happened to each pre-adapt leaf, (n,)
        int8. realized is read from the levels, not from the marks: HOLD if
        the leaf is still a leaf, COARSEN if its parent is one, REFINE if
        its four children are.

    Raises:
        ValueError: If marks has the wrong shape, a non-integer dtype or a
            value outside the vocabulary.
        RuntimeError: If a leaf moved more than one level, which a
            corner-balanced mesh cannot do.
    """
    marks = _check_marks(marks, len(leaves))
    mesh = _refine(grid, leaves, marks)
    _coarsen(mesh, leaves, marks)
    mesh = balance(grid, mesh)

    realized = np.empty(len(leaves), dtype=np.int8)
    for p, leaf in enumerate(leaves):
        if leaf in mesh:
            realized[p] = HOLD
        elif leaf[0] > 0 and parent(leaf) in mesh:
            realized[p] = COARSEN
        elif all(child in mesh for child in children(leaf)):
            realized[p] = REFINE
        else:
            raise RuntimeError(f"leaf {leaf}: moved more than one level; the mesh given was not corner balanced")
    return sorted(mesh), realized


def correspondence(old_leaves: Sequence[Leaf], new_leaves: Sequence[Leaf]) -> tuple[np.ndarray, np.ndarray]:
    """Map each position of the new mesh to the old positions it came from.

    Args:
        old_leaves: The pre-adapt leaves, in their element order.
        new_leaves: The post-adapt leaves, in their element order.

    Returns:
        new_to_old, (n_new, 4) int32, and new_kind, (n_new,) int8, as
        AdaptReport defines them: an unchanged leaf and a child carry one
        old position in slot 0; a merged parent carries its four children
        in child order.

    Raises:
        RuntimeError: If a new leaf is not within one level of the old mesh.
    """
    old = {leaf: p for p, leaf in enumerate(old_leaves)}
    new_to_old = np.full((len(new_leaves), 4), -1, dtype=np.int32)
    new_kind = np.empty(len(new_leaves), dtype=np.int8)
    for q, leaf in enumerate(new_leaves):
        if leaf in old:
            new_kind[q] = KIND_UNCHANGED
            new_to_old[q, 0] = old[leaf]
        elif leaf[0] > 0 and parent(leaf) in old:
            new_kind[q] = KIND_CHILD
            new_to_old[q, 0] = old[parent(leaf)]
        elif all(child in old for child in children(leaf)):
            new_kind[q] = KIND_MERGED
            new_to_old[q] = [old[child] for child in children(leaf)]
        else:
            raise RuntimeError(f"leaf {leaf}: not within one level of the old mesh")
    return new_to_old, new_kind