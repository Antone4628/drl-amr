"""Backend-independent checks on code written against the v2 contract.

training_only_params  which of a function's parameters could receive a
                      training-only type. Used to assert that the
                      observation builder shared with deployment accepts
                      none.

topology_violations   whether a SolverState's arrays describe a mesh:
                      families, neighbors, faces and, on a uniform base,
                      sizes and centroids. Each check returns a list of
                      findings, empty when the state is clean, and assumes
                      axis-aligned rectangular elements. Written once, for
                      the mock's tests and for the scripted-action harness
                      on any backend.
"""
from __future__ import annotations

import inspect
import typing
from collections.abc import Callable

import numpy as np

from contract.reports import AdaptReport, AuxFields, IntervalReport
from contract.state import SolverState

# Types that exist in training and not at deployment. AuxFields is the
# auxiliary channel; IntervalReport contains it; AdaptReport is a training
# diagnostic the deployed loop never builds.
TRAINING_ONLY = (AdaptReport, AuxFields, IntervalReport)


def _mentions_training_only(annotation: object) -> bool:
    """True if an annotation is, or contains at any depth, a training-only
    type or typing.Any."""
    if annotation is typing.Any:
        return True
    if isinstance(annotation, (list, tuple)):
        return any(_mentions_training_only(a) for a in annotation)
    if any(annotation is t for t in TRAINING_ONLY):
        return True
    return any(_mentions_training_only(a) for a in typing.get_args(annotation))


def training_only_params(fn: Callable) -> list[str]:
    """Names of fn's parameters that could receive a training-only type.

    A parameter is reported when its annotation is or contains one of
    TRAINING_ONLY (AuxFields | None, dict[str, AuxFields], ...), when it is
    annotated Any, and when it has no annotation at all: an unannotated
    parameter accepts everything.

    Args:
        fn: The function to inspect, e.g. the actor observation builder.

    Returns:
        The offending parameter names, in signature order; empty when the
        signature is clean.
    """
    hints = typing.get_type_hints(fn)
    offending = []
    for name in inspect.signature(fn).parameters:
        if name not in hints or _mentions_training_only(hints[name]):
            offending.append(name)
    return offending


def _note(found: list[str], rule: str, bad: np.ndarray) -> None:
    """Append one finding for a rule if any entry of bad is set.

    Args:
        found: The list of findings, appended to in place.
        rule: The rule's name, the stable prefix of the finding.
        bad: True where the rule is broken; (n_active,) or (n_active, faces).
    """
    if bad.any():
        first = np.argwhere(bad)[0]
        where = f"position {first[0]}" + (f", face {first[1]}" if bad.ndim == 2 else "")
        found.append(f"{rule}: {int(bad.sum())}, first at {where}")


def _range_violations(state: SolverState) -> list[str]:
    """The one finding that stops every other check: an index that names no
    element.

    -1 (absent) is allowed. Anything else outside [0, n_active) would either
    raise or, below -1, be read by numpy as counting from the end.
    """
    found: list[str] = []
    n = state.n_active
    bad = ((state.neighbors < -1) | (state.neighbors >= n)).any(axis=(1, 2))
    bad |= ((state.family < -1) | (state.family >= n)).any(axis=1)
    _note(found, "indices.range", bad)
    return found


def family_violations(state: SolverState, rtol: float = 1e-9) -> list[str]:
    """Check that every reported family is the children of one parent cell.

    Rules:
        family.partial_row: A row is all -1 or all positions.
        family.level: The members share one level, at least 1.
        family.rows_agree: Every member carries the same row, and an element
            is in its own row.
        family.quadrants: Member c sits in orthant c of the cell the members
            span, c = ix + 2*iy (+ 4*iz): the child order.

    That a family which exists is reported is uniform_base_violations' rule,
    since it needs the base grid.

    Args:
        state: The state to check.
        rtol: Tolerance on a coordinate, as a fraction of the domain's extent.

    Returns:
        One finding per broken rule; empty when clean.
    """
    found = _range_violations(state)
    if found:
        return found
    family, level, n, d = state.family, state.level, state.n_active, state.ndim
    tol = rtol * (state.domain_hi - state.domain_lo)

    filled = family >= 0
    complete = filled.all(axis=1)
    _note(found, "family.partial_row", filled.any(axis=1) & ~complete)

    # From here on, the complete rows only: rows holds their positions, and
    # members[i, c] is child c of the family of rows[i].
    rows = np.flatnonzero(complete)
    members = family[rows]

    bad = np.zeros(n, dtype=bool)
    bad[rows] = (level[rows] < 1) | (level[members] != level[rows, None]).any(axis=1)
    _note(found, "family.level", bad)

    bad = np.zeros(n, dtype=bool)
    bad[rows] = (family[members] != members[:, None, :]).any(axis=(1, 2))
    bad[rows] |= ~(members == rows[:, None]).any(axis=1)
    _note(found, "family.rows_agree", bad)

    # Bit a of the child index says which half of the parent along axis a.
    bits = (np.arange(2**d)[:, None] >> np.arange(d)) & 1
    center = state.centroid[members].mean(axis=1, keepdims=True)
    expected = center + (bits - 0.5) * state.h[members]
    bad = np.zeros(n, dtype=bool)
    bad[rows] = (np.abs(state.centroid[members] - expected) > tol).any(axis=(1, 2))
    _note(found, "family.quadrants", bad)
    return found


def neighbor_violations(state: SolverState) -> list[str]:
    """Check the neighbor array against itself and the levels: integers only.

    Rules:
        neighbors.slots: A face has no neighbor, one in slot 0, or one in
            every slot.
        neighbors.two_to_one: A face with every slot filled holds neighbors
            one level finer; a face with slot 0 alone holds one at the same
            level or one coarser. With one slot per face (1D) the neighbor
            is within one level either way.
        neighbors.symmetry: If q is a neighbor of p across a face, p is a
            neighbor of q across the opposite face.

    Where a neighbor sits, and which slot it belongs in, is geometry:
    face_violations.

    Args:
        state: The state to check.

    Returns:
        One finding per broken rule; empty when clean.
    """
    found = _range_violations(state)
    if found:
        return found
    neighbors, level = state.neighbors, state.level
    n, faces, slots = neighbors.shape

    filled = neighbors >= 0
    count = filled.sum(axis=2)
    full = count == slots
    _note(found, "neighbors.slots", ~((count == 0) | ((count == 1) & filled[:, :, 0]) | full))

    # Each neighbor's level against the element's own; 0 where absent.
    step = np.where(filled, level[neighbors] - level[:, None, None], 0)
    if slots == 1:
        bad = np.abs(step) > 1
    else:
        bad = filled & np.where(full[:, :, None], step != 1, (step != 0) & (step != -1))
    _note(found, "neighbors.two_to_one", bad.any(axis=2))

    # The opposite face of f = 2*axis + side is f ^ 1: same axis, other side.
    here = np.arange(n)
    bad = np.zeros((n, faces), dtype=bool)
    for face in range(faces):
        for slot in range(slots):
            q = neighbors[:, face, slot]
            back = neighbors[q, face ^ 1]
            bad[:, face] |= (q >= 0) & ~(back == here[:, None]).any(axis=1)
    _note(found, "neighbors.symmetry", bad)
    return found


def face_violations(state: SolverState, rtol: float = 1e-9) -> list[str]:
    """Check every face against the geometry: walls, the wrap, and where
    each neighbor sits.

    For an element p, a face on one axis with s = -1 on its minimum side and
    +1 on its maximum, and a neighbor q, along that axis

        delta = c_q - c_p - s * (h_p + h_q) / 2

    is 0 when q is adjacent and -s * L when q is adjacent through the
    periodic wrap, L being the domain's extent on the axis.

    Rules:
        faces.wall: A face has no neighbor exactly when it lies on the
            boundary of a walled axis.
        faces.normal: delta is one of the two values above.
        faces.wrap: neighbor_wrap is True exactly when delta is -s * L, and
            False on a face with no neighbor.
        faces.tangential: Along the face, the two centroids differ by
            |h_q - h_p| / 2. For finer neighbors the sign is the slot's:
            slot k has one bit per tangential axis, the lower axis first,
            0 on the lower half.

    Args:
        state: The state to check.
        rtol: Tolerance on a coordinate, as a fraction of the domain's extent.

    Returns:
        One finding per broken rule; empty when clean.
    """
    found = _range_violations(state)
    if found:
        return found
    neighbors, level, c, h = state.neighbors, state.level, state.centroid, state.h
    n, faces, slots = neighbors.shape
    extent = state.domain_hi - state.domain_lo
    tol = rtol * extent

    wall = np.zeros((n, faces), dtype=bool)
    normal = np.zeros((n, faces), dtype=bool)
    wrap = np.zeros((n, faces), dtype=bool)
    tangential = np.zeros((n, faces), dtype=bool)
    for face in range(faces):
        axis, side = divmod(face, 2)
        s = 2 * side - 1
        corner = state.domain_hi[axis] if side else state.domain_lo[axis]
        on_boundary = np.abs(c[:, axis] + s * h[:, axis] / 2 - corner) <= tol[axis]
        empty = (neighbors[:, face] < 0).all(axis=1)
        wall[:, face] = empty != (on_boundary & ~state.periodic[axis])
        wrap[:, face] = empty & state.neighbor_wrap[:, face]

        others = [a for a in range(state.ndim) if a != axis]
        for slot in range(slots):
            q = neighbors[:, face, slot]
            has = q >= 0
            delta = c[q, axis] - c[:, axis] - s * (h[:, axis] + h[q, axis]) / 2
            adjacent = np.abs(delta) <= tol[axis]
            across = state.periodic[axis] & (np.abs(delta + s * extent[axis]) <= tol[axis])
            normal[:, face] |= has & ~(adjacent | across)
            wrap[:, face] |= has & (adjacent | across) & (state.neighbor_wrap[:, face] != across)

            # half is positive when q is the finer of the two.
            finer = level[q] > level
            for j, t in enumerate(others):
                offset = c[q, t] - c[:, t]
                half = (h[:, t] - h[q, t]) / 2
                sign = 2 * ((slot >> j) & 1) - 1
                miss = np.where(finer, np.abs(offset - sign * half), np.abs(np.abs(offset) + half))
                tangential[:, face] |= has & (miss > tol[t])

    _note(found, "faces.wall", wall)
    _note(found, "faces.normal", normal)
    _note(found, "faces.wrap", wrap)
    _note(found, "faces.tangential", tangential)
    return found


def uniform_base_violations(state: SolverState, rtol: float = 1e-9) -> list[str]:
    """Check sizes, centroids and family completeness against a uniform base
    grid. Not for a state whose base mesh is not uniform.

    Rules:
        uniform_base.h: h * 2**level is one base cell size for every
            element, and it divides the domain into whole cells.
        uniform_base.centroid: Every centroid is a cell center of the
            uniform grid of its element's level, inside the domain.
        uniform_base.area: The elements' areas sum to the domain's.
        uniform_base.family_complete: Elements that are all the children of
            one parent cell carry a complete family row. Checked only when
            the three rules above hold, since it reads each element's cell
            from them.

    Args:
        state: The state to check.
        rtol: Tolerance on a coordinate, as a fraction of the domain's extent.

    Returns:
        One finding per broken rule; empty when clean.
    """
    found = _range_violations(state)
    if found:
        return found
    level, c, h, d = state.level, state.centroid, state.h, state.ndim
    lo = state.domain_lo
    extent = state.domain_hi - lo
    tol = rtol * extent

    # The base cell size each element implies; element 0's is the reference.
    base = h * 2.0 ** level[:, None]
    bad = (np.abs(base - base[0]) > tol).any(axis=1)
    bad |= (np.abs(np.round(extent / base[0]) * base[0] - extent) > tol).any()
    _note(found, "uniform_base.h", bad)

    # An element's column and row on the uniform grid of its own level.
    cell = np.round((c - lo) / h - 0.5)
    bad = (np.abs(lo + (cell + 0.5) * h - c) > tol).any(axis=1)
    bad |= ((cell < 0) | ((cell + 1) * h > extent + tol)).any(axis=1)
    _note(found, "uniform_base.centroid", bad)

    covered, area = float(h.prod(axis=1).sum()), float(extent.prod())
    if abs(covered - area) > rtol * area:
        found.append(f"uniform_base.area: the elements cover {covered}, the domain is {area}")
    if found:
        return found

    # Group the elements by parent cell; a full group is a family.
    groups: dict[tuple[int, ...], list[int]] = {}
    for p in np.flatnonzero(level >= 1):
        key = (int(level[p]), *(int(x) // 2 for x in cell[p]))
        groups.setdefault(key, []).append(int(p))
    bad = np.zeros(state.n_active, dtype=bool)
    for group in groups.values():
        if len(group) == 2**d:
            bad[group] = (state.family[group] < 0).any(axis=1)
    _note(found, "uniform_base.family_complete", bad)
    return found


def topology_violations(state: SolverState, *, uniform_base: bool = True, rtol: float = 1e-9) -> list[str]:
    """Run every topology check on a state.

    The checks are family_violations, neighbor_violations, face_violations
    and, unless turned off, uniform_base_violations. An index that names no
    element is reported alone, as indices.range.

    Args:
        state: The state to check.
        uniform_base: Whether the base mesh is uniform. Pass False for a
            state built on a non-uniform base grid.
        rtol: Tolerance on a coordinate, as a fraction of the domain's extent.

    Returns:
        Every finding, in the order above; empty when the state is clean.
    """
    found = _range_violations(state)
    if found:
        return found
    found = family_violations(state, rtol) + neighbor_violations(state) + face_violations(state, rtol)
    if uniform_base:
        found += uniform_base_violations(state, rtol)
    return found