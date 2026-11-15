"""Per-element state schema for the solver contract, v2 (any dimension).

One schema for every dimension; the dimension shows only in array shapes
(SOLVER_CONTRACT_V2 section 5). An element has 2*ndim faces, each face has
2**(ndim-1) neighbor slots, and a sibling family has 2**ndim members. The 1D
contract's left, right and sibling are the ndim = 1 case.

Conventions (spec section 5.2): positions are 0-based; -1 means absent; faces
are numbered f = 2*axis + side, side 0 the minimum; children are numbered
c = ix + 2*iy; half-neighbor slots ascend along the face's tangential axis.

This class lives beside the 1D SolverState in contract/element_state.py,
which stays until the 1D backend populates this schema.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from contract._arrays import check_and_freeze


@dataclass(frozen=True, slots=True, kw_only=True)
class SolverState:
    """Snapshot of the active mesh, read by the agent once per round.

    An element is identified by (mesh_epoch, position). A position is valid
    within one mesh_epoch only: the ordering is opaque and changes at every
    adapt. Every array is checked at construction and made read-only.

    Per-element arrays (first axis n_active, indexed by position):
        level: Refinement level, 0 at the base mesh. (n_active,) int32.
        neighbors: Positions of the face neighbors.
            (n_active, 2*ndim, 2**(ndim-1)) int32. A neighbor at the same
            level or coarser is in slot 0, the other slots -1. Finer
            half-neighbors fill the slots. A wall face is all -1. Whether a
            neighbor is coarser is read from level[neighbor].
        neighbor_wrap: True where the face's neighbor lies across a periodic
            wrap. (n_active, 2*ndim) bool.
        family: Positions of all siblings in child order, the element itself
            included; all -1 at level 0 or when a sibling is not an active
            leaf. (n_active, 2**ndim) int32.
        can_refine: level < max_level. (n_active,) bool.
        can_coarsen: The family row is complete. Structural only: no balance
            check, no round state, no budget. (n_active,) bool.
        centroid: Element center. (n_active, ndim) float64.
        h: Element extent along each axis, not derived from level.
            (n_active, ndim) float64.
        attributes: Per-equation element attributes; zero columns when the
            equation has none. (n_active, n_attr) float64.

    Globals:
        n_active: Number of active elements.
        ndim: Spatial dimension (1, 2 or 3).
        nvar: Number of solution variables.
        max_level: Maximum refinement level.
        mesh_epoch: Counter that rises by one at every apply_marks and reset.
        time: Physical time.
        interval_T: The remesh interval, amr_freq * dt_base.
        dt_base: The base time step.
        amr_freq: Base steps per remesh interval.
        dt_stable: The stable step on the current mesh. Informational: it
            varies with the realized finest level.
        domain_lo: Lower corner of the domain. (ndim,) float64.
        domain_hi: Upper corner of the domain. (ndim,) float64.
        periodic: Whether each axis is periodic. (ndim,) bool.

    Optional:
        state_mean: Element-averaged solution. Reserved; None unless adopted.
            (n_active, nvar) float64.
        element_id: Stable ids, for a backend that has them (the 1D backend
            in sequential mode). None at 2D. (n_active,) int64.
    """

    level: np.ndarray
    neighbors: np.ndarray
    neighbor_wrap: np.ndarray
    family: np.ndarray
    can_refine: np.ndarray
    can_coarsen: np.ndarray
    centroid: np.ndarray
    h: np.ndarray
    attributes: np.ndarray

    n_active: int
    ndim: int
    nvar: int
    max_level: int
    mesh_epoch: int
    time: float
    interval_T: float
    dt_base: float
    amr_freq: int
    dt_stable: float
    domain_lo: np.ndarray
    domain_hi: np.ndarray
    periodic: np.ndarray

    state_mean: np.ndarray | None = None
    element_id: np.ndarray | None = None

    def __post_init__(self) -> None:
        """Check the globals, then every array's dtype and shape, and freeze."""
        n, d = self.n_active, self.ndim
        if d not in (1, 2, 3):
            raise ValueError(f"ndim: {d}, expected 1, 2 or 3")
        if n < 1:
            raise ValueError(f"n_active: {n}, expected at least 1")
        if self.nvar < 1:
            raise ValueError(f"nvar: {self.nvar}, expected at least 1")

        faces, slots, children = 2 * d, 2 ** (d - 1), 2**d

        check_and_freeze("level", self.level, np.int32, (n,))
        check_and_freeze("neighbors", self.neighbors, np.int32, (n, faces, slots))
        check_and_freeze("neighbor_wrap", self.neighbor_wrap, bool, (n, faces))
        check_and_freeze("family", self.family, np.int32, (n, children))
        check_and_freeze("can_refine", self.can_refine, bool, (n,))
        check_and_freeze("can_coarsen", self.can_coarsen, bool, (n,))
        check_and_freeze("centroid", self.centroid, np.float64, (n, d))
        check_and_freeze("h", self.h, np.float64, (n, d))

        # attributes: the column count belongs to the equation, so only the
        # rank and the row count are fixed here.
        if not isinstance(self.attributes, np.ndarray) or self.attributes.ndim != 2:
            raise ValueError("attributes: expected a 2-D numpy array")
        n_attr = self.attributes.shape[1]
        check_and_freeze("attributes", self.attributes, np.float64, (n, n_attr))

        check_and_freeze("domain_lo", self.domain_lo, np.float64, (d,))
        check_and_freeze("domain_hi", self.domain_hi, np.float64, (d,))
        check_and_freeze("periodic", self.periodic, bool, (d,))

        if self.state_mean is not None:
            check_and_freeze("state_mean", self.state_mean, np.float64, (n, self.nvar))
        if self.element_id is not None:
            check_and_freeze("element_id", self.element_id, np.int64, (n,))  