"""Deployment/visualization snapshot schema for the solver contract.

RESTRUCTURE Phase 6 (deployment adapter, build-order step 1). A copy-safe,
per-frame record of the *physical* solver state — the solution field plus
enough mesh geometry to render it — emitted by SolverContract.get_snapshot()
and accumulated into a per-frame list by the deployment runner across a full
simulation (DEPLOYMENT_ADAPTER_DESIGN.md §3).

Distinct from SolverState (contract/element_state.py): SolverState is the
agent-facing *topology* view (neighbors, levels, refinability) read after every
mutating action during training; SolverSnapshot is the deployment-facing *viz*
view (solution + geometry) read at the output cadence. Different audience,
different lifecycle.

Lean by design (DEPLOYMENT_ADAPTER_DESIGN.md §3): solution + geometry only, NO
per-frame error recomputation (ZZ is expensive) and NO exact solution — error-
bearing data and exact overlays are handled at adaptation boundaries / the
final frame / the visualization layer, not on every frame.

Concrete for the 1D wave backend; dimension-agnostic generalization is parked
(P-012, Phase 9).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class SolverSnapshot:
    """Copy-safe per-frame snapshot of physical solution + mesh geometry.

    All array fields are independent copies of the solver's state at capture
    time, so a snapshot accumulated into a list is unaffected by subsequent
    solver mutation (time stepping, adaptation).

    Fields:
        time:      physical simulation time at capture.
        q:         DG nodal solution, shape (npoin_dg,), element-contiguous
                   (element e occupies q[e*ngl : (e+1)*ngl]).
        coord:     physical coordinate of each DG node, shape (npoin_dg,),
                   aligned with q.
        intma:     element-to-node connectivity (element-local LGL index ->
                   global DG-node index); lets the viz layer draw per-element
                   curves and resolve element-interface discontinuities.
        xelem:     element boundary coordinates, shape (n_active + 1,).
        active:    forest element IDs of the active leaves, shape (n_active,).
        levels:    refinement level of each active element, shape (n_active,),
                   aligned with active; for level-colored rendering and the
                   annotation box's max-level field.
        ngl:       LGL points per element (nop + 1); constant across a run.
        xgl:       LGL reference nodes on [-1, 1], shape (ngl,); constant.
        npoin_dg:  total DG node count = ngl * n_active.
        n_active:  number of active elements; element count for the annotation
                   box.
    """
    time: float
    q: np.ndarray
    coord: np.ndarray
    intma: np.ndarray
    xelem: np.ndarray
    active: np.ndarray
    levels: np.ndarray
    ngl: int
    xgl: np.ndarray
    npoin_dg: int
    n_active: int