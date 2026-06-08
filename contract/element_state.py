"""Per-element state schema for the solver contract (concrete 1D).

Phase 3 (RESTRUCTURE_ROADMAP, D-034). Structure-of-arrays snapshot of the
active mesh the agent reads whenever it needs a fresh view (after every
mutating action). All per-element arrays are indexed by *active position*
[0, n_active), ordered consistently with compute_error()'s output.

Concrete for the 1D wave backend; dimension-agnostic generalization is parked
(P-012, Phase 9). The advection-flavored globals (wave_speed, domain_length)
are the fields most likely to change shape under generalization.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class SolverState:
    """Snapshot of the active mesh exposed to the agent core.

    Per-element arrays (shape (n_active,), indexed by active position;
    position i pairs with compute_error(...)[i]):
        element_id:  stable integer identity, survives mesh mutation — the
                     agent's queue stores these to re-resolve an element's
                     position after cascades/coarsening shift the ordering.
        level:       refinement level (0 = base, up to max_level).
        left, right: active positions of the neighbors (periodic 1D — always
                     valid). Used for neighbor error/level observations.
        sibling:     active position of the coarsening sibling, or -1 if the
                     element is level 0 or its sibling is not an active leaf.
        can_refine:  structural refinability — level < max_level.
        can_coarsen: structural coarsenability — sibling is an active leaf AND
                     the post-coarsen mesh still satisfies 2:1 balance. Does
                     NOT include the agent's round state (cascade exclusion)
                     or budget — the agent ANDs those in.

    Globals:
        n_active:      number of active elements (length of each array).
        max_level:     maximum refinement level.
        stable_dt:     largest CFL-stable timestep for the current mesh. The
                       backend owns the stability constraint; the agent sub-
                       steps with this across the advance.
        wave_speed:    advection speed (1D-advection-specific; → P-012).
        domain_length: physical domain length, for the agent's remesh interval
                       T = step_domain_fraction * domain_length / wave_speed.
    """
    element_id: np.ndarray
    level: np.ndarray
    left: np.ndarray
    right: np.ndarray
    sibling: np.ndarray
    can_refine: np.ndarray
    can_coarsen: np.ndarray

    n_active: int
    max_level: int
    stable_dt: float
    wave_speed: float
    domain_length: float