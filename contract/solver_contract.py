"""Solver contract for DRL-AMR backends — fine-grained sequential surface.

Phase 3, step 1 (RESTRUCTURE_ROADMAP, D-034). This is the *verbs* slice:
lifecycle, error, per-element adaptation, balance, rebuild, time advance.
Deferred to later Phase 3 steps:
  - get_state + topology queries (the *nouns*) — defined with the 1D state schema
  - apply_marks + predict_post_balance_count + the iteration-mode seam (D-043)
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from contract.element_state import SolverState

# Mark vocabulary, shared by adapt_element (sequential) and apply_marks (batch).
COARSEN = -1
HOLD = 0
REFINE = +1


@runtime_checkable
class SolverContract(Protocol):
    """Interface a numerical backend exposes to the agent core.

    The agent decides *what* action on *which* element; the backend owns *how*
    its mesh realizes it. Implemented by python_1d now; later by python_2d and
    a Jexpresso-wrapping adapter — none of which the agent core sees.

    Granularity (D-043): this is the fine-grained sequential surface.
    `adapt_element` mutates one element; `balance` and `rebuild` are explicit so
    the agent owns the (expensive) rebuild boundary. The batch counterpart
    `apply_marks` (one pass, global balance, single rebuild) is added later.
    """

    # --- Lifecycle ---
    def reset(
        self,
        icase: int | None = None,
        refinement_mode: str = "none",
        refinement_level: int = 0,
    ) -> None:
        """Begin a fresh episode: rebuild base mesh, set IC (icase, or keep
        current if None), optionally apply `refinement_level` uniform passes
        ('fixed'/'random'/'none'), rebuild operators. State is read separately
        via get_state/compute_error. (icase is 1D-wave-specific; generalization
        parked → P-012.)
        """
    
    # --- State ---
    def get_state(self) -> SolverState:
        """Return a structure-of-arrays snapshot of the active mesh —
        topology, levels, refinability, and globals. Called whenever the
        agent needs a fresh view, notably after every mutating action (cascades
        and coarsening shift active positions). Error indicators are NOT
        included; the agent pairs this with compute_error(indicator).
        """

    # --- Error ---
    def compute_error(self, indicator: str) -> np.ndarray:
        """Per-element error indicator for all active elements, shape
        (n_active,), ordered with the active list. `indicator` selects the
        registered estimator ('zz_style' default, 'raw_jump' comparator). All
        normalization/threshold math lives in the agent, not here.
        """

    # --- Adaptation (sequential; mutates topology + projects solution, no rebuild) ---
    def adapt_element(self, active_idx: int, mark: int) -> bool:
        """Apply one mark to one element. mark ∈ {REFINE, HOLD, COARSEN}.
        Updates topology and projects the solution; does NOT enforce balance or
        rebuild operators. Returns True iff the mesh changed (False for HOLD or
        a rejected mark — e.g. refine at max_level, coarsen with no valid
        sibling). Dispatches to the backend's native refine/coarsen internally.
        """

    def balance(self) -> set[int]:
        """Enforce 2:1 balance. Returns the set of element IDs *created* by
        cascades (empty if none) — the agent's cascade bookkeeping
        (`consumed_elements`) consumes this directly instead of diffing active
        sets. Does NOT rebuild operators.
        """

    def rebuild(self) -> None:
        """Rebuild DG operators (mass/diff/flux) and forcing for the current
        mesh — the explicit commit after element ops. Sequential mode:
        adapt_element → balance → rebuild per element. This is the cost the
        batch mode amortizes to once per round.
        """

    # --- Time advance ---
    def step(self, dt: float) -> None:
        """Advance the solution by one time step of size dt. No mesh change."""