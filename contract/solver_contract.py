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
from contract.solver_snapshot import SolverSnapshot

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

    Iteration modes (D-043). One agent core drives this contract in either of
    two granularities; the driving loop (env step() vs. deployment-adapter
    internal loop) is orthogonal to the mode — both reduce to these primitives:

      Sequential (default, parity target): per element, the agent calls
        adapt_element(idx, mark) -> balance() -> rebuild(), tracks the returned
        cascade set in consumed_elements, then re-reads get_state() before the
        next element. One rebuild per element per round.

      Batch: the agent reads get_state() once at round start, builds a full
        marks array, and calls apply_marks(marks) — one global balance, one
        rebuild. No per-element re-observation, so consumed_elements is unused;
        budget stays accurate via the committed-mark counter (+ optional
        predict_post_balance_count).
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

    def reinitialize_ic(self) -> None:
        """Reset time to zero and reproject the initial condition onto the
        current (possibly adapted) mesh — mesh topology and operators
        unchanged.

        Deployment burn-in (DEPLOYMENT_ADAPTER_DESIGN.md §3): after the agent
        refines the mesh to resolve the IC, re-seed the solution from the exact
        IC sampled on the now-finer nodes (sharper than the L2-projected
        solution carried through refinement), then refine again. No operator
        rebuild — geometry is unchanged. (icase-based IC is 1D-wave-specific →
        P-012.)
        """

    # --- State ---
    def get_state(self) -> SolverState:
        """Return a structure-of-arrays snapshot of the active mesh —
        topology, levels, refinability, and globals. Called whenever the
        agent needs a fresh view, notably after every mutating action (cascades
        and coarsening shift active positions). Error indicators are NOT
        included; the agent pairs this with compute_error(indicator).
        """

    # --- Snapshot (deployment / visualization; no mesh change) ---
    def get_snapshot(self) -> SolverSnapshot:
        """Return a copy-safe snapshot of the physical solution + mesh geometry
        for deployment-time visualization (animations, plots, static frames).

        Distinct from get_state (agent-facing topology): this is the deployment
        artifact — the current solution field plus enough geometry to render it.
        No error recomputation (ZZ is expensive — error-bearing data lands at
        adaptation boundaries / the final frame instead) and no Gym/reward
        machinery. The returned arrays are independent copies, safe to
        accumulate into a per-frame list across a full simulation while the
        solver state evolves (DEPLOYMENT_ADAPTER_DESIGN.md §3).
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

    # --- Adaptation (batch; one pass, global balance, single rebuild) ---
    def apply_marks(self, marks: np.ndarray) -> SolverState:
        """Apply a full round of decisions in one pass and return the new state.

        The batch counterpart to the sequential adapt_element -> balance ->
        rebuild triad (D-043). `marks` has shape (n_active,), values in
        {REFINE, HOLD, COARSEN}, aligned with the current active ordering
        (same indexing as get_state / compute_error).

        Semantics:
          - Refine marks: each marked element is refined.
          - Coarsen marks: honored only for a *complete* sibling family — a
            parent coarsens iff all its children are marked COARSEN and
            eligible. A lone/partial coarsen mark is a no-op. This is the
            dimension-agnostic coarsening form (p4est collapses a complete
            family only) and resolves the D-034 arity hazard (1D=2, 2D=4,
            3D=8).
          - Ineligible marks (refine at max_level, ineligible coarsen) are
            silently ignored.
          - 2:1 balance is enforced globally over the resulting mesh in a
            single pass; operators are then rebuilt once.

        Unlike the sequential path this does NOT return a cascade set: all
        decisions are made on one pre-round snapshot, so there is no "later in
        the round" to protect and the sequential consumed_elements exclusion
        does not apply (D-043). Budget accounting uses the agent's committed-
        mark counter, optionally corrected by predict_post_balance_count.

        Returns:
            The new SolverState after marks + global balance + rebuild.
        """

    def predict_post_balance_count(self, marks: np.ndarray) -> int:
        """Predict the active-element count after applying `marks` + balance,
        WITHOUT mutating the mesh (D-043).

        A cheap topology-only dry run: lets the agent's budget logic (and a
        future alpha-scaled barrier, D-015) account for cascade-induced growth
        before committing. Optional — a backend that cannot predict cheaply may
        raise NotImplementedError, and the agent falls back to its committed-
        mark counter (refine +1 / coarsen -1 per decision in 1D; the family
        arity in higher dimensions).

        Args:
            marks: Proposed decision array, shape (n_active,), as in apply_marks.

        Returns:
            Predicted number of active elements post-balance.
        """

    # --- Time advance ---
    def step(self, dt: float) -> None:
        """Advance the solution by one time step of size dt. No mesh change."""