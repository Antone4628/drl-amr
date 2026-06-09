"""Python-1D backend implementation of the SolverContract (Phase 4; D-034/043/044).

Composition adapter: wraps a DGAdvectionSolver and exposes the agent-facing
SolverContract surface. It owns the *solver-side* mapping only — cascade
set-diff, explicit operator rebuild, the structure-of-arrays state snapshot,
and the raw per-element error vector. All *agent* logic (observation assembly,
alpha/beta normalization, action masking, the priority queue, reward, and the
budget counter) lives in the agent core (Phase 5), NOT here.

Parity target (D-038): in sequential mode the triad
    adapt_element(idx, mark) -> balance() -> rebuild()
reproduces the old DGAMREnvMultiround._execute_action behavior element-by-
element (refine/coarsen primitive, then explicit balance with cascade capture,
then a single matrix+forcing rebuild).

Scope: the sequential surface is implemented. The batch seam (apply_marks,
predict_post_balance_count; D-043) is deferred past Phase 4 — sequential is the
parity target and the old env never drove batch mode.
"""
from __future__ import annotations

import numpy as np

from contract.element_state import SolverState
from contract.solver_contract import COARSEN, HOLD, REFINE

from .solvers.dg_advection_solver_multiround import DGAdvectionSolver
from .solvers.error_indicators import compute_errors


class Python1DSolverContract:
    """SolverContract over the 1D DG advection solver (composition wrapper).

    Wrap a DGAdvectionSolver constructed with ``balance=False``: 2:1 balance is
    enforced explicitly via :meth:`balance` so cascade-created elements can be
    reported to the agent (mirrors the old env's explicit-balance requirement).
    """

    def __init__(self, solver: DGAdvectionSolver) -> None:
        self._solver = solver

    # --- Lifecycle ---------------------------------------------------------
    def reset(
        self,
        icase: int | None = None,
        refinement_mode: str = "none",
        refinement_level: int = 0,
    ) -> None:
        # refinement_level drives the 'fixed' knob; mirror it onto the 'random'
        # knob so the single contract parameter controls either mode. 'none'
        # ignores both. (The old env used only 'fixed'/'none'.)
        self._solver.reset(
            refinement_mode=refinement_mode,
            refinement_level=refinement_level,
            refinement_max_level=refinement_level,
            icase=icase,
        )

    # --- State -------------------------------------------------------------
    def get_state(self) -> SolverState:
        solver = self._solver
        active = np.asarray(solver.active)
        label_mat = solver.label_mat
        n = len(active)
        max_level = solver.max_level

        # Per-element refinement level (reuses the solver's own accessor).
        level = solver.get_active_levels()

        # Neighbor active-positions under periodic BC (matches find_neighbor_index).
        idx = np.arange(n)
        left = (idx - 1) % n
        right = (idx + 1) % n

        # Coarsening sibling as an active position, or -1. Forest-based, matching
        # the old env's _find_sibling: same parent, and the sibling must still be
        # an active leaf (present in `active`).
        sibling = np.full(n, -1, dtype=int)
        for i in range(n):
            elem_id = active[i]
            parent_id = label_mat[elem_id - 1][1]
            if parent_id == 0:  # level 0 — no parent
                continue
            child1, child2 = label_mat[parent_id - 1][2:4]
            sibling_id = child2 if elem_id == child1 else child1
            matches = np.where(active == sibling_id)[0]
            if len(matches) > 0:
                sibling[i] = int(matches[0])

        can_refine = level < max_level

        # Structural coarsenability: sibling is an active leaf AND the post-
        # coarsen mesh still satisfies 2:1 balance. Matches the old env's
        # _can_coarsen condition 4 (parent level vs both external neighbors).
        # Does NOT fold in cascade exclusion or budget — the agent ANDs those in.
        can_coarsen = np.zeros(n, dtype=bool)
        for i in range(n):
            s = int(sibling[i])
            if s == -1:
                continue
            parent_level = int(level[i]) - 1
            lo, hi = (i, s) if i < s else (s, i)
            left_ext = (lo - 1) % n
            right_ext = (hi + 1) % n
            if (
                abs(parent_level - int(level[left_ext])) <= 1
                and abs(parent_level - int(level[right_ext])) <= 1
            ):
                can_coarsen[i] = True

        # Largest CFL-stable dt for the current mesh. Matches the old env's
        # _advance_solver value: courant_max * min_dx / wave_speed, with NO /2
        # margin. (solver.dt carries an extra /2 from _compute_timestep; the
        # contract surfaces the env-equivalent dt the agent sub-steps with.)
        dx_min = float(np.min(np.diff(solver.xelem)))
        stable_dt = solver.courant_max * dx_min / solver.wave_speed
        domain_length = float(solver.xelem[-1] - solver.xelem[0])

        return SolverState(
            element_id=active.astype(int),
            level=np.asarray(level, dtype=int),
            left=left,
            right=right,
            sibling=sibling,
            can_refine=np.asarray(can_refine, dtype=bool),
            can_coarsen=can_coarsen,
            n_active=n,
            max_level=int(max_level),
            stable_dt=stable_dt,
            wave_speed=float(solver.wave_speed),
            domain_length=domain_length,
        )

    # --- Error -------------------------------------------------------------
    def compute_error(self, indicator: str = "zz_style") -> np.ndarray:
        # Raw per-element indicator vector; all normalization/threshold math is
        # agent-side (D-044). 'zz_style' is the primary indicator (D-032).
        return compute_errors(self._solver, indicator)

    # --- Adaptation (sequential; no balance, no rebuild) -------------------
    def adapt_element(self, active_idx: int, mark: int) -> bool:
        if mark == HOLD:
            return False
        if mark == REFINE:
            return bool(self._solver.refine_element(active_idx))
        if mark == COARSEN:
            return bool(self._solver.coarsen_element(active_idx))
        return False  # unknown mark — treat as no-op

    def balance(self) -> set[int]:
        # Cascade set via pre/post active-label diff (balance only ever refines,
        # so post superset of pre). Pulls the old env's _detect_cascade_elements
        # down into the contract layer. Does NOT rebuild operators.
        pre = {int(e) for e in self._solver.active}
        self._solver.balance_mesh(balance=True)
        post = {int(e) for e in self._solver.active}
        return post - pre

    def rebuild(self) -> None:
        self._solver._update_matrices()
        self._solver._update_forcing()

    # --- Adaptation (batch; deferred past Phase 4 — D-043) -----------------
    def apply_marks(self, marks: np.ndarray) -> SolverState:
        raise NotImplementedError(
            "Batch mode (apply_marks) is the D-043 seam, deferred past Phase 4. "
            "The sequential triad adapt_element -> balance -> rebuild is the "
            "parity target; the old env never drove batch mode."
        )

    def predict_post_balance_count(self, marks: np.ndarray) -> int:
        raise NotImplementedError(
            "predict_post_balance_count is part of the deferred batch seam "
            "(D-043); the agent falls back to its committed-mark counter."
        )

    # --- Time advance ------------------------------------------------------
    def step(self, dt: float) -> None:
        self._solver.step(dt=dt)