"""Agent core for multi-round sequential DRL-AMR (RESTRUCTURE Phase 5).

The single, backend-agnostic decision core extracted from the duplicated
per-element logic in the training env (dg_amr_env_multiround.py) and the
deployment adapter (multiround_adapter.py). It binds the hyperparameters and a
SolverContract, and exposes the agent's operations — threshold/observation/mask/
queue/reward — by delegating to the pure functions in this package. It is
policy-agnostic: the driver supplies the chosen action (SB3 in training via
env.step(action); model.predict in deployment), and the core executes it.

Iteration-mode seam (D-043). One core drives the contract in either granularity;
the driving loop (env step() vs. adapter internal loop) is orthogonal:

  Sequential (default, parity target): per element the driver calls
    execute_now(idx, action) -> adapt_element -> balance -> rebuild, takes the
    returned cascade set into its consumed_elements, and re-reads the state
    before the next element.

  Batch (post-parity, P-015): the driver reads state once at round start, calls
    stage_flag(idx, action) per element to accumulate a marks array, then
    apply_round() -> contract.apply_marks (one global balance, one rebuild).

The committed-budget counter (D-043) tracks the net element delta of committed
decisions (cheap arithmetic, decoupled from rebuilds) to preserve live
budget-awareness under batch mode (where there is no per-element rebuild to read
n_active from). In sequential mode the live state.n_active supersedes it.
"""
from __future__ import annotations

import numpy as np

from agent.masking import (
    ACTION_COARSEN,
    ACTION_HOLD,
    ACTION_REFINE,
    ACTION_TO_MARK,
)
from agent.masking import action_masks as _action_masks
from agent.normalization import alpha_thresholds as _alpha_thresholds
from agent.observation import build_observation as _build_observation
from agent.queue import build_queue as _build_queue
from agent.reward import global_reward as _global_reward
from agent.reward import local_reward as _local_reward
from contract.element_state import SolverState
from contract.solver_contract import HOLD, SolverContract

MODE_SEQUENTIAL = "sequential"
MODE_BATCH = "batch"

_ACTION_LABELS = {ACTION_COARSEN: "coarsen", ACTION_HOLD: "hold", ACTION_REFINE: "refine"}


class AgentCore:
    """Backend-agnostic agent core binding hyperparameters to a SolverContract."""

    def __init__(
        self,
        contract: SolverContract,
        *,
        alpha: float = 0.1,
        beta: float = 1.2,
        p_ur: float = 10.0,
        p_or: float = 5.0,
        p_cr: float = 2.0,
        lambda_local: float = 0.1,
        lambda_global: float = 1.0,
        element_budget: int = 30,
        error_indicator: str = "zz_style",
        mode: str = MODE_SEQUENTIAL,
        refine_arity: int = 2,
    ):
        """Bind the contract and the agent hyperparameters.

        Args:
            contract: The solver backend the core drives (SolverContract).
            alpha, beta: Error-normalization / threshold parameters.
            p_ur, p_or, p_cr: Reward weights (under/over penalty, coarsen reward).
            lambda_local, lambda_global: Dual-reward weighting.
            element_budget: Soft budget for the resource_usage observation.
            error_indicator: Key passed to contract.compute_error.
            mode: MODE_SEQUENTIAL (default, parity target) or MODE_BATCH.
            refine_arity: Children produced per refine (2 in 1D, 4 in 2D, 8 in
                3D). The sole dimension-specific quantity; injected so the core
                logic stays dimension-agnostic. Drives the committed-budget
                counter's net delta (refine_arity - 1 per refine/coarsen).
        """
        if mode not in (MODE_SEQUENTIAL, MODE_BATCH):
            raise ValueError(f"unknown mode {mode!r}")

        self._contract = contract
        self.alpha = alpha
        self.beta = beta
        self.p_ur = p_ur
        self.p_or = p_or
        self.p_cr = p_cr
        self.lambda_local = lambda_local
        self.lambda_global = lambda_global
        self.element_budget = element_budget
        self.error_indicator = error_indicator
        self.mode = mode
        self.refine_arity = refine_arity

        # Round-scoped state (set by begin_round).
        self._committed_count = 0
        self._staged_marks: np.ndarray | None = None

    @property
    def contract(self) -> SolverContract:
        """The bound solver backend (read-only). The driver uses it for the
        driver-owned operations that are not agent decisions: lifecycle
        (reset), state reads (get_state), and the solver-advance loop (step) —
        per D-046 the advance + max-interval accumulation live in the driver."""
        return self._contract

    # --- Bound delegations to the pure functions ---------------------------

    def compute_error(self) -> np.ndarray:
        """Raw per-element error vector from the backend (binds error_indicator)."""
        return self._contract.compute_error(self.error_indicator)

    def compute_thresholds(self, errors: np.ndarray) -> tuple[float, float]:
        """(e_max, e_min) for the current error distribution (binds alpha, beta)."""
        return _alpha_thresholds(errors, self.alpha, self.beta)

    def observe(
        self, state: SolverState, errors: np.ndarray, active_idx: int, round_number: int
    ) -> np.ndarray:
        """Observation vector for one element (binds alpha, element_budget)."""
        return _build_observation(
            state, errors, active_idx, self.alpha, self.element_budget, round_number
        )

    def action_mask(
        self, state: SolverState, active_idx: int, consumed_elements: set[int]
    ) -> np.ndarray:
        """Valid-action mask for one element."""
        return _action_masks(state, active_idx, consumed_elements)

    def build_queue(
        self, state: SolverState, errors: np.ndarray, e_max: float, e_min: float
    ) -> list[int]:
        """Priority-sorted element-ID queue for one round."""
        return _build_queue(state, errors, e_max, e_min)

    def local_reward(self, e_k: float, action: int, e_max: float, e_min: float) -> float:
        """Local shaping reward (binds p_ur, p_or, p_cr)."""
        return _local_reward(e_k, action, e_max, e_min, self.p_ur, self.p_or, self.p_cr)

    def global_reward(
        self,
        max_interval_errors: np.ndarray,
        levels: np.ndarray,
        e_max: float,
        e_min: float,
        max_level: int,
    ) -> float:
        """Global retrospective reward (binds p_ur, p_or)."""
        return _global_reward(
            max_interval_errors, levels, e_max, e_min, max_level, self.p_ur, self.p_or
        )

    def combine_reward(self, r_local: float, r_global: float) -> float:
        """Weighted dual-reward combination (binds lambda_local, lambda_global)."""
        return self.lambda_local * r_local + self.lambda_global * r_global

    # --- Iteration-mode seam (D-043) ---------------------------------------

    def begin_round(self, state: SolverState) -> None:
        """Reset per-round state: committed counter and (batch) the marks buffer."""
        self._committed_count = state.n_active
        if self.mode == MODE_BATCH:
            self._staged_marks = np.full(state.n_active, HOLD, dtype=int)

    def execute_now(self, active_idx: int, action: int) -> dict:
        """Sequential execute: adapt_element -> balance -> rebuild for one element.

        Hold short-circuits with no contract calls. Returns the cascade set from
        balance() for the driver to fold into its consumed_elements (the driver
        owns that set, per the extraction's canonical choice).

        Returns:
            dict with 'action_taken' (str), 'cascade' (set[int]), 'changed' (bool).
        """
        if self.mode != MODE_SEQUENTIAL:
            raise RuntimeError("execute_now is only valid in sequential mode")

        label = _ACTION_LABELS[action]
        if action == ACTION_HOLD:
            return {"action_taken": label, "cascade": set(), "changed": False}

        mark = ACTION_TO_MARK[action]
        changed = self._contract.adapt_element(active_idx, mark)
        cascade = self._contract.balance()
        self._contract.rebuild()
        self._committed_count += self._committed_delta(action)

        return {"action_taken": label, "cascade": cascade, "changed": changed}

    def stage_flag(self, active_idx: int, action: int) -> None:
        """Batch stage: record one element's mark, no mesh change."""
        if self.mode != MODE_BATCH:
            raise RuntimeError("stage_flag is only valid in batch mode")
        self._staged_marks[active_idx] = ACTION_TO_MARK[action]
        self._committed_count += self._committed_delta(action)

    def apply_round(self) -> SolverState:
        """Batch apply: one pass (global balance + single rebuild) via the contract.

        NOTE: contract.apply_marks is a post-parity stub (P-015) and currently
        raises NotImplementedError on the python_1d backend.
        """
        if self.mode != MODE_BATCH:
            raise RuntimeError("apply_round is only valid in batch mode")
        return self._contract.apply_marks(self._staged_marks)

    @property
    def committed_count(self) -> int:
        """Net committed element count this round (cheap, rebuild-decoupled)."""
        return self._committed_count

    def _committed_delta(self, action: int) -> int:
        """Net element delta for one committed decision (refine_arity - 1 in 1D)."""
        if action == ACTION_REFINE:
            return self.refine_arity - 1
        if action == ACTION_COARSEN:
            return -(self.refine_arity - 1)
        return 0