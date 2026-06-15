"""Shared front-end engine for multi-round sequential DRL-AMR (RESTRUCTURE Phase 6).

The single re-entrant state machine that drives an AgentCore + SolverContract
through an episode (N_remesh intervals x max_level rounds x all active elements).
It is the one place the round/interval/element loop, the solver-advance +
max-over-interval error accumulation (D-046), and the dual-reward combination
live.

Both front-end shells are *thin wrappers* over this driver (D-047):
  - the gym.Env (envs/) calls driver.step(action) re-entrantly — SB3 owns the loop;
  - the deployment adapter (deployment/) drives the same driver.step in an
    internal loop, calling model.predict and discarding the reward.
Behavioral env<->adapter identity is therefore by construction; the old "must
mirror the env exactly" instruction is retired.

Mode-aware (D-043). Per-element execution dispatches on core.mode:
  sequential (default, parity target): core.execute_now per element (one rebuild
    each); the driver folds the returned cascade into consumed_elements and
    re-reads state before the next element.
  batch (post-parity, P-015; structurally present, exercised at Phase 7.5):
    core.stage_flag per element, then core.apply_round once at round end (one
    global balance + rebuild). apply_marks is currently a stub, so batch does
    not yet run end to end.

The sequential path is extracted verbatim from dg_amr_env_multiround.py's
step / reset / _advance_queue / _advance_solver / _start_new_interval; the D-038
parity target is this driver in sequential mode reproducing that env.
"""
from __future__ import annotations

import numpy as np

from agent.core import MODE_BATCH, MODE_SEQUENTIAL, AgentCore
from agent.masking import ACTION_COARSEN, ACTION_HOLD, ACTION_REFINE
from agent.observation import OBS_DIM
from contract.element_state import SolverState

_ACTION_LABELS = {ACTION_COARSEN: "coarsen", ACTION_HOLD: "hold", ACTION_REFINE: "refine"}

# Stage 1A default IC pool (D-026 / env default): includes ICs with negative
# regions to break the u > 0 spurious correlation found in the thesis.
_DEFAULT_IC_POOL = [1, 10, 12, 13, 14, 15, 16]


class MultiroundDriver:
    """Re-entrant episode driver over an AgentCore + SolverContract.

    Holds the episode-orchestration state (interval/round/queue position,
    interval-fixed thresholds, max-over-interval accumulator) and delegates
    every per-element decision and the reward math to the core. The backend is
    reached via core.contract for the driver-owned operations only: reset,
    get_state, and the advance step (D-046).
    """

    def __init__(
        self,
        core: AgentCore,
        *,
        n_remesh: int = 4,
        step_domain_fraction: float = 0.05,
        initial_refinement_level: int = 0,
        ic_pool: list[int] | None = None,
        verbosity: int = 0,
    ):
        """Bind the core and the episode-orchestration parameters.

        Args:
            core: AgentCore (already bound to the SolverContract backend). The
                driver reads the backend via core.contract for reset/get_state/
                step; all agent decisions + reward go through core methods.
            n_remesh: remesh intervals per episode (D-027).
            step_domain_fraction: fraction of the domain the wave traverses per
                interval; sets T = fraction * domain_length / wave_speed.
            initial_refinement_level: uniform refinement passes at episode start
                (0 = base mesh). Overridable per-episode via options.
            ic_pool: icase pool for IC sampling; defaults to the Stage 1A pool.
            verbosity: 0 silent / 1 summary / 2 detail.
        """
        self.core = core
        self.contract = core.contract  # backend, for driver-owned ops (D-046)
        self.n_remesh = n_remesh
        self.step_domain_fraction = step_domain_fraction
        self.initial_refinement_level = initial_refinement_level
        self.ic_pool = ic_pool if ic_pool is not None else list(_DEFAULT_IC_POOL)
        self.verbosity = verbosity

        # RNG for IC sampling. The gym wrapper passes its Gymnasium-seeded
        # np_random into reset(rng=...) so the IC stream is bit-identical to
        # the env (D-038 parity).
        self._rng = np.random.default_rng()

        # Episode state — set properly in reset().
        self.remesh_step = 0
        self.round_number = 1
        self.queue: list[int] = []
        self.queue_position = 0
        self.current_element_idx = 0
        self.consumed_elements: set[int] = set()
        self.e_max = 0.0          # interval-fixed upper threshold (D-021)
        self.e_min = 0.0          # interval-fixed lower threshold (D-021)
        self.max_interval_errors: np.ndarray | None = None  # D-008 accumulator
        self._episode_steps = 0
        self._total_episodes = 0

    # --- Public surface (consumed by both shells) --------------------------

    def reset(self, *, options: dict | None = None, rng=None) -> tuple[np.ndarray, dict]:
        """Begin a new episode and return the first element's (obs, info).

        The gym wrapper should call super().reset(seed=seed) itself and pass its
        seeded generator here as rng, so IC sampling reproduces the env's RNG
        stream exactly (the IC choice; D-029 pre-advance removed per D-049).

        options keys: 'icase' (force IC), 'refinement_level' (override).
        """
        if rng is not None:
            self._rng = rng

        # --- IC + initial refinement -----------------------------------------
        if options is not None and "icase" in options:
            icase = options["icase"]
        else:
            icase = int(self._rng.choice(self.ic_pool))

        if options is not None:
            refinement_level = options.get("refinement_level", self.initial_refinement_level)
        else:
            refinement_level = self.initial_refinement_level

        if refinement_level > 0:
            self.contract.reset(icase=icase, refinement_mode="fixed", refinement_level=refinement_level)
        else:
            self.contract.reset(icase=icase)

        self._log(1, f"\n{'=' * 60}\n  EPISODE {self._total_episodes + 1} START (icase={icase})\n{'=' * 60}")

        # --- Episode + first-interval state (thresholds + queue via begin_interval) ---
        self.remesh_step = 0
        self._episode_steps = 0
        state, errors = self.begin_interval()
        self.max_interval_errors = np.zeros(state.n_active)
        obs = self.core.observe(state, errors, self.current_element_idx, self.round_number)

        info = {
            "icase": icase,
            "n_active": state.n_active,
            "e_max": self.e_max,
            "e_min": self.e_min,
            "resource_usage": state.n_active / self.core.element_budget,
        }
        self._total_episodes += 1
        return obs, info

    def action_masks(self) -> np.ndarray:
        """Valid-action mask for the current element (MaskablePPO interface)."""
        state = self.contract.get_state()
        return self.core.action_mask(state, self.current_element_idx, self.consumed_elements)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Execute one element decision; advance state; return SB3-style tuple."""
        # 1. Pre-action snapshot (error captured BEFORE the mesh changes).
        state = self.contract.get_state()
        errors = self.core.compute_error()
        pre_action_error = float(errors[self.current_element_idx])
        pre_action_elem_id = int(state.element_id[self.current_element_idx])
        pre_action_n_active = state.n_active

        # 2. Execute (mode-aware) + local shaping reward.
        exec_result = self._commit_element(self.current_element_idx, action)
        r_local = self.core.local_reward(pre_action_error, action, self.e_max, self.e_min)

        post_state = self.contract.get_state()
        n_active_post = post_state.n_active

        # 3. Advance the queue (round transitions handled internally).
        queue_result = self._advance_queue(post_state)
        transition = queue_result["transition"]

        # 4. Interval boundary: solver advance + global retrospective reward.
        r_global = 0.0
        solver_info: dict = {}
        if transition in ("interval", "done"):
            solver_info = self._advance_solver()
            adv_state = self.contract.get_state()  # mesh fixed during advance
            r_global = self.core.global_reward(
                self.max_interval_errors, adv_state.level, self.e_max, self.e_min, adv_state.max_level
            )

        reward = self.core.combine_reward(r_local, r_global)

        # 5. Post-transition setup.
        terminated = False
        if transition == "interval":
            self._start_new_interval()
        elif transition == "done":
            terminated = True

        # 6. Next observation (zeros at episode end).
        if terminated:
            obs = np.zeros(OBS_DIM, dtype=np.float32)
        else:
            state_now = self.contract.get_state()
            errors_now = self.core.compute_error()
            obs = self.core.observe(state_now, errors_now, self.current_element_idx, self.round_number)

        self._episode_steps += 1

        info = {
            "element_id": pre_action_elem_id,
            "action": exec_result["action_taken"],
            "pre_action_error": pre_action_error,
            "n_active_pre": pre_action_n_active,
            "n_active_post": n_active_post,
            "n_cascade": len(exec_result["cascade"]),
            "resource_usage": n_active_post / self.core.element_budget,
            "r_local": r_local,
            "r_global": r_global,
            "reward": reward,
            "transition": transition,
            "queue_skipped": queue_result["skipped"],
            "remesh_step": self.remesh_step,
            "round_number": self.round_number,
            "episode_steps": self._episode_steps,
        }
        if solver_info:
            info["solver_T"] = solver_info["T"]
            info["solver_n_steps"] = solver_info["n_steps"]
            info["solver_max_error_peak"] = solver_info["max_error_peak"]

        return obs, reward, terminated, False, info

    def begin_interval(self) -> tuple[SolverState, np.ndarray]:
        """Set up an interval from the current (post-advance) mesh: recompute
        interval-fixed thresholds, reset the round counter and cascade-exclusion
        set, build the round-1 queue, and resolve the first element (D-021).

        Shared by reset() (first interval), _start_new_interval() (training
        transitions), and the deployment runner (each interval). Does NOT
        advance the solver or touch remesh_step / episode counters — the caller
        owns those. Returns (state, errors) for callers that need them (e.g. the
        first observation).
        """
        errors = self.core.compute_error()
        self.e_max, self.e_min = self.core.compute_thresholds(errors)
        self.round_number = 1
        self.consumed_elements = set()
        state = self.contract.get_state()
        self.core.begin_round(state)
        self.queue = self.core.build_queue(state, errors, self.e_max, self.e_min)
        self.queue_position = 0
        self.current_element_idx = self._resolve(state, self.queue[0])
        return state, errors

    def run_adaptation_phase(self, decide_fn) -> dict:
        """Run one interval's adaptation — max_level rounds over the current
        mesh — with NO solver advance (the deployment runner advances between
        phases). Each element's action comes from decide_fn(obs, masks) -> int.

        A second loop-driver over the same per-element internals step() uses
        (_commit_element + _advance_queue, mode-aware), so train/deploy
        adaptation is identical by construction (D-047). Precondition:
        begin_interval() has set up the queue + first element. Stops at the
        interval boundary; treats 'interval' and 'done' identically — whether
        more intervals follow is the caller's decision (deployment counts
        intervals by physical time, not remesh_step).

        Returns {'n_decisions', 'transition'}.
        """
        n_decisions = 0
        while True:
            state = self.contract.get_state()
            errors = self.core.compute_error()
            obs = self.core.observe(state, errors, self.current_element_idx, self.round_number)
            masks = self.core.action_mask(state, self.current_element_idx, self.consumed_elements)
            action = decide_fn(obs, masks)
            self._commit_element(self.current_element_idx, action)
            n_decisions += 1
            queue_result = self._advance_queue(self.contract.get_state())
            if queue_result["transition"] in ("interval", "done"):
                return {"n_decisions": n_decisions, "transition": queue_result["transition"]}

    # --- Internal driver mechanics -----------------------------------------

    def _commit_element(self, active_idx: int, action: int) -> dict:
        """Sequential: execute now + fold cascade. Batch: stage the mark only."""
        if self.core.mode == MODE_SEQUENTIAL:
            result = self.core.execute_now(active_idx, action)
            self.consumed_elements.update(result["cascade"])
            return result
        # Batch (post-parity): no mesh change here; committed at round end.
        self.core.stage_flag(active_idx, action)
        return {"action_taken": _ACTION_LABELS[action], "cascade": set(), "changed": False}

    def _advance_queue(self, state) -> dict:
        """Find the next element, handling round transitions internally.

        Returns {'transition': 'element'|'interval'|'done', 'skipped': int}.
        Round transitions (queue exhausted, more rounds remain) are absorbed
        here and surface as 'element' (the first element of the next round) —
        matching the env. Interval/done are signalled back to step() to run the
        solver advance + global reward. In batch mode the staged round is
        committed (core.apply_round) at each round/interval boundary.
        """
        skipped = 0
        while True:
            self.queue_position += 1
            while self.queue_position < len(self.queue):
                elem_id = self.queue[self.queue_position]
                matches = np.where(state.element_id == elem_id)[0]
                if len(matches) > 0:
                    self.current_element_idx = int(matches[0])
                    return {"transition": "element", "skipped": skipped}
                skipped += 1  # element consumed (refined/coarsened/cascade) — skip
                self.queue_position += 1

            # Queue exhausted.
            if self.round_number < state.max_level:
                if self.core.mode == MODE_BATCH:
                    state = self.core.apply_round()  # commit staged round (one rebuild)
                self.round_number += 1
                self.consumed_elements = set()
                self.core.begin_round(state)
                errors = self.core.compute_error()
                self.queue = self.core.build_queue(state, errors, self.e_max, self.e_min)
                self.queue_position = -1  # incremented to 0 at loop top
                skipped = 0
                continue

            # All rounds complete for this interval.
            if self.core.mode == MODE_BATCH:
                state = self.core.apply_round()  # commit final staged round
            if self.remesh_step + 1 >= self.n_remesh:
                return {"transition": "done", "skipped": skipped}
            return {"transition": "interval", "skipped": skipped}

    def advance(self, duration: float, substep_callback=None) -> dict:
        """Advance the PDE by `duration` with exact-landing CFL sub-stepping.

        The generalized advance both shells reuse (D-047): the mesh is fixed
        for the whole call, so dt = state.stable_dt is constant, and the final
        sub-step is clamped so the total lands exactly on `duration` (no
        overshoot). Mesh topology and operators are untouched.

        substep_callback (optional, zero-arg) is invoked once at the start
        (before any step, i.e. at the t_start state) and once after each
        completed sub-step. Training passes an error-accumulation callback
        (D-008 max-over-interval); deployment passes a frame-capture callback.
        It is opt-in precisely so a deployment advance pays NO per-sub-step
        error recompute (ZZ is expensive — DEPLOYMENT_ADAPTER_DESIGN.md §3).

        dt note (D-050): uses state.stable_dt (actual-mesh, no /2) — the current
        training value. The fixed worst-case-dt switch is a separate, isolated
        change (build-order step 8); keeping it out of here makes this advance
        behavior-neutral.
        """
        state = self.contract.get_state()
        dt = state.stable_dt  # actual-mesh, no /2 (carry-forward 2; D-050 step 8)
        n_steps = max(1, int(np.ceil(duration / dt)))

        if substep_callback is not None:
            substep_callback()  # t_start hook (e.g. the t_tau error snapshot)

        time_advanced = 0.0
        for _ in range(n_steps):
            step_dt = min(dt, duration - time_advanced)
            if step_dt <= 1e-15:
                break
            self.contract.step(dt=step_dt)
            time_advanced += step_dt
            if substep_callback is not None:
                substep_callback()

        return {"T": duration, "dt": dt, "n_steps": n_steps}

    def _advance_solver(self) -> dict:
        """Advance the PDE by one interval T, accumulating max-over-interval
        errors (D-008). The driver owns this loop (D-046); the mesh is fixed
        throughout, so stable_dt and the error array stay valid. Thin wrapper
        over advance(): the D-008 accumulation is the per-sub-step callback."""
        state = self.contract.get_state()
        T = self.step_domain_fraction * state.domain_length / state.wave_speed
        self.max_interval_errors = np.zeros(state.n_active)

        def _accumulate() -> None:
            errors = self.core.compute_error()
            self.max_interval_errors = np.maximum(self.max_interval_errors, errors)

        adv = self.advance(T, substep_callback=_accumulate)

        return {
            "T": adv["T"],
            "dt": adv["dt"],
            "n_steps": adv["n_steps"],
            "max_error_peak": float(np.max(self.max_interval_errors)),
        }

    def _start_new_interval(self) -> None:
        """Set up the next interval AFTER the advance (thresholds from the
        post-advance distribution, D-021). Interval setup is shared with reset()
        and the deployment adaptation phase via begin_interval()."""
        self.remesh_step += 1
        self.begin_interval()
        self._log(1, f"\n  Remesh interval {self.remesh_step + 1}/{self.n_remesh} "
                     f"(e_max={self.e_max:.6f}, e_min={self.e_min:.6f})")

    @staticmethod
    def _resolve(state, elem_id: int) -> int:
        """Resolve a stable element ID to its current active-array position."""
        return int(np.where(state.element_id == elem_id)[0][0])

    def _log(self, level: int, msg: str) -> None:
        if self.verbosity >= level:
            print(msg)