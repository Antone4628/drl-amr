"""Deployment runner — the PDE-solver front-end (RESTRUCTURE Phase 6, deployment
adapter build-order step 5).

Runs a full simulation to a physical end time over the shared MultiroundDriver
(D-047) using any decide_fn (a trained policy or random), collecting a lean
per-frame SolverSnapshot list for the visualization layer (step 6). Env-free, no
reward.

Shape (DEPLOYMENT_ADAPTER_DESIGN.md §2-§3):
  init: driver.reset (IC + initial_refinement_level), then optional burn-in —
        repeat [run_adaptation_phase -> reinitialize_ic -> begin_interval] to
        refine the mesh to the IC before timestepping (D-049: burn-in replaces
        the retired pre-advance; deployment assumes pre_advance is off).
  loop: per interval, run_adaptation_phase(decide_fn) (rounds, no advance) ->
        advance(T_interval) with sim-time output cadence + exact landing ->
        begin_interval. Runs to time_final (final interval shortened to land
        exactly).

Output cadence is sim-time (constant playback speed regardless of mesh
adaptation): frames at output_dt multiples, plus time_final, plus remesh
boundaries when capture_remesh. output_dt None/<=0 is the every-CFL high-def
limit. Snapshots are lean (no per-frame error recompute — ZZ is expensive).
"""
from __future__ import annotations

import numpy as np

from contract.solver_snapshot import SolverSnapshot
from drivers.multiround_driver import MultiroundDriver


def random_decide_fn(rng: np.random.Generator):
    """decide_fn drawing a uniform action among the currently-valid ones."""
    def decide(obs, masks):
        valid = np.flatnonzero(masks)
        return int(rng.choice(valid))
    return decide


def model_decide_fn(model, *, deterministic: bool = True):
    """decide_fn from a trained MaskablePPO model (mask-aware predict)."""
    def decide(obs, masks):
        action, _ = model.predict(obs, action_masks=masks, deterministic=deterministic)
        return int(action)
    return decide


class DeploymentRunner:
    """Single-simulation deployment runner over a MultiroundDriver."""

    _EPS = 1e-9

    def __init__(
        self,
        driver: MultiroundDriver,
        *,
        time_final: float,
        output_dt: float | None = None,
        burnin: bool = False,
        n_burnin: int = 0,
        capture_remesh: bool = True,
    ):
        self.driver = driver
        self.contract = driver.contract
        self.time_final = float(time_final)
        self.output_dt = output_dt
        self.burnin = burnin
        self.n_burnin = n_burnin
        self.capture_remesh = capture_remesh
        self.snapshots: list[SolverSnapshot] = []

    # --- Public --------------------------------------------------------------
    def run(
        self, decide_fn, *, icase: int, rng: np.random.Generator | None = None
    ) -> list[SolverSnapshot]:
        """Run the full simulation to time_final and return the snapshot list."""
        self.snapshots = []
        self._init(decide_fn, icase, rng)

        state = self.contract.get_state()
        t_interval = self.driver.step_domain_fraction * state.domain_length / state.wave_speed

        t = 0.0
        self._capture()  # initial post-burn-in frame (t=0), always

        while t < self.time_final - self._EPS:
            # Adapt this interval's mesh (rounds, no advance), then advance it.
            self.driver.run_adaptation_phase(decide_fn)
            t1 = min(t + t_interval, self.time_final)
            t = self._advance_interval(t, t1)
            is_final = t >= self.time_final - self._EPS
            if self.capture_remesh and not is_final:
                self._capture()  # remesh-boundary frame (mesh about to change)
            if not is_final:
                self.driver.begin_interval()  # set up next interval (D-021)

        return self.snapshots

    # --- Internal ------------------------------------------------------------
    def _init(self, decide_fn, icase: int, rng) -> None:
        """Reset to IC (+ initial_refinement_level), then optional burn-in.

        Deployment assumes pre-advance is OFF (config pre_advance_range=[0,0];
        D-049). Burn-in is the deployment-time mesh warm-up: each pass refines on
        the current solution, re-seeds the exact IC on the finer mesh, and
        re-sets up the interval. The post-burn-in mesh determines the whole
        deterministic rollout, so it is first-class, not optional polish.
        """
        if rng is None:
            rng = np.random.default_rng()
        self.driver.reset(options={"icase": icase}, rng=rng)
        for _ in range(self.n_burnin if self.burnin else 0):
            self.driver.run_adaptation_phase(decide_fn)
            self.contract.reinitialize_ic()
            self.driver.begin_interval()

    def _advance_interval(self, t0: float, t1: float) -> float:
        """Advance [t0, t1] over the (fixed) current mesh, capturing at sim-time
        output instants with exact landing. Returns t1."""
        od = self.output_dt
        if not od or od <= 0:
            # Every-CFL high-def limit: capture after each sub-step.
            self.driver.advance(t1 - t0, substep_callback=self._capture)
            return t1

        # Interior output instants in (t0, t1), then the interval end.
        targets: list[tuple[float, bool]] = []
        j = int(np.floor(t0 / od + self._EPS)) + 1
        while j * od < t1 - self._EPS:
            targets.append((j * od, True))
            j += 1
        on_grid = abs(t1 / od - round(t1 / od)) < self._EPS
        targets.append((t1, on_grid or t1 >= self.time_final - self._EPS))

        t = t0
        for tgt, capture_here in targets:
            self.driver.advance(tgt - t)
            t = tgt
            if capture_here:
                self._capture()
        return t

    def _capture(self) -> None:
        """Append a lean snapshot, de-duplicating same-time captures (e.g. a
        remesh boundary coinciding with an output instant)."""
        snap = self.contract.get_snapshot()
        if self.snapshots and abs(self.snapshots[-1].time - snap.time) < self._EPS:
            return
        self.snapshots.append(snap)