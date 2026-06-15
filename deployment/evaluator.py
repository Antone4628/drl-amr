"""Deployment evaluator — paired-run orchestration over the DeploymentRunner
(RESTRUCTURE Phase 6, deployment adapter build-order step 7).

A thin layer ABOVE the runner. The runner is single-simulation; the evaluator
runs one or two simulations with FRESH state each and pairs their results, for
metrics that need a baseline (e.g. cost-savings vs a full-refinement run,
DEPLOYMENT_ADAPTER_DESIGN.md §2-§3). It does not touch the per-interval loop.

Freshness: each rollout gets a fresh driver (via build_driver) and a fresh set
of metric instances (via metrics_factory) — metrics are stateful, so primary and
baseline must not share them.

Strategy boundary (v1, capability only): a rollout is run via a decide_fn — the
in-loop agent strategy (model.predict / random), driven through the runner's
run_adaptation_phase. full-refinement and threshold-AMR are DIFFERENT adaptation
strategies (they bypass the queue/observation) and are designed-not-built; they
will plug in as alternative run_strategy implementations, not as decide_fns. The
cost-savings metric pairs the agent run against a full-refinement baseline run.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from contract.solver_snapshot import SolverSnapshot
from deployment.metrics import Metric
from deployment.runner import DeploymentRunner
from drivers.multiround_driver import MultiroundDriver


@dataclass(frozen=True, slots=True)
class RunResult:
    """One rollout: captured frames + finalized metric results (positional,
    aligned to the metrics_factory's list)."""
    snapshots: list[SolverSnapshot]
    metrics: list


@dataclass(frozen=True, slots=True)
class PairedResult:
    """A primary rollout and an optional baseline rollout."""
    primary: RunResult
    baseline: RunResult | None


class Evaluator:
    """Orchestrates one or two fresh-state rollouts over the DeploymentRunner."""

    def __init__(
        self,
        build_driver: Callable[[], MultiroundDriver],
        *,
        time_final: float,
        output_dt: float | None = None,
        burnin: bool = False,
        n_burnin: int = 0,
        capture_remesh: bool = True,
        metrics_factory: Callable[[], list[Metric]] | None = None,
    ):
        self._build_driver = build_driver
        self.time_final = float(time_final)
        self.output_dt = output_dt
        self.burnin = burnin
        self.n_burnin = n_burnin
        self.capture_remesh = capture_remesh
        self._metrics_factory = metrics_factory

    def run_strategy(
        self, decide_fn, *, icase: int, rng: np.random.Generator | None = None
    ) -> RunResult:
        """Run one rollout on a fresh driver with fresh metrics; return its
        snapshots + finalized metric results. (decide_fn is the agent strategy —
        see the strategy-boundary note in the module docstring.)"""
        runner = DeploymentRunner(
            self._build_driver(),
            time_final=self.time_final,
            output_dt=self.output_dt,
            burnin=self.burnin,
            n_burnin=self.n_burnin,
            capture_remesh=self.capture_remesh,
            metrics=self._metrics_factory() if self._metrics_factory else None,
        )
        snaps = runner.run(decide_fn, icase=icase, rng=rng)
        return RunResult(snapshots=snaps, metrics=runner.metric_results)

    def evaluate(
        self,
        primary_decide_fn,
        *,
        icase: int,
        baseline_decide_fn=None,
        rng: np.random.Generator | None = None,
        baseline_rng: np.random.Generator | None = None,
    ) -> PairedResult:
        """Run the primary rollout and, if a baseline strategy is given, a paired
        baseline rollout (fresh state). Returns both for comparative metrics."""
        primary = self.run_strategy(primary_decide_fn, icase=icase, rng=rng)
        baseline = None
        if baseline_decide_fn is not None:
            baseline = self.run_strategy(baseline_decide_fn, icase=icase, rng=baseline_rng)
        return PairedResult(primary=primary, baseline=baseline)