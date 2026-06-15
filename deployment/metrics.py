"""Deployment metrics — pluggable performance measures over a rollout
(RESTRUCTURE Phase 6, deployment adapter build-order step 7).

A metric is any object with update(snapshot) (called once per captured frame —
i.e. at the output cadence, inside DeploymentRunner._capture) and finalize()
(terminal result, computed once at end of run). The runner holds a list of them;
the evaluator (evaluator.py) orchestrates paired runs when a metric needs a
baseline (cost-vs-full-refinement).

Capability only (step 7): this defines the plug interface. Concrete metric
definitions (headline accuracy / cost-savings) are deferred per
DEPLOYMENT_ADAPTER_DESIGN.md §3. Metrics are STATEFUL across a single run — pass
fresh instances per run() (the evaluator does this via a metrics_factory).
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from contract.solver_snapshot import SolverSnapshot


@runtime_checkable
class Metric(Protocol):
    """A pluggable rollout metric, stateful across one run."""

    def update(self, snapshot: SolverSnapshot) -> None:
        """Accumulate from one captured frame (called at the output cadence)."""
        ...

    def finalize(self) -> Any:
        """Return the terminal metric result (called once at end of run)."""
        ...