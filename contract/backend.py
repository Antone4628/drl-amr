"""The v2 solver contract's verbs: what the agent core may ask of a backend.

BackendInfo     returned by describe(): what this backend is and offers.
BackendFailure  the one error a backend raises when a verb cannot complete.
SolverBackend   the protocol: nine verbs (SOLVER_CONTRACT_V2 section 6).

This is the 2D contract. The 1D SolverContract in contract/solver_contract.py,
with its sequential adapt_element / balance / rebuild triad, stays beside it
until the 1D backend implements this protocol.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from contract.reports import AdaptReport, IntervalReport, PathSnapshot
from contract.state import SolverState

# The verbs a backend may leave out. An absent one raises NotImplementedError
# and is missing from BackendInfo.optional_verbs.
OPTIONAL_VERBS = frozenset({"dry_run", "reinitialize_ic", "get_snapshot"})


class BackendFailure(RuntimeError):
    """A verb could not complete: the solve threw, or the state is not finite.

    The one error class every backend raises for a failed solve, so the
    environment catches one thing and ends the episode.
    """


@dataclass(frozen=True, slots=True, kw_only=True)
class BackendInfo:
    """What a backend is and what it offers; returned by describe().

    Fields:
        ndim: Spatial dimension (1, 2 or 3).
        nvar: Number of solution variables.
        indicators: Names compute_error accepts.
        reductions: Names of the reductions from nvar to n_comp.
        ic_params: The initial condition's parameters, name -> (low, high).
            reset takes one float per name, within its bounds.
        optional_verbs: Which of OPTIONAL_VERBS this backend implements.
    """

    ndim: int
    nvar: int
    indicators: tuple[str, ...]
    reductions: tuple[str, ...]
    ic_params: Mapping[str, tuple[float, float]]
    optional_verbs: frozenset[str]

    def __post_init__(self) -> None:
        """Check the dimension, the verb names and the parameter bounds."""
        if self.ndim not in (1, 2, 3):
            raise ValueError(f"ndim: {self.ndim}, expected 1, 2 or 3")
        if self.nvar < 1:
            raise ValueError(f"nvar: {self.nvar}, expected at least 1")
        if not isinstance(self.optional_verbs, frozenset):
            raise ValueError("optional_verbs: expected a frozenset of verb names")
        unknown = self.optional_verbs - OPTIONAL_VERBS
        if unknown:
            raise ValueError(f"optional_verbs: unknown names {sorted(unknown)}")
        for name, (low, high) in self.ic_params.items():
            if not low <= high:
                raise ValueError(f"ic_params[{name!r}]: low {low} above high {high}")


@runtime_checkable
class SolverBackend(Protocol):
    """The interface a 2D-capable backend exposes to the agent core.

    One remesh interval is, for each round: compute_error, the agent's
    decisions on the frozen state, apply_marks; then one step_interval.
    Every return is in bulk; nothing is asked per element.

    isinstance(x, SolverBackend) checks that the nine verbs exist by name,
    not their signatures or behavior.
    """

    def describe(self) -> BackendInfo:
        """Return what this backend is and offers. No side effects."""

    def reset(
        self, ic: Mapping[str, float], amr_freq: int, initial_level: int = 1
    ) -> SolverState:
        """Begin an episode and return its first state.

        Args:
            ic: One float per name in BackendInfo.ic_params.
            amr_freq: Base steps per remesh interval; the realized interval
                is read back from the state's interval_T.
            initial_level: The mesh is refined uniformly to this level by
                all-refine adapts, and the initial condition is then sampled
                on the refined mesh.

        mesh_epoch rises.
        """

    def get_state(self) -> SolverState:
        """Return the current state. No side effects."""

    def compute_error(
        self, indicators: Sequence[str], reduction: str
    ) -> dict[str, np.ndarray]:
        """Return one (n_active, n_comp) float64 array per indicator name.

        Stateless: the indicator is an argument, never a backend setting.
        The reduction maps nvar to n_comp. No normalization here; that is
        the agent's.
        """

    def dry_run(self, marks: np.ndarray) -> tuple[int, np.ndarray]:
        """Predict what apply_marks(marks) would realize, changing nothing.

        Optional (BackendInfo.optional_verbs).

        Returns:
            The predicted n_active, and the predicted realized flags,
            (n_active,) int8.
        """

    def apply_marks(self, marks: np.ndarray) -> tuple[SolverState, AdaptReport]:
        """Apply one round of marks and return the new state and the report.

        marks is (n_active,), any integer dtype, values COARSEN, HOLD or
        REFINE, on the current ordering. Realization: refine, then coarsen
        complete families only, then balance; one level per call; ineligible
        marks ignored. The solution is transferred. mesh_epoch rises by one;
        time does not advance. A round of all HOLD still adapts and rebuilds.
        """

    def step_interval(
        self, accumulate: Sequence[str], cadence: int, aux: Sequence[str] = ()
    ) -> IntervalReport:
        """Advance exactly one remesh interval. No mesh change.

        Args:
            accumulate: Indicator names whose maximum over the interval is
                returned in IntervalReport.acc.
            cadence: Sample the accumulated indicators every this many
                solver steps.
            aux: Names of training-only fields to return in
                IntervalReport.aux.

        Raises:
            BackendFailure: If the solve throws or the state is not finite.
        """

    def get_snapshot(self, write: bool = False) -> PathSnapshot:
        """Return the latest frame on disk, writing one first if write.

        Optional (BackendInfo.optional_verbs).
        """

    def reinitialize_ic(self) -> None:
        """Re-sample the initial condition on the current mesh, time to zero.

        Optional (BackendInfo.optional_verbs). Training does not call it.
        """