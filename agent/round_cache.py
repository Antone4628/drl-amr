"""The round cache: one read of the backend per adaptation round.

Within a batch round the mesh does not change (SOLVER_CONTRACT_V2 section 7),
so the state and the indicator arrays are read once and held. read_round
makes that one read; RoundCache holds it. Everything the 2D agent core does
within a round takes a RoundCache and no backend, so it cannot call one.

This is the 2D core's first module (AGENT_2D_ROADMAP item 2.1). It stands
beside the sequential core (core.py, masking.py, observation.py, queue.py),
which reads the 1D SolverState and is not on this path.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from contract._arrays import check_and_freeze
from contract.backend import BackendFailure, SolverBackend
from contract.state import SolverState


def _names(indicators: Sequence[str]) -> tuple[str, ...]:
    """Return the indicator names as a tuple, or reject them.

    Raises:
        ValueError: Starting with "indicators", for a bare string, an empty
            list or a repeated name.
    """
    if isinstance(indicators, str):
        raise ValueError(f"indicators: expected a sequence of names, got the string {indicators!r}")
    names = tuple(indicators)
    if not names:
        raise ValueError("indicators: expected at least one name")
    if len(set(names)) != len(names):
        raise ValueError(f"indicators: repeated names in {list(names)}")
    return names


@dataclass(frozen=True, slots=True, kw_only=True)
class RoundCache:
    """One round's read of the backend: a state and its indicator arrays.

    A snapshot of one mesh_epoch. Row p of every array belongs to the element
    at position p of state, and to no element of any later state: the order
    changes at every adapt. Every array is checked against the state at
    construction and made read-only, and the mapping is sealed.

    Fields:
        state: The v2 SolverState the arrays were computed on.
        errors: Indicator name -> (n_active, n_comp) float64 array.
        indicators: The names, in the order they were asked for.
        reduction: The reduction the arrays were computed under.
    """

    state: SolverState
    errors: Mapping[str, np.ndarray]
    indicators: tuple[str, ...]
    reduction: str

    def __post_init__(self) -> None:
        """Check the arrays against the state, freeze them, seal the mapping.

        Raises:
            ValueError: Naming the indicator, for a missing or extra name, a
                wrong dtype or a wrong shape.
            BackendFailure: If an array holds a non-finite value.
        """
        names = _names(self.indicators)
        if set(self.errors) != set(names):
            raise ValueError(f"errors: names {sorted(self.errors)}, expected {sorted(names)}")

        n = self.state.n_active
        first = self.errors[names[0]]
        if not isinstance(first, np.ndarray) or first.ndim != 2 or first.shape[1] < 1:
            raise ValueError(f"errors[{names[0]!r}]: expected a 2-D numpy array with at least one column")
        n_comp = first.shape[1]
        for name in names:
            check_and_freeze(f"errors[{name!r}]", self.errors[name], np.float64, (n, n_comp))
            if not np.isfinite(self.errors[name]).all():
                raise BackendFailure(f"errors[{name!r}]: not finite")

        # A frozen dataclass forbids assignment, in __post_init__ too; this
        # is the one sanctioned way around it.
        object.__setattr__(self, "indicators", names)
        object.__setattr__(self, "errors", MappingProxyType({name: self.errors[name] for name in names}))

    @property
    def n_active(self) -> int:
        """Number of active elements in the round's state."""
        return self.state.n_active

    @property
    def mesh_epoch(self) -> int:
        """The mesh_epoch this cache is a snapshot of."""
        return self.state.mesh_epoch

    @property
    def n_comp(self) -> int:
        """Columns per indicator array, set by the reduction."""
        return self.errors[self.indicators[0]].shape[1]


def read_round(
    backend: SolverBackend, state: SolverState, indicators: Sequence[str], reduction: str
) -> RoundCache:
    """Read the indicator arrays for one round and pair them with its state.

    Makes exactly one backend call, compute_error, and never get_state: the
    state is the one reset or apply_marks just returned. Passing an older
    state pairs the arrays with the wrong mesh, and when the element count
    happens to match, nothing here can tell.

    Args:
        backend: The backend the state came from.
        state: The current state, as reset or apply_marks returned it.
        indicators: Indicator names, at least one, none repeated.
        reduction: The reduction from nvar to n_comp, as compute_error takes it.

    Raises:
        ValueError: Starting with "indicators", before the backend is called;
            or naming an array the backend returned under the wrong name,
            dtype or shape.
        BackendFailure: If a returned array holds a non-finite value.
    """
    names = _names(indicators)
    errors = backend.compute_error(names, reduction)
    return RoundCache(state=state, errors=errors, indicators=names, reduction=reduction)