"""Tests for the v2 backend protocol and the training-only check
(contract/backend.py, contract/checks.py).

AGENT_2D_ROADMAP item 1.1, signals S7-S9.

Run: pytest tests/contract/test_contract_backend.py -v
"""
from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from agent.observation import build_observation
from contract.backend import (
    OPTIONAL_VERBS,
    BackendFailure,
    BackendInfo,
    SolverBackend,
)
from contract.checks import TRAINING_ONLY, training_only_params
from contract.reports import AdaptReport, AuxFields, IntervalReport
from contract.solver_contract import SolverContract
from contract.state import SolverState

VERBS = (
    "describe",
    "reset",
    "get_state",
    "compute_error",
    "dry_run",
    "apply_marks",
    "step_interval",
    "get_snapshot",
    "reinitialize_ic",
)


def info_fields():
    return dict(
        ndim=2,
        nvar=1,
        indicators=("ind_a", "ind_b"),
        reductions=("scalar", "per_variable"),
        ic_params={"x0": (0.0, 2.0), "width": (0.05, 0.3)},
        optional_verbs=frozenset({"dry_run"}),
    )


class StubBackend:
    """Nine verbs by name and nothing behind them."""

    def describe(self): ...
    def reset(self, ic, amr_freq, initial_level=1): ...
    def get_state(self): ...
    def compute_error(self, indicators, reduction): ...
    def dry_run(self, marks): ...
    def apply_marks(self, marks): ...
    def step_interval(self, accumulate, cadence, aux=()): ...
    def get_snapshot(self, write=False): ...
    def reinitialize_ic(self): ...


# --- S7: the protocol ------------------------------------------------------

def test_protocol_has_exactly_the_nine_verbs():
    declared = {
        name
        for name, value in vars(SolverBackend).items()
        if callable(value) and not name.startswith("_")
    }
    assert declared == set(VERBS)


def test_stub_with_nine_verbs_is_a_backend():
    assert isinstance(StubBackend(), SolverBackend)


@pytest.mark.parametrize("missing", VERBS)
def test_a_backend_missing_one_verb_is_not_a_backend(missing):
    members = {name: getattr(StubBackend, name) for name in VERBS if name != missing}
    incomplete = type("Incomplete", (), members)
    assert not isinstance(incomplete(), SolverBackend)


def test_the_1d_backend_is_a_solver_contract_and_not_a_solver_backend():
    from backends.python_1d.contract_impl import Python1DSolverContract

    assert issubclass(Python1DSolverContract, SolverContract)
    assert not issubclass(Python1DSolverContract, SolverBackend)


# --- S9: BackendInfo and BackendFailure ------------------------------------

def test_backend_info_constructs_and_is_frozen():
    info = BackendInfo(**info_fields())
    assert info.optional_verbs == {"dry_run"}
    with pytest.raises(dataclasses.FrozenInstanceError):
        info.ndim = 3


def test_optional_verb_names():
    assert OPTIONAL_VERBS == {"dry_run", "reinitialize_ic", "get_snapshot"}


def test_unknown_optional_verb_rejected():
    fields = info_fields()
    fields["optional_verbs"] = frozenset({"dry_run", "apply_marks"})
    with pytest.raises(ValueError, match=r"^optional_verbs: unknown names \['apply_marks'\]"):
        BackendInfo(**fields)


def test_optional_verbs_must_be_a_frozenset():
    fields = info_fields()
    fields["optional_verbs"] = ["dry_run"]
    with pytest.raises(ValueError, match=r"^optional_verbs: expected a frozenset"):
        BackendInfo(**fields)


@pytest.mark.parametrize("name, bad_value", [("ndim", 4), ("nvar", 0)])
def test_backend_info_bad_globals_rejected(name, bad_value):
    fields = info_fields()
    fields[name] = bad_value
    with pytest.raises(ValueError, match=rf"^{name}: "):
        BackendInfo(**fields)


def test_inverted_ic_bounds_rejected():
    fields = info_fields()
    fields["ic_params"] = {"x0": (2.0, 0.0)}
    with pytest.raises(ValueError, match=r"^ic_params\['x0'\]: "):
        BackendInfo(**fields)


def test_backend_failure_is_a_runtime_error():
    assert issubclass(BackendFailure, RuntimeError)
    with pytest.raises(RuntimeError, match="solve diverged"):
        raise BackendFailure("solve diverged")


# --- S8: the training-only check -------------------------------------------

def test_training_only_types():
    assert set(TRAINING_ONLY) == {AdaptReport, AuxFields, IntervalReport}


def test_todays_1d_builder_is_clean():
    assert training_only_params(build_observation) == []


def test_a_clean_v2_shaped_builder_is_clean():
    def builder(
        state: SolverState, errors: dict[str, np.ndarray], round_index: int
    ) -> np.ndarray: ...

    assert training_only_params(builder) == []


def test_aux_fields_parameter_is_reported():
    def builder(state: SolverState, aux: AuxFields) -> np.ndarray: ...

    assert training_only_params(builder) == ["aux"]


def test_optional_interval_report_is_reported():
    def builder(state: SolverState, report: IntervalReport | None = None) -> np.ndarray: ...

    assert training_only_params(builder) == ["report"]


def test_nested_aux_fields_is_reported():
    def builder(state: SolverState, extras: dict[str, AuxFields]) -> np.ndarray: ...

    assert training_only_params(builder) == ["extras"]


def test_adapt_report_parameter_is_reported():
    def builder(state: SolverState, last: AdaptReport) -> np.ndarray: ...

    assert training_only_params(builder) == ["last"]


def test_aux_inside_a_callable_is_reported():
    def builder(state: SolverState, hook: Callable[[AuxFields], float]) -> np.ndarray: ...

    assert training_only_params(builder) == ["hook"]


def test_unannotated_parameter_is_reported():
    def builder(state: SolverState, extra) -> np.ndarray: ...

    assert training_only_params(builder) == ["extra"]


def test_any_parameter_is_reported():
    def builder(state: SolverState, extra: Any) -> np.ndarray: ...

    assert training_only_params(builder) == ["extra"]


def test_star_kwargs_are_reported():
    def builder(state: SolverState, **kwargs) -> np.ndarray: ...

    assert training_only_params(builder) == ["kwargs"]


def test_offenders_come_in_signature_order():
    def builder(a: AuxFields, state: SolverState, b: AdaptReport) -> np.ndarray: ...

    assert training_only_params(builder) == ["a", "b"]


def test_return_annotation_is_ignored():
    def builder(state: SolverState) -> AuxFields: ...

    assert training_only_params(builder) == []
