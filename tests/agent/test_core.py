"""Tests for agent/core.py — the binder, iteration-mode seam, and budget counter.

Drives AgentCore against a fake SolverContract that records call order, so the
seam is verified in isolation (real contract_impl-backed parity is the Phase 7
gate). Covers: sequential triad order + cascade return + hold short-circuit,
mode guards, the committed-budget counter (incl. refine_arity), batch staging +
apply_round, and that the bound delegations match the underlying pure functions.
"""

import numpy as np
import pytest

from agent.core import MODE_BATCH, MODE_SEQUENTIAL, AgentCore
from agent.masking import ACTION_COARSEN, ACTION_HOLD, ACTION_REFINE
from agent.masking import action_masks as action_masks_fn
from agent.normalization import alpha_thresholds
from agent.observation import build_observation
from agent.queue import build_queue as build_queue_fn
from agent.reward import global_reward, local_reward
from contract.element_state import SolverState
from contract.solver_contract import COARSEN, HOLD, REFINE


class FakeContract:
    """Records contract calls; returns a configurable cascade from balance()."""

    def __init__(self, cascade=None):
        self.calls = []
        self._cascade = set(cascade) if cascade is not None else set()
        self.applied_marks = None

    def adapt_element(self, active_idx, mark):
        self.calls.append(("adapt_element", active_idx, mark))
        return True

    def balance(self):
        self.calls.append(("balance",))
        return set(self._cascade)

    def rebuild(self):
        self.calls.append(("rebuild",))

    def apply_marks(self, marks):
        self.calls.append(("apply_marks",))
        self.applied_marks = np.asarray(marks).copy()
        return "NEW_STATE"

    def compute_error(self, indicator):
        self.calls.append(("compute_error", indicator))
        return np.array([0.1, 0.2])


def make_state(n=4, *, max_level=3):
    return SolverState(
        element_id=np.arange(1, n + 1),
        level=np.zeros(n, dtype=int),
        left=np.asarray([(i - 1) % n for i in range(n)]),
        right=np.asarray([(i + 1) % n for i in range(n)]),
        sibling=np.full(n, -1),
        can_refine=np.ones(n, dtype=bool),
        can_coarsen=np.zeros(n, dtype=bool),
        n_active=n,
        max_level=max_level,
        stable_dt=0.01,
        wave_speed=1.0,
        domain_length=2.0,
    )


# --- sequential seam --------------------------------------------------------

def test_execute_now_triad_order():
    fake = FakeContract()
    core = AgentCore(fake, mode=MODE_SEQUENTIAL)
    core.begin_round(make_state(4))
    result = core.execute_now(2, ACTION_REFINE)
    assert fake.calls == [("adapt_element", 2, REFINE), ("balance",), ("rebuild",)]
    assert result["action_taken"] == "refine"
    assert result["changed"] is True
    assert result["cascade"] == set()


def test_execute_now_returns_cascade():
    fake = FakeContract(cascade={99})
    core = AgentCore(fake, mode=MODE_SEQUENTIAL)
    core.begin_round(make_state(4))
    assert core.execute_now(0, ACTION_REFINE)["cascade"] == {99}


def test_execute_now_hold_short_circuits():
    fake = FakeContract()
    core = AgentCore(fake, mode=MODE_SEQUENTIAL)
    core.begin_round(make_state(4))
    result = core.execute_now(0, ACTION_HOLD)
    assert fake.calls == []
    assert result == {"action_taken": "hold", "cascade": set(), "changed": False}


def test_execute_now_coarsen_uses_coarsen_mark():
    fake = FakeContract()
    core = AgentCore(fake, mode=MODE_SEQUENTIAL)
    core.begin_round(make_state(4))
    core.execute_now(1, ACTION_COARSEN)
    assert fake.calls[0] == ("adapt_element", 1, COARSEN)


def test_execute_now_rejects_batch_mode():
    core = AgentCore(FakeContract(), mode=MODE_BATCH)
    core.begin_round(make_state(4))
    with pytest.raises(RuntimeError):
        core.execute_now(0, ACTION_REFINE)


# --- committed-budget counter ----------------------------------------------

def test_committed_counter_sequential():
    core = AgentCore(FakeContract(), mode=MODE_SEQUENTIAL)
    core.begin_round(make_state(4))
    assert core.committed_count == 4
    core.execute_now(0, ACTION_REFINE)
    assert core.committed_count == 5
    core.execute_now(1, ACTION_REFINE)
    assert core.committed_count == 6
    core.execute_now(2, ACTION_COARSEN)
    assert core.committed_count == 5
    core.execute_now(3, ACTION_HOLD)
    assert core.committed_count == 5


def test_committed_counter_refine_arity():
    core = AgentCore(FakeContract(), mode=MODE_SEQUENTIAL, refine_arity=4)
    core.begin_round(make_state(10))
    core.execute_now(0, ACTION_REFINE)
    assert core.committed_count == 13  # +3
    core.execute_now(1, ACTION_COARSEN)
    assert core.committed_count == 10  # -3


# --- batch seam -------------------------------------------------------------

def test_stage_flag_and_apply_round():
    fake = FakeContract()
    core = AgentCore(fake, mode=MODE_BATCH)
    core.begin_round(make_state(3))
    core.stage_flag(0, ACTION_REFINE)
    core.stage_flag(2, ACTION_COARSEN)
    assert core.committed_count == 3 + 1 - 1
    out = core.apply_round()
    assert out == "NEW_STATE"
    assert ("apply_marks",) in fake.calls
    assert np.array_equal(fake.applied_marks, np.array([REFINE, HOLD, COARSEN]))


def test_stage_flag_rejects_sequential():
    core = AgentCore(FakeContract(), mode=MODE_SEQUENTIAL)
    core.begin_round(make_state(3))
    with pytest.raises(RuntimeError):
        core.stage_flag(0, ACTION_REFINE)


def test_apply_round_rejects_sequential():
    core = AgentCore(FakeContract(), mode=MODE_SEQUENTIAL)
    core.begin_round(make_state(3))
    with pytest.raises(RuntimeError):
        core.apply_round()


def test_unknown_mode_rejected():
    with pytest.raises(ValueError):
        AgentCore(FakeContract(), mode="bogus")


# --- bound delegations match the pure functions -----------------------------

def test_compute_error_binds_indicator():
    fake = FakeContract()
    core = AgentCore(fake, error_indicator="zz_style")
    core.compute_error()
    assert ("compute_error", "zz_style") in fake.calls


def test_compute_thresholds_binds_params():
    core = AgentCore(FakeContract(), alpha=0.2, beta=1.5)
    errors = np.array([0.01, 0.1, 0.4])
    assert core.compute_thresholds(errors) == alpha_thresholds(errors, 0.2, 1.5)


def test_observe_binds_params():
    core = AgentCore(FakeContract(), alpha=0.1, element_budget=8)
    state = make_state(4)
    errors = np.array([0.5, 0.1, 0.01, 0.001])
    got = core.observe(state, errors, 1, round_number=2)
    expected = build_observation(state, errors, 1, 0.1, 8, 2)
    assert np.allclose(got, expected)


def test_action_mask_delegates():
    core = AgentCore(FakeContract())
    state = make_state(4)
    assert np.array_equal(
        core.action_mask(state, 0, set()), action_masks_fn(state, 0, set())
    )


def test_build_queue_delegates():
    core = AgentCore(FakeContract())
    state = make_state(4)
    errors = np.array([1.0, 0.05, 0.05, 0.001])
    assert core.build_queue(state, errors, 0.1, 0.01) == build_queue_fn(
        state, errors, 0.1, 0.01
    )


def test_reward_delegations_bind_weights():
    core = AgentCore(FakeContract(), p_ur=10.0, p_or=5.0, p_cr=2.0)
    assert core.local_reward(1.0, ACTION_COARSEN, 0.1, 0.01) == local_reward(
        1.0, ACTION_COARSEN, 0.1, 0.01, 10.0, 5.0, 2.0
    )
    mie = np.array([1.0, 0.001])
    levels = np.array([0, 2])
    assert core.global_reward(mie, levels, 0.1, 0.01, 3) == global_reward(
        mie, levels, 0.1, 0.01, 3, 10.0, 5.0
    )


def test_combine_reward_binds_lambdas():
    core = AgentCore(FakeContract(), lambda_local=0.1, lambda_global=1.0)
    assert core.combine_reward(2.0, -5.0) == pytest.approx(0.1 * 2.0 + 1.0 * -5.0)