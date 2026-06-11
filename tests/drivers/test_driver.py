"""Behavioral tests for the shared MultiroundDriver (RESTRUCTURE Phase 6).

Drives MultiroundDriver against a controllable fake SolverContract (the
test_core.py pattern), with a real AgentCore in between. These target the
driver's ORCHESTRATION — the interval/round/element state machine, interval-
fixed thresholds (D-021), dual-reward delivery timing (global only at interval
boundaries, D-046), cascade -> consumed_elements (D-045), and termination after
N_remesh intervals — not the core's per-element math, which is covered in
tests/agent/. Sequential is the parity target (D-038); one batch-structural
test exercises the mode-aware seam (D-043) against a batch-capable fake.
"""
from __future__ import annotations

import numpy as np
import pytest

from agent.core import MODE_BATCH, MODE_SEQUENTIAL, AgentCore
from agent.masking import ACTION_HOLD, ACTION_REFINE
from agent.observation import OBS_DIM
from contract.element_state import SolverState
from contract.solver_contract import COARSEN, REFINE
from drivers.multiround_driver import MultiroundDriver


# ---------------------------------------------------------------------------
# Controllable fake backend
# ---------------------------------------------------------------------------
class FakeContract:
    """Minimal but coherent 1D-like mesh backend for driver orchestration tests.

    Models a flat, ordered active list with stable element IDs, a parent/children
    map for sibling lookup, periodic neighbors, and a per-element error field that
    grows with advance time (so max-over-interval accumulation is observable).
    refine grows the mesh by 1 (arity 2), coarsen merges a sibling pair. balance()
    returns a *configurable* cascade set (default empty) so cascade->consumed can
    be tested without modelling real balance physics.
    """

    def __init__(self, *, n_base=4, max_level=2, wave_speed=1.0,
                 domain_length=2.0, stable_dt=0.05):
        self.max_level = max_level
        self.wave_speed = wave_speed
        self.domain_length = domain_length
        self.stable_dt = stable_dt
        self._n_base = n_base
        # Instrumentation + injectable behavior.
        self._next_cascade: set[int] = set()
        self.rebuild_count = 0
        self.step_count = 0
        self.reset()

    # --- Lifecycle ---
    def reset(self, icase=None, refinement_mode="none", refinement_level=0):
        self._ids = list(range(self._n_base))
        self._level_of = {e: 0 for e in self._ids}
        self._parent_of = {e: -1 for e in self._ids}
        self._children_of: dict[int, tuple[int, int]] = {}
        self._error_of = {e: 1.0 + 0.1 * e for e in self._ids}  # mild gradient
        self._next = self._n_base
        self._time = 0.0
        self.step_count = 0
        if refinement_level > 0 and refinement_mode != "none":
            for _ in range(refinement_level):
                for e in list(self._ids):          # snapshot — list mutates
                    self._refine(self._ids.index(e))

    # --- State ---
    def get_state(self) -> SolverState:
        ids = self._ids
        n = len(ids)
        level = np.array([self._level_of[e] for e in ids], dtype=int)
        left = np.array([(i - 1) % n for i in range(n)], dtype=int)
        right = np.array([(i + 1) % n for i in range(n)], dtype=int)
        sibling = np.full(n, -1, dtype=int)
        can_coarsen = np.zeros(n, dtype=bool)
        for i, e in enumerate(ids):
            p = self._parent_of.get(e, -1)
            if p == -1:
                continue
            c1, c2 = self._children_of[p]
            other = c2 if e == c1 else c1
            if other in ids:
                sib_pos = ids.index(other)
                sibling[i] = sib_pos
                parent_level = self._level_of[e] - 1
                ln = self._level_of[ids[left[i]]]
                rn = self._level_of[ids[right[i]]]
                can_coarsen[i] = abs(parent_level - ln) <= 1 and abs(parent_level - rn) <= 1
        can_refine = level < self.max_level
        return SolverState(
            element_id=np.array(ids, dtype=np.int64),
            level=level, left=left, right=right, sibling=sibling,
            can_refine=can_refine, can_coarsen=can_coarsen,
            n_active=n, max_level=self.max_level, stable_dt=self.stable_dt,
            wave_speed=self.wave_speed, domain_length=self.domain_length,
        )

    # --- Error ---
    def compute_error(self, indicator):
        return np.array([self._error_of[e] * (1.0 + self._time) for e in self._ids])

    # --- Adaptation (sequential) ---
    def adapt_element(self, active_idx, mark):
        if mark == REFINE:
            return self._refine(active_idx)
        if mark == COARSEN:
            return self._coarsen(active_idx)
        return False  # HOLD

    def balance(self):
        cascade, self._next_cascade = self._next_cascade, set()
        return set(cascade)

    def rebuild(self):
        self.rebuild_count += 1

    # --- Adaptation (batch) ---
    def apply_marks(self, marks):
        # Refine first (resolve target IDs before mutation); HOLD is a no-op.
        # Complete-family coarsen is unused by current tests, so omitted here.
        refine_ids = [self._ids[i] for i, m in enumerate(marks)
                      if m == REFINE and self._level_of[self._ids[i]] < self.max_level]
        for e in refine_ids:
            if e in self._ids:
                self._refine(self._ids.index(e))
        self.rebuild_count += 1
        return self.get_state()

    def predict_post_balance_count(self, marks):
        raise NotImplementedError  # optional (D-043); agent falls back to its counter

    # --- Time advance ---
    def step(self, dt):
        self._time += dt
        self.step_count += 1

    # --- internals ---
    def _refine(self, idx):
        e = self._ids[idx]
        if self._level_of[e] >= self.max_level:
            return False
        c1, c2 = self._next, self._next + 1
        self._next += 2
        for c in (c1, c2):
            self._level_of[c] = self._level_of[e] + 1
            self._parent_of[c] = e
            self._error_of[c] = self._error_of[e]
        self._children_of[e] = (c1, c2)
        self._ids[idx:idx + 1] = [c1, c2]
        return True

    def _coarsen(self, idx):
        e = self._ids[idx]
        p = self._parent_of.get(e, -1)
        if p == -1:
            return False
        c1, c2 = self._children_of[p]
        sib = c2 if e == c1 else c1
        if sib not in self._ids:
            return False
        i, j = sorted((self._ids.index(e), self._ids.index(sib)))
        if j != i + 1:
            return False
        self._ids[i:j + 1] = [p]
        return True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_driver(*, mode=MODE_SEQUENTIAL, pre_advance=(0.0, 0.0),
                n_remesh=2, max_level=2, n_base=4, alpha=0.1):
    fake = FakeContract(n_base=n_base, max_level=max_level)
    core = AgentCore(fake, alpha=alpha, mode=mode, element_budget=30)
    driver = MultiroundDriver(
        core, n_remesh=n_remesh, step_domain_fraction=0.05,
        pre_advance_range=pre_advance, verbosity=0,
    )
    return driver, fake, core


def run_episode(driver, action=ACTION_HOLD, rng=None, options=None, cap=10000):
    _, info0 = driver.reset(rng=rng, options=options)
    steps = []
    done = False
    while not done:
        out = driver.step(action)
        steps.append(out)
        done = out[2] or out[3]
        if len(steps) > cap:
            raise RuntimeError("episode did not terminate")
    return info0, steps


# ---------------------------------------------------------------------------
# reset / basic surface
# ---------------------------------------------------------------------------
def test_reset_returns_obs_and_info():
    driver, _, _ = make_driver()
    obs, info = driver.reset(options={"icase": 1})
    assert obs.shape == (OBS_DIM,)
    assert obs.dtype == np.float32
    assert set(info) >= {"icase", "n_active", "e_max", "e_min", "resource_usage"}
    assert info["e_max"] >= 0.0
    assert driver.remesh_step == 0 and driver.round_number == 1
    assert len(driver.queue) == info["n_active"]


def test_forced_icase_respected():
    driver, _, _ = make_driver()
    _, info = driver.reset(options={"icase": 12})
    assert info["icase"] == 12


def test_rng_handoff_reproducible():
    d1, _, _ = make_driver()
    _, i1 = d1.reset(rng=np.random.default_rng(0))
    d2, _, _ = make_driver()
    _, i2 = d2.reset(rng=np.random.default_rng(0))
    assert i1["icase"] == i2["icase"]
    assert i1["icase"] in [1, 10, 12, 13, 14, 15, 16]


def test_pre_advance_steps_solver_only_when_enabled():
    d_off, f_off, _ = make_driver(pre_advance=(0.0, 0.0))
    d_off.reset(options={"icase": 1})
    assert f_off.step_count == 0

    d_on, f_on, _ = make_driver(pre_advance=(1.0, 1.0))
    d_on.reset(options={"icase": 1})
    assert f_on.step_count > 0


def test_step_returns_five_tuple_truncated_false():
    driver, _, _ = make_driver()
    driver.reset(options={"icase": 1})
    obs, reward, terminated, truncated, info = driver.step(ACTION_HOLD)
    assert obs.shape == (OBS_DIM,)
    assert isinstance(reward, float)
    assert truncated is False
    assert info["transition"] in ("element", "interval", "done")


def test_action_masks_shape_and_hold_always_valid():
    driver, _, _ = make_driver()
    driver.reset(options={"icase": 1})
    mask = driver.action_masks()
    assert mask.shape == (3,) and mask.dtype == bool
    assert mask[ACTION_HOLD]  # hold is always valid


# ---------------------------------------------------------------------------
# Execution path
# ---------------------------------------------------------------------------
def test_refine_grows_mesh_and_rebuilds():
    driver, fake, _ = make_driver()
    driver.reset(options={"icase": 1})
    rb = fake.rebuild_count
    _, _, _, _, info = driver.step(ACTION_REFINE)
    assert info["n_active_post"] == info["n_active_pre"] + 1
    assert fake.rebuild_count == rb + 1


def test_cascade_elements_recorded_in_consumed():
    driver, fake, _ = make_driver()
    driver.reset(options={"icase": 1})
    fake._next_cascade = {777, 888}            # injected for the next balance()
    _, _, _, _, info = driver.step(ACTION_REFINE)
    assert info["n_cascade"] == 2
    assert {777, 888} <= driver.consumed_elements


# ---------------------------------------------------------------------------
# Episode skeleton + reward timing
# ---------------------------------------------------------------------------
def test_episode_terminates_after_n_remesh_intervals():
    driver, _, _ = make_driver(n_remesh=2, max_level=2, n_base=4)
    _, steps = run_episode(driver)
    transitions = [s[4]["transition"] for s in steps]
    assert transitions.count("done") == 1
    assert transitions.count("interval") == driver.n_remesh - 1
    assert steps[-1][2] is True                 # terminated only at the end
    assert all(s[2] is False for s in steps[:-1])
    assert np.array_equal(steps[-1][0], np.zeros(OBS_DIM, dtype=np.float32))


def test_hold_episode_visits_every_element_every_round():
    # No mesh change under HOLD -> exactly n_remesh * max_level * n_base steps,
    # and every round index 1..max_level appears (D-018/D-019).
    driver, _, _ = make_driver(n_remesh=2, max_level=2, n_base=4)
    _, steps = run_episode(driver)
    assert len(steps) == 2 * 2 * 4
    rounds_seen = {s[4]["round_number"] for s in steps}
    assert rounds_seen == {1, 2}


def test_global_reward_only_at_interval_boundaries():
    driver, _, core = make_driver()
    _, steps = run_episode(driver)
    for _, reward, _, _, info in steps:
        if info["transition"] == "element":
            assert info["r_global"] == 0.0
        else:                                   # interval / done
            assert info["r_global"] != 0.0      # under-refined HOLD mesh penalised
        expected = core.lambda_local * info["r_local"] + core.lambda_global * info["r_global"]
        assert reward == pytest.approx(expected)


def test_thresholds_fixed_within_interval_change_across():
    driver, _, _ = make_driver(n_remesh=2, max_level=2, n_base=4)
    driver.reset(options={"icase": 1})
    seen = []  # (e_max, transition) captured AFTER each step
    done = False
    while not done:
        _, _, term, trunc, info = driver.step(ACTION_HOLD)
        seen.append((driver.e_max, info["transition"]))
        done = term or trunc
    # First interval boundary = where thresholds are recomputed (D-021).
    boundary = next(i for i, (_, t) in enumerate(seen) if t in ("interval", "done"))
    emax_interval1 = seen[0][0]
    # Every decision strictly before the boundary shares interval-1 thresholds
    # (round transitions do NOT recompute them).
    assert all(e == pytest.approx(emax_interval1) for e, _ in seen[:boundary])
    # The boundary step recomputes from the post-advance error distribution.
    assert seen[boundary][0] != pytest.approx(emax_interval1)


def test_max_interval_errors_accumulated_during_advance():
    driver, _, _ = make_driver(n_remesh=2, max_level=2, n_base=4)
    _, steps = run_episode(driver)
    boundary = next(s[4] for s in steps if s[4]["transition"] in ("interval", "done"))
    assert boundary["solver_n_steps"] >= 1
    assert boundary["solver_max_error_peak"] > 0.0
    assert len(driver.max_interval_errors) >= 1


# ---------------------------------------------------------------------------
# Batch-structural (mode-aware seam, D-043) — exercised against a batch-capable
# fake; the real backend's apply_marks lands at Phase 7.5.
# ---------------------------------------------------------------------------
def test_batch_episode_completes():
    driver, _, _ = make_driver(mode=MODE_BATCH, n_remesh=2, max_level=2, n_base=4)
    _, steps = run_episode(driver, action=ACTION_HOLD)
    assert steps[-1][2] is True
    transitions = [s[4]["transition"] for s in steps]
    assert transitions.count("done") == 1
    assert transitions.count("interval") == driver.n_remesh - 1