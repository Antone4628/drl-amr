"""Behavioral tests for the MultiroundEnv gym wrapper (RESTRUCTURE Phase 6).

The env is a thin wrapper (D-047), so these target the WRAPPER's contract, not
the orchestration (covered in tests/drivers/) or the per-element math (covered in
tests/agent/): the declared gym spaces, the seeded-RNG handoff (D-038), pure
delegation of reset/step/action_masks, and the MaskablePPO plumbing. Driven
against the same controllable fake SolverContract pattern used by the driver
tests, trimmed to the sequential surface (the env runs sequential only).

NOTE: the alpha-normalized error components obs[0:3] are negative across the
healthy regime (boundary o = -1), so the declared Box low = 0.0 on them is too
tight (see MultiroundEnv module docstring). These tests therefore do NOT assert
observation_space.contains(obs) and do NOT run gymnasium env_checker — both would
flag that known-loose bound, which is held verbatim until the parity gate. Only
the genuinely-bounded components are range-checked.
"""
from __future__ import annotations

import numpy as np
import pytest
from gymnasium import spaces

from agent.core import AgentCore
from agent.masking import ACTION_HOLD
from agent.observation import OBS_DIM
from contract.element_state import SolverState
from contract.solver_contract import COARSEN, REFINE
from drivers.multiround_driver import MultiroundDriver
from envs.multiround_env import MultiroundEnv


# ---------------------------------------------------------------------------
# Controllable fake backend (sequential surface; trimmed from test_driver.py)
# ---------------------------------------------------------------------------
class FakeContract:
    """Minimal coherent 1D-like backend: ordered active list with stable IDs,
    periodic neighbors, a parent/children sibling map, a time-growing error
    field, and refine/coarsen/balance/rebuild/step. Batch apply_marks is omitted
    — the env drives sequential mode only."""

    def __init__(self, *, n_base=4, max_level=2, wave_speed=1.0,
                 domain_length=2.0, stable_dt=0.05):
        self.max_level = max_level
        self.wave_speed = wave_speed
        self.domain_length = domain_length
        self.stable_dt = stable_dt
        self._n_base = n_base
        self.rebuild_count = 0
        self.step_count = 0
        self.reset()

    def reset(self, icase=None, refinement_mode="none", refinement_level=0):
        self._ids = list(range(self._n_base))
        self._level_of = {e: 0 for e in self._ids}
        self._parent_of = {e: -1 for e in self._ids}
        self._children_of: dict[int, tuple[int, int]] = {}
        self._error_of = {e: 1.0 + 0.1 * e for e in self._ids}
        self._next = self._n_base
        self._time = 0.0
        self.step_count = 0
        if refinement_level > 0 and refinement_mode != "none":
            for _ in range(refinement_level):
                for e in list(self._ids):
                    self._refine(self._ids.index(e))

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

    def compute_error(self, indicator):
        return np.array([self._error_of[e] * (1.0 + self._time) for e in self._ids])

    def adapt_element(self, active_idx, mark):
        if mark == REFINE:
            return self._refine(active_idx)
        if mark == COARSEN:
            return self._coarsen(active_idx)
        return False  # HOLD

    def balance(self):
        return set()  # no cascades in this fake

    def rebuild(self):
        self.rebuild_count += 1

    def step(self, dt):
        self._time += dt
        self.step_count += 1

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
# Helper
# ---------------------------------------------------------------------------
def make_env(*, pre_advance=(0.0, 0.0), n_remesh=2, max_level=2, n_base=4, alpha=0.1):
    fake = FakeContract(n_base=n_base, max_level=max_level)
    core = AgentCore(fake, alpha=alpha, element_budget=30)
    driver = MultiroundDriver(
        core, n_remesh=n_remesh, step_domain_fraction=0.05,
        pre_advance_range=pre_advance, verbosity=0,
    )
    return MultiroundEnv(driver), fake


# ---------------------------------------------------------------------------
# Spaces
# ---------------------------------------------------------------------------
def test_action_space_is_discrete_three():
    env, _ = make_env()
    assert isinstance(env.action_space, spaces.Discrete)
    assert env.action_space.n == 3


def test_observation_space_matches_old_env():
    env, _ = make_env()
    box = env.observation_space
    assert isinstance(box, spaces.Box)
    assert box.shape == (OBS_DIM,)
    assert box.dtype == np.float32
    # Bounds lifted verbatim from dg_amr_env_multiround.py (the parity target).
    assert np.array_equal(box.low, np.zeros(OBS_DIM, dtype=np.float32))
    assert np.array_equal(
        box.high,
        np.array([np.inf, np.inf, np.inf, 1.0, 1.0, 1.0, 2.0, 1.0], dtype=np.float32),
    )


# ---------------------------------------------------------------------------
# reset / seeding handoff
# ---------------------------------------------------------------------------
def test_reset_returns_obs_and_info():
    env, _ = make_env()
    obs, info = env.reset(options={"icase": 1})
    assert obs.shape == (OBS_DIM,)
    assert obs.dtype == np.float32
    assert set(info) >= {"icase", "n_active", "e_max", "e_min", "resource_usage"}
    # Only the genuinely-bounded components are range-checked (see module note).
    assert np.all(obs[3:6] >= 0.0) and np.all(obs[3:6] <= 1.0)   # levels
    assert obs[6] >= 0.0                                          # resource_usage
    assert 0.0 <= obs[7] <= 1.0                                   # round_progress


def test_reset_forwards_options_icase():
    env, _ = make_env()
    _, info = env.reset(options={"icase": 13})
    assert info["icase"] == 13


def test_seeded_reset_reproducible_icase():
    # super().reset(seed=s) seeds self.np_random, handed to the driver; same
    # seed -> same IC draw (the D-038 RNG handoff).
    env1, _ = make_env()
    env2, _ = make_env()
    _, i1 = env1.reset(seed=0)
    _, i2 = env2.reset(seed=0)
    assert i1["icase"] == i2["icase"]
    assert i1["icase"] in [1, 10, 12, 13, 14, 15, 16]


# ---------------------------------------------------------------------------
# step / action_masks delegation
# ---------------------------------------------------------------------------
def test_step_returns_five_tuple_truncated_false():
    env, _ = make_env()
    env.reset(options={"icase": 1})
    obs, reward, terminated, truncated, info = env.step(ACTION_HOLD)
    assert obs.shape == (OBS_DIM,) and obs.dtype == np.float32
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert truncated is False
    assert info["transition"] in ("element", "interval", "done")


def test_action_masks_delegates_to_driver():
    env, _ = make_env()
    env.reset(options={"icase": 1})
    mask = env.action_masks()
    assert mask.shape == (3,) and mask.dtype == bool
    assert mask[ACTION_HOLD]  # hold is always valid


def test_hold_episode_terminates_with_zero_terminal_obs():
    env, _ = make_env(n_remesh=2, max_level=2, n_base=4)
    env.reset(options={"icase": 1})
    last_obs, terminated, steps = None, False, 0
    while not terminated:
        last_obs, _, terminated, _, _ = env.step(ACTION_HOLD)
        steps += 1
        assert steps < 1000
    assert steps == 2 * 2 * 4  # n_remesh * max_level * n_base under pure HOLD
    assert np.array_equal(last_obs, np.zeros(OBS_DIM, dtype=np.float32))


# ---------------------------------------------------------------------------
# MaskablePPO plumbing (the Phase 6 env-shell verification)
# ---------------------------------------------------------------------------
def test_maskable_ppo_smoke():
    sb3_contrib = pytest.importorskip("sb3_contrib")
    maskable_ppo = sb3_contrib.MaskablePPO
    env, _ = make_env(n_remesh=2, max_level=2, n_base=4)
    # SB3 auto-wraps in Monitor + DummyVecEnv; get_action_masks finds the env's
    # action_masks() through env_method. Tiny rollout just exercises the plumbing.
    model = maskable_ppo("MlpPolicy", env, n_steps=64, batch_size=64, n_epochs=1, verbose=0)
    model.learn(total_timesteps=128)