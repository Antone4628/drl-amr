"""Diagnostics-callback aggregation tests for the training port (RESTRUCTURE Phase 6).

Drives MultiroundDiagnosticsCallback._on_step over synthetic info dicts (the
same keys MultiroundDriver.step emits) against a fake model/locals, and pins:
  - episode-level aggregation (returns, lengths, action/coarsen/mask fractions);
  - the lambda_local reach through env.unwrapped.driver.core (not the 0.1 fallback);
  - the report-generation path (JSON + PDF) and the numpy JSON serializer.

No RL, no real env — fast. The full train loop is covered by the smoke run.
"""
import types

import numpy as np
import pytest

from training.diagnostics import MultiroundDiagnosticsCallback


def _make_callback(tmp_path, *, lambda_local=0.5):
    """Callback wired to a fake model exposing env.envs[0].unwrapped.driver.core."""
    cb = MultiroundDiagnosticsCallback(log_dir=str(tmp_path), log_freq=10_000, verbose=0)
    cb.num_timesteps = 1  # 1 % 10_000 != 0 -> _log_to_tensorboard never fires
    core = types.SimpleNamespace(lambda_local=lambda_local)
    driver = types.SimpleNamespace(core=core)
    env0 = types.SimpleNamespace(unwrapped=types.SimpleNamespace(driver=driver))
    cb.model = types.SimpleNamespace(env=types.SimpleNamespace(envs=[env0]), logger=None)
    return cb


def _step(cb, *, action, r_local, r_global, resource_usage, masks, done, n_cascade=0):
    """Feed one synthetic step through the callback."""
    info = {
        "action": action,
        "r_local": r_local,
        "r_global": r_global,
        "resource_usage": resource_usage,
        "n_cascade": n_cascade,
    }
    cb.locals = {
        "infos": [info],
        "dones": [done],
        "action_masks": np.array([masks], dtype=bool),  # (1, 3): [coarsen, hold, refine]
    }
    cb._on_step()


def _run_two_episodes(cb):
    # Episode 1 (3 steps): refine, coarsen, hold; global reward 5.0 at the boundary.
    _step(cb, action="refine", r_local=1.0, r_global=0.0, resource_usage=0.4, masks=[1, 1, 1], done=False)
    _step(cb, action="coarsen", r_local=2.0, r_global=0.0, resource_usage=0.5, masks=[1, 1, 0], done=False)
    _step(cb, action="hold", r_local=0.0, r_global=5.0, resource_usage=0.6, masks=[0, 1, 1], done=True)
    # Episode 2 (2 steps): refine, refine; global reward 3.0 at the boundary.
    _step(cb, action="refine", r_local=0.5, r_global=0.0, resource_usage=0.3, masks=[1, 1, 1], done=False)
    _step(cb, action="refine", r_local=0.5, r_global=3.0, resource_usage=0.7, masks=[1, 1, 1], done=True)


def test_aggregation_over_two_episodes(tmp_path):
    cb = _make_callback(tmp_path, lambda_local=0.5)
    _run_two_episodes(cb)

    assert cb.episodes_completed == 2
    assert cb.episode_lengths == [3, 2]

    # Return = lambda_local * sum(r_local) + sum(r_global), with lambda_local read
    # from the core (0.5). The 0.1 fallback would give [5.3, 3.05], so these
    # values prove the reach resolves.
    assert cb.episode_returns == pytest.approx([0.5 * 3.0 + 5.0, 0.5 * 1.0 + 3.0])  # [6.5, 3.5]

    assert cb.episode_total_global == pytest.approx([5.0, 3.0])
    assert cb.episode_mean_local == pytest.approx([1.0, 0.5])

    assert cb.episode_coarsen_freq == pytest.approx([1 / 3, 0.0])
    assert cb.episode_mean_coarsen_reward == pytest.approx([2.0, 0.0])

    assert cb.episode_final_resource == pytest.approx([0.6, 0.7])
    assert cb.episode_coarsen_masked_frac == pytest.approx([1 / 3, 0.0])
    assert cb.episode_refine_masked_frac == pytest.approx([1 / 3, 0.0])

    assert cb.action_history.count("refine") == 3
    assert cb.action_history.count("coarsen") == 1
    assert cb.action_history.count("hold") == 1


def test_lambda_falls_back_when_chain_unavailable(tmp_path):
    cb = MultiroundDiagnosticsCallback(log_dir=str(tmp_path), log_freq=10_000, verbose=0)
    cb.num_timesteps = 1
    # Fake model whose env[0] has no .unwrapped -> AttributeError -> 0.1 fallback.
    cb.model = types.SimpleNamespace(env=types.SimpleNamespace(envs=[types.SimpleNamespace()]), logger=None)

    _step(cb, action="refine", r_local=10.0, r_global=0.0, resource_usage=0.5, masks=[1, 1, 1], done=True)

    assert cb.episode_returns == pytest.approx([0.1 * 10.0])  # 1.0 — fallback lambda


def test_on_training_end_writes_reports(tmp_path):
    cb = _make_callback(tmp_path)
    cb._on_training_start()
    _run_two_episodes(cb)
    cb.on_training_end()

    assert (tmp_path / "training_diagnostics.json").exists()
    assert (tmp_path / "training_report.pdf").exists()