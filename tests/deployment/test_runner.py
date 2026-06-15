"""Integration tests for the deployment runner (RESTRUCTURE Phase 6, step 5).

Exercises the runner end-to-end on a REAL small solver chain (built via
build_driver_from_config) with the random decide_fn — the model path is
exercised at the manual Phase-6-exit run, not here, so the suite stays free of
old-repo artifacts. Targets the deployment SHAPE: bracketing [0, time_final] with
exact landing, sim-time output cadence, capture_remesh, and burn-in warm-up.
"""
from __future__ import annotations

import numpy as np
import pytest

from agent.masking import ACTION_HOLD, ACTION_REFINE
from agent.observation import OBS_DIM
from deployment.build import build_driver_from_config
from deployment.runner import DeploymentRunner, random_decide_fn


def _small_config() -> dict:
    return {
        "solver": {
            "nop": 4, "xelem": [-1.0, -0.4, 0.0, 0.4, 1.0],
            "max_level": 2, "max_elements": 120, "courant_max": 0.1,
        },
        "environment": {
            "alpha": 0.1, "beta": 1.2, "element_budget": 30, "n_remesh": 4,
            "step_domain_fraction": 0.05, "initial_refinement_level": 1,
            "error_indicator": "zz_style",
            "ic_pool": [1], "verbosity": 0,
        },
        "reward": {
            "p_ur": 10.0, "p_or": 5.0, "p_cr": 2.0,
            "lambda_local": 0.1, "lambda_global": 1.0,
        },
    }


def _t_interval(driver) -> float:
    driver.reset(options={"icase": 1})
    st = driver.contract.get_state()
    return driver.step_domain_fraction * st.domain_length / st.wave_speed


def _refine_fn(obs, mask):
    return ACTION_REFINE if mask[ACTION_REFINE] else ACTION_HOLD


# --- builder ---------------------------------------------------------------
def test_build_driver_from_config_threads_params():
    cfg = _small_config()
    driver = build_driver_from_config(cfg)
    obs, info = driver.reset(options={"icase": 1})
    assert obs.shape == (OBS_DIM,)
    assert info["icase"] == 1
    assert driver.core.alpha == cfg["environment"]["alpha"]
    assert driver.core.element_budget == cfg["environment"]["element_budget"]
    assert driver.core.error_indicator == "zz_style"
    assert driver.n_remesh == cfg["environment"]["n_remesh"]


# --- decide_fn -------------------------------------------------------------
def test_random_decide_fn_returns_valid_action():
    decide = random_decide_fn(np.random.default_rng(0))
    mask = np.array([True, False, True])      # coarsen ok, hold no, refine ok
    for _ in range(20):
        a = decide(None, mask)
        assert mask[a]


# --- runner shape ----------------------------------------------------------
def test_runner_brackets_and_output_cadence():
    cfg = _small_config()
    driver = build_driver_from_config(cfg)
    T = _t_interval(driver)
    runner = DeploymentRunner(driver, time_final=4 * T, output_dt=T / 2, capture_remesh=True)
    snaps = runner.run(random_decide_fn(np.random.default_rng(0)), icase=1)
    times = np.array([s.time for s in snaps])
    assert len(snaps) > 1
    assert times[0] == pytest.approx(0.0)
    assert times[-1] == pytest.approx(4 * T)
    assert np.all(np.diff(times) >= -1e-9)                 # monotonic
    assert np.max(np.diff(times)) <= T / 2 + 1e-9          # output cadence bound
    # snapshots are real meshes
    assert all(s.q.shape == (s.npoin_dg,) for s in snaps)
    assert all(s.n_active >= 1 for s in snaps)


def test_runner_every_cfl_limit_is_denser():
    cfg = _small_config()
    d_coarse = build_driver_from_config(cfg)
    T = _t_interval(d_coarse)
    s_coarse = DeploymentRunner(d_coarse, time_final=2 * T, output_dt=T).run(
        random_decide_fn(np.random.default_rng(0)), icase=1)
    d_dense = build_driver_from_config(cfg)
    s_dense = DeploymentRunner(d_dense, time_final=2 * T, output_dt=None).run(
        random_decide_fn(np.random.default_rng(0)), icase=1)
    assert len(s_dense) > len(s_coarse)


def test_capture_remesh_adds_boundary_frames():
    cfg = _small_config()
    d_on = build_driver_from_config(cfg)
    T = _t_interval(d_on)
    # output_dt > interval -> no interior output frames; isolates remesh capture
    s_on = DeploymentRunner(d_on, time_final=4 * T, output_dt=10 * T, capture_remesh=True).run(
        random_decide_fn(np.random.default_rng(0)), icase=1)
    d_off = build_driver_from_config(cfg)
    s_off = DeploymentRunner(d_off, time_final=4 * T, output_dt=10 * T, capture_remesh=False).run(
        random_decide_fn(np.random.default_rng(0)), icase=1)
    assert len(s_on) > len(s_off)
    assert s_off[0].time == pytest.approx(0.0)
    assert s_off[-1].time == pytest.approx(4 * T)


def test_burnin_refines_initial_mesh():
    cfg = _small_config()
    d_no = build_driver_from_config(cfg)
    T = _t_interval(d_no)
    s_no = DeploymentRunner(d_no, time_final=2 * T, output_dt=T, burnin=False).run(
        _refine_fn, icase=1)
    d_yes = build_driver_from_config(cfg)
    s_yes = DeploymentRunner(d_yes, time_final=2 * T, output_dt=T, burnin=True, n_burnin=2).run(
        _refine_fn, icase=1)
    # burn-in applies adaptation passes before t=0 -> finer starting mesh
    assert s_yes[0].n_active > s_no[0].n_active

# --- metrics seam (step 7) -------------------------------------------------
class _CountMetric:
    """Fake metric for the seam test: records n_active per captured frame."""

    def __init__(self):
        self.seen = []

    def update(self, snap):
        self.seen.append(snap.n_active)

    def finalize(self):
        return {"n_frames": len(self.seen), "max_n_active": max(self.seen)}


def test_count_metric_satisfies_protocol():
    from deployment.metrics import Metric
    assert isinstance(_CountMetric(), Metric)


def test_runner_drives_metrics():
    cfg = _small_config()
    driver = build_driver_from_config(cfg)
    T = _t_interval(driver)
    m = _CountMetric()
    runner = DeploymentRunner(driver, time_final=4 * T, output_dt=T / 2, metrics=[m])
    snaps = runner.run(random_decide_fn(np.random.default_rng(0)), icase=1)
    # one update per captured frame; finalized result available positionally
    assert m.seen and len(m.seen) == len(snaps)
    assert len(runner.metric_results) == 1
    assert runner.metric_results[0]["n_frames"] == len(snaps)
    assert runner.metric_results[0]["max_n_active"] == max(s.n_active for s in snaps)