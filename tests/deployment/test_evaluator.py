"""Tests for the deployment evaluator (RESTRUCTURE Phase 6, step 7).

Exercises the paired-run seam on a REAL small chain with random/refine/hold
decide_fns and a fake metric: fresh driver + fresh metrics per rollout, primary
and baseline pairing, and metric-instance independence across paired runs.
"""
from __future__ import annotations

import numpy as np

from agent.masking import ACTION_HOLD, ACTION_REFINE
from deployment.build import build_driver_from_config
from deployment.evaluator import Evaluator, PairedResult, RunResult
from deployment.runner import random_decide_fn


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


def _t_interval(cfg) -> float:
    d = build_driver_from_config(cfg)
    d.reset(options={"icase": 1})
    st = d.contract.get_state()
    return d.step_domain_fraction * st.domain_length / st.wave_speed


def _refine_fn(obs, mask):
    return ACTION_REFINE if mask[ACTION_REFINE] else ACTION_HOLD


def _hold_fn(obs, mask):
    return ACTION_HOLD


class _CountMetric:
    """Fake metric: records n_active per captured frame."""

    def __init__(self):
        self.seen = []

    def update(self, snap):
        self.seen.append(snap.n_active)

    def finalize(self):
        return {"n_frames": len(self.seen), "max_n_active": max(self.seen)}


def _evaluator(cfg, T, **kw) -> Evaluator:
    return Evaluator(
        lambda: build_driver_from_config(cfg),
        time_final=4 * T,
        output_dt=T / 2,
        metrics_factory=lambda: [_CountMetric()],
        **kw,
    )


def test_run_strategy_returns_runresult():
    cfg = _small_config()
    T = _t_interval(cfg)
    res = _evaluator(cfg, T).run_strategy(_refine_fn, icase=1)
    assert isinstance(res, RunResult)
    assert len(res.snapshots) > 1
    assert len(res.metrics) == 1
    assert res.metrics[0]["n_frames"] == len(res.snapshots)


def test_evaluate_baseline_none_by_default():
    cfg = _small_config()
    T = _t_interval(cfg)
    res = _evaluator(cfg, T).evaluate(_refine_fn, icase=1)
    assert isinstance(res, PairedResult)
    assert res.baseline is None
    assert len(res.primary.snapshots) > 1


def test_evaluate_pairs_two_runs():
    cfg = _small_config()
    T = _t_interval(cfg)
    res = _evaluator(cfg, T).evaluate(
        _refine_fn, icase=1, baseline_decide_fn=random_decide_fn(np.random.default_rng(0))
    )
    assert res.baseline is not None
    assert len(res.primary.snapshots) > 1
    assert len(res.baseline.snapshots) > 1


def test_metrics_factory_makes_independent_instances():
    cfg = _small_config()
    T = _t_interval(cfg)
    # primary refines (mesh grows), baseline holds (mesh stays at the initial level)
    res = _evaluator(cfg, T).evaluate(_refine_fn, icase=1, baseline_decide_fn=_hold_fn)
    # each run's metric saw exactly its own frames — no cross-run state bleed
    assert res.primary.metrics[0]["n_frames"] == len(res.primary.snapshots)
    assert res.baseline.metrics[0]["n_frames"] == len(res.baseline.snapshots)
    # refining drives a higher active-element count than holding
    assert res.primary.metrics[0]["max_n_active"] > res.baseline.metrics[0]["max_n_active"]