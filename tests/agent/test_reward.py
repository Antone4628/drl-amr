"""Tests for agent/reward.py — dual reward (local shaping + global retrospective).

Covers every cell of the local reward table (under/over/neutral x coarsen/hold/
refine), logarithmic scaling, and the degenerate-threshold case; and the global
reward's level-guards (max_level / base level), n_active normalization (D-030),
non-positivity, and empty/neutral cases.
"""

import math

import numpy as np

from agent.masking import ACTION_COARSEN, ACTION_HOLD, ACTION_REFINE
from agent.reward import global_reward, local_reward

P_UR, P_OR, P_CR = 10.0, 5.0, 2.0
E_MAX, E_MIN = 0.1, 0.01


def lr(e_k, thr):
    return abs(math.log10(e_k / thr))


# --- local_reward -----------------------------------------------------------

def test_local_under_coarsen_penalized():
    r = local_reward(1.0, ACTION_COARSEN, E_MAX, E_MIN, P_UR, P_OR, P_CR)
    assert math.isclose(r, -P_UR * lr(1.0, E_MAX))  # -10 * 1


def test_local_under_refine_and_hold_zero():
    assert local_reward(1.0, ACTION_REFINE, E_MAX, E_MIN, P_UR, P_OR, P_CR) == 0.0
    assert local_reward(1.0, ACTION_HOLD, E_MAX, E_MIN, P_UR, P_OR, P_CR) == 0.0


def test_local_over_coarsen_rewarded():
    r = local_reward(0.001, ACTION_COARSEN, E_MAX, E_MIN, P_UR, P_OR, P_CR)
    assert math.isclose(r, P_CR * lr(0.001, E_MIN))  # +2 * 1
    assert r > 0


def test_local_over_refine_penalized():
    r = local_reward(0.001, ACTION_REFINE, E_MAX, E_MIN, P_UR, P_OR, P_CR)
    assert math.isclose(r, -P_OR * lr(0.001, E_MIN))  # -5 * 1


def test_local_over_hold_zero():
    assert local_reward(0.001, ACTION_HOLD, E_MAX, E_MIN, P_UR, P_OR, P_CR) == 0.0


def test_local_neutral_all_zero():
    for a in (ACTION_COARSEN, ACTION_HOLD, ACTION_REFINE):
        assert local_reward(0.05, a, E_MAX, E_MIN, P_UR, P_OR, P_CR) == 0.0


def test_local_log_scaling():
    # 100x above threshold -> log_ratio = 2
    r = local_reward(10.0, ACTION_COARSEN, E_MAX, E_MIN, P_UR, P_OR, P_CR)
    assert math.isclose(r, -P_UR * 2.0)


def test_local_degenerate_thresholds_zero():
    # e_max = e_min = 0 disables both branches
    for a in (ACTION_COARSEN, ACTION_HOLD, ACTION_REFINE):
        assert local_reward(5.0, a, 0.0, 0.0, P_UR, P_OR, P_CR) == 0.0


# --- global_reward ----------------------------------------------------------

def test_global_all_neutral_zero():
    errors = np.array([0.05, 0.05, 0.05])
    levels = np.array([0, 1, 1])
    assert global_reward(errors, levels, E_MAX, E_MIN, 3, P_UR, P_OR) == 0.0


def test_global_under_refinable_penalized_and_normalized():
    errors = np.array([1.0, 0.05])  # pos0 under, pos1 neutral
    levels = np.array([0, 0])
    r = global_reward(errors, levels, E_MAX, E_MIN, 3, P_UR, P_OR)
    assert math.isclose(r, -5.0)  # penalty 10*1, /2


def test_global_under_at_maxlevel_not_penalized():
    errors = np.array([1.0])
    levels = np.array([3])  # at max_level -> guarded out
    assert global_reward(errors, levels, E_MAX, E_MIN, 3, P_UR, P_OR) == 0.0


def test_global_over_refined_penalized():
    errors = np.array([0.001])
    levels = np.array([2])  # level > 0
    r = global_reward(errors, levels, E_MAX, E_MIN, 3, P_UR, P_OR)
    assert math.isclose(r, -5.0)  # -(5*1)/1


def test_global_over_at_base_level_not_penalized():
    errors = np.array([0.001])
    levels = np.array([0])  # base level -> guarded out
    assert global_reward(errors, levels, E_MAX, E_MIN, 3, P_UR, P_OR) == 0.0


def test_global_normalization_by_n_active():
    err2, lvl2 = np.array([1.0, 0.05]), np.array([0, 0])
    err4, lvl4 = np.array([1.0, 0.05, 0.05, 0.05]), np.array([0, 0, 0, 0])
    r2 = global_reward(err2, lvl2, E_MAX, E_MIN, 3, P_UR, P_OR)
    r4 = global_reward(err4, lvl4, E_MAX, E_MIN, 3, P_UR, P_OR)
    assert math.isclose(r2, -5.0)
    assert math.isclose(r4, -2.5)


def test_global_empty_zero():
    assert global_reward(np.array([]), np.array([]), E_MAX, E_MIN, 3, P_UR, P_OR) == 0.0


def test_global_mixed_nonpositive():
    errors = np.array([1.0, 0.001, 0.05])  # under, over, neutral
    levels = np.array([0, 2, 1])
    r = global_reward(errors, levels, E_MAX, E_MIN, 3, P_UR, P_OR)
    assert math.isclose(r, -5.0)  # (10*1 + 5*1)/3
    assert r <= 0