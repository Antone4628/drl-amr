"""Tests for agent/normalization.py — alpha/beta error normalization.

Relocated from tests/python_1d/solvers/test_error_indicators.py (RESTRUCTURE
Phase 5): the normalization/threshold math moved to the agent layer, so its
coverage moves with it. Converted from the custom runner to pytest (clean
pass/fail invariants).

Locks in the observation sign convention found earlier: the prose/architecture
description claims o_k ~ 1.0 at the decision boundary, but the code matches
DynAMO Eq. 15 and produces o_k = -1.0 in the realistic subunit-error regime.

Run:
    pytest tests/agent/test_normalization.py
"""

import math

import numpy as np

from agent.normalization import alpha_thresholds, normalized_error

# =============================================================================
# normalized_error
# =============================================================================

def test_subunit_regime_directionality():
    """Realistic regime (alpha*e_inf < 1): larger e_k -> larger (less negative) o_k."""
    alpha = 0.1
    e_inf = 1.0  # alpha * e_inf = 0.1 < 1, so log10(...) < 0

    o_small = normalized_error(0.001, alpha, e_inf)
    o_med = normalized_error(0.01, alpha, e_inf)
    o_large = normalized_error(0.5, alpha, e_inf)

    assert o_small < o_med < o_large, (
        f"Directionality violated: o(0.001)={o_small}, "
        f"o(0.01)={o_med}, o(0.5)={o_large}"
    )


def test_decision_boundary_is_negative_one():
    """At e_k = e_max = alpha * e_inf, the formula gives o_k = -1 exactly."""
    alpha = 0.1
    e_inf = 1.0
    e_max = alpha * e_inf  # = 0.1, the refinement threshold

    o = normalized_error(e_max, alpha, e_inf)

    assert math.isclose(o, -1.0, abs_tol=1e-12), (
        f"Expected o = -1 at decision boundary, got {o}"
    )


def test_above_threshold_means_o_greater_than_minus_one():
    """e_k > e_max should produce o_k > -1 (refinement candidate region)."""
    alpha = 0.1
    e_inf = 1.0

    o_above = normalized_error(0.5, alpha, e_inf)
    assert o_above > -1.0, f"Expected o > -1 above threshold, got {o_above}"

    o_below = normalized_error(0.001, alpha, e_inf)
    assert o_below < -1.0, f"Expected o < -1 below threshold, got {o_below}"


def test_known_log_values():
    """Spot-check against hand-computed values.

    With (alpha=0.1, e_inf=1.0) the denominator is log10(0.1) = -1, so the
    formula reduces to o_k = -log10(e_k) / -1 = log10(e_k).
    """
    alpha = 0.1
    e_inf = 1.0

    cases = [
        (1.0, 0.0),
        (0.1, -1.0),
        (0.01, -2.0),
        (0.001, -3.0),
        (0.5, math.log10(0.5)),  # ~ -0.301
    ]
    for e_k, expected in cases:
        o = normalized_error(e_k, alpha, e_inf)
        assert math.isclose(o, expected, abs_tol=1e-12), (
            f"e_k={e_k}: expected {expected}, got {o}"
        )


def test_eps_floor_on_zero_error():
    """e_k = 0 should not produce NaN/Inf — eps floor kicks in."""
    o = normalized_error(0.0, alpha=0.1, e_inf=1.0)
    assert np.isfinite(o), f"Expected finite o for e_k=0, got {o}"


def test_zero_e_inf_returns_zero():
    """e_inf = 0 (degenerate mesh) returns a finite value without crashing."""
    o = normalized_error(1e-10, alpha=0.1, e_inf=0.0)
    assert np.isfinite(o), f"Expected finite o for e_inf=0, got {o}"


def test_supraunit_regime_flips_directionality():
    """Documentation-only: when alpha*e_inf > 1, directionality reverses.

    This regime should never occur in a healthy DG run (boundary jumps are
    bounded by the O(1) solution amplitude). The test locks in the assumption —
    if a future indicator returns values outside O(1), this needs rethinking.
    """
    alpha = 0.1
    e_inf = 100.0  # alpha*e_inf = 10, log10 = +1 (positive denominator)

    o_at_boundary = normalized_error(alpha * e_inf, alpha, e_inf)  # e_k = 10
    assert math.isclose(o_at_boundary, -1.0, abs_tol=1e-12)

    o_higher_error = normalized_error(50.0, alpha, e_inf)
    o_lower_error = normalized_error(1.0, alpha, e_inf)
    assert o_higher_error < o_lower_error, (
        f"Expected directionality flip: o(50)={o_higher_error} should be < "
        f"o(1)={o_lower_error} in supraunit regime"
    )


# =============================================================================
# alpha_thresholds
# =============================================================================

def test_thresholds_basic():
    """e_max = alpha * max(errors); e_min = e_max ** beta."""
    errors = np.array([0.001, 0.01, 0.05, 0.1, 0.5])
    alpha, beta = 0.1, 1.2

    e_max, e_min = alpha_thresholds(errors, alpha, beta)

    expected_e_max = 0.1 * 0.5  # = 0.05
    expected_e_min = expected_e_max ** 1.2

    assert math.isclose(e_max, expected_e_max, abs_tol=1e-12), \
        f"e_max: expected {expected_e_max}, got {e_max}"
    assert math.isclose(e_min, expected_e_min, abs_tol=1e-12), \
        f"e_min: expected {expected_e_min}, got {e_min}"


def test_thresholds_hysteresis_ordering():
    """For beta > 1 and e_max < 1, we need e_min < e_max (neutral zone non-empty)."""
    errors = np.array([0.01, 0.1, 0.3])
    e_max, e_min = alpha_thresholds(errors, alpha=0.1, beta=1.2)
    assert e_min < e_max, f"Hysteresis ordering violated: e_min={e_min}, e_max={e_max}"


def test_thresholds_empty_errors():
    """Empty error array should not crash. Returns (0.0, 0.0)."""
    e_max, e_min = alpha_thresholds(np.array([]), alpha=0.1, beta=1.2)
    assert e_max == 0.0 and e_min == 0.0, f"Got ({e_max}, {e_min}) for empty input"


def test_thresholds_all_zero_errors():
    """All-zero error array (e.g. degenerate IC). Returns (0.0, 0.0)."""
    e_max, e_min = alpha_thresholds(np.zeros(5), alpha=0.1, beta=1.2)
    assert e_max == 0.0 and e_min == 0.0, f"Got ({e_max}, {e_min}) for all-zero input"