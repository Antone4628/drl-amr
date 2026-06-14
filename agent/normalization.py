"""Alpha/beta error normalization for the agent core (RESTRUCTURE Phase 5).

Pure math on the per-element error vector — no solver/backend coupling. The
backend (contract.compute_error) returns raw per-element error indicators; the
agent maps them to classification thresholds and to the normalized observation
the policy sees. Relocated here from the python_1d backend's error_indicators.py
so normalization is shared agent-layer logic that any backend's raw errors flow
through unchanged.

Architecture Decisions:
    D-004: alpha-based error normalization (DynAMO Eq. 15, 16, 21)

References:
    - Dzanic et al. (2024), DynAMO — Eq. 15 (normalized error), 16/21 (thresholds)
"""
from __future__ import annotations

import numpy as np


def alpha_thresholds(
    errors: np.ndarray, alpha: float, beta: float
) -> tuple[float, float]:
    """Compute DynAMO-style error thresholds for element classification.

    Elements with error above e_max are under-refined; below e_min are
    over-refined; between e_min and e_max is the neutral zone (Architecture
    Spec §5.3, DynAMO Eq. 16/21).

    Args:
        errors: Per-element error indicators (shape (n_active,)).
        alpha: Error tolerance (0 < alpha < 1). Smaller alpha -> more
            aggressive refinement.
        beta: Hysteresis exponent (beta > 1). Larger beta -> wider neutral zone.

    Returns:
        (e_max, e_min) where e_max = alpha * max(errors) and
        e_min = e_max ** beta. Returns (0.0, 0.0) for an empty or all-zero
        error vector.
    """
    e_inf = np.max(errors) if len(errors) > 0 else 0.0
    e_max = alpha * e_inf
    e_min = e_max ** beta if e_max > 0 else 0.0
    return e_max, e_min


def normalized_error(
    e_k: float, alpha: float, e_inf: float, eps: float = 1e-30
) -> float:
    """Compute the alpha-normalized log-error observation (DynAMO Eq. 15).

        o = -log10(e_k) / log10(alpha * e_inf)

    In the operating regime where alpha * e_inf < 1 (guaranteed for any healthy
    DG run, since boundary jumps are bounded by the O(1) solution amplitude),
    the refinement threshold e_max = alpha * e_inf maps to o = -1 exactly:
        o > -1  ->  e_k > e_max  ->  refinement candidate
        o < -1  ->  e_k < e_max  ->  below refinement threshold
    Larger raw errors produce larger (less negative) observation values.

    Sign convention: the boundary value is o = -1, NOT +1. (Some older prose /
    architecture-description text claims o ~ 1.0 at the boundary; the code
    follows DynAMO Eq. 15 and produces -1.0. The unit tests lock this in.)

    Args:
        e_k: Error indicator for the element.
        alpha: Error tolerance parameter.
        e_inf: Max error across all active elements.
        eps: Floor value to prevent log(0).

    Returns:
        Normalized error scalar (0.0 fallback if the denominator degenerates).
    """
    e_k = max(e_k, eps)
    denominator = np.log10(alpha * max(e_inf, eps))

    if abs(denominator) < 1e-30:
        return 0.0

    return -np.log10(e_k) / denominator