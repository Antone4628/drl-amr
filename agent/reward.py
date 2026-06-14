"""Dual-reward computation for the agent core (RESTRUCTURE Phase 5).

Pure functions implementing the dual reward (Architecture Spec §8, D-003):

  - local_reward: immediate per-element shaping. Classifies one action against
    the element's pre-action error using the fixed interval thresholds
    (e_max, e_min). Do-nothing is never penalized; correct coarsening of an
    over-refined element earns a positive reward (D-020).

  - global_reward: delayed per-interval retrospective. After the solver advances
    by one remesh interval, penalizes the adapted mesh using max-over-interval
    errors (D-008) against the same thresholds, normalized by element count
    (D-030/G1). Always <= 0.

Penalty/reward scaling is logarithmic: an element 10x past a threshold scores
weight x 1, 100x past scores weight x 2 — smooth gradients without extremes.

Seam (a) (RESTRUCTURE Phase 5): global_reward is a pure function of the
max-over-interval error vector and the post-advance levels. The driver owns the
solver-advance loop and accumulates max_interval_errors across sub-steps, then
hands the vector + levels here, keeping the reward free of solver-stepping
assumptions.
"""
from __future__ import annotations

import numpy as np

from agent.masking import ACTION_COARSEN, ACTION_REFINE

_EPS = 1e-30


def local_reward(
    e_k: float,
    action: int,
    e_max: float,
    e_min: float,
    p_ur: float,
    p_or: float,
    p_cr: float,
) -> float:
    """Local shaping reward for a single element action (Spec §8.1).

    Args:
        e_k: Raw (unnormalized) pre-action error indicator for the element.
        action: Action taken (0 = coarsen, 1 = hold, 2 = refine).
        e_max: Upper threshold (above it = under-refined).
        e_min: Lower threshold (below it = over-refined).
        p_ur: Under-refinement penalty weight (wrong action above e_max).
        p_or: Over-refinement penalty weight (wrong action below e_min).
        p_cr: Correct-coarsening reward weight (D-020).

    Returns:
        Scalar reward: negative = penalty, positive = correct coarsen, 0 =
        correct/acceptable/neutral.
    """
    e_k = max(e_k, _EPS)

    if e_k > e_max and e_max > _EPS:
        # Under-refined: refine/hold acceptable (0), coarsen is wrong.
        if action == ACTION_COARSEN:
            return -p_ur * abs(np.log10(e_k / e_max))
        return 0.0

    if e_k < e_min and e_min > _EPS:
        # Over-refined: coarsen correct (+p_cr), refine wrong, hold acceptable.
        log_ratio = abs(np.log10(e_k / e_min))
        if action == ACTION_REFINE:
            return -p_or * log_ratio
        if action == ACTION_COARSEN:
            return p_cr * log_ratio
        return 0.0

    # Neutral zone: all actions acceptable.
    return 0.0


def global_reward(
    max_interval_errors: np.ndarray,
    levels: np.ndarray,
    e_max: float,
    e_min: float,
    max_level: int,
    p_ur: float,
    p_or: float,
) -> float:
    """Global retrospective reward after one remesh interval (Spec §8.2).

    Penalizes the adapted mesh using max-over-interval errors against the
    interval's fixed thresholds. Conditional guards: under-refinement is only
    the agent's fault if the element could refine further (level < max_level);
    over-refinement only if it is actually refined (level > 0). Normalized by
    element count (D-030/G1) so the reward measures average per-element mesh
    quality rather than total penalty mass (removing the coarsen-everything
    attractor).

    Args:
        max_interval_errors: Max-over-interval error per active element
            (shape (n_active,)), accumulated by the driver across the advance.
        levels: Refinement level per active element (shape (n_active,)),
            aligned with max_interval_errors (post-advance state.level).
        e_max: Upper threshold (interval-fixed, D-021).
        e_min: Lower threshold (interval-fixed, D-021).
        max_level: Maximum refinement level.
        p_ur: Under-refinement penalty weight.
        p_or: Over-refinement penalty weight.

    Returns:
        Scalar reward <= 0. Zero means the mesh matched the error distribution
        across the advance.
    """
    n_active = len(max_interval_errors)
    if n_active == 0:
        return 0.0

    # Sequential left-to-right accumulation, matching the env exactly so the
    # reward is bit-identical at the parity gate (np.sum would reorder adds).
    total_penalty = 0.0
    for i in range(n_active):
        e_k = max(max_interval_errors[i], _EPS)
        level = int(levels[i])

        if e_k > e_max and e_max > _EPS:
            if level < max_level:
                total_penalty += p_ur * abs(np.log10(e_k / e_max))
        elif e_k < e_min and e_min > _EPS:
            if level > 0:
                total_penalty += p_or * abs(np.log10(e_k / e_min))
        # else neutral, or guarded-out -> no penalty

    return -total_penalty / n_active