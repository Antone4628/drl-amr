"""Tests for agent/observation.py — 8-component observation assembly.

Verifies the observation vector wiring against synthetic SolverState snapshots:
correct element vs. neighbor slots, level normalization, resource_usage,
round_progress, dtype/shape. The alpha-normalization math is covered by
test_normalization.py; here normalized_error is used as the oracle for the error
slots so the test checks assembly, not the formula.
"""

import numpy as np

from agent.normalization import normalized_error
from agent.observation import OBS_DIM, build_observation
from contract.element_state import SolverState


def make_state(level, left, right, *, max_level=3, element_id=None):
    """Build a minimal SolverState for observation tests.

    Only level/left/right/n_active/max_level are read by build_observation;
    sibling/can_refine/can_coarsen are set to inert defaults.
    """
    n = len(level)
    level = np.asarray(level)
    if element_id is None:
        element_id = np.arange(1, n + 1)
    return SolverState(
        element_id=np.asarray(element_id),
        level=level,
        left=np.asarray(left),
        right=np.asarray(right),
        sibling=np.full(n, -1),
        can_refine=level < max_level,
        can_coarsen=np.zeros(n, dtype=bool),
        n_active=n,
        max_level=max_level,
        stable_dt=0.01,
        wave_speed=1.0,
        domain_length=2.0,
    )


def test_full_vector_wiring():
    """Each slot carries the right quantity for the right element."""
    state = make_state(level=[0, 1, 2, 3], left=[3, 0, 1, 2], right=[1, 2, 3, 0], max_level=3)
    errors = np.array([0.5, 0.1, 0.01, 0.001])
    alpha, e_inf = 0.1, 0.5

    obs = build_observation(
        state, errors, active_idx=1, alpha=alpha, element_budget=8, round_number=2
    )

    expected = np.array([
        normalized_error(0.1, alpha, e_inf),    # current (idx 1)
        normalized_error(0.5, alpha, e_inf),    # left neighbor (idx 0)
        normalized_error(0.01, alpha, e_inf),   # right neighbor (idx 2)
        1 / 3,                                   # level[1] / max_level
        0 / 3,                                   # left level[0]
        2 / 3,                                   # right level[2]
        4 / 8,                                   # n_active / budget
        2 / 3,                                   # round_number / max_level
    ], dtype=np.float32)

    assert obs.shape == (OBS_DIM,)
    assert obs.dtype == np.float32
    assert np.allclose(obs, expected, atol=1e-6), f"\nobs     ={obs}\nexpected={expected}"


def test_left_right_not_swapped():
    """Distinct neighbor errors confirm left/right slots aren't transposed."""
    state = make_state(level=[0, 0, 0, 0], left=[3, 0, 1, 2], right=[1, 2, 3, 0])
    errors = np.array([0.9, 0.5, 0.2, 0.05])
    alpha, e_inf = 0.1, 0.9

    obs = build_observation(state, errors, 0, alpha, element_budget=8, round_number=1)
    # idx 0: left = pos 3 (err 0.05), right = pos 1 (err 0.5)
    assert np.isclose(obs[1], normalized_error(0.05, alpha, e_inf), atol=1e-6)
    assert np.isclose(obs[2], normalized_error(0.5, alpha, e_inf), atol=1e-6)


def test_resource_usage_can_exceed_one():
    """resource_usage = n_active / element_budget, may exceed 1.0 post-cascade."""
    n = 40
    left = [(i - 1) % n for i in range(n)]
    right = [(i + 1) % n for i in range(n)]
    state = make_state(level=[0] * n, left=left, right=right)
    errors = np.full(n, 0.1)
    obs = build_observation(state, errors, 0, alpha=0.1, element_budget=30, round_number=1)
    assert np.isclose(obs[6], 40 / 30, atol=1e-6)


def test_round_progress_endpoints():
    """round_progress hits 1.0 on the final round."""
    state = make_state(level=[0, 0], left=[1, 0], right=[1, 0], max_level=3)
    errors = np.array([0.1, 0.1])
    obs_first = build_observation(state, errors, 0, 0.1, element_budget=8, round_number=1)
    obs_last = build_observation(state, errors, 0, 0.1, element_budget=8, round_number=3)
    assert np.isclose(obs_first[7], 1 / 3, atol=1e-6)
    assert np.isclose(obs_last[7], 1.0, atol=1e-6)


def test_level_normalization():
    """level slot = level / max_level."""
    state = make_state(level=[0, 2], left=[1, 0], right=[1, 0], max_level=4)
    errors = np.array([0.1, 0.1])
    obs = build_observation(state, errors, 1, 0.1, element_budget=8, round_number=1)
    assert np.isclose(obs[3], 2 / 4, atol=1e-6)