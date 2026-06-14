"""Tests for agent/masking.py — action mask construction.

Verifies the mask against synthetic SolverState snapshots: hold always valid,
refine follows can_refine, coarsen requires backend eligibility AND a
non-cascade sibling, sibling-id resolution is correct, and an ineligible
element never indexes its (-1) sibling.
"""

import numpy as np

from agent.masking import ACTION_COARSEN, ACTION_HOLD, ACTION_REFINE, action_masks
from contract.element_state import SolverState


def make_state(*, can_refine, can_coarsen, sibling, element_id=None, max_level=3):
    """Build a minimal SolverState for masking tests.

    Masking reads only can_refine/can_coarsen/sibling/element_id; level/left/
    right/globals are inert defaults.
    """
    n = len(can_refine)
    if element_id is None:
        element_id = [10 * (i + 1) for i in range(n)]
    return SolverState(
        element_id=np.asarray(element_id),
        level=np.zeros(n, dtype=int),
        left=np.asarray([(i - 1) % n for i in range(n)]),
        right=np.asarray([(i + 1) % n for i in range(n)]),
        sibling=np.asarray(sibling),
        can_refine=np.asarray(can_refine, dtype=bool),
        can_coarsen=np.asarray(can_coarsen, dtype=bool),
        n_active=n,
        max_level=max_level,
        stable_dt=0.01,
        wave_speed=1.0,
        domain_length=2.0,
    )


def test_hold_always_valid():
    state = make_state(can_refine=[False], can_coarsen=[False], sibling=[-1])
    assert action_masks(state, 0, set())[ACTION_HOLD]


def test_refine_follows_can_refine():
    state = make_state(can_refine=[True, False], can_coarsen=[False, False], sibling=[-1, -1])
    assert action_masks(state, 0, set())[ACTION_REFINE]
    assert not action_masks(state, 1, set())[ACTION_REFINE]


def test_coarsen_allowed_when_eligible_and_sibling_free():
    # idx0 coarsenable, sibling at pos1 (id 20)
    state = make_state(can_refine=[False, False], can_coarsen=[True, True], sibling=[1, 0])
    assert action_masks(state, 0, set())[ACTION_COARSEN]


def test_coarsen_blocked_by_cascade_sibling():
    state = make_state(can_refine=[False, False], can_coarsen=[True, True], sibling=[1, 0])
    # idx0's sibling is element_id 20; excluding it blocks coarsen
    assert not action_masks(state, 0, {20})[ACTION_COARSEN]
    # excluding an unrelated id does not block
    assert action_masks(state, 0, {30})[ACTION_COARSEN]


def test_coarsen_blocked_when_not_structurally_eligible():
    # can_coarsen False, sibling -1 — must not index sibling, must be False
    state = make_state(can_refine=[False], can_coarsen=[False], sibling=[-1])
    assert not action_masks(state, 0, set())[ACTION_COARSEN]


def test_sibling_id_resolution():
    # distinct ids; only the *sibling's* id should gate the coarsen mask
    state = make_state(
        can_refine=[False, False, False, False],
        can_coarsen=[True, False, False, False],
        sibling=[2, -1, 0, -1],          # idx0's sibling is pos2 -> id 30
        element_id=[10, 20, 30, 40],
    )
    assert not action_masks(state, 0, {30})[ACTION_COARSEN]   # sibling id excluded
    assert action_masks(state, 0, {10})[ACTION_COARSEN]       # own id irrelevant
    assert action_masks(state, 0, {20, 40})[ACTION_COARSEN]   # non-sibling ids irrelevant


def test_mask_shape_and_dtype():
    state = make_state(can_refine=[True], can_coarsen=[False], sibling=[-1])
    mask = action_masks(state, 0, set())
    assert mask.shape == (3,)
    assert mask.dtype == bool