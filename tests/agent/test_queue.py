"""Tests for agent/queue.py — priority-magnitude queue ordering.

Verifies descending-priority order, that under- and over-refined both outrank
neutral, that neutral elements fall to the end, that the queue returns element
IDs (not positions), and the degenerate-threshold (all-neutral) case.
"""

import numpy as np

from agent.queue import build_queue
from contract.element_state import SolverState


def make_state(element_id, *, max_level=3):
    """Build a minimal SolverState for queue tests (reads n_active, element_id)."""
    n = len(element_id)
    return SolverState(
        element_id=np.asarray(element_id),
        level=np.zeros(n, dtype=int),
        left=np.asarray([(i - 1) % n for i in range(n)]),
        right=np.asarray([(i + 1) % n for i in range(n)]),
        sibling=np.full(n, -1),
        can_refine=np.ones(n, dtype=bool),
        can_coarsen=np.zeros(n, dtype=bool),
        n_active=n,
        max_level=max_level,
        stable_dt=0.01,
        wave_speed=1.0,
        domain_length=2.0,
    )


def test_priority_ordering_distinct():
    """Distinct priorities give an unambiguous descending order."""
    state = make_state([10, 20, 30, 40])
    errors = np.array([100.0, 1.0, 0.05, 0.0001])
    e_max, e_min = 0.1, 0.01
    # priorities: pos0=log10(1000)=3, pos1=log10(10)=1, pos2 neutral=0, pos3=log10(100)=2
    assert build_queue(state, errors, e_max, e_min) == [10, 40, 20, 30]


def test_returns_element_ids_not_positions():
    """Non-identity ids: a position-returning impl would give [0, 2, 1]."""
    state = make_state([70, 80, 90])
    errors = np.array([10.0, 0.05, 1.0])  # pos0 under-hi, pos1 neutral, pos2 under-lo
    assert build_queue(state, errors, 0.1, 0.01) == [70, 90, 80]


def test_under_and_over_both_outrank_neutral():
    """Over-refined elements get positive priority too, so they beat neutral."""
    state = make_state([10, 20, 30])
    errors = np.array([1.0, 0.05, 0.001])  # under, neutral, over
    queue = build_queue(state, errors, 0.1, 0.01)
    assert queue[-1] == 20            # neutral last
    assert set(queue[:2]) == {10, 30}  # under & over lead (tie between them)


def test_neutral_elements_sort_to_end():
    state = make_state([1, 2, 3, 4, 5])
    errors = np.array([10.0, 0.05, 0.05, 0.05, 0.05])  # only pos0 non-neutral
    queue = build_queue(state, errors, 0.1, 0.01)
    assert queue[0] == 1
    assert set(queue[1:]) == {2, 3, 4, 5}


def test_all_neutral_is_permutation():
    state = make_state([10, 20, 30])
    errors = np.array([0.05, 0.05, 0.05])
    queue = build_queue(state, errors, 0.1, 0.01)
    assert sorted(queue) == [10, 20, 30]
    assert len(queue) == 3


def test_degenerate_thresholds_all_neutral():
    """e_max/e_min <= eps disables both branches -> all neutral (t=0 case)."""
    state = make_state([10, 20])
    errors = np.array([1.0, 2.0])
    queue = build_queue(state, errors, e_max=0.0, e_min=0.0)
    assert sorted(queue) == [10, 20]