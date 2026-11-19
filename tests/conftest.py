"""Fixtures shared by every test under tests/.

counting    the counting wrapper's class. counting(backend) wraps a real
            backend, forwards every verb to it and counts the calls by name,
            so "one compute_error call per round" is read off a Counter.

A fixture needs no import: a test asks for it by naming it as an argument.
The wrapper was first defined in tests/agent/test_round_cache.py
(AGENT_2D_ROADMAP item 2.1) and moved here when a second test file needed it
(item 2.2).
"""
from __future__ import annotations

from collections import Counter

import pytest


class CountingBackend:
    """Forward every verb to a real backend and count the calls by name.

    Only calls made through the wrapper are counted. A call the inner
    backend makes on itself, or one made on the inner backend directly, is
    not.
    """

    def __init__(self, inner):
        self.inner = inner
        self.calls = Counter()

    def __getattr__(self, name):
        attr = getattr(self.inner, name)

        def counted(*args, **kwargs):
            self.calls[name] += 1
            return attr(*args, **kwargs)

        return counted


@pytest.fixture
def counting() -> type[CountingBackend]:
    """The counting wrapper's class: counting(backend) wraps one backend."""
    return CountingBackend
