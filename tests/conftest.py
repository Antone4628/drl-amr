"""Fixtures shared by every test under tests/.

counting    the counting wrapper's class. counting(backend) wraps a real
            backend, forwards every verb to it and counts the calls by name,
            so "one compute_error call per round" is read off a Counter.
failing     the failing wrapper's class. failing(backend, verb, at) wraps a
            real backend and raises BackendFailure at the at-th call of one
            named verb, so "the environment ends the episode" is tested at
            any point of an episode without a backend that can fail.

A fixture needs no import: a test asks for it by naming it as an argument.
The wrapper was first defined in tests/agent/test_round_cache.py
(AGENT_2D_ROADMAP item 2.1) and moved here when a second test file needed it
(item 2.2). The failing wrapper is item 3.1's: item 1.2's mock raises no
BackendFailure on demand, and it is not edited.
"""
from __future__ import annotations

from collections import Counter

import pytest

from contract.backend import BackendFailure


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


class FailingBackend:
    """Forward every verb to a real backend, but fail one call of one verb.

    The at-th call of the named verb raises BackendFailure before the inner
    backend is reached, so that call changes nothing. Every other call, of
    that verb or another, is forwarded. Calls of the named verb are counted
    in seen, across resets.
    """

    def __init__(self, inner, verb, at=1):
        self.inner = inner
        self.verb = verb
        self.at = at
        self.seen = 0

    def __getattr__(self, name):
        attr = getattr(self.inner, name)
        if name != self.verb:
            return attr

        def failing(*args, **kwargs):
            self.seen += 1
            if self.seen == self.at:
                raise BackendFailure(f"{name}: failed on demand at call {self.at}")
            return attr(*args, **kwargs)

        return failing


@pytest.fixture
def failing() -> type[FailingBackend]:
    """The failing wrapper's class: failing(backend, verb, at) wraps one backend."""
    return FailingBackend
