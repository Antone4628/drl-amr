"""Plumbing tests for the error-indicator registry + dispatcher (RESTRUCTURE
Phase 7).

Ports the registry/dispatcher keepers from the Stage 1A indicator-plumbing smoke
(drl-amr-1d tests/multiround_buildout/smoke_test_indicator_plumbing.py, tasks
Z3.1 / Z3.2). The other tasks from that smoke are not ported: the pre-advance
episode and the pre-advance warning (Z3.3 / Z3.6) are obsolete (pre-advance
retired, D-049); the zz_style full episode (Z3.4) is covered by
tests/integration/test_multiround_smoke.py; and the ZZ-nonzero-at-t=0 structural
property (Z3.5) is covered by test_zz_indicator_sanity.py.
"""

import numpy as np
import pytest

from backends.python_1d.solvers.dg_advection_solver_multiround import DGAdvectionSolver
from backends.python_1d.solvers.error_indicators import (
    INDICATOR_REGISTRY,
    compute_element_errors,
    compute_element_errors_zz,
    compute_errors,
)

_BASE_XELEM = np.array([-1.0, -0.4, 0.0, 0.4, 1.0])


def make_solver(icase=1):
    """Bare solver for indicator-level tests (no env / contract)."""
    return DGAdvectionSolver(
        nop=4,
        xelem=_BASE_XELEM.copy(),
        max_elements=120,
        max_level=3,
        icase=icase,
        balance=False,
    )


def _advance(solver, n_steps=10):
    """Advance a few CFL steps so raw_jump errors are meaningfully nonzero
    (raw jump is ~0 at t=0). Matches the advance idiom in
    test_zz_indicator_sanity.py."""
    dx_min = np.min(np.diff(solver.xelem))
    dt = solver.courant_max * dx_min / solver.wave_speed
    for _ in range(n_steps):
        solver.step(dt=dt)


def test_registry_completeness():
    """Z3.1: the registry has exactly the expected keys, all callable, mapped to
    the right indicator functions."""
    assert set(INDICATOR_REGISTRY) == {"raw_jump", "zz_style"}
    for key, fn in INDICATOR_REGISTRY.items():
        assert callable(fn), f"registry['{key}'] is not callable: {type(fn)}"
    assert INDICATOR_REGISTRY["raw_jump"] is compute_element_errors
    assert INDICATOR_REGISTRY["zz_style"] is compute_element_errors_zz


def test_dispatcher_matches_direct_calls():
    """Z3.2: compute_errors(solver, key) returns exactly what the underlying
    indicator function returns."""
    solver = make_solver(icase=1)
    _advance(solver)  # make raw_jump nonzero so the equality is non-trivial

    raw_dispatch = compute_errors(solver, "raw_jump")
    raw_direct = compute_element_errors(solver)
    assert np.allclose(raw_dispatch, raw_direct)
    assert np.max(raw_dispatch) > 0.0, "raw_jump should be nonzero after advancing"

    zz_dispatch = compute_errors(solver, "zz_style")
    zz_direct = compute_element_errors_zz(solver)
    assert np.allclose(zz_dispatch, zz_direct)


def test_dispatcher_default_is_zz_style():
    """The dispatcher default is zz_style (D-048) — a bare compute_errors(solver)
    must not silently fall back to raw jumps. At t=0 zz is nonzero and raw is
    ~zero, so matching zz_direct (not raw) confirms the default routes to zz."""
    solver = make_solver(icase=1)
    default = compute_errors(solver)
    zz_direct = compute_element_errors_zz(solver)
    assert np.allclose(default, zz_direct)


def test_dispatcher_rejects_unknown_indicator():
    """Z3.2: an unregistered key raises ValueError."""
    solver = make_solver(icase=1)
    with pytest.raises(ValueError):
        compute_errors(solver, "nonexistent")