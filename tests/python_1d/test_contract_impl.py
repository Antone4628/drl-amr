"""Contract-surface tests for Python1DSolverContract (RESTRUCTURE Phase 4).

The durable form of the Phase-4 excision smoke. Exercises the *sequential*
SolverContract surface — the parity target (D-038) — over the ported 1D DG
multiround solver: lifecycle (reset), state snapshot (get_state), error vector
(compute_error), the adapt_element -> balance -> rebuild triad with cascade
reporting, and step. Also pins two deliberate design choices (D-044): the
stable_dt no-`/2` value and the deferred batch-seam stubs.

Pytest regime (per the 2026-06-09 test-convention note): clean pass/fail
invariants. Richer behavioral exploration (cascade visualization) lives in the
tools/ diagnostics, not here.

Run: pytest tests/python_1d/test_contract_impl.py -v
"""

import numpy as np
import pytest

from backends.python_1d.contract_impl import Python1DSolverContract
from backends.python_1d.solvers.dg_advection_solver_multiround import DGAdvectionSolver
from contract.element_state import SolverState
from contract.solver_contract import COARSEN, HOLD, REFINE, SolverContract

# Base mesh: 4 non-uniform elements symmetric about 0 — matches the solver's
# reset() default and the env default.
BASE_XELEM = np.array([-1.0, -0.4, 0.0, 0.4, 1.0])
MAX_LEVEL = 3


@pytest.fixture
def contract():
    """A fresh contract over a balance=False solver (balance is enforced
    explicitly via .balance() so cascades are reported — see contract_impl)."""
    solver = DGAdvectionSolver(
        nop=4,
        xelem=BASE_XELEM.copy(),
        max_elements=120,
        max_level=MAX_LEVEL,
        icase=1,
        balance=False,
        verbose=False,
    )
    return Python1DSolverContract(solver)


# --- helpers ---------------------------------------------------------------

def assert_2to1_balanced(state):
    """Every adjacent active pair (periodic) differs by <= 1 refinement level."""
    lvl = state.level
    right = state.right
    for i in range(state.n_active):
        gap = abs(int(lvl[i]) - int(lvl[int(right[i])]))
        assert gap <= 1, f"2:1 balance violated at position {i}: gap={gap}"


def refine_at(contract, active_idx):
    """Sequential triad on one element: adapt -> balance -> rebuild.

    Returns the cascade set from balance().
    """
    contract.adapt_element(active_idx, REFINE)
    cascade = contract.balance()
    contract.rebuild()
    return cascade


# --- structural conformance ------------------------------------------------

def test_isinstance_solver_contract(contract):
    assert isinstance(contract, SolverContract)


# --- reset / get_state -----------------------------------------------------

def test_reset_returns_base_mesh(contract):
    contract.reset(icase=1)
    st = contract.get_state()
    assert st.n_active == 4
    assert np.all(st.level == 0)
    assert st.max_level == MAX_LEVEL


def test_get_state_is_solver_state(contract):
    assert isinstance(contract.get_state(), SolverState)


def test_get_state_array_lengths(contract):
    st = contract.get_state()
    n = st.n_active
    for arr in (st.element_id, st.level, st.left, st.right,
                st.sibling, st.can_refine, st.can_coarsen):
        assert len(arr) == n


def test_get_state_globals(contract):
    st = contract.get_state()
    assert st.stable_dt > 0
    assert st.wave_speed > 0
    assert st.domain_length == pytest.approx(2.0)


def test_base_mesh_topology(contract):
    st = contract.get_state()
    n = st.n_active
    # periodic neighbor wiring
    assert st.left[0] == n - 1
    assert st.right[n - 1] == 0
    for i in range(n):
        assert st.left[i] == (i - 1) % n
        assert st.right[i] == (i + 1) % n
    # level-0 base mesh: all refinable, none coarsenable (no parent), no sibling
    assert np.all(st.can_refine)
    assert not np.any(st.can_coarsen)
    assert np.all(st.sibling == -1)


def test_stable_dt_has_no_half_factor(contract):
    """Pins the deliberate no-`/2` choice (D-044, carry-forward parity item):
    get_state().stable_dt is the env-equivalent courant_max*min_dx/wave_speed,
    NOT solver.dt (which carries an extra /2 from _compute_timestep)."""
    st = contract.get_state()
    solver = contract._solver
    dx_min = float(np.min(np.diff(solver.xelem)))
    expected = solver.courant_max * dx_min / solver.wave_speed
    assert st.stable_dt == pytest.approx(expected)
    # ...and it is exactly twice solver.dt (which carries the /2 margin)
    assert st.stable_dt == pytest.approx(2.0 * solver.dt)


# --- compute_error ---------------------------------------------------------

@pytest.mark.parametrize("indicator", ["zz_style", "raw_jump"])
def test_compute_error_shape_and_finite(contract, indicator):
    st = contract.get_state()
    err = np.asarray(contract.compute_error(indicator))
    assert err.shape == (st.n_active,)
    assert np.all(np.isfinite(err))


def test_zz_error_positive_at_t0(contract):
    """ZZ indicator is generically nonzero at t=0 (the D-032 property)."""
    err = np.asarray(contract.compute_error("zz_style"))
    assert np.all(err > 0)


# --- adapt_element ---------------------------------------------------------

def test_hold_is_noop(contract):
    n0 = contract.get_state().n_active
    changed = contract.adapt_element(0, HOLD)
    assert changed is False
    assert contract.get_state().n_active == n0


def test_refine_grows_mesh_by_one(contract):
    n0 = contract.get_state().n_active
    changed = contract.adapt_element(0, REFINE)
    assert changed is True
    contract.balance()
    contract.rebuild()
    assert contract.get_state().n_active == n0 + 1


def test_refine_rejected_at_max_level(contract):
    """Driving one lineage down to max_level; a further refine is rejected."""
    for _ in range(MAX_LEVEL):
        contract.adapt_element(0, REFINE)
        contract.balance()
        contract.rebuild()
    st = contract.get_state()
    assert st.level.max() == MAX_LEVEL
    idx = int(np.argmax(st.level))  # a max-level element
    assert contract.adapt_element(idx, REFINE) is False


def test_coarsen_shrinks_mesh(contract):
    """Refine to make a sibling pair, then coarsen one back."""
    n0 = contract.get_state().n_active
    refine_at(contract, 1)
    st = contract.get_state()
    idx = int(np.argmax(st.can_coarsen))
    assert st.can_coarsen[idx]
    changed = contract.adapt_element(idx, COARSEN)
    assert changed is True
    contract.balance()
    contract.rebuild()
    assert contract.get_state().n_active == n0
    assert_2to1_balanced(contract.get_state())


# --- balance / cascade -----------------------------------------------------

def test_balance_reports_cascade(contract):
    """A level-2 element next to a level-0 forces a cascade; balance() returns
    the created element IDs, and the result is 2:1 balanced."""
    refine_at(contract, 1)  # interior element to L1 (no cascade expected)
    st = contract.get_state()
    target = None
    for i in range(st.n_active):
        if st.level[i] == 1 and (st.level[int(st.left[i])] == 0
                                 or st.level[int(st.right[i])] == 0):
            target = i
            break
    assert target is not None, "expected a level-1 element adjacent to a level-0"
    contract.adapt_element(target, REFINE)   # L1 -> L2, next to L0
    cascade = contract.balance()
    contract.rebuild()
    assert isinstance(cascade, set)
    assert len(cascade) >= 1
    assert_2to1_balanced(contract.get_state())
    post_ids = {int(e) for e in contract.get_state().element_id}
    assert cascade <= post_ids  # cascade IDs are real post-balance elements


# --- step ------------------------------------------------------------------

def test_step_advances_time_without_remeshing(contract):
    st0 = contract.get_state()
    i = int(np.argmax(st0.can_refine))
    refine_at(contract, i)
    st1 = contract.get_state()
    assert st1.n_active > st0.n_active

    t0 = contract._solver.time
    contract.step(st1.stable_dt)
    assert contract._solver.time > t0
    # step does not change the mesh
    assert contract.get_state().n_active == st1.n_active
    # solution stays finite (white-box guard on the wrapped solver)
    assert np.all(np.isfinite(contract._solver.q))


# --- batch seam (deferred, D-043) ------------------------------------------

def test_batch_seam_stubs_raise(contract):
    st = contract.get_state()
    marks = np.zeros(st.n_active, dtype=int)
    with pytest.raises(NotImplementedError):
        contract.apply_marks(marks)
    with pytest.raises(NotImplementedError):
        contract.predict_post_balance_count(marks)