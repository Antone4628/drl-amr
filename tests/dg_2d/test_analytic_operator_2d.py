"""Regression pins for the analytic 2D DG operator builder (step-7 gate).

The gate (tools/gate_2d_compare.py) compares Jexpresso's extracted 2D
operator against tools/analytic_operator_2d.py. If the analytic builder
itself were wrong, a gate failure could not be attributed to either side --
so the builder gets its own independent verification here, runnable in CI
with no Jexpresso artifact:

  1. Reduction to 1D (the load-bearing test). With cy = 0 the 2D operator
     must decouple into independent x-lines, each block equal to the
     F1-verified analytic 1D operator (tests/python_1d/dg/test_operator.py,
     verified entrywise against BOTH codes and a Julia-side independent
     construction at ~1e-12, 2026-07-27). Same along y with cx = 0 -- and on
     the anisotropic gate domain (dx = 2.5 != dy = 5.0) the two reductions
     exercise the two distinct face-lift factors, so a Jx/Jy or
     J_face/J_vol confusion cannot pass both.
  2. Structural invariants at the gate configuration: conservation,
     consistency, dissipative spectrum with exactly one zero mode.
  3. Upwind direction under negative velocity: for cx < 0 the flux takes the
     right state, so the downwind (L) trace row couples to its +x neighbour
     and the R rows receive no face contribution.

The inline 1D reference below is a documented copy of
analytic_collocated_upwind_operator from tests/python_1d/dg/test_operator.py
(the F1-verified construction), duplicated rather than imported so this
module does not depend on test-file importability. If that construction ever
changes, change this copy with it.

Run with: pytest tests/dg_2d/test_analytic_operator_2d.py -v
"""

import numpy as np
import pytest

from tools.analytic_operator_2d import (
    GATE_CX,
    GATE_CY,
    GATE_DOMAIN,
    GATE_NEX,
    GATE_NEY,
    GATE_NOP,
    build_analytic_operator_2d,
    lgl_and_diff,
    lumped_mass_vector_2d,
    structural_report,
)

NGL = GATE_NOP + 1


def _analytic_1d(nop, nelem, U, x0, x1):
    """The F1-verified analytic 1D collocated upwind operator (U > 0).

    Copy of tests/python_1d/dg/test_operator.py::
    analytic_collocated_upwind_operator, restated on an explicit interval.
    """
    assert U > 0.0
    ngl = nop + 1
    _, wgl, D = lgl_and_diff(nop)
    J = (x1 - x0) / nelem / 2.0
    lift = U / (J * wgl[0])
    n = nelem * ngl
    L = np.zeros((n, n))
    for e in range(nelem):
        s = e * ngl
        L[s:s + ngl, s:s + ngl] = -(U / J) * D
        left = (e - 1) % nelem
        L[s, s] -= lift
        L[s, left * ngl + ngl - 1] += lift
    return L


def _dof(e, i, j, ngl=NGL):
    return e * ngl * ngl + j * ngl + i


# =============================================================================
# 1. Reduction to the F1-verified 1D operator
# =============================================================================

def test_reduction_to_1d_along_x():
    """cy = 0: every (ey, j) line is an independent periodic 1D problem in x,
    equal to the verified 1D operator; and those rows couple to nothing
    outside their own line."""
    cx = GATE_CX
    L2 = build_analytic_operator_2d(GATE_NOP, GATE_NEX, GATE_NEY,
                                    cx, 0.0, domain=GATE_DOMAIN)
    (xmin, xmax), _ = GATE_DOMAIN
    L1 = _analytic_1d(GATE_NOP, GATE_NEX, cx, xmin, xmax)

    for ey in (0, GATE_NEY - 1):
        for j in (0, 2, NGL - 1):
            idx = [_dof(ey * GATE_NEX + ex, i, j)
                   for ex in range(GATE_NEX) for i in range(NGL)]
            sub = L2[np.ix_(idx, idx)]
            assert np.allclose(sub, L1, atol=1e-12), (
                f"x-line (ey={ey}, j={j}) block differs from the verified "
                f"1D operator: max|diff| = {np.max(np.abs(sub - L1)):.3e}")
            other = np.delete(L2[idx, :], idx, axis=1)
            assert np.max(np.abs(other)) == 0.0, (
                f"x-line (ey={ey}, j={j}) rows couple outside their line "
                f"with cy = 0")


def test_reduction_to_1d_along_y():
    """cx = 0: same reduction along y -- on the gate domain this exercises
    the OTHER element size (dy = 5.0 vs dx = 2.5), i.e. the y-face lift
    factor 1/(w0*Jy), so a Jx/Jy confusion cannot pass both reductions."""
    cy = GATE_CY
    L2 = build_analytic_operator_2d(GATE_NOP, GATE_NEX, GATE_NEY,
                                    0.0, cy, domain=GATE_DOMAIN)
    _, (ymin, ymax) = GATE_DOMAIN
    L1 = _analytic_1d(GATE_NOP, GATE_NEY, cy, ymin, ymax)

    for ex in (0, GATE_NEX - 1):
        for i in (0, 1, NGL - 1):
            idx = [_dof(ey * GATE_NEX + ex, i, j)
                   for ey in range(GATE_NEY) for j in range(NGL)]
            sub = L2[np.ix_(idx, idx)]
            assert np.allclose(sub, L1, atol=1e-12), (
                f"y-line (ex={ex}, i={i}) block differs from the verified "
                f"1D operator: max|diff| = {np.max(np.abs(sub - L1)):.3e}")
            other = np.delete(L2[idx, :], idx, axis=1)
            assert np.max(np.abs(other)) == 0.0, (
                f"y-line (ex={ex}, i={i}) rows couple outside their line "
                f"with cx = 0")


# =============================================================================
# 2. Structural invariants at the gate configuration
# =============================================================================

@pytest.fixture(scope="module")
def gate_operator():
    L = build_analytic_operator_2d(GATE_NOP, GATE_NEX, GATE_NEY,
                                   GATE_CX, GATE_CY, domain=GATE_DOMAIN)
    m = lumped_mass_vector_2d(GATE_NOP, GATE_NEX, GATE_NEY,
                              domain=GATE_DOMAIN)
    return L, m


def test_gate_dof_count(gate_operator):
    L, m = gate_operator
    n = GATE_NEX * GATE_NEY * NGL * NGL
    assert L.shape == (n, n)
    assert m.shape == (n,)


def test_gate_conservation(gate_operator):
    L, m = gate_operator
    assert np.max(np.abs(m @ L)) < 1e-10


def test_gate_consistency(gate_operator):
    L, _ = gate_operator
    assert np.max(np.abs(L @ np.ones(L.shape[0]))) < 1e-9


def test_gate_spectrum(gate_operator):
    """Dissipativity and kernel structure.

    The genuine kernel (the constant, |lambda| ~ 1e-15) is counted at the
    tight 1e-10*rho threshold: exactly 1. The looser 1e-6*rho band contains
    3 at the gate configuration -- the kernel plus one conjugate-cancelled
    physical pair at |lambda| = 2|Re(lam_phys)| ~ 3.7e-7: the 2D operator is
    a Kronecker sum (and cx/dx = cy/dy makes the two 1D sub-operators the
    same matrix here), so lam + conj(lam) = 2*Re(lam) enters the spectrum
    with imaginary parts cancelled, and p=4 upwind's physical dissipation
    (~theta^(2p+2)) is ~2e-7 on a 4-element chain. Sandbox-verified
    2026-08-31: smallest |lambda| = 1.8e-15, 3.73e-7 (x2), 2.9e-4 (x2).
    Pinning the band count documents the phenomenon and catches spectral
    regressions; the margins are factor ~15 below and ~50 above the
    threshold at this configuration."""
    L, m = gate_operator
    rep = structural_report(L, m)
    assert rep["max_re_lambda"] < 1e-8 * rep["spectral_radius"], (
        f"max Re(lambda) = {rep['max_re_lambda']:.3e} -- not dissipative")
    assert rep["n_zero_modes"] == 1, (
        f"expected exactly one kernel mode at |lam| < 1e-10*rho, "
        f"found {rep['n_zero_modes']}")
    assert rep["n_near_zero_modes"] == 3, (
        f"expected 3 modes in the near-zero band at the gate config "
        f"(kernel + one conjugate-cancelled physical pair), "
        f"found {rep['n_near_zero_modes']}; smallest |lambda| = "
        f"{rep['smallest_abs_lambda']}")


# =============================================================================
# 3. Upwind direction under negative velocity
# =============================================================================

def test_negative_cx_upwinds_from_the_right():
    """cx < 0: the numerical flux takes the RIGHT state. The L element's
    x-max trace row must couple to its +x neighbour, and the R element's
    x-min rows must receive no face contribution (volume-only)."""
    nex, ney = 3, 5
    cx, cy = -0.7, 0.3
    L = build_analytic_operator_2d(GATE_NOP, nex, ney, cx, cy,
                                   domain=GATE_DOMAIN)
    e_l = 0 * nex + 0          # cell (0, 0)
    e_r = 0 * nex + 1          # cell (1, 0)
    k = 1
    row_l = _dof(e_l, NGL - 1, k)
    row_r = _dof(e_r, 0, k)
    assert abs(L[row_l, _dof(e_r, 0, k)]) > 0.0, (
        "cx < 0: downwind (L) trace row does not couple to its +x neighbour")
    assert L[row_r, _dof(e_l, NGL - 1, k)] == 0.0, (
        "cx < 0: upwind (R) trace row received a face contribution")

    m = lumped_mass_vector_2d(GATE_NOP, nex, ney, domain=GATE_DOMAIN)
    rep = structural_report(L, m)
    assert rep["conservation_mTL"] < 1e-10
    assert rep["consistency_L1"] < 1e-9
    assert rep["max_re_lambda"] < 1e-8 * rep["spectral_radius"]
