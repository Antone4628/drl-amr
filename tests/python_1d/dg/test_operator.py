"""Regression pins for the assembled 1D DG semi-discrete operator.

The DG-roadmap Phase-2F verification (2026-07-27c, D-064) extracted the
semi-discrete operator L (du/dt = L q) from both the Jexpresso and Python
codes and compared each entrywise against an analytic collocated upwind DG
operator constructed from first principles. All three agreed to ~1e-12
absolute / ~4e-15 relative. This module makes the Python leg of that
verification permanent: the analytic construction lives here (it was
otherwise session-ephemeral -- the same loss mode that cost us the original
convergence study), and the entrywise comparison re-runs on every test pass.

It also pins the welded-wrap assertion from tools/dump_dg_operator.py: the
solver's step() contains a conditional `qp[-1] = qp[0]` that lives OUTSIDE
Dhat, so a clean operator cannot expose it. It is the same defect shape found
and fixed in Jexpresso (commit 586e7ba3) after surviving four days behind a
green smoke test.

Reference values (2026-07-27c, F1):

    nop=4 nelem=50: Python vs analytic max|diff| = 3.75e-12 (rel 3.8e-15)
    nop=3 nelem=20: Python vs analytic max|diff| = 3.69e-13 (rel 1.5e-15)
    wrap coupling entry U/(w0*J): 1000.0 and 240.0 respectively
    conservation w^T L ~ 1e-15; consistency L@1 ~ 1e-13
    spectrum: max Re(lambda) <= 0, single zero mode

Scope: collocated LGL (lumped mass), upwind flux, uniform periodic mesh --
the Jexpresso-matching convention. Over-integration is covered by order in
test_convergence.py; its operator is structurally different (dense M^-1
within elements) and is not pinned entrywise here.

Independence caveat: the analytic construction reuses lgl_gen/Lagrange_basis
for nodes, weights, and the differentiation matrix. Those are shared with the
solver, but the defect classes this module guards -- assembly, flux
correction, periodic wrap, mass lumping -- all live above the basis level,
and the basis itself was verified against the fully independent Julia-side
analytic construction in F1.

Run with: pytest tests/python_1d/dg/test_operator.py -v
"""

import numpy as np
import pytest

from backends.python_1d.dg.basis import Lagrange_basis, lgl_gen
from tools.convergence_study import DOMAIN
from tools.dump_dg_operator import build_operator

# The two F1-verified configurations. (F1's second config exercised rusanov
# on the Jexpresso side; the Python solver's flux is upwind, which for linear
# advection rusanov with lambda = U reproduces exactly -- verified in F1.)
CONFIGS = [
    pytest.param((4, 50), id="nop4_nelem50"),
    pytest.param((3, 20), id="nop3_nelem20"),
]

ABS_TOL = 1e-8   # F1's own criterion: zero entries differing above 1e-8


# =============================================================================
# Analytic construction
# =============================================================================

def analytic_collocated_upwind_operator(nop, nelem, U, domain=DOMAIN):
    """Collocated (DGSEM) upwind DG operator for du/dt + U du/dx = 0, U > 0,
    on a uniform periodic mesh, built from first principles.

    Strong form on each element (Jacobian J = h/2, LGL nodes/weights xi_i,
    w_i, differentiation matrix D_ij = psi_j'(xi_i)):

        du_i/dt = -(U/J) sum_j D_ij u_j
                  + delta_{i,0} * (U/(J w_0)) * (u_N^{left nbr} - u_0)

    The outflow-face correction vanishes identically for U > 0 (upwind takes
    the interior state there), so the flux correction lands in exactly one
    row per element -- the structural signature of lumped mass + upwind that
    F1 confirmed in both codes.
    """
    assert U > 0.0
    ngl = nop + 1
    xgl, wgl = lgl_gen(ngl)
    psi, dpsi = Lagrange_basis(ngl, ngl, xgl, xgl)

    # Collocated evaluation: the basis at its own nodes must be the identity.
    assert np.allclose(psi, np.eye(ngl), atol=1e-12), (
        "Lagrange_basis at its own LGL nodes is not the identity -- "
        "orientation or node-generation assumptions are wrong"
    )

    # Orientation self-check via exact derivatives: D @ xi = 1.
    D = dpsi.T
    if not np.allclose(D @ xgl, np.ones(ngl), atol=1e-10):
        D = dpsi
    assert np.allclose(D @ xgl, np.ones(ngl), atol=1e-10), (
        "neither dpsi nor dpsi.T differentiates xi to 1 -- Lagrange_basis "
        "conventions have changed"
    )

    h = (domain[1] - domain[0]) / nelem
    J = h / 2.0
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


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(scope="module", params=CONFIGS)
def config(request):
    nop, nelem = request.param
    solver, L = build_operator(icase=8, nop=nop, nelem=nelem,
                               nq_mode="collocated")
    return solver, L, nop, nelem


# =============================================================================
# The CI-critical assertion
# =============================================================================

def test_wrap_is_not_welded(config):
    """build_operator raises on a welded wrap; reaching here means it didn't.

    Re-asserted explicitly so this test names the failure if the fixture
    ever stops going through build_operator.
    """
    solver, _, _, _ = config
    assert solver.periodicity[0] != solver.periodicity[-1]


def test_dof_count_has_no_shared_nodes(config):
    """A DG numbering duplicates interface DOFs: npoin_dg == nelem * ngl.

    This is the check that exposed the Jexpresso collapse (249 vs 250)."""
    solver, L, _, nelem = config
    n = nelem * solver.ngl
    assert solver.npoin_dg == n
    assert L.shape == (n, n)


# =============================================================================
# Entrywise verification (the F1 method, D-064)
# =============================================================================

def test_operator_matches_analytic_entrywise(config):
    solver, L, nop, nelem = config
    A = analytic_collocated_upwind_operator(nop, nelem, U=solver.wave_speed)

    diff = np.max(np.abs(L - A))
    rel = diff / np.max(np.abs(A))
    assert diff < ABS_TOL, (
        f"nop={nop} nelem={nelem}: max entrywise |L - L_analytic| = {diff:.3e} "
        f"(rel {rel:.3e}); F1 reference was ~4e-12 abs. The assembled operator "
        f"has structurally diverged from the verified discretization."
    )


def test_periodic_wrap_is_flux_coupling(config):
    """The wrap must appear as a genuine off-diagonal flux entry of exactly
    U/(w0*J) -- element 0's inflow row coupled to the last element's last
    node -- not as a shared or welded DOF."""
    solver, L, nop, nelem = config
    ngl = nop + 1
    _, wgl = lgl_gen(ngl)
    J = (DOMAIN[1] - DOMAIN[0]) / nelem / 2.0
    expected = solver.wave_speed / (J * wgl[0])   # 1000.0 / 240.0 in F1

    entry = L[0, (nelem - 1) * ngl + (ngl - 1)]
    assert np.isclose(entry, expected, rtol=1e-10), (
        f"wrap coupling entry {entry} != U/(w0*J) = {expected}"
    )


# =============================================================================
# Structural properties (independent of the analytic helper)
# =============================================================================

def test_conservation(config):
    """m^T L = 0 with m the lumped mass diag(J*w): d/dt of total mass is
    zero for every state. F1 reference ~1e-15."""
    solver, L, nop, nelem = config
    ngl = nop + 1
    _, wgl = lgl_gen(ngl)
    J = (DOMAIN[1] - DOMAIN[0]) / nelem / 2.0
    m = np.tile(J * wgl, nelem)
    assert np.max(np.abs(m @ L)) < 1e-10


def test_consistency(config):
    """L @ 1 = 0: a constant state has zero time derivative. F1 ~1e-13."""
    _, L, _, _ = config
    n = L.shape[0]
    assert np.max(np.abs(L @ np.ones(n))) < 1e-9


def test_spectrum_is_dissipative(config):
    """Upwind DG: max Re(lambda) <= 0 with exactly one zero mode (the
    constant). A positive real part means anti-dissipation -- the signature
    of a sign error in the flux correction."""
    _, L, _, _ = config
    lam = np.linalg.eigvals(L)
    rho = np.max(np.abs(lam))

    assert np.max(lam.real) < 1e-8 * rho, (
        f"max Re(lambda) = {np.max(lam.real):.3e} > 0 at spectral radius "
        f"{rho:.1f} -- the operator is not dissipative"
    )
    n_zero = int(np.sum(np.abs(lam) < 1e-6 * rho))
    assert n_zero == 1, f"expected exactly one zero mode, found {n_zero}"