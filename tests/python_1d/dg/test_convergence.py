"""Regression pins for the spatial accuracy of the Python 1D DG operator.

These tests assert that the reference solver converges at p+1 for a smooth
periodic solution, that the observed order *tracks* the polynomial degree, and
that mass is conserved to machine precision. They exist because the original
convergence study was lost across repo restructures, leaving the reference
operator -- the oracle for the Jexpresso DG cross-validation (DG roadmap
Phase 2F) -- with no artifact establishing its order of accuracy.

The diagnostic that produces the full tables, CSVs, and log-log plots is
`tools/convergence_study.py`. This module deliberately imports its machinery
rather than reimplementing it, so the pin and the diagnostic cannot drift
apart, and deleting the tool breaks the suite loudly.

Reference values (2026-07-27, nop=4, icase 8, meshes 8/16/32/64, courant 0.2):

    collocated (nq=ngl, Jexpresso DGSEM convention)
        rates 4.972 / 4.993 / 5.000     L2 at h=1/32: 4.253e-10
    overint (nq=nop+2, production default)
        rates 4.962 / 4.934 / 4.979     L2 at h=1/32: 1.527e-10

    order tracking (collocated): p=2 -> 3.008, p=3 -> 4.000, p=4 -> 5.000
    mass drift: 1e-16 .. 1e-18 in all runs
    dt/2 check: relative change < 1e-6 (time error ~6 orders below spatial)

Overint is ~2.8-3.0x more accurate than collocated at equal DOF -- the same
order with a smaller constant, the expected signature of two discretizations
differing only in mass-matrix quadrature. That gap is legitimate and is why
the Phase-2F operator comparison against Jexpresso must use collocated mode.

Scope: this covers the spatial operator only -- volume term, upwind flux, mass
matrix, periodic wrap -- on uniform conforming meshes. It does not cover the
AMR projection path (see tests/python_1d/amr/test_projection_exactness.py),
the Rusanov flux, non-periodic boundaries, or non-uniform meshes.

Runtime: roughly 8 s (dominated by the pure-Python RK stage loop).

Run with: pytest tests/python_1d/dg/test_convergence.py -v
"""

import numpy as np
import pytest

from tools.convergence_study import pairwise_rates, run_case

# =============================================================================
# Study configuration
# =============================================================================

# icase 8 (sin(pi*x)) is SMOOTH and exactly periodic, but its branch in
# `exact_solution` is *static* -- it ignores `time`. That formula is correct
# only at integer multiples of the traversal period, where the advected
# solution coincides with the initial condition. Domain length 2 and wave
# speed 2 give a period of exactly 1.0, so T_FINAL must stay an integer.
# Do not shorten it to speed these tests up; the comparison would become
# meaningless rather than merely less accurate.
ICASE_SMOOTH = 8
T_FINAL = 1.0

# icase 1 (Gaussian, beta=256) has nonzero mean, so mass conservation can be
# reported relative to a nonzero integral. sin(pi*x) integrates to zero and
# gives no relative scale.
ICASE_NONZERO_MEAN = 1

WAVE_SPEED = 2.0
COURANT = 0.2

# Loose bands. The point is to catch a broken operator, not to pin digits that
# legitimately move with numpy/BLAS versions.
RATE_TOLERANCE = 0.5
MASS_DRIFT_ABS = 1e-12


def _n_steps(nop, nelem_finest, t_final=T_FINAL, courant=COURANT):
    """Steps for a sweep, sized on the finest mesh and landing exactly on t_final.

    Mirrors the dt policy in `tools/convergence_study.main`: one dt for every
    mesh in the sweep, from the standard DG h/(2p+1) scaling. Holding dt fixed
    across meshes is what makes the measured slope purely spatial. If the dt
    policy in the tool changes, this must follow.
    """
    h_fine = 2.0 / nelem_finest
    dt_target = courant * h_fine / ((2 * nop + 1) * WAVE_SPEED)
    return int(np.ceil(t_final / dt_target))


def _sweep(nop, nelems, nq_mode, icase=ICASE_SMOOTH):
    """Run a mesh sweep and return the per-mesh result dicts."""
    n_steps = _n_steps(nop, max(nelems))
    return [
        run_case(icase, nop, nelem, T_FINAL, n_steps, nq_mode, verbose=False)
        for nelem in nelems
    ]


# =============================================================================
# Fixtures (module-scoped: the sweeps are the expensive part)
# =============================================================================

@pytest.fixture(scope="module")
def sweep_p4_collocated():
    """nop=4, collocated LGL (nq=ngl) -- the Jexpresso-matching convention."""
    return _sweep(nop=4, nelems=[16, 32, 64], nq_mode="collocated")


@pytest.fixture(scope="module")
def sweep_p4_overint():
    """nop=4, over-integrated (nq=nop+2) -- the production default."""
    return _sweep(nop=4, nelems=[16, 32, 64], nq_mode="overint")


# =============================================================================
# Order of accuracy
# =============================================================================

@pytest.mark.parametrize("fixture_name", ["sweep_p4_collocated", "sweep_p4_overint"])
def test_spatial_order_is_p_plus_one(fixture_name, request):
    """The finest pairwise rate reaches p+1 in both quadrature modes."""
    results = request.getfixturevalue(fixture_name)
    expected = results[0]["nop"] + 1

    rates = pairwise_rates(results)
    finest = rates[-1]

    assert finest is not None, "no usable rate -- errors may have hit zero"
    assert abs(finest - expected) < RATE_TOLERANCE, (
        f"{fixture_name}: finest pairwise rate {finest:.3f}, expected "
        f"~{expected}. Rates across the sweep: "
        f"{[None if r is None else round(r, 3) for r in rates]}"
    )


@pytest.mark.parametrize("nop", [2, 3])
def test_order_tracks_polynomial_degree(nop):
    """Observed order follows p+1 as p varies.

    A single correct order at one degree can be produced by a wrong operator
    for the wrong reason. Requiring the exponent to move with p is a much
    stronger statement, and it is cheap at low order.
    """
    results = _sweep(nop=nop, nelems=[8, 16, 32], nq_mode="collocated")
    expected = nop + 1

    finest = pairwise_rates(results)[-1]

    assert finest is not None
    assert abs(finest - expected) < RATE_TOLERANCE, (
        f"nop={nop}: finest pairwise rate {finest:.3f}, expected ~{expected}"
    )


def test_errors_decrease_monotonically(sweep_p4_collocated):
    """Refinement must reduce the error at every step of the sweep.

    Cheap guard against a sweep that has drifted into the round-off floor,
    where a fitted slope can still look plausible while the errors stagnate.
    """
    errs = [r["l2_abs"] for r in sweep_p4_collocated]
    for coarse, fine in zip(errs[:-1], errs[1:]):
        assert fine < coarse, f"error did not decrease under refinement: {errs}"


# =============================================================================
# Quadrature-mode relationship (the Phase-2F premise)
# =============================================================================

def test_overint_is_more_accurate_at_equal_dof(sweep_p4_collocated, sweep_p4_overint):
    """Over-integration beats collocation by a bounded, roughly constant factor.

    Both modes are p+1; they differ only in the mass-matrix quadrature, so the
    error ratio should be O(1) and near-constant across the sweep (~2.8-3.0x
    as measured). This is pinned because it underwrites the Phase-2F decision
    to compare Jexpresso against *collocated* mode: a comparison against
    overint would show a ~3x gap that is entirely legitimate and would be easy
    to misread as a defect in the Jexpresso operator.
    """
    for coll, over in zip(sweep_p4_collocated, sweep_p4_overint):
        assert coll["nelem"] == over["nelem"]
        ratio = coll["l2_abs"] / over["l2_abs"]
        assert 1.5 < ratio < 6.0, (
            f"nelem={coll['nelem']}: collocated/overint error ratio {ratio:.2f} "
            f"outside the expected O(1) band -- the two quadrature conventions "
            f"may no longer be the only difference between the modes"
        )


# =============================================================================
# Conservation
# =============================================================================

def test_mass_conserved_smooth_ic(sweep_p4_collocated):
    """Mass drift stays at machine precision across the sweep."""
    for r in sweep_p4_collocated:
        assert r["mass_drift_abs"] < MASS_DRIFT_ABS, (
            f"nelem={r['nelem']}: mass drift {r['mass_drift_abs']:.3e}"
        )


def test_mass_conserved_nonzero_mean_ic():
    """Relative mass conservation on an IC with nonzero integral.

    sin(pi*x) integrates to zero, so its drift has no relative scale. The
    Gaussian does, which makes this the meaningful conservation baseline --
    and the number the Jexpresso conservation check (Phase 2F) must match.
    """
    n_steps = _n_steps(nop=4, nelem_finest=32)
    r = run_case(ICASE_NONZERO_MEAN, 4, 32, T_FINAL, n_steps,
                 "collocated", verbose=False)

    assert abs(r["mass_initial"]) > 1e-3, (
        "expected a nonzero-mean IC -- the relative drift below is meaningless "
        "otherwise"
    )

    relative_drift = r["mass_drift_abs"] / abs(r["mass_initial"])
    assert relative_drift < 1e-12, (
        f"relative mass drift {relative_drift:.3e} over {n_steps} steps"
    )


# =============================================================================
# Time-error isolation
# =============================================================================

def test_time_error_is_negligible():
    """Halving dt must not move the L2 error.

    If it does, the 4th-order LSERK time error is contaminating the measured
    spatial slope and every order assertion above is void -- the tests would
    then be pinning a blend of space and time error rather than the operator.
    """
    nop, nelem = 4, 32
    n_steps = _n_steps(nop, nelem)

    base = run_case(ICASE_SMOOTH, nop, nelem, T_FINAL, n_steps,
                    "collocated", verbose=False)
    refined = run_case(ICASE_SMOOTH, nop, nelem, T_FINAL, 2 * n_steps,
                       "collocated", verbose=False)

    change = abs(refined["l2_abs"] - base["l2_abs"]) / base["l2_abs"]
    assert change < 1e-3, (
        f"L2 error moved {change:.2%} when dt was halved -- the spatial order "
        f"measured by this module is not isolated from time error"
    )
