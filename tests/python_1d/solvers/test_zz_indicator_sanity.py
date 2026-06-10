"""Phase Z2.2 sanity check for compute_element_errors_zz().

Verifies the ZZ error indicator on full-solver state (not isolated linear
algebra — that's Phase Z1's job). Targets the failure mode that motivated
D-032: at t=0, the raw boundary-jump indicator returns identically zero for
every element because adjacent elements share their interface LGL node and
nodal interpolation produces identically zero interface jumps. The ZZ
indicator should be generically nonzero at t=0.

Five checks:
  1. ZZ errors are all > 0 at t=0 for each of icases {1, 14, 15, 16}.
  2. Raw-jump errors are all (essentially) zero at t=0 — confirms that
     the comparison set up by D-032 is the right one.
  3. For symmetric IC + symmetric mesh (icase=1, default 4-element mesh),
     the ZZ error array is symmetric (errors[i] ≈ errors[n_active-1-i]).
  4. For Gaussian IC, the elements covering the peak (middle of domain)
     have larger ZZ errors than the elements covering the flat tails.
  5. After a small (sub-T) solver advance, ZZ errors remain nonzero —
     i.e., ZZ does not have a reciprocal "blind at t > 0" pathology.

Run from repo root: python tests/python_1d/solvers/test_zz_indicator_sanity.py
"""

import sys

import numpy as np

from backends.python_1d.solvers.dg_advection_solver_multiround import DGAdvectionSolver
from backends.python_1d.solvers.error_indicators import (
    compute_element_errors,
    compute_element_errors_zz,
)

# Default base mesh — matches solver and env defaults: 4 non-uniform elements,
# symmetric about x=0.
BASE_XELEM = np.array([-1.0, -0.4, 0.0, 0.4, 1.0])


def make_solver(icase):
    """Build a solver with the standard Stage 1A multiround configuration."""
    return DGAdvectionSolver(
        nop=4,
        xelem=BASE_XELEM.copy(),
        max_elements=120,
        max_level=3,
        icase=icase,
        balance=False,
        verbose=False,
    )


def check_initial_state(icase):
    """Checks 1, 2, 4 for one IC."""
    solver = make_solver(icase)

    e_zz = compute_element_errors_zz(solver)
    e_raw = compute_element_errors(solver)

    print(f"  icase={icase:2d}:")
    print(f"    ZZ  errors: {np.array2string(e_zz,  formatter={'float_kind': lambda x: f'{x:.3e}'})}")
    print(f"    raw errors: {np.array2string(e_raw, formatter={'float_kind': lambda x: f'{x:.3e}'})}")

    # Check 1: ZZ errors all > 0
    if np.any(e_zz <= 0):
        print("    ✗ FAIL: ZZ errors should all be > 0 (some are <= 0)")
        return False
    else:
        print(f"    ✓ ZZ errors all positive (min = {e_zz.min():.3e})")

    # Check 2: raw errors all near zero (the t=0 blindness D-032 addresses)
    if np.max(np.abs(e_raw)) > 1e-10:
        print(f"    ✗ FAIL: raw errors should be ~0 at t=0 "
              f"(max abs = {np.max(np.abs(e_raw)):.3e}). "
              f"Either initialization changed or this isn't a clean t=0 state.")
        return False
    else:
        print(f"    ✓ raw errors ~0 at t=0 (max abs = {np.max(np.abs(e_raw)):.3e})")

    # Check 4: peak elements > tail elements (Gaussian / bump / soliton / Mexican hat
    # are all centered at x=0 with their largest features in the middle two elements).
    # Default mesh: elements 0 and 3 are tails, elements 1 and 2 cover the central feature.
    if len(e_zz) == 4:
        center_max = max(e_zz[1], e_zz[2])
        tail_max = max(e_zz[0], e_zz[3])
        if center_max > tail_max:
            print(f"    ✓ center elements > tail elements "
                  f"(center max = {center_max:.3e}, tail max = {tail_max:.3e})")
        else:
            print(f"    ✗ FAIL: expected center elements > tail elements, "
                  f"got center max = {center_max:.3e}, tail max = {tail_max:.3e}")
            return False

    return True


def check_symmetric_ic_produces_symmetric_errors():
    """Check 3: icase=1 (Gaussian centered at 0) on symmetric mesh → symmetric errors."""
    solver = make_solver(icase=1)
    e_zz = compute_element_errors_zz(solver)

    # Default mesh has 4 elements, symmetric about x=0: pairs (0,3) and (1,2).
    rel_diff_outer = abs(e_zz[0] - e_zz[3]) / max(e_zz[0], e_zz[3], 1e-30)
    rel_diff_inner = abs(e_zz[1] - e_zz[2]) / max(e_zz[1], e_zz[2], 1e-30)

    print("  icase=1 (Gaussian) symmetry check:")
    print(f"    e[0] = {e_zz[0]:.6e}, e[3] = {e_zz[3]:.6e}  → rel diff = {rel_diff_outer:.2e}")
    print(f"    e[1] = {e_zz[1]:.6e}, e[2] = {e_zz[2]:.6e}  → rel diff = {rel_diff_inner:.2e}")

    tol = 1e-10  # symmetry should hold to round-off
    if rel_diff_outer < tol and rel_diff_inner < tol:
        print(f"    ✓ symmetric to {tol:.0e}")
        return True
    else:
        print(f"    ✗ FAIL: not symmetric to {tol:.0e}")
        return False


def check_post_advance_still_nonzero():
    """Check 5: ZZ errors remain nonzero after a sub-T solver advance."""
    solver = make_solver(icase=1)

    # Advance by ~0.5 * T_remesh-interval, where step_domain_fraction defaults
    # to 0.05 in the env. domain_length = 2.0, wave_speed = 1.0 → T = 0.1.
    domain_length = solver.xelem[-1] - solver.xelem[0]
    T = 0.05 * domain_length / solver.wave_speed
    advance_duration = 0.5 * T

    dx_min = np.min(np.diff(solver.xelem))
    dt = solver.courant_max * dx_min / solver.wave_speed

    time_advanced = 0.0
    while time_advanced < advance_duration - 1e-15:
        step_dt = min(dt, advance_duration - time_advanced)
        solver.step(dt=step_dt)
        time_advanced += step_dt

    e_zz = compute_element_errors_zz(solver)

    print(f"  Post-advance check (icase=1, advanced {time_advanced:.4f}s ≈ 0.5 T):")
    print(f"    ZZ errors: {np.array2string(e_zz, formatter={'float_kind': lambda x: f'{x:.3e}'})}")

    if np.all(e_zz > 0):
        print(f"    ✓ all ZZ errors still positive (min = {e_zz.min():.3e})")
        return True
    else:
        print("    ✗ FAIL: some ZZ errors are zero or negative after advance")
        return False


def main():
    print("Test 1+2+4: ZZ nonzero at t=0, raw zero at t=0, center > tails")
    print("-" * 70)
    pass_initial = all(check_initial_state(icase) for icase in [1, 14, 15, 16])

    print("\nTest 3: symmetric IC + symmetric mesh → symmetric ZZ errors")
    print("-" * 70)
    pass_symmetry = check_symmetric_ic_produces_symmetric_errors()

    print("\nTest 5: ZZ remains nonzero post-advance")
    print("-" * 70)
    pass_post_advance = check_post_advance_still_nonzero()

    print("\n" + "=" * 70)
    if pass_initial and pass_symmetry and pass_post_advance:
        print("All Phase Z2 sanity checks passed.")
    else:
        print("Some checks FAILED — see above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
