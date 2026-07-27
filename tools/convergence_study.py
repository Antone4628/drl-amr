"""Grid-convergence study for the Python 1D DG reference solver.

Establishes the observed spatial order of accuracy of `DGAdvectionSolver` by
refining a uniform mesh at fixed polynomial order and measuring the L2 error
against the analytic advected solution. For a smooth solution the expected
rate is p+1 (= nop+1).

Why this exists
---------------
The original convergence study predates several restructures and no longer
exists in the repo. The June parity gate (D-052) verified *foundation
soundness* -- deterministic correctness, deployment fidelity, and
training-soundness-in-regime -- but none of those legs establishes that the
spatial operator is order-accurate. This script is the missing artifact, and
it is the oracle for the Jexpresso DG cross-validation (DG roadmap Phase 2F):
if the two codes disagree there, a recorded order study on this side is what
tells us which one is wrong.

Two quadrature modes
--------------------
The production solver over-integrates (`nq = nop + 2`), which yields an exact,
non-diagonal mass matrix. Jexpresso's DG path is collocated LGL, giving the
lumped diagonal mass `M = Je * omega`. Both discretizations converge at p+1,
but their operators are *not* entrywise identical. `--nq-mode collocated`
reproduces the Jexpresso convention, and is the mode the Phase-2F operator
comparison must use.

    overint     nq = nop + 2   (production default; exact mass matrix)
    collocated  nq = ngl       (DGSEM / Jexpresso convention; lumped mass)

Error norm
----------
The L2 error is computed with the assembled mass matrix,

    ||e||_L2 = sqrt( e^T M e ),      e = q_numerical - q_exact(t_final)

which is the correct discrete L2 norm (the DG mass matrix is block diagonal
here because `periodicity_dg` is the identity). The exact solution is sampled
at the DG nodes, so the measured error includes the interpolation error of the
exact solution -- standard, and also O(h^(p+1)).

Time-error isolation
--------------------
A convergence study that lets dt shrink with h measures a moving mixture of
space and time error and can report a plausible-looking but meaningless slope.
This script therefore fixes a single dt for the entire sweep, chosen from the
*finest* mesh, and lands exactly on t_final. It then (by default) re-runs the
finest mesh at dt/2 and reports whether the error moved; if it did, the time
error of the 4th-order LSERK integrator is polluting the spatial slope and the
study is invalid -- lower --courant and re-run.

Usage
-----
Run from the project root (`~/projects/drl-amr`) with the `rl-amr` env active:

    python -m tools.convergence_study
    python -m tools.convergence_study --icase 8 10 --nop 4
    python -m tools.convergence_study --nq-mode collocated
    python -m tools.convergence_study --icase 1 --nelem 16 32 64 128
    python -m tools.convergence_study --icase 2 --allow-nonsmooth   # see the ~0.5 rate

Outputs (default `tools/`): a printed table, `convergence_study.csv`,
`convergence_study.json`, and `convergence_study.png`.
"""

import argparse
import json
import os
import sys
import time as _time

import numpy as np

from backends.python_1d.dg.basis import Lagrange_basis, lgl_gen
from backends.python_1d.solvers.dg_advection_solver_multiround import DGAdvectionSolver
from backends.python_1d.solvers.utils import exact_solution

# =============================================================================
# Initial-condition classification
# =============================================================================
# Convergence order is a property of the operator, not the IC -- one smooth,
# exactly-periodic case is sufficient. The classification below exists to stop
# an invalid case from silently producing a number that looks like a result.

# Smooth and periodic on [-1, 1]: a valid p+1 fit is expected.
SMOOTH_CASES = {
    1: "Gaussian pulse, beta=256 (sharp: pre-asymptotic below ~32 elements)",
    3: "Diffusing Gaussian, visc=0 (periodicity error ~1e-14)",
    8: "sin(pi*x) -- analytic, exactly periodic. RECOMMENDED DEFAULT "
        "(static formula: t_final must be an integer multiple of 1.0)",
    10: "tanh(5 sin(pi x)) smooth square (has negative values)",
    11: "erf(5 sin(pi x)) smooth square",
    12: "sigmoid smooth square (has negative values)",
    13: "Two Gaussian pulses, beta=256 (sharp)",
    14: "Compact-support bump (C^inf, not analytic)",
    15: "sech^2 soliton (periodicity floor ~1e-9 -- error stalls there)",
    16: "Mexican hat / Ricker wavelet (has negative values)",
}

# Cannot produce a valid order fit. Reason is printed on use.
BLOCKED_CASES = {
    2: "discontinuous square pulse -- L2 rate is ~0.5, not p+1",
    4: "discontinuous square pulse -- L2 rate is ~0.5, not p+1",
    5: "discontinuous step -- L2 rate is ~0.5, not p+1",
    6: "sin((x+1)pi/2) -- value is periodic but the derivative is not",
    7: "STATIC: this case ignores `time` entirely; the 'exact' solution "
       "does not advect, so the measured error is meaningless",
    9: "defined on [-4, 4] with wave speed 1.0 -- incompatible with this "
       "study's [-1, 1] mesh",
}

DOMAIN = (-1.0, 1.0)
WAVE_SPEED = 2.0
PERIOD = (DOMAIN[1] - DOMAIN[0]) / WAVE_SPEED  # = 1.0

# ICs whose branch in `exact_solution` ignores `time` entirely. Their formula
# is the correct advected solution ONLY at integer multiples of PERIOD, where
# the wave has returned to its initial position. At any other t_final they
# compare the numerical solution against a stationary profile, yielding a
# meaningless error that presents as a failed convergence test.
STATIC_CASES = {6, 7, 8}


# =============================================================================
# Solver construction
# =============================================================================

def make_solver(icase, nop, nelem, nq_mode):
    """Build a fixed-mesh (no AMR) solver on a uniform mesh over DOMAIN.

    A uniform mesh is required for a clean h-study, so the solver is
    constructed directly rather than via `reset()` (which hardcodes the
    non-uniform 4-element base mesh [-1, -0.4, 0, 0.4, 1]).

    AMR is inert here: `balance=False` and no adaptation call is ever made.
    `max_level` is left at 1 only because the forest builder expects a
    positive depth; it has no effect on the discretization.
    """
    xelem = np.linspace(DOMAIN[0], DOMAIN[1], nelem + 1)

    solver = DGAdvectionSolver(
        nop=nop,
        xelem=xelem,
        max_elements=max(64, 4 * nelem),
        max_level=1,
        courant_max=0.1,      # unused: every step is driven with an explicit dt
        icase=icase,
        periodic=True,
        verbose=False,
        balance=False,
    )

    if nq_mode == "collocated":
        _make_collocated(solver)

    return solver


def _make_collocated(solver):
    """Switch the solver to collocated LGL quadrature (nq == ngl).

    Reproduces the Jexpresso DGSEM convention: quadrature points coincide with
    the interpolation nodes, so the mass matrix lumps to diag(Je * omega).
    Rebuilds the bases, the projection operators, and all DG matrices.
    """
    solver.nq = solver.ngl
    solver.xnq, solver.wnq = lgl_gen(solver.nq)
    solver.psi, solver.dpsi = Lagrange_basis(
        solver.ngl, solver.nq, solver.xgl, solver.xnq
    )
    solver._initialize_projections()
    solver._update_matrices()


# =============================================================================
# Diagnostics
# =============================================================================

def l2_error(solver, t_final, icase):
    """Mass-matrix L2 error against the analytic solution at t_final.

    Returns (absolute, relative). The relative error divides by ||q_exact||,
    which is the quantity to compare across ICs of different amplitude.
    """
    qe, _ = exact_solution(solver.coord, solver.npoin_dg, t_final, icase)
    e = solver.q - qe

    num = float(e @ (solver.M @ e))
    den = float(qe @ (solver.M @ qe))

    # Guard against tiny negatives from round-off in a near-zero norm.
    abs_err = np.sqrt(max(num, 0.0))
    rel_err = abs_err / np.sqrt(den) if den > 0.0 else np.nan
    return abs_err, rel_err


def total_mass(solver):
    """Integral of q over the domain, via the assembled mass matrix.

    Uses the partition-of-unity property of the Lagrange basis:
    ones^T M q = integral( sum_i phi_i * sum_j q_j phi_j ) = integral(q).
    Conserved exactly (to round-off) for periodic linear advection -- this is
    the Python-side baseline for the Phase-2F conservation check.
    """
    return float(np.ones(solver.npoin_dg) @ (solver.M @ solver.q))


# =============================================================================
# Single run
# =============================================================================

def run_case(icase, nop, nelem, t_final, n_steps, nq_mode, verbose=True):
    """Advance one mesh to t_final with a fixed dt and report the error."""
    t0 = _time.perf_counter()

    solver = make_solver(icase, nop, nelem, nq_mode)
    dt = t_final / n_steps

    mass_initial = total_mass(solver)

    for _ in range(n_steps):
        solver.step(dt=dt)

    if not np.all(np.isfinite(solver.q)):
        raise RuntimeError(
            f"Solution went non-finite (icase={icase}, nelem={nelem}). "
            f"dt={dt:.3e} is likely above the stability limit -- lower --courant."
        )

    abs_err, rel_err = l2_error(solver, t_final, icase)
    mass_final = total_mass(solver)

    result = {
        "icase": icase,
        "nop": nop,
        "nelem": nelem,
        "h": (DOMAIN[1] - DOMAIN[0]) / nelem,
        "npoin_dg": int(solver.npoin_dg),
        "n_steps": n_steps,
        "dt": dt,
        "wave_speed": float(solver.wave_speed),
        "l2_abs": abs_err,
        "l2_rel": rel_err,
        "mass_initial": mass_initial,
        "mass_final": mass_final,
        "mass_drift_abs": abs(mass_final - mass_initial),
        "wall_time_s": _time.perf_counter() - t0,
    }

    if verbose:
        print(
            f"    nelem={nelem:4d}  h={result['h']:.6f}  "
            f"L2={abs_err:.6e}  rel={rel_err:.6e}  "
            f"mass drift={result['mass_drift_abs']:.2e}  "
            f"({result['wall_time_s']:.1f}s)"
        )

    return result


# =============================================================================
# Rates and fitting
# =============================================================================

def pairwise_rates(results):
    """Observed order between consecutive meshes: log(e1/e2) / log(h1/h2).

    More informative than a single fitted slope, because it shows where the
    sweep enters the asymptotic range (rates climbing to p+1) and where it
    leaves it (rates collapsing at the round-off floor).
    """
    rates = [None]
    for prev, cur in zip(results[:-1], results[1:]):
        if prev["l2_abs"] <= 0.0 or cur["l2_abs"] <= 0.0:
            rates.append(None)
            continue
        rate = np.log(prev["l2_abs"] / cur["l2_abs"]) / np.log(prev["h"] / cur["h"])
        rates.append(float(rate))
    return rates


def fitted_slope(results):
    """Least-squares slope of log(L2) vs log(h) over the whole sweep."""
    usable = [r for r in results if r["l2_abs"] > 0.0]
    if len(usable) < 2:
        return float("nan")
    logh = np.log([r["h"] for r in usable])
    loge = np.log([r["l2_abs"] for r in usable])
    return float(np.polyfit(logh, loge, 1)[0])


# =============================================================================
# Reporting
# =============================================================================

def print_table(icase, results, rates, slope, nop):
    expected = nop + 1
    print()
    print(f"  icase {icase}: {SMOOTH_CASES.get(icase, BLOCKED_CASES.get(icase, '?'))}")
    print("  " + "-" * 76)
    print(f"  {'nelem':>6} {'h':>10} {'DOF':>7} {'L2 error':>14} "
          f"{'rel L2':>12} {'rate':>7} {'mass drift':>12}")
    print("  " + "-" * 76)
    for r, rate in zip(results, rates):
        rate_s = "  --  " if rate is None else f"{rate:6.3f}"
        print(f"  {r['nelem']:>6d} {r['h']:>10.6f} {r['npoin_dg']:>7d} "
              f"{r['l2_abs']:>14.6e} {r['l2_rel']:>12.4e} {rate_s:>7} "
              f"{r['mass_drift_abs']:>12.2e}")
    print("  " + "-" * 76)
    print(f"  fitted slope over full sweep : {slope:.3f}   (expected {expected})")

    # Verdict based on the last pairwise rate -- the most asymptotic estimate.
    tail = [r for r in rates if r is not None]
    if tail:
        last = tail[-1]
        if last >= expected - 0.25:
            verdict = "PASS -- consistent with p+1"
        elif last < expected - 1.0:
            verdict = ("FAIL -- well below p+1. Check: IC smoothness, "
                       "time-error pollution, filtering/limiting")
        else:
            verdict = ("MARGINAL -- possibly pre-asymptotic (coarse meshes) "
                       "or approaching the round-off floor (fine meshes)")
        print(f"  finest pairwise rate         : {last:.3f}   -> {verdict}")
    print()


def dt_refinement_check(icase, nop, nelem, t_final, n_steps, nq_mode, baseline_err):
    """Re-run the finest mesh at dt/2 to confirm the spatial error is isolated.

    If halving dt moves the L2 error appreciably, the 4th-order LSERK time
    error is contaminating the spatial slope and the whole study is void.
    """
    print("  Time-error isolation check (finest mesh at dt/2)...")
    refined = run_case(icase, nop, nelem, t_final, 2 * n_steps, nq_mode, verbose=False)

    if baseline_err <= 0.0:
        print("    baseline error is zero -- check skipped")
        return None

    change = abs(refined["l2_abs"] - baseline_err) / baseline_err
    print(f"    L2 at dt   : {baseline_err:.6e}")
    print(f"    L2 at dt/2 : {refined['l2_abs']:.6e}")
    print(f"    relative change: {change:.2%}", end="  ")

    if change < 0.01:
        print("-> CLEAN (spatial error dominates; the slope is meaningful)")
    elif change < 0.05:
        print("-> BORDERLINE (consider a smaller --courant)")
    else:
        print("-> POLLUTED: time error is significant. "
              "The reported order is NOT a spatial order. Lower --courant.")
    print()
    return change


# =============================================================================
# Plotting
# =============================================================================

def make_plot(all_results, nop, nq_mode, outpath):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.0, 5.5))

    for icase, results in sorted(all_results.items()):
        h = np.array([r["h"] for r in results])
        e = np.array([r["l2_abs"] for r in results])
        ax.loglog(h, e, "o-", label=f"icase {icase}")

    # Reference triangle at the expected rate, anchored to the first curve.
    first = next(iter(sorted(all_results.items())))[1]
    h_ref = np.array([r["h"] for r in first])
    e_ref = np.array([r["l2_abs"] for r in first])
    if len(h_ref) >= 2 and e_ref[0] > 0.0:
        expected = nop + 1
        h_line = np.array([h_ref[0], h_ref[-1]])
        e_line = e_ref[0] * (h_line / h_ref[0]) ** expected
        ax.loglog(h_line, e_line, "k--", alpha=0.55,
                  label=f"$h^{{{expected}}}$ reference")

    ax.set_xlabel("element size $h$")
    ax.set_ylabel(r"$\|q_h - q_{exact}\|_{L^2}$")
    ax.set_title(f"1D DG convergence  (nop={nop}, quadrature={nq_mode})")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9)
    ax.invert_xaxis()
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"  plot -> {outpath}")


def write_csv(all_results, rates_by_case, outpath):
    header = ("icase,nop,nelem,h,npoin_dg,n_steps,dt,l2_abs,l2_rel,rate,"
              "mass_initial,mass_final,mass_drift_abs,wall_time_s\n")
    with open(outpath, "w") as fh:
        fh.write(header)
        for icase, results in sorted(all_results.items()):
            for r, rate in zip(results, rates_by_case[icase]):
                rate_s = "" if rate is None else f"{rate:.6f}"
                fh.write(
                    f"{r['icase']},{r['nop']},{r['nelem']},{r['h']:.10g},"
                    f"{r['npoin_dg']},{r['n_steps']},{r['dt']:.10g},"
                    f"{r['l2_abs']:.10e},{r['l2_rel']:.10e},{rate_s},"
                    f"{r['mass_initial']:.16e},{r['mass_final']:.16e},"
                    f"{r['mass_drift_abs']:.6e},{r['wall_time_s']:.3f}\n"
                )
    print(f"  csv  -> {outpath}")


# =============================================================================
# CLI
# =============================================================================

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Grid-convergence study for the Python 1D DG reference solver.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Smooth cases (valid order fit):\n  "
            + "\n  ".join(f"{k:>3}: {v}" for k, v in sorted(SMOOTH_CASES.items()))
            + "\n\nBlocked cases (require --allow-nonsmooth):\n  "
            + "\n  ".join(f"{k:>3}: {v}" for k, v in sorted(BLOCKED_CASES.items()))
        ),
    )
    p.add_argument("--icase", type=int, nargs="+", default=[8],
                   help="Initial condition(s). Default: 8 (sin(pi x)).")
    p.add_argument("--nop", type=int, default=4,
                   help="Polynomial order. Default: 4 (the project default).")
    p.add_argument("--nelem", type=int, nargs="+", default=[8, 16, 32, 64],
                   help="Mesh sweep (uniform). Default: 8 16 32 64.")
    p.add_argument("--t-final", type=float, default=1.0,
                   help="Final time. Default: 1.0 = exactly one traversal "
                        "(domain length 2, wave speed 2). Static ICs (6, 7, 8) "
                        "require an integer multiple of 1.0.")
    p.add_argument("--courant", type=float, default=0.2,
                   help="Courant number setting the single dt used for the "
                        "whole sweep, computed on the finest mesh as "
                        "courant * h / ((2*nop+1) * c). Default: 0.2.")
    p.add_argument("--nq-mode", choices=["overint", "collocated"], default="overint",
                   help="Quadrature. 'overint' (nq=nop+2) is the production "
                        "default; 'collocated' (nq=ngl) matches Jexpresso DGSEM "
                        "and is the mode required for the Phase-2F operator "
                        "comparison. Default: overint.")
    p.add_argument("--allow-nonsmooth", action="store_true",
                   help="Permit blocked ICs. The reported rate will not be p+1; "
                        "useful only for demonstrating that fact.")
    p.add_argument("--no-dt-check", action="store_true",
                   help="Skip the dt/2 time-error isolation check.")
    p.add_argument("--no-plot", action="store_true", help="Skip the PNG.")
    p.add_argument("--outdir", default=os.path.dirname(os.path.abspath(__file__)),
                   help="Output directory. Default: alongside this script.")
    p.add_argument("--tag", default="",
                   help="Suffix appended to output filenames.")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    # --- validate ICs -----------------------------------------------------
    cases = []
    for icase in args.icase:
        if icase in BLOCKED_CASES:
            msg = f"icase {icase}: {BLOCKED_CASES[icase]}"
            if args.allow_nonsmooth:
                print(f"WARNING -- {msg}\n         proceeding under --allow-nonsmooth; "
                      f"the reported rate is not a valid order estimate.\n")
                cases.append(icase)
            else:
                print(f"ERROR -- {msg}\n       pass --allow-nonsmooth to run anyway.")
                return 2
        elif icase in SMOOTH_CASES:
            cases.append(icase)
        else:
            print(f"ERROR -- icase {icase} is not a known case.")
            return 2

    nelems = sorted(set(args.nelem))
    if len(nelems) < 2:
        print("ERROR -- need at least two meshes for a convergence study.")
        return 2

    # --- one dt for the whole sweep, set by the finest mesh ---------------
    # Standard DG scaling: the stable dt shrinks like h / (2p+1). Holding dt
    # fixed across meshes is what makes the measured slope purely spatial.
    # --- static-IC / t_final compatibility --------------------------------
    static_requested = sorted(set(cases) & STATIC_CASES)
    if static_requested:
        n_periods = args.t_final / PERIOD
        if args.t_final <= 0 or abs(n_periods - round(n_periods)) > 1e-12:
            suggestion = max(1, round(n_periods)) * PERIOD
            print(
                f"ERROR -- icase(s) {static_requested} use a time-independent "
                f"formula in `exact_solution`.\n"
                f"       That formula is the correct advected solution only at "
                f"integer multiples of the\n"
                f"       traversal period ({PERIOD:g}). t_final="
                f"{args.t_final} is {n_periods:.4f} periods, so the study\n"
                f"       would compare against a stationary profile and report "
                f"a meaningless rate.\n"
                f"       Use --t-final {suggestion:g} (or another integer "
                f"multiple), or choose a\n"
                f"       time-dependent IC such as 1, 10, 13, or 16."
            )
            return 2

    h_fine = (DOMAIN[1] - DOMAIN[0]) / nelems[-1]
    dt_target = args.courant * h_fine / ((2 * args.nop + 1) * WAVE_SPEED)
    n_steps = int(np.ceil(args.t_final / dt_target))
    dt = args.t_final / n_steps  # exact landing on t_final

    print("=" * 80)
    print("1D DG CONVERGENCE STUDY -- Python reference solver")
    print("=" * 80)
    print(f"  domain          : [{DOMAIN[0]}, {DOMAIN[1]}] periodic")
    print(f"  nop             : {args.nop}  (ngl={args.nop + 1}, expected order "
          f"{args.nop + 1})")
    print(f"  quadrature      : {args.nq_mode}  "
          f"(nq={args.nop + 2 if args.nq_mode == 'overint' else args.nop + 1})")
    print(f"  mesh sweep      : {nelems}")
    print(f"  t_final         : {args.t_final}")
    print(f"  dt (all meshes) : {dt:.6e}   over {n_steps} steps  "
          f"(courant={args.courant})")
    print(f"  flux            : upwind (Fmatrix_upwind_flux_bc), periodic")
    print("=" * 80)

    all_results = {}
    rates_by_case = {}
    summary = {}

    for icase in cases:
        print(f"\n  running icase {icase} ...")
        results = []
        for nelem in nelems:
            r = run_case(icase, args.nop, nelem, args.t_final, n_steps, args.nq_mode)
            results.append(r)

        # Sanity: the study's dt assumed u = 2.0.
        if not np.isclose(results[0]["wave_speed"], WAVE_SPEED):
            print(f"  WARNING -- icase {icase} reports wave speed "
                  f"{results[0]['WAVE_SPEED']}, not {WAVE_SPEED}. "
                  f"The dt for this sweep was sized for {WAVE_SPEED}.")

        rates = pairwise_rates(results)
        slope = fitted_slope(results)
        print_table(icase, results, rates, slope, args.nop)

        dt_change = None
        if not args.no_dt_check:
            dt_change = dt_refinement_check(
                icase, args.nop, nelems[-1], args.t_final, n_steps,
                args.nq_mode, results[-1]["l2_abs"],
            )

        all_results[icase] = results
        rates_by_case[icase] = rates
        summary[icase] = {
            "fitted_slope": slope,
            "final_pairwise_rate": rates[-1],
            "expected_order": args.nop + 1,
            "dt_halving_relative_change": dt_change,
            "max_mass_drift": max(r["mass_drift_abs"] for r in results),
        }

    # --- outputs ----------------------------------------------------------
    os.makedirs(args.outdir, exist_ok=True)
    tag = f"_{args.tag}" if args.tag else ""
    stem = os.path.join(args.outdir, f"convergence_study{tag}")

    print("=" * 80)
    print("OUTPUTS")
    write_csv(all_results, rates_by_case, f"{stem}.csv")

    meta = {
        "nop": args.nop,
        "nelem": nelems,
        "t_final": args.t_final,
        "courant": args.courant,
        "dt": dt,
        "n_steps": n_steps,
        "nq_mode": args.nq_mode,
        "domain": list(DOMAIN),
        "summary": summary,
        "runs": {str(k): v for k, v in all_results.items()},
    }
    with open(f"{stem}.json", "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"  json -> {stem}.json")

    if not args.no_plot:
        make_plot(all_results, args.nop, args.nq_mode, f"{stem}.png")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
