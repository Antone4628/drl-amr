"""Step-7 gate: compare the extracted Jexpresso 2D DG operator against the
analytic construction, entrywise and structurally (D-064, one dimension up).

Consumes the two files written by the GATE2D probe (a temporary,
never-committed instrumentation of Jexpresso's problems/drivers.jl -- see
JEXPRESSO_ENVIRONMENT.md):

    GATE2D_operator_<tag>.txt   n x n matrix, row r = d(rhs_r)/d(u_c), i.e.
                                column c is the RHS response to the unit
                                vector at DOF c (space-separated, one row
                                per line)
    GATE2D_nodes_<tag>.txt      n lines "x y" -- the coordinates of DOF ip
                                (1-based line number = ip)

Pipeline:
  [1] DOF count: n == nex*ney*ngl^2 (the periodic-collapse detector, the
      check family that exposed the 1D weld at 249 vs 250)
  [2] geometry: derive (iel, i, j) from the DG numbering
      ip = (iel-1)*ngl^2 + (j-1)*ngl + i; map each element to its grid cell
      from its (i=1, j=1) corner node; require the map bijective and EVERY
      node coordinate on the analytic LGL lattice
  [3] assemble the analytic operator directly in Jexpresso's element order
      (no matrix permutation) and compare entrywise: max |diff|, relative,
      count above --abs-tol, sparsity-support equality
  [4] conservation m^T L, [5] consistency L @ 1, [6] spectrum
      (max Re(lambda) <= 0, single zero mode) -- on the EXTRACTED operator
  Verdict PASS/FAIL; exit code 0/1.

Run (paths from the probe's own final println):

    cd ~/projects/drl-amr
    conda activate rl-amr
    python -m tools.gate_2d_compare \
        --operator <...>/GATE2D_operator_<tag>.txt \
        --nodes <...>/GATE2D_nodes_<tag>.txt

Defaults are the gate configuration (4x4, nop 4, domain (-5,5)x(0,20),
c = (0.5, 1.0)); every parameter is overridable for reuse as the
rebase-regression instrument on other configurations.
"""

import argparse
import sys

import numpy as np

from backends.python_1d.dg.basis import lgl_gen
from tools.analytic_operator_2d import (
    GATE_CX,
    GATE_CY,
    GATE_DOMAIN,
    GATE_NEX,
    GATE_NEY,
    GATE_NOP,
    build_analytic_operator_2d,
    lumped_mass_vector_2d,
    structural_report,
)

ABS_TOL_DEFAULT = 1e-8       # F1's own criterion: zero entries differing above this
GEOM_TOL_DEFAULT = 1e-8      # node-lattice tolerance (gmsh transfinite noise ~1e-11)
SUPPORT_TOL = 1e-8           # threshold defining an operator's sparsity support


def decode(idx, ngl, elem_cells):
    """0-based flat DOF index -> (e, (ex, ey), i, j), plus 1-based Julia view."""
    e = idx // (ngl * ngl)
    r = idx % (ngl * ngl)
    j = r // ngl
    i = r % ngl
    ex, ey = elem_cells[e]
    return e, ex, ey, i, j


def fmt_dof(idx, ngl, elem_cells):
    e, ex, ey, i, j = decode(idx, ngl, elem_cells)
    return (f"ip={idx + 1} [iel={e + 1} cell=({ex},{ey}) "
            f"i={i + 1} j={j + 1}]")


def derive_elem_cells(x, y, nex, ney, nop, domain, tol):
    """Map each Jexpresso element to its (ex, ey) grid cell and verify every
    DOF coordinate against the analytic LGL lattice.

    Returns (elem_cells, max_node_err). Raises with a diagnostic on any
    failure -- a node off the lattice means the numbering, the mesh, or the
    stated domain/discretization parameters are wrong, and the entrywise
    comparison would be meaningless.
    """
    ngl = nop + 1
    (xmin, xmax), (ymin, ymax) = domain
    dx = (xmax - xmin) / nex
    dy = (ymax - ymin) / ney
    xgl, _ = lgl_gen(ngl)
    nelem = nex * ney
    n = nelem * ngl * ngl
    assert len(x) == n, f"nodes file has {len(x)} entries, expected {n}"

    elem_cells = []
    max_node_err = 0.0
    for e in range(nelem):
        ip00 = e * ngl * ngl          # (i=1, j=1) corner in 1-based terms
        ex = round((x[ip00] - xmin) / dx)
        ey = round((y[ip00] - ymin) / dy)
        if not (0 <= ex < nex and 0 <= ey < ney):
            raise AssertionError(
                f"element {e + 1}: corner ({x[ip00]}, {y[ip00]}) maps to grid "
                f"cell ({ex}, {ey}) outside [0,{nex})x[0,{ney}) -- wrong "
                f"domain/mesh parameters?")
        for j in range(ngl):
            for i in range(ngl):
                ip = e * ngl * ngl + j * ngl + i
                xe = xmin + ex * dx + (xgl[i] + 1.0) / 2.0 * dx
                ye = ymin + ey * dy + (xgl[j] + 1.0) / 2.0 * dy
                err = max(abs(x[ip] - xe), abs(y[ip] - ye))
                if err > max_node_err:
                    max_node_err = err
                if err > tol:
                    raise AssertionError(
                        f"DOF ip={ip + 1} (iel={e + 1}, i={i + 1}, j={j + 1}): "
                        f"coords ({x[ip]}, {y[ip]}) vs lattice ({xe}, {ye}), "
                        f"err {err:.3e} > {tol:.1e} -- numbering or lattice "
                        f"convention broken")
        elem_cells.append((ex, ey))

    if len(set(elem_cells)) != nelem:
        seen = {}
        for e, c in enumerate(elem_cells):
            if c in seen:
                raise AssertionError(
                    f"elements {seen[c] + 1} and {e + 1} both map to grid "
                    f"cell {c} -- element->cell map is not a bijection")
            seen[c] = e
    return elem_cells, max_node_err


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--operator", required=True,
                    help="path to GATE2D_operator_<tag>.txt")
    ap.add_argument("--nodes", required=True,
                    help="path to GATE2D_nodes_<tag>.txt")
    ap.add_argument("--nex", type=int, default=GATE_NEX)
    ap.add_argument("--ney", type=int, default=GATE_NEY)
    ap.add_argument("--nop", type=int, default=GATE_NOP)
    ap.add_argument("--xmin", type=float, default=GATE_DOMAIN[0][0])
    ap.add_argument("--xmax", type=float, default=GATE_DOMAIN[0][1])
    ap.add_argument("--ymin", type=float, default=GATE_DOMAIN[1][0])
    ap.add_argument("--ymax", type=float, default=GATE_DOMAIN[1][1])
    ap.add_argument("--cx", type=float, default=GATE_CX)
    ap.add_argument("--cy", type=float, default=GATE_CY)
    ap.add_argument("--abs-tol", type=float, default=ABS_TOL_DEFAULT)
    ap.add_argument("--geom-tol", type=float, default=GEOM_TOL_DEFAULT)
    ap.add_argument("--top-k", type=int, default=5,
                    help="largest-diff entries to list when any exceed tol")
    args = ap.parse_args(argv)

    domain = ((args.xmin, args.xmax), (args.ymin, args.ymax))
    ngl = args.nop + 1
    n_expect = args.nex * args.ney * ngl * ngl
    failures = []

    print("=" * 72)
    print(f"GATE 2D operator comparison (D-064): nop={args.nop} "
          f"mesh={args.nex}x{args.ney} c=({args.cx}, {args.cy})")
    print(f"domain = {domain}")
    print("=" * 72)

    L = np.loadtxt(args.operator)
    xy = np.loadtxt(args.nodes)
    x, y = xy[:, 0], xy[:, 1]

    # [1] DOF count -- the periodic-collapse detector.
    ok = (L.shape == (n_expect, n_expect) and len(x) == n_expect)
    print(f"[1] DOF count: matrix {L.shape}, nodes {len(x)}, expected "
          f"{n_expect}  {'PASS' if ok else 'FAIL'}")
    if not ok:
        if L.shape[0] != len(x):
            print(f"    NOTE: matrix dim {L.shape[0]} != node count {len(x)} "
                  f"-- if it is an integer multiple, the state vector "
                  f"carries neqs > 1 and the probe/compare need a per-"
                  f"equation slicing decision before proceeding.")
        failures.append("dof_count")
        _verdict(failures)
        return 1

    # [2] geometry: element->cell map + full node-lattice verification.
    try:
        elem_cells, max_node_err = derive_elem_cells(
            x, y, args.nex, args.ney, args.nop, domain, args.geom_tol)
        print(f"[2] geometry: element->cell bijection OK; max node-lattice "
              f"err = {max_node_err:.3e}  PASS")
    except AssertionError as exc:
        print(f"[2] geometry: FAIL -- {exc}")
        failures.append("geometry")
        _verdict(failures)
        return 1

    # [3] entrywise comparison against the analytic operator, assembled
    # directly in Jexpresso's element order.
    A = build_analytic_operator_2d(args.nop, args.nex, args.ney,
                                   args.cx, args.cy, domain=domain,
                                   elem_cells=elem_cells)
    diff = np.abs(L - A)
    max_diff = float(diff.max())
    scale = float(np.max(np.abs(A)))
    n_over = int(np.sum(diff > args.abs_tol))
    supp_l = np.abs(L) > SUPPORT_TOL
    supp_a = np.abs(A) > SUPPORT_TOL
    supp_eq = bool(np.array_equal(supp_l, supp_a))
    ok = (n_over == 0)
    print(f"[3] entrywise: max|diff| = {max_diff:.3e} "
          f"(rel {max_diff / scale:.3e}); entries > {args.abs_tol:.0e}: "
          f"{n_over}; nnz {int(supp_l.sum())} vs {int(supp_a.sum())}, "
          f"support {'identical' if supp_eq else 'DIFFERS'}  "
          f"{'PASS' if ok and supp_eq else 'FAIL'}")
    if not (ok and supp_eq):
        failures.append("entrywise")
        order = np.argsort(diff, axis=None)[::-1][:args.top_k]
        print(f"    top {args.top_k} differences:")
        for flat in order:
            r, c = np.unravel_index(flat, diff.shape)
            print(f"      row {fmt_dof(r, ngl, elem_cells)}  "
                  f"col {fmt_dof(c, ngl, elem_cells)}  "
                  f"jex={L[r, c]:+.12e}  analytic={A[r, c]:+.12e}  "
                  f"diff={diff[r, c]:.3e}")

    # [4]-[6] structural checks on the EXTRACTED operator (mass vector is
    # legitimate here because [2] verified the geometry it encodes).
    m = lumped_mass_vector_2d(args.nop, args.nex, args.ney, domain=domain)
    rep = structural_report(L, m)
    ok = rep["conservation_mTL"] < 1e-10
    print(f"[4] conservation m^T L = {rep['conservation_mTL']:.3e}  "
          f"{'PASS' if ok else 'FAIL'}")
    if not ok:
        failures.append("conservation")
    ok = rep["consistency_L1"] < 1e-9
    print(f"[5] consistency  L @ 1 = {rep['consistency_L1']:.3e}  "
          f"{'PASS' if ok else 'FAIL'}")
    if not ok:
        failures.append("consistency")
    ok = (rep["max_re_lambda"] < 1e-8 * rep["spectral_radius"]
          and rep["n_zero_modes"] == 1)
    print(f"[6] spectrum: max Re(lambda) = {rep['max_re_lambda']:.3e} at "
          f"rho = {rep['spectral_radius']:.3f}; kernel modes "
          f"(|lam| < 1e-10*rho) = {rep['n_zero_modes']} (expect 1)  "
          f"{'PASS' if ok else 'FAIL'}")
    print(f"    near-zero band (|lam| < 1e-6*rho): "
          f"{rep['n_near_zero_modes']} -- kernel + conjugate-cancelled "
          f"physical pairs (Kronecker-sum spectrum; informational, see "
          f"analytic_operator_2d.structural_report)")
    print(f"    smallest |lambda|: ["
          + ", ".join(f"{v:.3e}" for v in rep["smallest_abs_lambda"]) + "]")
    if not ok:
        failures.append("spectrum")

    _verdict(failures)
    return 0 if not failures else 1


def _verdict(failures):
    print("=" * 72)
    if not failures:
        print("VERDICT: PASS -- the extracted 2D operator matches the "
              "analytic construction entrywise and is structurally sound.")
    else:
        print(f"VERDICT: FAIL ({', '.join(failures)})")
    print("=" * 72)


if __name__ == "__main__":
    sys.exit(main())
