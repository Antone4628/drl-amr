"""Compare two GATE2D probe dumps of the same geometry whose DOFs are numbered
differently -- the bridge between the tag-only 4x4 gate family and the natively
periodic one, and the standing comparator for any future renumbering.

A Jexpresso DG mesh built from a gmsh file with Periodic constraints has the
same elements and the same LGL lattices as the tag-only mesh of the same
geometry, but Gridap numbers its cells and nodes differently, so the probe's
dumps (operator, nodes, mass, ad_lvl -- all in Jexpresso `ip` order) cannot be
compared byte for byte or entrywise as written.  A node permutation is needed,
and coordinates alone do not give one: DG keeps one DOF per element at a
shared vertex or edge, so several `ip` carry the same (x, y).  The permutation
is therefore built per element -- elements are matched by centroid, then nodes
within a matched pair by coordinate, where the LGL lattice makes them unique.

Consumes, from two directories, the files the GATE2D probe writes:

    GATE2D_operator_<tag>.txt   n x n matrix (space-separated rows)
    GATE2D_nodes_<tag>.txt      n lines "x y", line number = ip
    GATE2D_mass_<tag>.txt       n lines, params.M diagonal (optional)
    GATE2D_adlvl_<tag>.txt      nelem lines, mesh.ad_lvl (optional)

where <tag> is "nop<N>_<nelem>elem"; nop and nelem are read from it.

Reports, with P the permutation taking reference ip to new ip:
  [1] element matching: every reference element has exactly one partner at
      the same centroid, and every node of it one partner at the same
      coordinate inside that element
  [2] operator: max |L_new[P,P] - L_ref|, the same relative to max |L_ref|,
      and whether the sparsity supports coincide
  [3] mass: max |M_new[P] - M_ref| relative to max M_ref (when both exist)
  [4] ad_lvl: levels agree element by element (when both exist)
Verdict PASS when [1] holds, [2] is within --tol, supports coincide, and [3]
and [4] agree where present; exit code 0/1.

Run:

    cd ~/projects/drl-amr
    conda activate rl-amr
    python -m tools.gate_2d_permute_compare --ref <dir of the reference dumps> --new <dir of the new dumps>

--tol defaults to 1e-10: the two meshes are written by gmsh from different
.geo files, so their node coordinates differ by the generator's round-off
(~1e-11 relative on this family); the operators differ by the same order.
"""

import argparse
import glob
import os
import re
import sys

import numpy as np

TOL_DEFAULT = 1e-10
SUPPORT_TOL = 1e-8
TAG_RE = re.compile(r"GATE2D_operator_nop(\d+)_(\d+)elem\.txt$")


def find_dumps(d):
    """Locate the probe's files in directory d; return (tag, nop, nelem, paths)."""
    ops = sorted(glob.glob(os.path.join(d, "GATE2D_operator_nop*_*elem.txt")))
    if len(ops) != 1:
        raise SystemExit(f"{d}: expected exactly one GATE2D_operator dump, found {len(ops)}")
    m = TAG_RE.search(ops[0])
    nop, nelem = int(m.group(1)), int(m.group(2))
    tag = f"nop{nop}_{nelem}elem"
    paths = {
        "operator": ops[0],
        "nodes": os.path.join(d, f"GATE2D_nodes_{tag}.txt"),
        "mass": os.path.join(d, f"GATE2D_mass_{tag}.txt"),
        "adlvl": os.path.join(d, f"GATE2D_adlvl_{tag}.txt"),
    }
    if not os.path.exists(paths["nodes"]):
        raise SystemExit(f"{d}: missing {paths['nodes']}")
    return tag, nop, nelem, paths


def load(paths):
    L = np.loadtxt(paths["operator"], dtype=float)
    xy = np.loadtxt(paths["nodes"], dtype=float)
    if xy.ndim == 1:
        xy = xy.reshape(-1, 2)
    M = np.loadtxt(paths["mass"], dtype=float) if os.path.exists(paths["mass"]) else None
    lv = (np.loadtxt(paths["adlvl"], dtype=int).reshape(-1)
          if os.path.exists(paths["adlvl"]) else None)
    return L, xy, M, lv


def element_blocks(xy, k):
    """Split node coordinates into consecutive blocks of k (one element each)."""
    n = xy.shape[0]
    if n % k != 0:
        raise SystemExit(f"node count {n} is not a multiple of ngl^2 = {k}")
    return xy.reshape(n // k, k, 2)


def build_permutation(xy_ref, xy_new, k, tol):
    """Return P (len n) with new ip = P[ref ip], plus the element map and the
    worst coordinate residuals, matching elements by centroid and nodes within
    an element by coordinate."""
    bref = element_blocks(xy_ref, k)
    bnew = element_blocks(xy_new, k)
    if bref.shape[0] != bnew.shape[0]:
        raise SystemExit(f"element counts differ: ref {bref.shape[0]}, new {bnew.shape[0]}")
    ne = bref.shape[0]
    cref = bref.mean(axis=1)
    cnew = bnew.mean(axis=1)
    scale = max(np.ptp(xy_ref[:, 0]), np.ptp(xy_ref[:, 1]), 1.0)
    elem_map = -np.ones(ne, dtype=int)
    used = np.zeros(ne, dtype=bool)
    worst_centroid = 0.0
    for e in range(ne):
        d = np.linalg.norm(cnew - cref[e], axis=1)
        j = int(np.argmin(d))
        if d[j] > tol * scale or used[j]:
            raise SystemExit(f"element {e + 1} (ref) has no unique partner at its centroid "
                             f"(nearest new element {j + 1}, distance {d[j]:.3e})")
        used[j] = True
        elem_map[e] = j
        worst_centroid = max(worst_centroid, d[j])
    P = -np.ones(ne * k, dtype=int)
    worst_node = 0.0
    for e in range(ne):
        j = elem_map[e]
        taken = np.zeros(k, dtype=bool)
        for a in range(k):
            d = np.linalg.norm(bnew[j] - bref[e, a], axis=1)
            b = int(np.argmin(d))
            if d[b] > tol * scale or taken[b]:
                raise SystemExit(f"node {a + 1} of ref element {e + 1} has no unique partner "
                                 f"in new element {j + 1} (distance {d[b]:.3e})")
            taken[b] = True
            P[e * k + a] = j * k + b
            worst_node = max(worst_node, d[b])
    return P, elem_map, worst_centroid, worst_node


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ref", required=True, help="directory of the reference dumps")
    ap.add_argument("--new", required=True, help="directory of the dumps to compare")
    ap.add_argument("--tol", type=float, default=TOL_DEFAULT,
                    help="pass threshold on max |L_new[P,P] - L_ref| (and on coordinate matching, "
                         "relative to the domain size)")
    ap.add_argument("--top", type=int, default=5, help="how many largest differences to list")
    args = ap.parse_args(argv)

    tag_r, nop_r, ne_r, pr = find_dumps(args.ref)
    tag_n, nop_n, ne_n, pn = find_dumps(args.new)
    if (nop_r, ne_r) != (nop_n, ne_n):
        raise SystemExit(f"tags differ: ref {tag_r}, new {tag_n}")
    ngl = nop_r + 1
    k = ngl * ngl
    Lr, xr, Mr, lr = load(pr)
    Ln, xn, Mn, ln = load(pn)
    n = Lr.shape[0]
    if Lr.shape != (n, n) or Ln.shape != (n, n) or xr.shape[0] != n or xn.shape[0] != n:
        raise SystemExit(f"shape mismatch: L_ref {Lr.shape}, L_new {Ln.shape}, "
                         f"nodes {xr.shape[0]} / {xn.shape[0]}")

    print(f"== permute-compare  tag {tag_r}  n = {n}  ngl = {ngl}  nelem = {ne_r}")
    print(f"   ref {args.ref}")
    print(f"   new {args.new}")

    P, emap, wc, wn = build_permutation(xr, xn, k, args.tol)
    identity = bool(np.all(P == np.arange(n)))
    print(f"[1] elements matched 1:1 by centroid (worst {wc:.3e}); nodes within elements "
          f"(worst {wn:.3e}); permutation {'is the identity' if identity else 'is non-trivial'}")

    Lp = Ln[np.ix_(P, P)]
    D = np.abs(Lp - Lr)
    maxabs = float(D.max())
    scale_L = float(np.abs(Lr).max())
    rel = maxabs / scale_L if scale_L > 0 else maxabs
    sr = np.abs(Lr) > SUPPORT_TOL
    sn = np.abs(Lp) > SUPPORT_TOL
    same_support = bool(np.array_equal(sr, sn))
    print(f"[2] operator: max |L_new[P,P] - L_ref| = {maxabs:.3e}  (relative {rel:.3e}; "
          f"nnz ref {int(sr.sum())}, new {int(sn.sum())}, support "
          f"{'identical' if same_support else 'DIFFERENT'})")
    if args.top > 0 and maxabs > 0:
        flat = np.argsort(D, axis=None)[::-1][:args.top]
        for f in flat:
            r, c = np.unravel_index(f, D.shape)
            print(f"      row ip={r + 1} [iel={r // k + 1}]  col ip={c + 1} [iel={c // k + 1}]  "
                  f"ref {Lr[r, c]: .12e}  new {Lp[r, c]: .12e}  diff {D[r, c]:.3e}")

    mass_ok = True
    if Mr is not None and Mn is not None:
        dm = float(np.abs(Mn[P] - Mr).max())
        rm = dm / float(np.abs(Mr).max())
        mass_ok = rm <= args.tol
        print(f"[3] mass: max |M_new[P] - M_ref| = {dm:.3e}  (relative {rm:.3e}; "
              f"sums {Mr.sum():.15g} / {Mn.sum():.15g})")
    else:
        print("[3] mass: not compared (dump missing on one side)")

    lvl_ok = True
    if lr is not None and ln is not None:
        if lr.shape[0] != ne_r or ln.shape[0] != ne_r:
            raise SystemExit(f"ad_lvl length ref {lr.shape[0]} / new {ln.shape[0]} vs nelem {ne_r}")
        mism = int(np.sum(ln[emap] != lr))
        lvl_ok = mism == 0
        counts_r = {int(l): int(np.sum(lr == l)) for l in np.unique(lr)}
        print(f"[4] ad_lvl: {mism} element(s) differ after the element map; counts {counts_r}")
    else:
        print("[4] ad_lvl: not compared (dump missing on one side)")

    ok = maxabs <= args.tol and same_support and mass_ok and lvl_ok
    print(f"VERDICT: {'PASS' if ok else 'FAIL'}  (tol {args.tol:.1e})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
