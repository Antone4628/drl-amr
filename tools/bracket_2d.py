"""Phase-6 gate -- the solution-level bracket: L2 error and discrete mass of
Jexpresso 2D DG runs on conforming, 2:1, and in-solve-adapted meshes, from
their VTU output alone.

Companion to tools/gate_2d_mortar.py (the operator half of the gate). Reads
each run's iter_N/iter_N_1.vtu frames, derives element boxes and LGL
quadrature weights from the node coordinates with the gate's own Mesh2 (the
DG output writes points in ip order, ip = (iel-1)*ngl^2 + (j-1)*ngl + i --
Mesh2 asserts every 25-point block is one element's LGL lattice, so a
reordered file fails loudly), and reports per run:

  mass      w^T q at every frame (w = w_i w_j Jvol(e), the diagonal DG mass),
            and its relative drift over the run
  L2        sqrt(sum_k w_k e_k^2) at the final frame, e = q - q_exact, with
            q_exact the translated Gaussian as a PERIODIC IMAGE SUM (the
            free-space form floors a measurement at the omitted image's norm
            -- the 2026-08-16 lesson, JEXPRESSO_ENVIRONMENT.md section 10)

and, over the runs in the order given (coarse to fine), the bracket verdict:
every run's mass drift within --mass-tol, and L2 strictly decreasing (unless
--no-l2-gate, which still reports the L2 values and ratios).

Frame times must increase strictly with the frame number N: an output counter
that restarted, or files overwritten across solve segments, would otherwise
pair a field with the wrong time and the wrong mesh, silently.

Two modes (item 8a, 2026-10-03):

  default     static mesh: every frame must carry the same mesh, or the run
              is refused -- a static run whose mesh changes is a defect.
  --adaptive  in-solve adaptation: each frame is scored on ITS OWN mesh
              (derived from its own node coordinates, as above), and a
              per-frame table reports t, nelem, the level histogram, and the
              mass change from frame 0, marking every frame whose mesh differs
              from the previous frame's. Mass drift is measured across all
              frames and all meshes. Every frame's mesh must still pass
              Mesh2's own checks (LGL lattice, tiling, conforming-or-2:1
              contacts, every side covered once), so a level jump of two or an
              overlapped refinement stops the scorer.

What the mass check cannot see -- by construction, pinned in the tests: a
coarsen or refine that puts a child's data in the wrong quadrant conserves
mass exactly, because the four quadrants of a parent have equal area. Child
ordering is checked by the geometric Q1b-coarsen probe in Jexpresso's
transfer, never by this tool. Mass at the instant of each adapt (before and
after the transfer) is the amr_strategy! probe's measurement; this tool sees
output frames only.

The exact solution defaults to the advection2d_dg family's IC (initialize.jl:
A = 1, centre (domain x-centre, 8), sigma 1) advected by c = (0.5, 1.0); every
parameter is overridable.

Run -- the item-7 bracket:

    cd ~/projects/drl-amr
    conda activate rl-amr
    python -m tools.bracket_2d \\
        --run 4x4=<abs dir of the 4x4 run> \\
        --run 2to1=<abs dir of the static 2:1 run> \\
        --run 8x8=<abs dir of the 8x8 run>

Run -- an in-solve adaptive run (item 8a), L2 informational:

    python -m tools.bracket_2d --adaptive --no-l2-gate \\
        --run 4x4=<abs dir> --run insolve=<abs dir> --run 8x8=<abs dir>

Named regressions:
  - reproduces tools/jexpresso/vtu_error.jl's banked 10x20 value
    3.042960147959e-03 from the convergence study's vtu:
        python -m tools.bracket_2d --nex 10 --ney 20 --run 10x20=<abs dir>
  - reproduces the item-7 bracket (JEXPRESSO_ENVIRONMENT.md section 8.7.5),
    with and without --adaptive: L2 5.788928e-01 / 4.278274e-01 /
    1.070981e-01, 2:1 mass drift 3.0e-13 over 17 frames.
"""

import argparse
import glob
import os
import re
import sys

import numpy as np

from tools.analytic_operator_2d import GATE_CX, GATE_CY, GATE_DOMAIN, GATE_NEX, GATE_NEY, GATE_NOP
from tools.analytic_operator_2d_mortar import geometric_mass
from tools.gate_2d_mortar import GEOM_TOL, Mesh2

_DTYPES = {"Float64": "<f8", "Float32": "<f4", "Int64": "<i8", "UInt8": "u1"}


def read_vtu(path):
    """The arrays of a VTK XML UnstructuredGrid written with raw appended data
    and UInt64 block headers (Jexpresso's writer). Refuses anything else."""
    b = open(path, "rb").read()
    i = b.find(b"<AppendedData")
    if i < 0:
        raise ValueError(f"{path}: no AppendedData section")
    xml = b[:i].decode()
    if 'type="UnstructuredGrid"' not in xml or 'header_type="UInt64"' not in xml:
        raise ValueError(f"{path}: not an UnstructuredGrid with UInt64 headers")
    if 'encoding="raw"' not in b[i:i + 200].decode(errors="replace"):
        raise ValueError(f"{path}: appended data is not raw-encoded")
    base = b.index(b"_", i) + 1
    out = {}
    pat = r'<DataArray type="(\w+)" Name="(\w+)" NumberOfComponents="(\d+)"[^>]*offset="(\d+)"'
    for m in re.finditer(pat, xml):
        typ, name, ncomp, off = m.group(1), m.group(2), int(m.group(3)), int(m.group(4))
        start = base + off
        nbytes = int(np.frombuffer(b[start:start + 8], dtype="<u8")[0])
        arr = np.frombuffer(b[start + 8:start + 8 + nbytes], dtype=_DTYPES[typ])
        out[name] = arr.reshape(-1, ncomp) if ncomp > 1 else arr
    return out


def pvd_times(run_dir):
    """{N: t} from simulation.pvd (DataSet timestep=... file="iter_N.pvtu"), or {}."""
    p = os.path.join(run_dir, "simulation.pvd")
    if not os.path.isfile(p):
        return {}
    txt = open(p).read()
    return {int(n): float(t) for t, n in
            re.findall(r'timestep="([^"]+)"\s+file="iter_(\d+)\.pvtu"', txt)}


def load_frames(run_dir, field="q"):
    """[(iter, t, x, y, q)] for every iter_N/iter_N_1.vtu, ascending in N. The
    frame time comes from the file's TimeValue field, else simulation.pvd;
    a frame with neither is an error (an L2 against the wrong time is silent)."""
    times = pvd_times(run_dir)
    frames = []
    for p in glob.glob(os.path.join(run_dir, "iter_*", "iter_*_1.vtu")):
        n = int(re.search(r"iter_(\d+)_1\.vtu$", p).group(1))
        v = read_vtu(p)
        if field not in v:
            raise ValueError(f"{p}: no point field {field!r} (fields: {sorted(v)})")
        q = np.asarray(v[field], dtype=float)
        if v[field].dtype != np.float64:
            raise ValueError(f"{p}: field {field!r} is {v[field].dtype}, need Float64 "
                             f"for a round-off mass check")
        if "TimeValue" in v:
            t = float(v["TimeValue"][0])
            if n in times and abs(times[n] - t) > 1e-12:
                raise ValueError(f"{p}: TimeValue {t} disagrees with simulation.pvd {times[n]}")
        elif n in times:
            t = times[n]
        else:
            raise ValueError(f"{p}: no TimeValue field and no simulation.pvd entry for iter_{n}")
        P = v["Points"]
        frames.append((n, t, P[:, 0].copy(), P[:, 1].copy(), q))
    if not frames:
        raise ValueError(f"{run_dir}: no iter_*/iter_*_1.vtu frames")
    frames = sorted(frames, key=lambda f: f[0])
    for (na, ta, *_), (nb, tb, *_) in zip(frames, frames[1:], strict=False):
        if not tb > ta:
            raise ValueError(
                f"{run_dir}: frame times do not increase with the frame number "
                f"(iter_{na} t={ta}, iter_{nb} t={tb}) -- an output counter that "
                f"restarted, or frames overwritten across solve segments")
    return frames


def gaussian_exact(x, y, t, domain=GATE_DOMAIN, centre0=None, c=(GATE_CX, GATE_CY),
                   sigma=(1.0, 1.0), amp=1.0):
    """The IC Gaussian translated by c*t on the doubly periodic domain, as the
    image sum over the nearest images in each direction."""
    (x0, x1), (y0, y1) = domain
    lx, ly = x1 - x0, y1 - y0
    if centre0 is None:
        centre0 = (0.5 * (x0 + x1), 8.0)
    xc = x0 + (centre0[0] + c[0] * t - x0) % lx
    yc = y0 + (centre0[1] + c[1] * t - y0) % ly
    gx = sum(np.exp(-((x - xc - k * lx) / sigma[0]) ** 2) for k in (-1, 0, 1))
    gy = sum(np.exp(-((y - yc - k * ly) / sigma[1]) ** 2) for k in (-1, 0, 1))
    return amp * gx * gy


def _same_mesh(a, b):
    """The same set of element boxes, to the node-geometry tolerance, in any
    element order -- a renumbering of the same elements, or a rebuild of the
    same mesh with node noise, is not a mesh change."""
    if a.nelem != b.nelem:
        return False
    ka, kb = np.round(a.box, 6), np.round(b.box, 6)   # sort keys immune to node noise
    sa = a.box[np.lexsort(ka.T[::-1])]
    sb = b.box[np.lexsort(kb.T[::-1])]
    return bool(np.allclose(sa, sb, rtol=0.0, atol=GEOM_TOL))


def score_run(run_dir, nop=GATE_NOP, domain=GATE_DOMAIN, nex=GATE_NEX, ney=GATE_NEY,
              exact=gaussian_exact, adaptive=False):
    """Score one run. Each frame's mesh is derived from that frame's own node
    coordinates (rebuilt only when the coordinates change). Static mode
    (adaptive=False) refuses a run whose mesh changes; adaptive mode scores
    every frame on its own mesh and records where the mesh changed."""
    frames = load_frames(run_dir)
    per_frame, mesh, prev_xy = [], None, None
    for n, t, x, y, q in frames:
        changed = False
        if prev_xy is None or not (np.array_equal(x, prev_xy[0]) and np.array_equal(y, prev_xy[1])):
            m = Mesh2(x, y, nop, domain, nex, ney)
            changed = mesh is not None and not _same_mesh(m, mesh)
            if changed and not adaptive:
                raise ValueError(f"{run_dir}: the mesh changes between frames (iter_{n}, "
                                 f"t={t}); static mode refuses this -- use --adaptive "
                                 f"for an in-solve adaptive run")
            mesh = m
            prev_xy = (x, y)
        per_frame.append(dict(iter=n, t=t, nelem=mesh.nelem,
                              levels=np.bincount(mesh.level).tolist(), changed=changed,
                              mass=float(geometric_mass(mesh) @ q)))
    _, t, x, y, q = frames[-1]
    w = geometric_mass(mesh)
    e = q - exact(x, y, t, domain=domain)
    masses = np.array([f["mass"] for f in per_frame])
    nelems = [f["nelem"] for f in per_frame]
    return dict(
        n=len(q), nelem=mesh.nelem, levels=np.bincount(mesh.level).tolist(),
        nframes=len(frames), t_final=t,
        n_meshes=1 + sum(f["changed"] for f in per_frame),
        nelem_min=min(nelems), nelem_max=max(nelems),
        frames=per_frame,
        l2=float(np.sqrt(w @ e ** 2)), maxabs=float(np.max(np.abs(e))),
        mass0=masses[0], mass_drift=float(np.max(np.abs(masses - masses[0])) / abs(masses[0])),
    )


def print_frames(label, r):
    """The per-frame table of an adaptive run; '*' marks a frame whose mesh
    differs from the previous frame's."""
    print(f"  {label}: {r['nframes']} frames, {r['n_meshes']} meshes, "
          f"nelem {r['nelem_min']}..{r['nelem_max']}")
    print(f"    {'iter':>5s} {'t':>7s} {'nelem':>5s} {'levels':>12s} {'mesh':>4s} "
          f"{'(m - m0)/m0':>12s}")
    m0 = r["frames"][0]["mass"]
    for f in r["frames"]:
        print(f"    {f['iter']:5d} {f['t']:7.3f} {f['nelem']:5d} {str(f['levels']):>12s} "
              f"{'*' if f['changed'] else '':>4s} {(f['mass'] - m0) / abs(m0):12.3e}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", action="append", required=True, metavar="LABEL=DIR",
                    help="a run, coarse to fine; repeat")
    ap.add_argument("--nex", type=int, default=GATE_NEX, help="base-grid elements in x")
    ap.add_argument("--ney", type=int, default=GATE_NEY, help="base-grid elements in y")
    ap.add_argument("--nop", type=int, default=GATE_NOP)
    ap.add_argument("--mass-tol", type=float, default=1e-12,
                    help="max relative mass drift per run")
    ap.add_argument("--adaptive", action="store_true",
                    help="allow the mesh to change between frames (in-solve adaptation); "
                         "score each frame on its own mesh and print a per-frame table")
    ap.add_argument("--no-l2-gate", action="store_true",
                    help="report L2 and its ratios, but do not gate the verdict on "
                         "L2 decreasing across the runs (mass still gates)")
    args = ap.parse_args(argv)

    runs = []
    for spec in args.run:
        label, _, d = spec.partition("=")
        if not d:
            ap.error(f"--run needs LABEL=DIR, got {spec!r}")
        runs.append((label, d))

    print("=" * 96)
    print(f"BRACKET 2D: nop={args.nop} base {args.nex}x{args.ney} domain={GATE_DOMAIN} "
          f"mode={'adaptive' if args.adaptive else 'static'} "
          f"L2 gate={'off' if args.no_l2_gate else 'on'}")
    print("=" * 96)
    print(f"{'run':8s} {'n':>6s} {'nelem':>5s} {'levels':>10s} {'frames':>6s} {'t':>5s} "
          f"{'L2':>14s} {'max|e|':>11s} {'mass(t=0)':>19s} {'mass drift':>11s}")
    res = []
    for label, d in runs:
        r = score_run(d, nop=args.nop, nex=args.nex, ney=args.ney, adaptive=args.adaptive)
        res.append((label, r))
        print(f"{label:8s} {r['n']:6d} {r['nelem']:5d} {str(r['levels']):>10s} {r['nframes']:6d} "
              f"{r['t_final']:5.2f} {r['l2']:14.6e} {r['maxabs']:11.3e} {r['mass0']:19.12e} "
              f"{r['mass_drift']:11.3e}")
        print(f"         {d}")
    failures = []
    for label, r in res:
        if r["mass_drift"] > args.mass_tol:
            failures.append(f"mass drift {label}")
    for (la, ra), (lb, rb) in zip(res, res[1:], strict=False):
        if not rb["l2"] < ra["l2"]:
            if args.no_l2_gate:
                print(f"  (not gated) L2 not decreasing {la} -> {lb}")
            else:
                failures.append(f"L2 not decreasing {la} -> {lb}")
    if len(res) > 1:
        print("  L2 ratios: " + ", ".join(
            f"{la}/{lb} = {ra['l2'] / rb['l2']:.3f}" for (la, ra), (lb, rb) in zip(res, res[1:], strict=False)))
    if args.adaptive:
        print("-" * 96)
        for label, r in res:
            print_frames(label, r)
    print("=" * 96)
    print("VERDICT: " + ("PASS" if not failures else "FAIL (" + "; ".join(failures) + ")"))
    print("=" * 96)
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
