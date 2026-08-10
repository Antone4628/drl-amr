#!/usr/bin/env python3
"""DG-correct renderer for 1D Jexpresso solution dumps.

Standalone: reads only the plain-text dumps written by the `JE_DUMP` probe and
depends on nothing in `drl-amr` beyond numpy/matplotlib. It deliberately does
NOT go through `contract.SolverSnapshot` -- that type carries AMR fields
(levels, budget) about which a fixed-mesh Jexpresso run has nothing truthful to
say, and populating them with zeros to satisfy a type is the kind of small lie
that gets read back as fact later.

Why this exists at all: a nodal line plot of a DG solution is misleading in two
specific ways.

  1. Element-interface discontinuities are drawn as steep line segments,
     indistinguishable from a genuine gradient. The jump magnitude IS the DG
     observable of interest -- it is the error indicator this project feeds the
     RL agent -- so smearing it into a slope discards the main thing you want
     to read off the picture.

  2. Within-element oscillation is invisible. A degree-(ngl-1) polynomial can
     wiggle substantially between its nodes, so a nodal plot is only a LOWER
     bound on how oscillatory the solution really is.

This tool fixes both: each element is drawn as its actual polynomial, densely
sampled, and each element gets its OWN plot call so matplotlib never connects
the last node of element e to the first node of element e+1. The interface
discontinuity therefore renders as a true break.

The interpolation math is the same formulation used by
`analysis/deployment_viz._lagrange_interpolate`, deliberately, so the two
renderers cannot disagree about what a DG element looks like.

Usage
-----
    # one snapshot
    python -m tools.plot_dg_1d dg_dump_it004.txt --periodic --out dg.png

    # selected times, four panels
    python -m tools.plot_dg_1d dg_dump_it*.txt --times 0 0.3 1.5 1.7 --out fig.png

    # every 4th dump, wrapped into a grid
    python -m tools.plot_dg_1d dg_dump_it*.txt --stride 4 --max-cols 4 --out grid.png

    # animation (.gif needs no ffmpeg; .mp4 does)
    python -m tools.plot_dg_1d dg_dump_it*.txt --animate --fps 8 --out dg.gif
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: never open a window, never call plt.show()

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter  # noqa: E402

_HEADER_RE = re.compile(r"^#\s*(\w+)\s+(\S+)")

_DG_BLUE = "#1f77b4"


# --------------------------------------------------------------------------
# I/O
# --------------------------------------------------------------------------
def read_dump(path: Path) -> tuple[dict, np.ndarray, np.ndarray]:
    """Parse a JE_DUMP text file.

    Returns (meta, x, q) where

        x : (nelem, ngl)        element-major nodal coordinates
        q : (nelem, ngl, nvar)  element-major nodal values

    The dump is written element-major (iel outer, node inner) straight off
    `mesh.connijk`, so the reshape below is exact for both DG (duplicated
    interface DOFs) and CG (shared ones). Under CG the two "sides" of an
    interface simply carry the same value and the rendered gap has zero height
    -- which is itself the correct picture, and makes a CG/DG A-B comparison
    with this same tool meaningful.
    """
    meta: dict[str, str] = {}
    with open(path) as fh:
        for line in fh:
            if not line.startswith("#"):
                break
            m = _HEADER_RE.match(line)
            if m:
                meta[m.group(1)] = m.group(2)

    for key in ("t", "nelem", "ngl", "nvar"):
        if key not in meta:
            raise ValueError(f"{path}: dump header is missing '# {key} <value>'")

    nelem = int(meta["nelem"])
    ngl = int(meta["ngl"])
    nvar = int(meta["nvar"])

    raw = np.atleast_2d(np.loadtxt(path, comments="#"))
    expected = nelem * ngl
    if raw.shape[0] != expected:
        raise ValueError(
            f"{path}: header says nelem*ngl = {expected} rows, file has {raw.shape[0]}. "
            "Stale dump, or the probe and the run disagree about the mesh."
        )
    if raw.shape[1] != 2 + nvar:
        raise ValueError(
            f"{path}: expected {2 + nvar} columns (iel, x, q1..q{nvar}), "
            f"file has {raw.shape[1]}."
        )

    x = raw[:, 1].reshape(nelem, ngl)
    q = raw[:, 2 : 2 + nvar].reshape(nelem, ngl, nvar)

    info = {
        "t": float(meta["t"]),
        "nelem": nelem,
        "ngl": ngl,
        "nvar": nvar,
        "AD": meta.get("AD", "unknown"),
        "path": Path(path),
    }
    return info, x, q


# --------------------------------------------------------------------------
# Interpolation and diagnostics
# --------------------------------------------------------------------------
def lagrange_eval(nodes: np.ndarray, values: np.ndarray, sample: np.ndarray) -> np.ndarray:
    """Evaluate the Lagrange interpolant through (nodes, values) at `sample`.

    Built directly in PHYSICAL coordinates rather than mapping back to the
    reference element first. The element map is affine, and every term of the
    basis is a ratio of coordinate differences, so the two are algebraically
    identical -- this way just has fewer moving parts and needs no knowledge of
    the reference LGL nodes.
    """
    nodes = np.asarray(nodes, dtype=float)
    values = np.asarray(values, dtype=float)
    sample = np.asarray(sample, dtype=float)
    n = nodes.size
    basis = np.ones((n, sample.size))
    for b in range(n):
        for k in range(n):
            if k != b:
                basis[b] *= (sample - nodes[k]) / (nodes[b] - nodes[k])
    return basis.T @ values


def interface_jumps(q: np.ndarray, ivar: int, *, periodic: bool = False) -> np.ndarray:
    """Signed jumps q(+) - q(-) at each element interface.

    Interior interfaces are element e's right trace against element e+1's left
    trace. With `periodic`, the wrap face (last element's right trace against
    the first element's left trace) is appended -- worth checking explicitly,
    since a collapsed wrap node is exactly the defect class this project has
    hit before and it shows up here as an identically-zero wrap jump.
    """
    right = q[:-1, -1, ivar]
    left = q[1:, 0, ivar]
    jumps = left - right
    if periodic:
        jumps = np.append(jumps, q[0, 0, ivar] - q[-1, -1, ivar])
    return jumps


# --------------------------------------------------------------------------
# Plotting primitive
# --------------------------------------------------------------------------
def plot_dg(
    x: np.ndarray,
    q: np.ndarray,
    ax,
    *,
    ivar: int = 0,
    n_dense: int = 40,
    show_nodes: bool = True,
    show_bounds: bool = True,
    gap_frac: float = 0.0,
    color: str = _DG_BLUE,
    label: str | None = "DG polynomial",
):
    """Draw one variable as per-element polynomials with true interface breaks.

    `gap_frac` insets each element's DRAWN extent by that fraction of its width
    at both ends (0.01 is plenty). This is a presentation device only -- it
    makes the discontinuity unmistakable at a glance on a conforming mesh,
    where the true gap has zero width in x. It very slightly misrepresents the
    domain, so it is off by default; turn it on for a slide, off for analysis.
    """
    nelem, _ngl = x.shape

    if show_bounds:
        for e in range(nelem):
            ax.axvline(x[e, 0], color="0.85", lw=0.6, ls=":", zorder=0)
        ax.axvline(x[-1, -1], color="0.85", lw=0.6, ls=":", zorder=0)

    for e in range(nelem):
        xe = x[e]
        x_lo, x_hi = float(xe[0]), float(xe[-1])
        inset = gap_frac * (x_hi - x_lo)
        xs = np.linspace(x_lo + inset, x_hi - inset, n_dense)
        ys = lagrange_eval(xe, q[e, :, ivar], xs)

        # One plot call PER ELEMENT. This is the whole point: matplotlib draws
        # each element as an independent Line2D, so nothing is ever drawn
        # between element e's last point and element e+1's first point.
        ax.plot(xs, ys, color=color, lw=1.7, zorder=3,
                label=label if (e == 0 and label) else None)

        if show_nodes:
            ax.plot(xe, q[e, :, ivar], ls="none", marker="o", ms=3.0,
                    mfc="white", mec=color, mew=0.9, zorder=4)

    ax.set_xlabel("x")
    return ax


def _global_ylim(dumps, ivar):
    lo = min(float(q[:, :, ivar].min()) for _, _, q in dumps)
    hi = max(float(q[:, :, ivar].max()) for _, _, q in dumps)
    pad = 0.08 * (hi - lo) + 1e-12
    return lo - pad, hi + pad


def _var_names(nvar, requested):
    names = list(requested or [])
    names += [f"q{i + 1}" for i in range(len(names), nvar)]
    return names[:nvar]


# --------------------------------------------------------------------------
# Static figure (grid, wrapped)
# --------------------------------------------------------------------------
def build_figure(dumps, args):
    """Grid of panels. Columns wrap at `--max-cols` so a long time series stays
    viewable instead of becoming one unreadable strip. With multiple solution
    variables each variable gets its own block of rows."""
    nvar = dumps[0][0]["nvar"]
    n = len(dumps)
    ncol = max(1, min(args.max_cols, n))
    rows_per_var = -(-n // ncol)  # ceil
    names = _var_names(nvar, args.var_names)

    fig, axes = plt.subplots(
        nvar * rows_per_var, ncol,
        figsize=(5.0 * ncol, 3.4 * nvar * rows_per_var),
        squeeze=False,
    )

    for ivar in range(nvar):
        ylim = tuple(args.ylim) if args.ylim else _global_ylim(dumps, ivar)
        for k, (info, x, q) in enumerate(dumps):
            ax = axes[ivar * rows_per_var + k // ncol, k % ncol]
            plot_dg(
                x, q, ax,
                ivar=ivar,
                n_dense=args.n_dense,
                show_nodes=not args.no_nodes,
                show_bounds=not args.no_bounds,
                gap_frac=args.gap_frac,
                label="DG polynomial" if (ivar == 0 and k == 0) else None,
            )
            ax.set_ylim(*ylim)

            jumps = interface_jumps(q, ivar, periodic=args.periodic)
            max_jump = float(np.abs(jumps).max()) if jumps.size else 0.0
            ax.set_title(
                f"{names[ivar]}   t = {info['t']:.4f}\n"
                f"nelem = {info['nelem']}, p = {info['ngl'] - 1}, "
                f"max |jump| = {max_jump:.2e}",
                fontsize=10,
            )
            if k % ncol == 0:
                ax.set_ylabel(names[ivar])
                if ivar == 0 and k == 0:
                    ax.legend(loc="upper right", fontsize=8)
            else:
                ax.set_ylabel("")

        # Blank any unused cells in this variable's block.
        for k in range(n, rows_per_var * ncol):
            axes[ivar * rows_per_var + k // ncol, k % ncol].axis("off")

    if args.suptitle:
        fig.suptitle(args.suptitle, fontsize=11, fontweight="bold")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# Animation
# --------------------------------------------------------------------------
def build_animation(dumps, args):
    """One panel per solution variable, one frame per dump, fixed y-limits.

    y-limits are computed once over the whole series so amplitude is comparable
    frame to frame -- an animation that rescales per frame hides exactly the
    dissipation you want to see.
    """
    nvar = dumps[0][0]["nvar"]
    names = _var_names(nvar, args.var_names)
    ylims = [tuple(args.ylim) if args.ylim else _global_ylim(dumps, i)
             for i in range(nvar)]

    fig, axes = plt.subplots(nvar, 1, figsize=(9.0, 3.8 * nvar), squeeze=False)
    if args.suptitle:
        fig.suptitle(args.suptitle, fontsize=11, fontweight="bold")
    n = len(dumps)

    def draw(k):
        info, x, q = dumps[k]
        for ivar in range(nvar):
            ax = axes[ivar, 0]
            ax.clear()
            plot_dg(
                x, q, ax,
                ivar=ivar,
                n_dense=args.n_dense,
                show_nodes=not args.no_nodes,
                show_bounds=not args.no_bounds,
                gap_frac=args.gap_frac,
                label="DG polynomial" if ivar == 0 else None,
            )
            ax.set_ylim(*ylims[ivar])
            ax.set_ylabel(names[ivar])

            jumps = interface_jumps(q, ivar, periodic=args.periodic)
            max_jump = float(np.abs(jumps).max()) if jumps.size else 0.0
            ax.text(
                0.02, 0.97,
                f"t = {info['t']:.4f}  |  max |jump| = {max_jump:.3e}\n"
                f"frame {k + 1}/{n}",
                transform=ax.transAxes, va="top", ha="left", fontsize=9,
                family="monospace",
                bbox={"boxstyle": "round,pad=0.3", "fc": "white", "alpha": 0.85},
            )
            if ivar == 0:
                ax.legend(loc="upper right", fontsize=8)
        return ()

    anim = FuncAnimation(fig, draw, frames=n, interval=1000.0 / args.fps, blit=False)
    return fig, anim


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def _report(dumps, periodic):
    """Print the interface-jump summary. The numbers are the point, not decoration:
    a strictly-zero jump at every face means the run was not actually DG."""
    for info, _x, q in dumps:
        print(f"{info['path'].name}  t = {info['t']:.4f}  AD = {info['AD']}  "
              f"nelem = {info['nelem']}  p = {info['ngl'] - 1}  nvar = {info['nvar']}")
        for ivar in range(info["nvar"]):
            j = interface_jumps(q, ivar, periodic=periodic)
            print(f"    var {ivar + 1}: max |jump| = {np.abs(j).max():.6e}   "
                  f"mean |jump| = {np.abs(j).mean():.6e}   n_faces = {j.size}")
            if periodic:
                print(f"             wrap-face jump = {j[-1]:+.6e}")


def _select(dumps, args):
    """Apply --times (nearest match) then --stride."""
    if args.times:
        available = np.array([d[0]["t"] for d in dumps])
        picked = [dumps[int(np.argmin(np.abs(available - t)))] for t in args.times]
        seen, out = set(), []
        for d in picked:  # de-duplicate, preserving order
            key = d[0]["path"]
            if key not in seen:
                seen.add(key)
                out.append(d)
        dumps = out
    if args.stride > 1:
        dumps = dumps[:: args.stride]
    return dumps


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="DG-correct renderer for 1D Jexpresso dumps.",
    )
    ap.add_argument("dumps", nargs="+", type=Path,
                    help="JE_DUMP text files, one per output time")
    ap.add_argument("--out", type=Path, default=Path("dg_plot.png"),
                    help="output path; .gif/.mp4 with --animate, else an image")
    ap.add_argument("--animate", action="store_true",
                    help="write an animation instead of a static grid "
                         "(.gif uses Pillow, no ffmpeg needed; .mp4 needs ffmpeg)")
    ap.add_argument("--fps", type=int, default=8,
                    help="animation frames per second (default: 8)")
    ap.add_argument("--max-cols", type=int, default=4,
                    help="static mode: wrap the panel grid at this many columns "
                         "(default: 4)")
    ap.add_argument("--times", nargs="*", type=float,
                    help="keep only the dumps nearest these simulation times")
    ap.add_argument("--stride", type=int, default=1,
                    help="keep every Nth dump (applied after --times)")
    ap.add_argument("--n-dense", type=int, default=40,
                    help="samples per element for the polynomial curve (default: 40)")
    ap.add_argument("--no-nodes", action="store_true",
                    help="hide the LGL nodal markers")
    ap.add_argument("--no-bounds", action="store_true",
                    help="hide the dotted element-boundary lines")
    ap.add_argument("--gap-frac", type=float, default=0.0,
                    help="presentation only: inset each element's drawn extent by "
                         "this fraction of its width so the interface break is "
                         "visibly wide (try 0.01; default 0 = true geometry)")
    ap.add_argument("--periodic", action="store_true",
                    help="also report the periodic wrap-face jump")
    ap.add_argument("--var-names", nargs="*",
                    help="labels for the solution variables, e.g. --var-names u v")
    ap.add_argument("--ylim", nargs=2, type=float, metavar=("LO", "HI"))
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--suptitle")
    ap.add_argument("--quiet", action="store_true",
                    help="suppress the per-dump jump report")
    args = ap.parse_args(argv)

    dumps = [read_dump(p) for p in args.dumps]
    dumps.sort(key=lambda d: d[0]["t"])

    nvars = {d[0]["nvar"] for d in dumps}
    if len(nvars) != 1:
        raise SystemExit(f"dumps disagree about nvar: {sorted(nvars)}")

    dumps = _select(dumps, args)
    if not dumps:
        raise SystemExit("no dumps left after --times/--stride selection")

    if not args.quiet:
        _report(dumps, args.periodic)

    args.out.parent.mkdir(parents=True, exist_ok=True)

    if args.animate:
        fig, anim = build_animation(dumps, args)
        writer = (PillowWriter(fps=args.fps)
                  if str(args.out).lower().endswith(".gif")
                  else FFMpegWriter(fps=args.fps))
        anim.save(str(args.out), writer=writer, dpi=args.dpi)
        plt.close(fig)
    else:
        fig = build_figure(dumps, args)
        fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)

    print(f"\nwrote {args.out}  ({len(dumps)} frames/panels)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
