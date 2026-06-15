"""Visualization for deployment rollouts (RESTRUCTURE Phase 6, deployment adapter
build-order step 6).

A pure consumer of the SolverSnapshot list the DeploymentRunner produces — no
backend, no solver, no Gym. Renders per-element DG curves (so element-interface
discontinuities are visible rather than smeared), an optional exact-solution
overlay that is AUTO-OFF unless an exact_fn is supplied (required: we are heading
to SWE/2D where no analytic solution exists, DEPLOYMENT_ADAPTER_DESIGN.md §3), and
an optional annotation box that is EMPTY by default and takes a label->value dict
so new fields (element count, resource usage, max level, ...) are cheap to add.

Headless-safe: never calls plt.show(); the caller/test picks the backend
(matplotlib.use("Agg") for file-only output).
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter

from contract.solver_snapshot import SolverSnapshot


def default_annotation(snap: SolverSnapshot) -> dict[str, str]:
    """Convenience label->value box: time, element count, max level."""
    return {
        "t": f"{snap.time:.4f}",
        "elements": str(snap.n_active),
        "max level": str(int(snap.levels.max())),
    }

def _lagrange_interpolate(nodes, values, sample):
    """Evaluate the Lagrange interpolant through (nodes, values) at `sample`.

    Pure numpy; `nodes` are the reference LGL points in [-1, 1], `values` the
    element's nodal solution. Backend-free — no dependency on backends/.
    """
    nodes = np.asarray(nodes, dtype=float)
    values = np.asarray(values, dtype=float)
    sample = np.asarray(sample, dtype=float)
    n = len(nodes)
    basis = np.ones((n, len(sample)))
    for b in range(n):
        for k in range(n):
            if k != b:
                basis[b] *= (sample - nodes[k]) / (nodes[b] - nodes[k])
    return basis.T @ values


def _element_polynomial(snap, e, n_dense=30):
    """Densely sample element e's DG polynomial -> (x_dense, u_dense).

    Reads the element-contiguous nodal slice of q and the element's physical
    extent from the snapshot, maps the reference LGL nodes (xgl) onto the
    element, and evaluates the degree-(ngl-1) interpolant on a dense grid so the
    true polynomial (and any inter-element jump) renders correctly.
    """
    ngl = snap.ngl
    u_elem = snap.q[e * ngl:(e + 1) * ngl]
    x_left = float(snap.xelem[e])
    x_right = float(snap.xelem[e + 1])
    zeta = np.linspace(-1.0, 1.0, n_dense)
    x_dense = x_left + 0.5 * (1.0 + zeta) * (x_right - x_left)
    return x_dense, _lagrange_interpolate(snap.xgl, u_elem, zeta)

def plot_snapshot(
    snap: SolverSnapshot,
    ax=None,
    *,
    exact_fn=None,
    annotate=None,
    color_by_level: bool = False,
    ylim: tuple[float, float] | None = None,
    n_dense: int = 30,
    legend: bool = True,
):
    """Render one snapshot to an Axes as DG-correct per-element polynomials.

    Each element is drawn as its actual degree-(ngl-1) polynomial, densely
    sampled (n_dense points), so the curve is the true DG solution and genuine
    inter-element jumps show as jumps rather than straight lines between nodes.
    Single blue by default (level is conveyed by the composite's level bars);
    color_by_level=True grades the curves by refinement level instead.

    exact_fn(coord, time) -> q overlays the analytic solution on a dense grid
    when supplied (auto-off when None). annotate(snap) -> dict[str, str] draws a
    label box when supplied (no box when None). legend draws the DG/Exact key.
    """
    if ax is None:
        _, ax = plt.subplots()

    cmap = plt.get_cmap("viridis")
    max_lvl = max(int(snap.levels.max()), 1)

    # Element boundaries (behind the curves).
    for xb in snap.xelem:
        ax.axvline(xb, color="0.8", lw=0.6, ls=":", zorder=0)

    # Per-element DG-correct polynomial curves. Label only the first element so
    # the legend carries a single "DG solution" entry.
    for e in range(snap.n_active):
        x_dense, u_dense = _element_polynomial(snap, e, n_dense=n_dense)
        color = cmap(int(snap.levels[e]) / max_lvl) if color_by_level else "#1f77b4"
        ax.plot(x_dense, u_dense, color=color, lw=1.5,
                label="DG solution" if e == 0 else None)

    # Optional exact overlay on a dense grid (auto-off when exact_fn is None).
    if exact_fn is not None:
        x_exact = np.linspace(float(snap.xelem[0]), float(snap.xelem[-1]), 500)
        ax.plot(x_exact, exact_fn(x_exact, snap.time), "r--", lw=1.0, alpha=0.8,
                label="Exact")

    if legend:
        ax.legend(loc="upper right", fontsize=8)

    # Optional annotation box (default empty).
    if annotate is not None:
        text = "\n".join(f"{k}: {v}" for k, v in annotate(snap).items())
        ax.text(
            0.02, 0.98, text, transform=ax.transAxes, va="top", ha="left",
            fontsize=8, family="monospace",
            bbox={"boxstyle": "round", "fc": "white", "ec": "0.7", "alpha": 0.8},
        )

    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlabel("x")
    ax.set_ylabel("u(x)")
    ax.set_title(f"t = {snap.time:.4f}")
    return ax


def _global_ylim(snapshots: list[SolverSnapshot]) -> tuple[float, float]:
    lo = min(float(s.q.min()) for s in snapshots)
    hi = max(float(s.q.max()) for s in snapshots)
    pad = 0.1 * (hi - lo) + 1e-12
    return lo - pad, hi + pad

def _frame_info(snap, frame_idx, n_frames, *, budget=None):
    """Animation info-box string (pre-restructure style): a pipe-joined metrics
    line then a frame counter. resource is omitted when budget is None."""
    parts = [f"t = {snap.time:.4f}", f"N = {snap.n_active}"]
    if budget:
        parts.append(f"resource = {snap.n_active / budget:.2f}")
    parts.append(f"max level = {int(snap.levels.max())}")
    return "  |  ".join(parts) + f"\nframe {frame_idx + 1}/{n_frames}"

def animate(
    snapshots: list[SolverSnapshot],
    *,
    exact_fn=None,
    ylim: tuple[float, float] | None = None,
    fps: int = 20,
    budget: int | None = None,
    suptitle: str | None = None,
    info_box: bool = True,
):
    """Build a FuncAnimation over the snapshots — single DG panel, no level bars.

    Each frame draws the DG-correct solution + optional exact overlay + element
    boundaries, with a per-frame info box (t | N | resource | max level, then
    frame i/total) when info_box is set, and an optional run-metadata suptitle.
    ylim is fixed across frames. Returns (fig, anim).
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    if ylim is None:
        ylim = _global_ylim(snapshots)
    n = len(snapshots)
    if suptitle:
        fig.suptitle(suptitle, fontsize=10, fontweight="bold")

    def draw(i):
        ax.clear()
        plot_snapshot(snapshots[i], ax=ax, exact_fn=exact_fn, ylim=ylim, legend=True)
        ax.set_title("")  # run metadata lives in the suptitle; per-frame in the box
        if info_box:
            ax.text(
                0.02, 0.97, _frame_info(snapshots[i], i, n, budget=budget),
                transform=ax.transAxes, va="top", ha="left", fontsize=9,
                bbox={"boxstyle": "round,pad=0.3", "fc": "white", "alpha": 0.8},
            )
        return ()

    anim = FuncAnimation(fig, draw, frames=n, interval=1000.0 / fps, blit=False)
    return fig, anim


def save_animation(
    snapshots: list[SolverSnapshot],
    path,
    *,
    fps: int = 20,
    exact_fn=None,
    ylim: tuple[float, float] | None = None,
    budget: int | None = None,
    suptitle: str | None = None,
    info_box: bool = True,
):
    """Build and save an animation. Writer inferred from the extension: .gif ->
    PillowWriter (no ffmpeg needed), anything else -> FFMpegWriter (.mp4)."""
    fig, anim = animate(
        snapshots, exact_fn=exact_fn, ylim=ylim, fps=fps,
        budget=budget, suptitle=suptitle, info_box=info_box,
    )
    writer = PillowWriter(fps=fps) if str(path).lower().endswith(".gif") else FFMpegWriter(fps=fps)
    anim.save(str(path), writer=writer)
    plt.close(fig)
    return path


def save_snapshot(snap: SolverSnapshot, path, *, exact_fn=None, annotate=None, ylim=None):
    """Save a single frame to an image file."""
    fig, ax = plt.subplots()
    plot_snapshot(snap, ax=ax, exact_fn=exact_fn, annotate=annotate, ylim=ylim)
    fig.savefig(str(path), dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path

def plot_level_bars(snap, ax, *, max_level=None, color_by_level=True):
    """Per-element refinement-level bars: height = level, width = element extent.

    Reads level straight off the lean snapshot (snap.levels) — no error or
    threshold data needed. max_level fixes the y-axis across a multi-column
    composite so bar heights are comparable column to column.
    """
    cmap = plt.get_cmap("viridis")
    mlvl = max(int(max_level if max_level is not None else snap.levels.max()), 1)
    for e in range(snap.n_active):
        x_left = float(snap.xelem[e])
        x_right = float(snap.xelem[e + 1])
        level = int(snap.levels[e])
        color = cmap(level / mlvl) if color_by_level else "#1f77b4"
        ax.add_patch(plt.Rectangle(
            (x_left, 0.0), x_right - x_left, level,
            facecolor=color, edgecolor="k", lw=0.5, alpha=0.8,
        ))
    ax.set_xlim(float(snap.xelem[0]), float(snap.xelem[-1]))
    ax.set_ylim(0.0, mlvl + 0.5)
    ax.set_yticks(range(mlvl + 1))
    ax.set_ylabel("Level")
    ax.set_xticklabels([])
    return ax


def select_snapshots(snapshots, target_times):
    """Return the snapshot nearest each target time (input order preserved)."""
    snap_times = np.array([s.time for s in snapshots])
    return [snapshots[int(np.argmin(np.abs(snap_times - t)))] for t in target_times]


def composite_snapshot(
    snapshots, *, exact_fn=None, show_bars=True, max_level=None,
    suptitle=None, ylim=None,
):
    """Multi-time composite figure: one column per snapshot, in time order.

    show_bars -> 2 rows/column (level bars on top, DG-vs-exact below); else
    1 row (DG-vs-exact only). The DG panels share a fixed y-range so amplitude
    is comparable across columns; the legend is drawn on the first column only.
    Returns the Figure.
    """
    n = len(snapshots)
    if n == 0:
        raise ValueError("composite_snapshot needs at least one snapshot")
    if ylim is None:
        ylim = _global_ylim(snapshots)
    mlvl = (max_level if max_level is not None
            else max(max(int(s.levels.max()) for s in snapshots), 1))
    nrows = 2 if show_bars else 1
    fig, axes = plt.subplots(nrows, n, figsize=(4 * n, 6 if show_bars else 3),
                             squeeze=False)
    for col, snap in enumerate(snapshots):
        title = f"t = {snap.time:.3f}\nN = {snap.n_active}"
        if show_bars:
            plot_level_bars(snap, axes[0, col], max_level=mlvl)
            axes[0, col].set_title(title, fontsize=9)
            ax_sol = axes[1, col]
        else:
            ax_sol = axes[0, col]
        plot_snapshot(snap, ax=ax_sol, exact_fn=exact_fn, ylim=ylim, legend=(col == 0))
        ax_sol.set_title("" if show_bars else title, fontsize=9)
        if col != 0:
            ax_sol.set_ylabel("")
    if suptitle:
        fig.suptitle(suptitle, fontsize=10, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig


def save_composite_snapshot(
    snapshots, path, *, exact_fn=None, show_bars=True, max_level=None,
    suptitle=None, ylim=None, dpi=150,
):
    """Build and save the multi-time composite figure (format by extension)."""
    fig = composite_snapshot(
        snapshots, exact_fn=exact_fn, show_bars=show_bars,
        max_level=max_level, suptitle=suptitle, ylim=ylim,
    )
    fig.savefig(str(path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path

def plot_diagnostics(snapshots: list[SolverSnapshot], axes=None):
    """Time-series diagnostics: active-element count and max refinement level."""
    times = [s.time for s in snapshots]
    n_active = [s.n_active for s in snapshots]
    max_level = [int(s.levels.max()) for s in snapshots]
    if axes is None:
        _, axes = plt.subplots(2, 1, sharex=True)
    axes[0].plot(times, n_active, "-o", ms=2)
    axes[0].set_ylabel("active elements")
    axes[1].plot(times, max_level, "-o", ms=2)
    axes[1].set_ylabel("max level")
    axes[1].set_xlabel("t")
    return axes