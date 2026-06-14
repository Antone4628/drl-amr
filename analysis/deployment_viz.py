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
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter

from contract.solver_snapshot import SolverSnapshot


def default_annotation(snap: SolverSnapshot) -> dict[str, str]:
    """Convenience label->value box: time, element count, max level."""
    return {
        "t": f"{snap.time:.4f}",
        "elements": str(snap.n_active),
        "max level": str(int(snap.levels.max())),
    }


def plot_snapshot(
    snap: SolverSnapshot,
    ax=None,
    *,
    exact_fn=None,
    annotate=None,
    color_by_level: bool = True,
    ylim: tuple[float, float] | None = None,
):
    """Render one snapshot to an Axes (per-element DG curves).

    exact_fn(coord, time) -> q overlays the analytic solution when supplied
    (auto-off when None). annotate(snap) -> dict[str, str] draws a label box when
    supplied (no box when None).
    """
    if ax is None:
        _, ax = plt.subplots()

    ngl = snap.ngl
    cmap = plt.get_cmap("viridis")
    max_lvl = max(int(snap.levels.max()), 1)

    # Element boundaries (behind the curves).
    for xb in snap.xelem:
        ax.axvline(xb, color="0.85", lw=0.5, zorder=0)

    # Per-element DG curves — slice the element-contiguous solution so the
    # line does not connect across element-interface jumps.
    for e in range(snap.n_active):
        sl = slice(e * ngl, (e + 1) * ngl)
        color = cmap(int(snap.levels[e]) / max_lvl) if color_by_level else "C0"
        ax.plot(snap.coord[sl], snap.q[sl], color=color, lw=1.5)

    # Optional exact overlay (auto-off when exact_fn is None).
    if exact_fn is not None:
        q_exact = exact_fn(snap.coord, snap.time)
        ax.plot(snap.coord, q_exact, "k--", lw=1.0, alpha=0.7, label="exact")
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
    ax.set_ylabel("q")
    ax.set_title(f"t = {snap.time:.4f}")
    return ax


def _global_ylim(snapshots: list[SolverSnapshot]) -> tuple[float, float]:
    lo = min(float(s.q.min()) for s in snapshots)
    hi = max(float(s.q.max()) for s in snapshots)
    pad = 0.1 * (hi - lo) + 1e-12
    return lo - pad, hi + pad


def animate(
    snapshots: list[SolverSnapshot],
    *,
    exact_fn=None,
    annotate=None,
    ylim: tuple[float, float] | None = None,
    fps: int = 20,
):
    """Build a FuncAnimation over the snapshot list. Returns (fig, anim).

    ylim is fixed across all frames (defaults to the global solution range) so
    the y-axis does not jump as the mesh/solution evolve.
    """
    fig, ax = plt.subplots()
    if ylim is None:
        ylim = _global_ylim(snapshots)

    def draw(i):
        ax.clear()
        plot_snapshot(snapshots[i], ax=ax, exact_fn=exact_fn, annotate=annotate, ylim=ylim)
        return ()

    anim = FuncAnimation(fig, draw, frames=len(snapshots), interval=1000.0 / fps, blit=False)
    return fig, anim


def save_animation(
    snapshots: list[SolverSnapshot],
    path,
    *,
    fps: int = 20,
    exact_fn=None,
    annotate=None,
    ylim: tuple[float, float] | None = None,
):
    """Build and save an animation. Writer inferred from the extension: .gif ->
    PillowWriter (no ffmpeg needed), anything else -> FFMpegWriter (.mp4)."""
    fig, anim = animate(snapshots, exact_fn=exact_fn, annotate=annotate, ylim=ylim, fps=fps)
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