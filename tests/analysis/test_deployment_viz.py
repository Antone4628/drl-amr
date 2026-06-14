"""Plumbing tests for deployment visualization (RESTRUCTURE Phase 6, step 6).

Headless (Agg). Exercises rendering on directly-constructed SolverSnapshots —
visual correctness is the eyeball gate at the Phase-6-exit run, not here.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from analysis.deployment_viz import (  # noqa: E402
    default_annotation,
    plot_diagnostics,
    plot_snapshot,
    save_animation,
    save_snapshot,
)
from contract.solver_snapshot import SolverSnapshot  # noqa: E402


def _snap(time=0.0, n_active=4, ngl=5, level_max=1) -> SolverSnapshot:
    npoin = ngl * n_active
    coord = np.linspace(-1.0, 1.0, npoin)
    q = np.sin(np.pi * coord)
    levels = np.zeros(n_active, dtype=int)
    levels[-1] = level_max
    return SolverSnapshot(
        time=time, q=q, coord=coord,
        intma=np.arange(npoin).reshape(n_active, ngl),
        xelem=np.linspace(-1.0, 1.0, n_active + 1),
        active=np.arange(n_active),
        levels=levels, ngl=ngl, xgl=np.linspace(-1.0, 1.0, ngl),
        npoin_dg=npoin, n_active=n_active,
    )


def test_plot_snapshot_draws_per_element_curves():
    snap = _snap(n_active=4, ngl=5)
    ax = plot_snapshot(snap)
    # element curves + boundary axvlines are all Line2D; at least one per element
    assert len(ax.get_lines()) >= snap.n_active
    plt.close(ax.figure)


def test_exact_overlay_auto_off_and_on():
    snap = _snap()
    ax_off = plot_snapshot(snap, exact_fn=None)
    n_off = len(ax_off.get_lines())
    plt.close(ax_off.figure)
    ax_on = plot_snapshot(snap, exact_fn=lambda coord, t: np.zeros_like(coord))
    n_on = len(ax_on.get_lines())
    plt.close(ax_on.figure)
    assert n_on == n_off + 1  # overlay adds exactly one line


def test_annotation_box_optional():
    snap = _snap()
    ax_no = plot_snapshot(snap, annotate=None)
    assert len(ax_no.texts) == 0
    plt.close(ax_no.figure)
    ax_yes = plot_snapshot(snap, annotate=default_annotation)
    assert len(ax_yes.texts) == 1
    plt.close(ax_yes.figure)


def test_default_annotation_keys():
    d = default_annotation(_snap(time=0.5, n_active=4, level_max=2))
    assert set(d) == {"t", "elements", "max level"}
    assert d["elements"] == "4"
    assert d["max level"] == "2"

def test_save_animation_writes_gif(tmp_path):
    snaps = [_snap(time=0.1 * i, n_active=4 + i) for i in range(4)]
    out = tmp_path / "rollout.gif"
    save_animation(snaps, out, fps=10, annotate=default_annotation)
    assert out.exists() and out.stat().st_size > 0


def test_save_snapshot_writes_png(tmp_path):
    out = tmp_path / "frame.png"
    save_snapshot(_snap(), out, annotate=default_annotation)
    assert out.exists() and out.stat().st_size > 0


def test_plot_diagnostics_returns_axes():
    snaps = [_snap(time=0.0, n_active=4), _snap(time=0.1, n_active=6)]
    axes = plot_diagnostics(snaps)
    assert len(axes) == 2
    plt.close(axes[0].figure)