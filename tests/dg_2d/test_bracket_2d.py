"""Pins for tools/bracket_2d.py -- the solution-level half of the Phase-6
item-7 gate.

The named regression against real data (reproducing vtu_error.jl's banked
10x20 value 3.042960147959e-03 from the convergence study's vtu) needs the
Jexpresso output tree and is run by hand; these pins cover what can go wrong
without it:

  1. the exact solution: at t = 4 it equals vtu_error.jl's image-sum Gaussian
     (centre (2, 12)), restated independently here; it wraps periodically
  2. the reader: a raw-appended VTU written here round-trips exactly, and a
     Float32 field or a frame with no time source is refused -- an L2 against
     the wrong time, or a mass check at single precision, would be silent
  3. the scorer: a static 2:1 run whose frames carry the exact field reads
     L2 = 0 at the final frame, and identical frames read zero mass drift

Run with: pytest tests/dg_2d/test_bracket_2d.py -v
"""

import os

import numpy as np
import pytest

from backends.python_1d.dg.basis import lgl_gen
from tools.bracket_2d import gaussian_exact, load_frames, read_vtu, score_run


# The static 2:1 gate mesh (4x4 on [-5,5]x[0,20], cell (2,2) refined), copied
# from tests/dg_2d/test_gate_2d_mortar.py rather than imported -- the repo's
# test files do not depend on each other's importability.
def boxes_2to1():
    dx, dy = 2.5, 5.0
    boxes = [(-5 + ex * dx, -5 + (ex + 1) * dx, ey * dy, (ey + 1) * dy)
             for ey in range(4) for ex in range(4) if (ex, ey) != (2, 2)]
    xa, xb, ya, yb = 0.0, 2.5, 10.0, 15.0
    xm, ym = 1.25, 12.5
    return boxes + [(xa, xm, ym, yb), (xa, xm, ya, ym), (xm, xb, ym, yb), (xm, xb, ya, ym)]


def nodes_from_boxes(boxes, nop=4):
    xgl, _ = lgl_gen(nop + 1)
    x, y = [], []
    for xa, xb, ya, yb in boxes:
        for j in range(nop + 1):
            for i in range(nop + 1):
                x.append(xa + (xgl[i] + 1) / 2 * (xb - xa))
                y.append(ya + (xgl[j] + 1) / 2 * (yb - ya))
    return np.array(x), np.array(y)


def write_vtu(path, x, y, q, t=None, qtype="<f8"):
    """Minimal writer in Jexpresso's layout: raw appended data, UInt64 headers."""
    pts = np.column_stack([x, y, np.zeros_like(x)]).astype("<f8").ravel()
    q = np.asarray(q).astype(qtype)
    blocks, arrays, off = [], [], 0
    specs = [("Points", "Float64", 3, pts)]
    if t is not None:
        specs.append(("TimeValue", "Float64", 1, np.array([t], dtype="<f8")))
    specs.append(("q", "Float64" if qtype == "<f8" else "Float32", 1, q))
    for name, typ, nc, arr in specs:
        raw = arr.tobytes()
        arrays.append(f'<DataArray type="{typ}" Name="{name}" NumberOfComponents="{nc}" '
                      f'format="appended" offset="{off}"/>')
        blocks.append(np.array([len(raw)], dtype="<u8").tobytes() + raw)
        off += 8 + len(raw)
    head = ('<?xml version="1.0" encoding="utf-8"?>\n'
            '<VTKFile type="UnstructuredGrid" version="1.0" byte_order="LittleEndian" '
            'header_type="UInt64">\n<UnstructuredGrid>\n' + "\n".join(arrays) +
            '\n</UnstructuredGrid>\n<AppendedData encoding="raw">\n_')
    with open(path, "wb") as f:
        f.write(head.encode() + b"".join(blocks) + b"\n</AppendedData>\n</VTKFile>\n")


def write_run(root, frames):
    """frames: [(t, x, y, q)] -> root/iter_N/iter_N_1.vtu, N = 1, 2, ..."""
    for n, (t, x, y, q) in enumerate(frames, start=1):
        d = os.path.join(root, f"iter_{n}")
        os.makedirs(d, exist_ok=True)
        write_vtu(os.path.join(d, f"iter_{n}_1.vtu"), x, y, q, t=t)


def test_exact_matches_vtu_error_formula_at_t4():
    rng = np.random.default_rng(3)
    x = rng.uniform(-5, 5, 200)
    y = rng.uniform(0, 20, 200)
    lx, ly = 10.0, 20.0

    def g(u, c, period):
        return sum(np.exp(-(u - c - k * period) ** 2) for k in (-1, 0, 1))

    ref = g(x, 2.0, lx) * g(y, 12.0, ly)          # vtu_error.jl: centre (2, 12)
    assert np.max(np.abs(gaussian_exact(x, y, 4.0) - ref)) < 1e-15


def test_exact_wraps_periodically():
    x = np.array([-4.0, 0.0, 4.9])
    y = np.array([1.0, 10.0, 19.0])
    # c = (0.5, 1.0): after one full y-period (t = 20) the centre is x = 0 + 10 = 0 mod 10
    assert np.allclose(gaussian_exact(x, y, 20.0), gaussian_exact(x, y, 0.0), atol=1e-14)


def test_vtu_round_trip(tmp_path):
    x, y = nodes_from_boxes(boxes_2to1())
    q = np.sin(x) * np.cos(y)
    p = str(tmp_path / "a.vtu")
    write_vtu(p, x, y, q, t=1.5)
    v = read_vtu(p)
    assert np.array_equal(v["Points"][:, 0], x) and np.array_equal(v["Points"][:, 1], y)
    assert np.array_equal(v["q"], q)
    assert v["TimeValue"][0] == 1.5


def test_float32_field_refused(tmp_path):
    x, y = nodes_from_boxes(boxes_2to1())
    d = tmp_path / "run"
    (d / "iter_1").mkdir(parents=True)
    write_vtu(str(d / "iter_1" / "iter_1_1.vtu"), x, y, np.ones_like(x), t=0.0, qtype="<f4")
    with pytest.raises(ValueError, match="Float64"):
        load_frames(str(d))


def test_missing_time_refused_and_pvd_used(tmp_path):
    x, y = nodes_from_boxes(boxes_2to1())
    d = tmp_path / "run"
    (d / "iter_1").mkdir(parents=True)
    write_vtu(str(d / "iter_1" / "iter_1_1.vtu"), x, y, np.ones_like(x), t=None)
    with pytest.raises(ValueError, match="no TimeValue"):
        load_frames(str(d))
    (d / "simulation.pvd").write_text(
        '<VTKFile type="Collection"><Collection>'
        '<DataSet timestep="4.0" file="iter_1.pvtu"/></Collection></VTKFile>')
    assert load_frames(str(d))[0][1] == 4.0


def test_score_exact_field_reads_zero(tmp_path):
    x, y = nodes_from_boxes(boxes_2to1())
    q0 = gaussian_exact(x, y, 0.0)
    q4 = gaussian_exact(x, y, 4.0)
    write_run(str(tmp_path), [(0.0, x, y, q0), (4.0, x, y, q4)])
    r = score_run(str(tmp_path))
    assert r["nelem"] == 19 and r["levels"] == [15, 4] and r["nframes"] == 2
    assert r["l2"] == 0.0


def test_identical_frames_zero_mass_drift(tmp_path):
    x, y = nodes_from_boxes(boxes_2to1())
    q = gaussian_exact(x, y, 0.0)
    write_run(str(tmp_path), [(0.0, x, y, q), (4.0, x, y, q)])
    assert score_run(str(tmp_path))["mass_drift"] == 0.0
