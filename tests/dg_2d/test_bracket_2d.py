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
  4. frame times must increase with the frame number (a restarted output
     counter would pair a field with the wrong time and mesh)
  5. the adaptive mode (item 8a): static mode refuses a run whose mesh
     changes; adaptive mode scores each frame on its own mesh, so a refine
     carried out by exact interpolation conserves mass to round-off; a static
     run scores identically in both modes; a renumbering of the same elements
     is not a mesh change
  6. THE BLIND SPOT, pinned on purpose: a refine that puts the children's
     data in the wrong quadrants also conserves mass to round-off (the four
     quadrants have equal area) while the field is wrong. Child ordering is
     the Q1b-coarsen probe's job; this pin exists so that probe is never
     dropped as redundant with the mass check.
  7. --no-l2-gate: L2 not decreasing fails the verdict by default and does not
     with the flag; mass still gates

Run with: pytest tests/dg_2d/test_bracket_2d.py -v
"""

import os

import numpy as np
import pytest

from backends.python_1d.dg.basis import Lagrange_basis, lgl_gen
from tools.bracket_2d import gaussian_exact, load_frames, main, read_vtu, score_run


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


# ---------------------------------------------------------------------------
# frame order, and the adaptive mode (item 8a)
# ---------------------------------------------------------------------------

def boxes_4x4():
    dx, dy = 2.5, 5.0
    return [(-5 + ex * dx, -5 + (ex + 1) * dx, ey * dy, (ey + 1) * dy)
            for ey in range(4) for ex in range(4)]


REFINED = (0.0, 2.5, 10.0, 15.0)           # cell (2, 2) of the 4x4, refined in boxes_2to1


def field_on(boxes, nop=4):
    """A smooth, non-symmetric field at the nodes of boxes -- no quadrant
    symmetry inside REFINED, so a child-order error changes it."""
    x, y = nodes_from_boxes(boxes, nop)
    return x, y, np.exp(-0.3 * (x - 0.4) ** 2 - 0.1 * (y - 11.0) ** 2) + 0.05 * x * y


def refine_by_interpolation(q_parent, nop=4, swap=False):
    """The four children of REFINED from the parent's nodal values by exact
    tensor Lagrange interpolation, in boxes_2to1's child order NW, SW, NE, SE.
    swap=True writes each child's data into the diagonally opposite quadrant
    (NW<->SE, SW<->NE) -- the child-order defect."""
    ngl = nop + 1
    xgl, _ = lgl_gen(ngl)
    Q = q_parent.reshape(ngl, ngl)                    # [j, i]: j along y, i along x
    # (sx, sy): -1 = lower half of the parent, +1 = upper half
    quads = {"NW": (-1, +1), "SW": (-1, -1), "NE": (+1, +1), "SE": (+1, -1)}
    order = ["NW", "SW", "NE", "SE"]
    source = {"NW": "SE", "SW": "NE", "NE": "SW", "SE": "NW"} if swap else {k: k for k in order}
    out = []
    for name in order:
        sx, sy = quads[source[name]]
        px, _ = Lagrange_basis(ngl, ngl, xgl, (xgl + sx) / 2.0)   # px[a, i] = l_a(xi_i)
        py, _ = Lagrange_basis(ngl, ngl, xgl, (xgl + sy) / 2.0)
        out.append((py.T @ Q @ px).ravel())               # [j, i] -> ip order j*ngl + i
    return np.concatenate(out)


def conforming_and_refined_frames(swap=False, nop=4):
    """Frame 1 on the 4x4; frame 2 on the static 2:1 mesh with the same field,
    the refined cell's children made by interpolation from its parent."""
    ngl2 = (nop + 1) ** 2
    b4 = boxes_4x4()
    x4, y4, q4 = field_on(b4, nop)
    ip = b4.index(REFINED)
    q_parent = q4[ip * ngl2:(ip + 1) * ngl2]
    q_keep = np.concatenate([q4[e * ngl2:(e + 1) * ngl2] for e in range(16) if e != ip])
    x2, y2 = nodes_from_boxes(boxes_2to1(), nop)
    q2 = np.concatenate([q_keep, refine_by_interpolation(q_parent, nop, swap)])
    return (x4, y4, q4), (x2, y2, q2)


def test_frame_times_must_increase(tmp_path):
    x, y = nodes_from_boxes(boxes_2to1())
    q = gaussian_exact(x, y, 0.0)
    write_run(str(tmp_path), [(0.0, x, y, q), (1.0, x, y, q), (0.5, x, y, q)])
    with pytest.raises(ValueError, match="do not increase"):
        load_frames(str(tmp_path))


def test_static_mode_refuses_a_mesh_change(tmp_path):
    (x4, y4, q4), (x2, y2, q2) = conforming_and_refined_frames()
    write_run(str(tmp_path), [(0.0, x4, y4, q4), (0.5, x2, y2, q2)])
    with pytest.raises(ValueError, match="mesh changes"):
        score_run(str(tmp_path))


def test_adaptive_refine_by_interpolation_conserves_mass(tmp_path):
    (x4, y4, q4), (x2, y2, q2) = conforming_and_refined_frames()
    write_run(str(tmp_path), [(0.0, x4, y4, q4), (0.5, x2, y2, q2), (1.0, x2, y2, q2)])
    r = score_run(str(tmp_path), adaptive=True)
    assert [f["nelem"] for f in r["frames"]] == [16, 19, 19]
    assert [f["levels"] for f in r["frames"]] == [[16], [15, 4], [15, 4]]
    assert [f["changed"] for f in r["frames"]] == [False, True, False]
    assert r["n_meshes"] == 2 and r["nelem_min"] == 16 and r["nelem_max"] == 19
    assert r["mass_drift"] < 1e-14


def test_adaptive_is_blind_to_child_order(tmp_path):
    """The blind spot, pinned: children in the wrong quadrants conserve mass
    to round-off while the field is wrong."""
    (x4, y4, q4), (x2, y2, q2) = conforming_and_refined_frames(swap=False)
    _, (_, _, q2s) = conforming_and_refined_frames(swap=True)
    assert np.max(np.abs(q2s - q2)) > 1e-2                 # the field really differs
    write_run(str(tmp_path), [(0.0, x4, y4, q4), (0.5, x2, y2, q2s)])
    assert score_run(str(tmp_path), adaptive=True)["mass_drift"] < 1e-14


def test_static_run_scores_identically_in_both_modes(tmp_path):
    x, y = nodes_from_boxes(boxes_2to1())
    write_run(str(tmp_path), [(0.0, x, y, gaussian_exact(x, y, 0.0)),
                              (4.0, x, y, 0.9 * gaussian_exact(x, y, 4.0))])
    a = score_run(str(tmp_path))
    b = score_run(str(tmp_path), adaptive=True)
    for k in ("l2", "maxabs", "mass0", "mass_drift", "nelem", "levels", "nframes"):
        assert a[k] == b[k], k
    assert b["n_meshes"] == 1


def test_renumbered_elements_are_not_a_mesh_change(tmp_path):
    b = boxes_2to1()
    x, y = nodes_from_boxes(b)
    q = gaussian_exact(x, y, 0.0)
    perm = list(range(len(b)))[::-1]
    xr, yr = nodes_from_boxes([b[e] for e in perm])
    ngl2 = 25
    qr = np.concatenate([q[e * ngl2:(e + 1) * ngl2] for e in perm])
    write_run(str(tmp_path), [(0.0, x, y, q), (0.5, xr, yr, qr)])
    r = score_run(str(tmp_path))                           # static mode accepts it
    assert r["n_meshes"] == 1 and r["mass_drift"] < 1e-14


def test_no_l2_gate(tmp_path):
    x, y = nodes_from_boxes(boxes_2to1())
    q0, q4 = gaussian_exact(x, y, 0.0), gaussian_exact(x, y, 4.0)
    write_run(str(tmp_path / "a"), [(0.0, x, y, q4), (4.0, x, y, q4)])         # L2 0, drift 0
    write_run(str(tmp_path / "b"), [(0.0, x, y, q0), (4.0, x, y, q0)])         # L2 > 0, drift 0
    write_run(str(tmp_path / "c"), [(0.0, x, y, q0), (4.0, x, y, 1.001 * q0)])  # drift 1e-3
    ab = ["--run", f"a={tmp_path / 'a'}", "--run", f"b={tmp_path / 'b'}"]
    ac = ["--run", f"a={tmp_path / 'a'}", "--run", f"c={tmp_path / 'c'}"]
    assert main(ab) == 1                                   # L2 not decreasing: gated
    assert main(ab + ["--no-l2-gate"]) == 0                # reported, not gated
    assert main(ac + ["--no-l2-gate"]) == 1                # mass still gates
