"""Pins for tools/amr_overlay_2d.py -- the element-outline overlay of the
Phase-6 item-10 demonstration runs.

The named regression against real data (the item-7 static 2:1 dump, the 8e
corner preadapts, the 41 frames of 8d-i) needs the Jexpresso output tree and
is run by hand; these pins hold the same counts on hand-built meshes:

  1. the static 2:1 gate mesh (item 7): conf_int 24, W 8, parent 4, halves 8,
     N1 0, N2 0, walls 0, 76 sides; the parent's four sides are kind 1, each
     child's two outer sides kind 2 and the two inner ones kind 0
  2. the walled 4x4 corner_ll preadapt (item 8e): conf_int 26, parent 2,
     halves 4, walls 18, W 0
  3. the channel 4x4 corner_ll preadapt (walls y): conf_int 26, W 3, parent 3
     with N1 1, halves 6, walls 9 -- and the wrap mortar's parent side is the
     element across the wrap (cell (3, 0), x-max), its halves the children's
     x-min sides
  4. a wrap face refined on both sides (the 8d configuration): W 9, N1 0,
     N2 1 -- six interior mortars, two conforming child faces across the wrap
  5. level 2 against level 1 across the wrap (8d-ii): N1 5, N2 1, W 4, with
     a level-2 child one level finer than every parent it meets
  6. the classification does not depend on the element order
  7. the polydata and pvd round-trip through an XML parser with 4 points and
     4 line cells per element and the census recoverable from `kind`
  8. the driver over a two-frame run writes one vtp per frame, the pvd, the
     census table, and gates on the --expect flags
  9. a nodes dump (no field) classifies, and a still renders to a PNG
 10. a level jump of two across a face is refused (Mesh2's own check), and
     the wrap identity is asserted by classify (a contact list with a wrap
     face removed fails it)

Run with: pytest tests/dg_2d/test_amr_overlay_2d.py -v
"""

import os
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from backends.python_1d.dg.basis import lgl_gen
from tools.amr_overlay_2d import (CONFORMING, MORTAR_CHILD, MORTAR_PARENT, WALL, WRAP,
                                  classify, glued_refinement_vertices, main, write_pvd,
                                  write_vtp)
from tools.gate_2d_mortar import Mesh2

DOMAIN = ((-5.0, 5.0), (0.0, 20.0))
DX, DY = 2.5, 5.0


# The fixtures are built from boxes, like tests/dg_2d/test_bracket_2d.py --
# copied rather than imported (test files do not depend on each other).
def boxes_4x4():
    return [(-5 + ex * DX, -5 + (ex + 1) * DX, ey * DY, (ey + 1) * DY)
            for ey in range(4) for ex in range(4)]


def cell(ex, ey):
    return (-5 + ex * DX, -5 + (ex + 1) * DX, ey * DY, (ey + 1) * DY)


def children(box):
    """NW, SW, NE, SE -- the gate fixtures' child order."""
    xa, xb, ya, yb = box
    xm, ym = 0.5 * (xa + xb), 0.5 * (ya + yb)
    return [(xa, xm, ym, yb), (xa, xm, ya, ym), (xm, xb, ym, yb), (xm, xb, ya, ym)]


def refine(boxes, box):
    out = [b for b in boxes if b != box]
    assert len(out) == len(boxes) - 1, f"{box} not in the mesh"
    return out + children(box)


def nodes_from_boxes(boxes, nop=4):
    xgl, _ = lgl_gen(nop + 1)
    x, y = [], []
    for xa, xb, ya, yb in boxes:
        for j in range(nop + 1):
            for i in range(nop + 1):
                x.append(xa + (xgl[i] + 1) / 2 * (xb - xa))
                y.append(ya + (xgl[j] + 1) / 2 * (yb - ya))
    return np.array(x), np.array(y)


def mesh_of(boxes, walls=()):
    x, y = nodes_from_boxes(boxes)
    return Mesh2(x, y, 4, DOMAIN, 4, 4, walls=walls)


def write_vtu(path, x, y, q, t):
    """Minimal writer in Jexpresso's layout: raw appended data, UInt64 headers."""
    pts = np.column_stack([x, y, np.zeros_like(x)]).astype("<f8").ravel()
    specs = [("Points", "Float64", 3, pts),
             ("TimeValue", "Float64", 1, np.array([t], dtype="<f8")),
             ("q", "Float64", 1, np.asarray(q).astype("<f8"))]
    blocks, arrays, off = [], [], 0
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
    for n, (t, x, y, q) in enumerate(frames, start=1):
        d = os.path.join(root, f"iter_{n}")
        os.makedirs(d, exist_ok=True)
        write_vtu(os.path.join(d, f"iter_{n}_1.vtu"), x, y, q, t)


def field(x, y):
    return np.exp(-0.3 * (x - 0.4) ** 2 - 0.1 * (y - 11.0) ** 2)


# ---------------------------------------------------------------------------
# 1. the static 2:1 gate mesh
# ---------------------------------------------------------------------------

def test_item7_mesh_census_and_kinds():
    boxes = refine(boxes_4x4(), cell(2, 2))
    mesh = mesh_of(boxes)
    cls = classify(mesh)
    c = cls["census"]
    assert c == dict(nelem=19, levels=[15, 4], conf_int=24, W=8, parent=4, halves=8,
                     N1=0, N2=0, W0=8, walls=0, sides=76)
    kind = cls["kind"]
    # the parent-adjacent sides: cell (1,2) x-max, (3,2) x-min, (2,1) y-max, (2,3) y-min
    assert kind[boxes.index(cell(1, 2)), 1] == MORTAR_PARENT
    assert kind[boxes.index(cell(3, 2)), 0] == MORTAR_PARENT
    assert kind[boxes.index(cell(2, 1)), 3] == MORTAR_PARENT
    assert kind[boxes.index(cell(2, 3)), 2] == MORTAR_PARENT
    # each child: two outer sides mortar child, two inner sides conforming
    for ch in children(cell(2, 2)):
        row = kind[boxes.index(ch)]
        assert sorted(row.tolist()) == [CONFORMING, CONFORMING, MORTAR_CHILD, MORTAR_CHILD]
    assert np.all(cls["wrap"][kind == MORTAR_CHILD] == 0)
    # the wrap sides sit on the domain boundary only
    for e in range(mesh.nelem):
        for s in range(4):
            if kind[e, s] == WRAP:
                col = (0, 1, 2, 3)[s]
                bound = (-5.0, 5.0, 0.0, 20.0)[s]
                assert abs(mesh.box[e, col] - bound) < 1e-12


# ---------------------------------------------------------------------------
# 2-3. the 8e corner preadapts
# ---------------------------------------------------------------------------

def test_walled_corner_ll():
    boxes = refine(boxes_4x4(), cell(0, 0))
    c = classify(mesh_of(boxes, walls="xy"))["census"]
    assert c["conf_int"] == 26 and c["W"] == 0 and c["parent"] == 2 and c["halves"] == 4
    assert c["walls"] == 18 and c["N1"] == 0 and c["N2"] == 0 and c["W0"] == 0
    assert c["sides"] == 76 == 2 * 26 + 2 + 4 + 18


def test_channel_corner_ll_wrap_mortar():
    boxes = refine(boxes_4x4(), cell(0, 0))
    mesh = mesh_of(boxes, walls="y")
    cls = classify(mesh)
    c = cls["census"]
    assert c["conf_int"] == 26 and c["W"] == 3 and c["parent"] == 3 and c["halves"] == 6
    assert c["walls"] == 9 and c["N1"] == 1 and c["N2"] == 0 and c["W0"] == 4
    kind, wrap = cls["kind"], cls["wrap"]
    e30 = boxes.index(cell(3, 0))
    assert kind[e30, 1] == MORTAR_PARENT and wrap[e30, 1] == 1
    nw, sw, ne, se = (boxes.index(ch) for ch in children(cell(0, 0)))
    assert kind[nw, 0] == MORTAR_CHILD and wrap[nw, 0] == 1
    assert kind[sw, 0] == MORTAR_CHILD and wrap[sw, 0] == 1
    assert kind[sw, 2] == WALL and kind[se, 2] == WALL
    assert kind[ne, 1] == MORTAR_CHILD and wrap[ne, 1] == 0


# ---------------------------------------------------------------------------
# 4-5. refinement across the wrap
# ---------------------------------------------------------------------------

def test_wrap_face_refined_on_both_sides():
    boxes = refine(refine(boxes_4x4(), cell(0, 1)), cell(3, 1))
    mesh = mesh_of(boxes)
    assert glued_refinement_vertices(mesh) == 1
    c = classify(mesh)["census"]
    assert c["nelem"] == 22 and c["W"] == 9 and c["N1"] == 0 and c["N2"] == 1
    assert c["conf_int"] == 26 and c["parent"] == 6 and c["halves"] == 12
    assert c["sides"] == 88 == 2 * (26 + 9) + 6 + 12


def test_level2_against_level1_across_the_wrap():
    boxes = boxes_4x4()
    for cl in (cell(0, 0), cell(0, 1), cell(0, 2), cell(3, 1)):
        boxes = refine(boxes, cl)
    nw, sw, _, _ = children(cell(0, 1))
    boxes = refine(refine(boxes, nw), sw)
    mesh = mesh_of(boxes)
    cls = classify(mesh)
    c = cls["census"]
    assert c["nelem"] == 34 and c["levels"] == [12, 14, 8]
    assert c["N1"] == 5 and c["N2"] == 1 and c["W"] == 4 and c["W0"] == 8
    # the x-wrap at row 1: two mortars whose children are level 2, parents level 1
    for ct in mesh.contacts:
        if ct["kind"] == "mortar":
            la = mesh.level[ct["a"]]
            lb = mesh.level[ct["b"]]
            assert abs(la - lb) == 1
    lvl2 = cls["level"][:, None] == 2
    assert int(np.sum((cls["kind"] == MORTAR_CHILD) & lvl2)) == 12
    assert int(np.sum((cls["kind"] == MORTAR_CHILD) & (cls["wrap"] == 1) & lvl2)) == 4


def test_order_independent():
    boxes = refine(boxes_4x4(), cell(2, 2))
    a = classify(mesh_of(boxes))["census"]
    b = classify(mesh_of(boxes[::-1]))["census"]
    assert a == b


# ---------------------------------------------------------------------------
# 7-9. files and the driver
# ---------------------------------------------------------------------------

def test_vtp_and_pvd_round_trip(tmp_path):
    boxes = refine(boxes_4x4(), cell(2, 2))
    mesh = mesh_of(boxes)
    cls = classify(mesh)
    p = str(tmp_path / "f.vtp")
    write_vtp(p, mesh, cls)
    piece = ET.parse(p).getroot().find("PolyData").find("Piece")
    assert piece.get("NumberOfPoints") == "76" and piece.get("NumberOfLines") == "76"
    cd = {d.get("Name"): np.array(d.text.split(), dtype=int)
          for d in piece.find("CellData")}
    assert set(cd) == {"kind", "wrap", "level", "elem", "side"}
    assert int(np.sum(cd["kind"] == MORTAR_PARENT)) == 4
    assert int(np.sum(cd["kind"] == MORTAR_CHILD)) == 8
    assert int(np.sum(cd["kind"] == WRAP)) == 16        # 8 faces, two sides each
    assert int(np.sum(cd["kind"] == CONFORMING)) == 48
    assert cd["elem"].min() == 1 and cd["elem"].max() == 19
    assert sorted(set(cd["side"].tolist())) == [0, 1, 2, 3]
    conn = np.array(piece.find("Lines").find("DataArray").text.split(), dtype=int)
    assert len(conn) == 2 * 76 and conn.max() == 75
    write_pvd(str(tmp_path / "o.pvd"), [(0.0, "a.vtp"), (0.5, "b.vtp")])
    ds = ET.parse(str(tmp_path / "o.pvd")).getroot().find("Collection").findall("DataSet")
    assert [(d.get("timestep"), d.get("file")) for d in ds] == [("0", "a.vtp"), ("0.5", "b.vtp")]


def test_driver_over_a_run(tmp_path):
    b4 = boxes_4x4()
    b2 = refine(b4, cell(2, 2))
    x4, y4 = nodes_from_boxes(b4)
    x2, y2 = nodes_from_boxes(b2)
    run = tmp_path / "run"
    write_run(str(run), [(0.0, x4, y4, field(x4, y4)), (0.5, x2, y2, field(x2, y2))])
    out = tmp_path / "out"
    assert main(["--run", str(run), "--out", str(out)]) == 0
    assert sorted(os.listdir(out)) == ["census.txt", "overlay.pvd", "overlay_0001.vtp",
                                       "overlay_0002.vtp"]
    lines = open(out / "census.txt").read().splitlines()
    assert len(lines) == 3
    assert lines[1].split()[:3] == ["1", "0.000", "16"] and lines[2].split()[:3] == ["2", "0.500", "19"]
    # --expect flags gate on every frame: 4 parents holds on frame 2 only
    assert main(["--run", str(run), "--out", str(out), "--no-vtp",
                 "--expect-mortar-parent", "4"]) == 1
    assert main(["--run", str(run), "--out", str(out), "--no-vtp",
                 "--expect-conforming", "32", "--expect-walls", "0"]) == 0


def test_nodes_dump_and_still(tmp_path):
    boxes = refine(boxes_4x4(), cell(0, 0))
    x, y = nodes_from_boxes(boxes)
    dump = tmp_path / "GATE2D_nodes_l1.txt"
    np.savetxt(str(dump), np.column_stack([x, y]))
    out = tmp_path / "out"
    assert main(["--nodes", str(dump), "--out", str(out), "--walls", "xy",
                 "--expect-conforming", "26", "--expect-mortar-parent", "2",
                 "--expect-walls", "18", "--still", "0.0"]) == 0
    assert (out / "still_t000.000.png").stat().st_size > 1000
    assert main(["--nodes", str(dump), "--out", str(out), "--walls", "y", "--no-vtp",
                 "--expect-walls", "18"]) == 1               # the channel has 9


# ---------------------------------------------------------------------------
# 10. refusals
# ---------------------------------------------------------------------------

def test_level_jump_refused():
    boxes = refine(boxes_4x4(), cell(2, 2))
    nw = children(cell(2, 2))[0]
    boxes = refine(boxes, nw)                    # level 2 beside level 0: not 2:1
    with pytest.raises(AssertionError, match="not conforming and not 2:1"):
        mesh_of(boxes)


def test_wrap_identity_is_asserted():
    """Drop one wrap contact behind Mesh2's back: the two sides it covered are
    unclassified, and classify refuses before the identity is reached; re-mark
    them as walls instead and the identity itself fails."""
    mesh = mesh_of(refine(boxes_4x4(), cell(2, 2)))
    gone = [c for c in mesh.contacts if c["periodic"]][0]
    mesh.contacts.remove(gone)
    with pytest.raises(AssertionError, match="unclassified"):
        classify(mesh)
    hi, lo = (1, 0) if gone["axis"] == 0 else (3, 2)
    mesh.walls = [(gone["a"], hi), (gone["b"], lo)]
    with pytest.raises(AssertionError, match="wrap identity"):
        classify(mesh)
