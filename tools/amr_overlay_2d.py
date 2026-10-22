"""Phase-6 item 10: element-outline overlay for the DG-AMR demonstration
runs -- every element side of every frame classified from the geometry alone.

Jexpresso's VTU frames carry the DG lattice as 16 bilinear sub-cells per
element (nop^2; measured 2026-10-20), so what ParaView draws as "cell edges"
is the LGL lattice, not the mesh. This tool derives the mesh a frame was
computed on from the frame's own node coordinates -- the geometry path of
tools/bracket_2d.py and tools/gate_2d_mortar.py (Mesh2: the DG point order
ip = (iel-1)*ngl^2 + (j-1)*ngl + i, element boxes from their corner nodes,
levels from box size, contacts along each axis with the period folded in on
the periodic axes and the domain boundaries recorded as walls on the walled
ones) -- and classifies every element side by what the DG surface term does
there:

  kind 0  conforming     one contact of equal length on each side (interior)
  kind 1  mortar parent  the long side of a 2:1 contact: one parent face
                         receiving two child halves (the gather)
  kind 2  mortar child   the short side of a 2:1 contact: a child half
                         (the scatter); its element is one level finer
  kind 3  wrap           a conforming contact across the periodic wrap
  kind 4  wall           a side on a walled domain boundary (no contact)

plus a `wrap` flag (0/1) on every side, so a 2:1 face ACROSS the wrap keeps
its mortar kind and is still findable (a wrap mortar's parent side is the
coarse element across the wrap, D-079 / item 8b). Jexpresso element ids,
levels and face lists are never read.

Per frame it writes a VTK polydata of element outlines (one line cell per
element side, cell data `kind`, `wrap`, `level`, `elem`, `side`) and a `.pvd`
time series, prints a census table, and can render stills and an animation
with matplotlib (the field as the 16 bilinear sub-cells per element, the
outlines coloured by kind with line width by level, mortar child sides inset
so they read beside the parent's line).

The census per frame, and the identity the classification must satisfy:

  conf_int   conforming interior faces (kind 0; two sides per face)
  W          wrap-matched conforming faces (kind 3; two sides per face)
  parent     mortar parent faces (kind 1; one side per face)
  halves     mortar child halves (kind 2; one side each; = 2 * parent)
  N1         mortar parent faces across the wrap
  N2         glued refinement vertices on the wrap lines: a corner shared by
             an element on each side of the wrap that is not a base vertex
  walls      wall sides (kind 4)
  sides      4 * nelem = 2 * (conf_int + W) + parent + halves + walls

and the three-number wrap identity of item 8d in its geometric form,

  W = W0 - N1 + N2,   W0 = ney (x periodic) + nex (y periodic),

which holds on every 2:1-balanced mesh (per base wrap face: the merged
vertex set has S shared + M hanging vertices and 1 + S + M segments, each
a conforming contact or a mortar half, so C = 1 - M + S). The tool asserts
it on every frame; a frame that fails it is a classification defect and the
run stops.

Named regression (item 10): on the certified frames and dumps the census
reproduces the record --
  item 7 / 8b step 4, the static 2:1 mesh (19 elements):   conf_int 24, W 8,
      parent 4, halves 8, N1 0, N2 0, walls 0
  item 8e, walled 4x4 corner_ll (--walls xy):               conf_int 26, W 0,
      parent 2, halves 4, walls 18
  item 8e, channel 4x4 corner_ll (--walls y):               conf_int 26, W 3,
      parent 3 (N1 1), halves 6, walls 9
  item 8d-i, the 41 frames of the level-1 traversal:        the record's
      per-build (W, conforming, parent, N1) sequence, frame k <-> build k
      (frame 0 repeats build 1: a frame written at an adapt time shows the
      mesh BEFORE that adapt)
The hand-counted fixtures of tests/dg_2d/test_amr_overlay_2d.py pin the same
numbers without the Jexpresso tree.

Run -- a demonstration run (frames every 0.125; polydata + census + stills):

    cd ~/projects/drl-amr
    conda activate rl-amr
    python -m tools.amr_overlay_2d --run <abs run dir> --out <abs out dir> \\
        --nex 8 --ney 8 --still 10.5 --still 12.0 --mp4 <abs out dir>/run.mp4

Run -- a single probe dump (no field; polydata and census only):

    python -m tools.amr_overlay_2d --nodes <abs GATE2D_nodes_*.txt> --out <dir> \\
        --walls xy --expect-conforming 26 --expect-mortar-parent 2 \\
        --expect-walls 18

--walls x|y|xy declares the walled axes as in the sibling tools; it is a
declaration the geometry cannot check (a walled mesh tiles exactly as a
periodic one), so pass what the run was.
"""

import argparse
import glob
import os
import shutil
import sys

import numpy as np

from tools.analytic_operator_2d import GATE_DOMAIN, GATE_NEX, GATE_NEY, GATE_NOP, wall_axes
from tools.bracket_2d import load_frames
from tools.gate_2d_mortar import GEOM_TOL, SIDE_NAMES, Mesh2

CONFORMING, MORTAR_PARENT, MORTAR_CHILD, WRAP, WALL = range(5)
KIND_NAMES = ("conforming", "mortar parent", "mortar child", "wrap", "wall")

# The data plates' palette (knowledge/explanations/figures, "textbook plate").
GROUND = "#FAF9F5"
INK, INK2, HAIR = "#1F1E1B", "#5E5B54", "#B9B4A8"
KIND_COLOR = {CONFORMING: INK, MORTAR_PARENT: "#1F5A96", MORTAR_CHILD: "#B8501A",
              WRAP: "#1E7A4E", WALL: "#8B1E2B"}
PARENT_LIGHT = "#6FA8DC"    # mortar parent under white ink: #1F5A96 sinks into the viridis floor


def kind_color(k, ink):
    """The drawn colour of a side of kind k when the conforming sides are drawn in ink."""
    if k == CONFORMING:
        return ink
    if k == MORTAR_PARENT and ink == "white":
        return PARENT_LIGHT
    return KIND_COLOR[k]


# ---------------------------------------------------------------------------
# classification
# ---------------------------------------------------------------------------

def _side_pair(axis):
    """(high side of a, low side of b) for a contact along axis."""
    return (1, 0) if axis == 0 else (3, 2)


def _tangential(axis):
    """(t0, t1): box columns of the tangential extent for a contact along axis."""
    return (2, 3) if axis == 0 else (0, 1)


def glued_refinement_vertices(mesh):
    """N2: corners on a wrap line shared by an element on each side of the
    wrap, excluding the base vertices (multiples of the base spacing)."""
    n2 = 0
    tol = mesh.tol
    for axis in (0, 1):
        if axis in mesh.wall_axes:
            continue
        lo_col, hi_col = (0, 1) if axis == 0 else (2, 3)
        t0, t1 = _tangential(axis)
        lo_bound = mesh.x0 if axis == 0 else mesh.y0
        hi_bound = mesh.x1 if axis == 0 else mesh.y1
        base = (mesh.Ly / mesh.ney) if axis == 0 else (mesh.Lx / mesh.nex)
        origin = mesh.y0 if axis == 0 else mesh.x0
        side_a = np.abs(mesh.box[:, hi_col] - hi_bound) <= tol
        side_b = np.abs(mesh.box[:, lo_col] - lo_bound) <= tol
        va = np.concatenate([mesh.box[side_a, t0], mesh.box[side_a, t1]])
        vb = np.concatenate([mesh.box[side_b, t0], mesh.box[side_b, t1]])
        shared = []
        for v in np.unique(np.round(va, 8)):
            if np.any(np.abs(vb - v) <= tol):
                shared.append(v)
        for v in shared:
            k = (v - origin) / base
            if abs(k - round(k)) > 1e-6:
                n2 += 1
    return n2


def classify(mesh):
    """Classify every element side of a gate_2d_mortar.Mesh2.

    Returns dict(kind, wrap, level, census) with kind and wrap (nelem, 4)
    int arrays over sides 0 x-min, 1 x-max, 2 y-min, 3 y-max, level the
    element levels, and the census of the module docstring. Raises
    AssertionError when a side is claimed inconsistently, a mortar's child is
    not one level finer than its parent, or the wrap identity fails."""
    nelem = mesh.nelem
    kind = -np.ones((nelem, 4), dtype=int)
    wrap = np.zeros((nelem, 4), dtype=int)

    def claim(e, s, k, w):
        if kind[e, s] == -1:
            kind[e, s], wrap[e, s] = k, w
        elif not (k == MORTAR_PARENT and kind[e, s] == k and wrap[e, s] == w):
            raise AssertionError(
                f"element {e + 1} side {SIDE_NAMES[s]} claimed as {KIND_NAMES[k]} "
                f"(wrap {w}) but already {KIND_NAMES[kind[e, s]]} (wrap {wrap[e, s]})")

    for c in mesh.contacts:
        a, b = c["a"], c["b"]
        hi, lo = _side_pair(c["axis"])
        per = int(c["periodic"])
        if c["kind"] == "conforming":
            k = WRAP if per else CONFORMING
            claim(a, hi, k, per)
            claim(b, lo, k, per)
            continue
        t0, t1 = _tangential(c["axis"])
        la = mesh.box[a, t1] - mesh.box[a, t0]
        lb = mesh.box[b, t1] - mesh.box[b, t0]
        parent_is_a = la > lb
        P, C = (a, b) if parent_is_a else (b, a)
        if mesh.level[C] != mesh.level[P] + 1:
            raise AssertionError(
                f"mortar contact between elements {a + 1} (level {mesh.level[a]}) and "
                f"{b + 1} (level {mesh.level[b]}): the child is not one level finer")
        claim(a, hi, MORTAR_PARENT if parent_is_a else MORTAR_CHILD, per)
        claim(b, lo, MORTAR_CHILD if parent_is_a else MORTAR_PARENT, per)
    for e, s in mesh.walls:
        claim(e, s, WALL, 0)
    if np.any(kind < 0):
        e, s = np.argwhere(kind < 0)[0]
        raise AssertionError(f"element {e + 1} side {SIDE_NAMES[s]} is unclassified")

    conf_int = sum(1 for c in mesh.contacts if c["kind"] == "conforming" and not c["periodic"])
    w_faces = sum(1 for c in mesh.contacts if c["kind"] == "conforming" and c["periodic"])
    parent = int(np.sum(kind == MORTAR_PARENT))
    halves = int(np.sum(kind == MORTAR_CHILD))
    n1 = int(np.sum((kind == MORTAR_PARENT) & (wrap == 1)))
    walls = int(np.sum(kind == WALL))
    n2 = glued_refinement_vertices(mesh)
    w0 = (mesh.ney if 0 not in mesh.wall_axes else 0) + (mesh.nex if 1 not in mesh.wall_axes else 0)
    sides = 4 * nelem
    if sides != 2 * (conf_int + w_faces) + parent + halves + walls:
        raise AssertionError(
            f"side census: {sides} sides != 2*({conf_int} + {w_faces}) + {parent} + "
            f"{halves} + {walls}")
    if halves != 2 * parent:
        raise AssertionError(f"{halves} mortar halves for {parent} parent faces")
    if w_faces != w0 - n1 + n2:
        raise AssertionError(
            f"wrap identity: W = {w_faces} but W0 - N1 + N2 = {w0} - {n1} + {n2} = "
            f"{w0 - n1 + n2}")
    census = dict(nelem=nelem, levels=np.bincount(mesh.level).tolist(),
                  conf_int=conf_int, W=w_faces, parent=parent, halves=halves,
                  N1=n1, N2=n2, W0=w0, walls=walls, sides=sides)
    return dict(kind=kind, wrap=wrap, level=mesh.level.copy(), census=census)


# ---------------------------------------------------------------------------
# geometry of the outlines
# ---------------------------------------------------------------------------

# corners 0 SW, 1 SE, 2 NE, 3 NW; side s = (corner p, corner q), p -> q along +t
SIDE_CORNERS = ((3, 0), (1, 2), (0, 1), (3, 2))


def element_corners(box):
    xa, xb, ya, yb = box
    return ((xa, ya), (xb, ya), (xb, yb), (xa, yb))


def side_segment(mesh, e, s, inset=0.0):
    """((x0, y0), (x1, y1)) of side s of element e, moved inward by
    inset * min(width, height) of the element when inset > 0."""
    xa, xb, ya, yb = mesh.box[e]
    corners = element_corners(mesh.box[e])
    p, q = SIDE_CORNERS[s]
    (x0, y0), (x1, y1) = corners[p], corners[q]
    if inset > 0.0:
        d = inset * min(xb - xa, yb - ya)
        dx, dy = ((d, 0.0), (-d, 0.0), (0.0, d), (0.0, -d))[s]   # inward normal per side
        x0, x1, y0, y1 = x0 + dx, x1 + dx, y0 + dy, y1 + dy
    return (x0, y0), (x1, y1)


# ---------------------------------------------------------------------------
# VTK output
# ---------------------------------------------------------------------------

def write_vtp(path, mesh, cls):
    """ASCII VTK XML polydata: 4 points and 4 line cells per element (one per
    side), cell data kind, wrap, level, elem (1-based, the DG element index),
    side (0..3 as SIDE_NAMES)."""
    nelem = mesh.nelem
    pts, conn, cell = [], [], {k: [] for k in ("kind", "wrap", "level", "elem", "side")}
    for e in range(nelem):
        corners = element_corners(mesh.box[e])
        pts.extend(corners)
        for s, (p, q) in enumerate(SIDE_CORNERS):
            conn.append((4 * e + p, 4 * e + q))
            cell["kind"].append(int(cls["kind"][e, s]))
            cell["wrap"].append(int(cls["wrap"][e, s]))
            cell["level"].append(int(cls["level"][e]))
            cell["elem"].append(e + 1)
            cell["side"].append(s)
    nl = len(conn)
    lines = ['<?xml version="1.0"?>',
             '<VTKFile type="PolyData" version="1.0" byte_order="LittleEndian">',
             '<PolyData>',
             f'<Piece NumberOfPoints="{len(pts)}" NumberOfVerts="0" NumberOfLines="{nl}" '
             'NumberOfStrips="0" NumberOfPolys="0">',
             '<Points>',
             '<DataArray type="Float64" NumberOfComponents="3" format="ascii">',
             "\n".join(f"{x:.15g} {y:.15g} 0" for x, y in pts),
             '</DataArray>', '</Points>', '<Lines>',
             '<DataArray type="Int64" Name="connectivity" format="ascii">',
             "\n".join(f"{p} {q}" for p, q in conn),
             '</DataArray>',
             '<DataArray type="Int64" Name="offsets" format="ascii">',
             "\n".join(str(2 * (k + 1)) for k in range(nl)),
             '</DataArray>', '</Lines>', '<CellData>']
    for name, vals in cell.items():
        lines.append(f'<DataArray type="Int32" Name="{name}" format="ascii">')
        lines.append("\n".join(str(v) for v in vals))
        lines.append('</DataArray>')
    lines += ['</CellData>', '</Piece>', '</PolyData>', '</VTKFile>', '']
    with open(path, "w") as f:
        f.write("\n".join(lines))


def write_pvd(path, entries):
    """entries: [(t, file name relative to the pvd)]."""
    lines = ['<?xml version="1.0"?>',
             '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">',
             '<Collection>']
    for t, fn in entries:
        lines.append(f'<DataSet timestep="{t:.15g}" group="" part="0" file="{fn}"/>')
    lines += ['</Collection>', '</VTKFile>', '']
    with open(path, "w") as f:
        f.write("\n".join(lines))


# ---------------------------------------------------------------------------
# census table
# ---------------------------------------------------------------------------

CENSUS_HEADER = (f"{'iter':>5s} {'t':>8s} {'nelem':>5s} {'levels':>14s} {'conf_int':>8s} "
                 f"{'W':>3s} {'parent':>6s} {'halves':>6s} {'N1':>3s} {'N2':>3s} "
                 f"{'walls':>5s} {'sides':>5s} {'W0-N1+N2':>8s}")


def census_line(n, t, c):
    return (f"{n:5d} {t:8.3f} {c['nelem']:5d} {str(c['levels']):>14s} {c['conf_int']:8d} "
            f"{c['W']:3d} {c['parent']:6d} {c['halves']:6d} {c['N1']:3d} {c['N2']:3d} "
            f"{c['walls']:5d} {c['sides']:5d} {c['W0'] - c['N1'] + c['N2']:8d}")


# ---------------------------------------------------------------------------
# matplotlib render
# ---------------------------------------------------------------------------

def render_frame(ax, mesh, cls, x=None, y=None, q=None, inset=0.05, lw=0.3,
                 ink="white", vmin=0.0, vmax=1.0, cmap="viridis", title=None):
    """Draw one frame on ax: the field (when given) as each element's 5x5
    nodal values on its 16 bilinear sub-cells, then every side as a line
    coloured by kind, width growing with level, mortar child sides inset."""
    from matplotlib.collections import LineCollection
    ngl = mesh.ngl
    if q is not None:
        for e in range(mesh.nelem):
            idx = np.array([mesh.dof(e, i, j) for j in range(ngl) for i in range(ngl)])
            X = x[idx].reshape(ngl, ngl)
            Y = y[idx].reshape(ngl, ngl)
            Q = q[idx].reshape(ngl, ngl)
            ax.pcolormesh(X, Y, Q, shading="gouraud", vmin=vmin, vmax=vmax, cmap=cmap,
                          rasterized=True)
    segs, colors, widths = [], [], []
    for k in (CONFORMING, WRAP, WALL, MORTAR_PARENT, MORTAR_CHILD):
        for e in range(mesh.nelem):
            for s in range(4):
                if cls["kind"][e, s] != k:
                    continue
                segs.append(side_segment(mesh, e, s, inset if k == MORTAR_CHILD else 0.0))
                colors.append(kind_color(k, ink))
                widths.append(lw * (1.0 + 0.4 * cls["level"][e]) * (1.4 if k != CONFORMING else 1.0))
    ax.add_collection(LineCollection(segs, colors=colors, linewidths=widths, zorder=5))
    pad = 0.01 * max(mesh.Lx, mesh.Ly)     # the boundary sides are drawn on the edge
    ax.set_xlim(mesh.x0 - pad, mesh.x1 + pad)
    ax.set_ylim(mesh.y0 - pad, mesh.y1 + pad)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    if title:
        ax.set_title(title, fontsize=8, family="monospace", color=INK2, loc="left")


def _figure(mesh):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    h = 7.0
    w = h * mesh.Lx / mesh.Ly + 1.2
    fig, ax = plt.subplots(figsize=(w, h), facecolor=GROUND)
    ax.set_facecolor(GROUND)
    return fig, ax


def _legend(ax, present, ink="white"):
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color=(INK if k == CONFORMING else kind_color(k, ink)),
                      lw=1.6, label=KIND_NAMES[k])
               for k in sorted(present)]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False,
              fontsize=7, labelcolor=INK2, handlelength=1.6)


def frame_title(t, c):
    return (f"t = {t:6.3f}   elements {c['nelem']:3d}  levels {c['levels']}   "
            f"W {c['W']}  N1 {c['N1']}  N2 {c['N2']}  walls {c['walls']}")


def render_still(path, mesh, cls, t, x=None, y=None, q=None, **kw):
    fig, ax = _figure(mesh)
    render_frame(ax, mesh, cls, x, y, q, title=frame_title(t, cls["census"]), **kw)
    _legend(ax, set(np.unique(cls["kind"]).tolist()), kw.get("ink", "white"))
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=GROUND)
    import matplotlib.pyplot as plt
    plt.close(fig)


def render_mp4(path, frames, meshes, classes, fps=8, **kw):
    """frames: [(iter, t, x, y, q)]; meshes/classes aligned. Needs ffmpeg."""
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg is not on PATH (brew install ffmpeg); write PNG frames "
                         "with --png-dir instead")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter
    fig, ax = _figure(meshes[0])
    writer = FFMpegWriter(fps=fps, metadata=dict(artist="amr_overlay_2d"))
    present = set()
    for cls in classes:
        present |= set(np.unique(cls["kind"]).tolist())
    with writer.saving(fig, path, dpi=150):
        for (n, t, x, y, q), mesh, cls in zip(frames, meshes, classes, strict=False):
            ax.clear()
            ax.set_facecolor(GROUND)
            render_frame(ax, mesh, cls, x, y, q, title=frame_title(t, cls["census"]), **kw)
            _legend(ax, present, kw.get("ink", "white"))
            writer.grab_frame(facecolor=GROUND)
    plt.close(fig)


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def meshes_for(frames, nop, domain, nex, ney, walls):
    """One Mesh2 per frame, rebuilt only when the node coordinates change."""
    meshes, prev = [], None
    for n, t, x, y, q in frames:
        if prev is None or not (np.array_equal(x, prev[0]) and np.array_equal(y, prev[1])):
            mesh = Mesh2(x, y, nop, domain, nex, ney, walls=walls)
            prev = (x, y)
        meshes.append(mesh)
    return meshes


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--run", help="a Jexpresso run dir (iter_N/iter_N_1.vtu frames)")
    src.add_argument("--nodes", help="a GATE2D_nodes_*.txt dump (one frame, no field)")
    ap.add_argument("--out", required=True, help="output dir for the polydata, pvd, census, stills")
    ap.add_argument("--nex", type=int, default=GATE_NEX, help="base-grid elements in x")
    ap.add_argument("--ney", type=int, default=GATE_NEY, help="base-grid elements in y")
    ap.add_argument("--nop", type=int, default=GATE_NOP)
    ap.add_argument("--xmin", type=float, default=GATE_DOMAIN[0][0])
    ap.add_argument("--xmax", type=float, default=GATE_DOMAIN[0][1])
    ap.add_argument("--ymin", type=float, default=GATE_DOMAIN[1][0])
    ap.add_argument("--ymax", type=float, default=GATE_DOMAIN[1][1])
    ap.add_argument("--walls", default="none", choices=("none", "x", "y", "xy"),
                    help="walled axes, as the run was (x = left/right walls, y = bottom/top)")
    ap.add_argument("--no-vtp", action="store_true", help="skip the polydata and pvd")
    ap.add_argument("--still", type=float, action="append", default=[],
                    help="render a PNG of the frame nearest this time; repeat")
    ap.add_argument("--png-dir", default=None, help="render every frame as a PNG here")
    ap.add_argument("--mp4", default=None, help="render the frames to this MP4 (needs ffmpeg)")
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--inset", type=float, default=0.05,
                    help="mortar child sides are drawn this fraction of the child's size "
                         "inside the element (0: on the face)")
    ap.add_argument("--lw", type=float, default=0.3,
                    help="base line width in points for a level-0 conforming side "
                         "(each level adds 40%%; non-conforming kinds 1.4x)")
    ap.add_argument("--ink", default="white", choices=("dark", "white"),
                    help="colour of the conforming sides: white (the default, for lines "
                         "over the field) or the plate's dark ink")
    ap.add_argument("--vmin", type=float, default=0.0)
    ap.add_argument("--vmax", type=float, default=1.0)
    ap.add_argument("--cmap", default="viridis")
    ap.add_argument("--expect-conforming", type=int, default=None,
                    help="every frame must have this many conforming faces (interior + wrap)")
    ap.add_argument("--expect-mortar-parent", type=int, default=None)
    ap.add_argument("--expect-wrap-mortar-parent", type=int, default=None, help="N1")
    ap.add_argument("--expect-walls", type=int, default=None)
    args = ap.parse_args(argv)

    domain = ((args.xmin, args.xmax), (args.ymin, args.ymax))
    walls = () if args.walls == "none" else args.walls
    os.makedirs(args.out, exist_ok=True)

    if args.run:
        frames = load_frames(args.run)
        source = args.run
    else:
        xy = np.loadtxt(args.nodes)
        frames = [(0, 0.0, xy[:, 0].copy(), xy[:, 1].copy(), None)]
        source = args.nodes
    meshes = meshes_for(frames, args.nop, domain, args.nex, args.ney, walls)

    print("=" * 100)
    print(f"AMR OVERLAY 2D: nop={args.nop} base {args.nex}x{args.ney} domain={domain} "
          f"walls={args.walls} frames={len(frames)}")
    print(f"  {source}")
    print("=" * 100)
    print(CENSUS_HEADER)
    classes, entries, failures, table = [], [], [], [CENSUS_HEADER]
    for (n, t, x, y, q), mesh in zip(frames, meshes, strict=False):
        cls = classify(mesh)
        classes.append(cls)
        c = cls["census"]
        line = census_line(n, t, c)
        print(line)
        table.append(line)
        for name, want, got in (("conforming", args.expect_conforming, c["conf_int"] + c["W"]),
                                ("mortar parent", args.expect_mortar_parent, c["parent"]),
                                ("wrap mortar parent", args.expect_wrap_mortar_parent, c["N1"]),
                                ("walls", args.expect_walls, c["walls"])):
            if want is not None and got != want:
                failures.append(f"iter {n}: {name} {got} != {want}")
        if not args.no_vtp:
            fn = f"overlay_{n:04d}.vtp"
            write_vtp(os.path.join(args.out, fn), mesh, cls)
            entries.append((t, fn))
    if entries:
        write_pvd(os.path.join(args.out, "overlay.pvd"), entries)
    with open(os.path.join(args.out, "census.txt"), "w") as f:
        f.write("\n".join(table) + "\n")

    kw = dict(inset=args.inset, lw=args.lw, ink=("white" if args.ink == "white" else INK),
              vmin=args.vmin, vmax=args.vmax, cmap=args.cmap)
    for ts in args.still:
        k = int(np.argmin([abs(fr[1] - ts) for fr in frames]))
        n, t, x, y, q = frames[k]
        p = os.path.join(args.out, f"still_t{t:07.3f}.png")
        render_still(p, meshes[k], classes[k], t, x, y, q, **kw)
        print(f"  still: {p} (iter {n}, t = {t})")
    if args.png_dir:
        os.makedirs(args.png_dir, exist_ok=True)
        for (n, t, x, y, q), mesh, cls in zip(frames, meshes, classes, strict=False):
            render_still(os.path.join(args.png_dir, f"frame_{n:04d}.png"), mesh, cls, t,
                         x, y, q, **kw)
        print(f"  {len(frames)} PNG frames in {args.png_dir}")
    if args.mp4:
        render_mp4(args.mp4, frames, meshes, classes, fps=args.fps, **kw)
        print(f"  animation: {args.mp4} ({len(frames)} frames at {args.fps} fps)")
    print("=" * 100)
    if failures:
        print("VERDICT: FAIL (" + "; ".join(failures) + ")")
    else:
        print(f"VERDICT: PASS -- every side classified once on {len(frames)} frame(s); "
              f"the wrap identity holds on all")
    print("=" * 100)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
