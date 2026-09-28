"""Phase-6 gate, part 1: structural verification of the extracted 2D DG
operator on a static 2:1 (non-conforming) mesh.

Sibling of tools/gate_2d_compare.py (the conforming entrywise gate, which
stays the conforming rebase instrument). On a 2:1 mesh there is no analytic
reference for the whole operator, so this tool asks questions whose answers
are known in advance and reads everything it needs about the mesh from the
probe's node dump -- Jexpresso element ids, levels, and face lists are never
trusted, only cross-checked.

Consumes the files written by the GATE2D probe (a temporary, never-committed
instrumentation of Jexpresso's problems/drivers.jl -- see
JEXPRESSO_ENVIRONMENT.md):

    GATE2D_operator_<tag>.txt   n x n; column c = RHS of the unit vector at DOF c
    GATE2D_nodes_<tag>.txt      n lines "x y" (line ip = DOF ip)
    GATE2D_mass_<tag>.txt       n lines, the diagonal DG mass params.M (optional)
    GATE2D_adlvl_<tag>.txt      nelem lines, mesh.ad_lvl (optional)

DOF numbering assumed (and verified through the geometry check): the DG
lattice ip = (iel-1)*ngl^2 + (j-1)*ngl + i, i along x, j along y.

Checks:
  [1] counts            n = nelem*ngl^2, dump lengths agree
  [2] geometry          element boxes from their nodes; every node on its box's
                        LGL lattice; boxes tile the domain; levels from size,
                        equal to the dumped ad_lvl; face topology incl. the
                        periodic wrap; every element side covered exactly once
  [3] mass              dumped M vs w_i*w_j*Jvol(e) from geometry
  [4] far rows          rows of level-0 elements with no mortar side vs the
                        analytic conforming operator, columns mapped by geometry
  [5] support           no entry outside the predicted upwind pattern; every
                        mortar half's forward block nonzero, reverse block zero
  [6] exactness         L u = -c.grad u for global polynomials of degree <= nop
                        per variable, on rows whose inflow faces are all
                        non-periodic; degree-(nop+1) controls must fail
  [7] conservation      w^T L with w = M
  [8] consistency       L 1
  [9] spectrum          max Re(lambda) <= 0, one kernel mode at 1e-10*rho
 [10] spec operator     INFORMATIONAL, does not gate: entrywise vs the operator
                        assembled from the section-5.17 mortar spec
                        (tools/analytic_operator_2d_mortar.py) on the mesh
                        derived here. Agreement is two-way certification only
                        alongside [5]-[9], which do not depend on the spec's
                        details; a misreading of the spec shared by Jexpresso
                        and that construction would agree.
Verdict PASS/FAIL (checks [1]-[9]); exit code 0/1.

Why [5] and [6] exist (explanation doc, Phase 6 section 7): conservation is
blind to any mortar defect that uses the same wrong flux on both sides (a
wrong half, a wrong interp); polynomial exactness sees it. Upwinding makes
each mortar face one-directional, so the reverse coupling blocks are
predictably zero.

Run -- the item-7 static 2:1 case (4x4 gate mesh, one element refined):

    cd ~/projects/drl-amr
    conda activate rl-amr
    python -m tools.gate_2d_mortar --dir <abs path to the probe's output dir> \
        --expect-nelem 19 --expect-mortar-halves 8

Leg 2 adds --cx 0.7 --cy 0.3. On the conforming 4x4 gate dumps (the named
regression of this instrument) drop the --expect flags; with no mass/adlvl
files the geometric mass is used and the ad_lvl cross-check is skipped.
"""

import argparse
import glob
import os
import sys

import numpy as np

from backends.python_1d.dg.basis import lgl_gen
from tools.analytic_operator_2d import (
    GATE_CX,
    GATE_CY,
    GATE_DOMAIN,
    GATE_NEX,
    GATE_NEY,
    GATE_NOP,
    build_analytic_operator_2d,
    structural_report,
)
from tools.analytic_operator_2d_mortar import build_spec_operator_2d

GEOM_TOL = 1e-8        # node-lattice / coincidence tolerance (gmsh noise ~1e-11)
MASS_TOL = 1e-10       # relative
FAR_TOL = 1e-9         # absolute, far rows vs analytic
SUPPORT_REL = 1e-12    # entries outside the pattern, relative to max|L|
REVERSE_REL = 1e-12    # reverse mortar blocks, relative to max|L|
FORWARD_REL = 1e-6     # forward mortar blocks must exceed this, relative
EXACT_TOL = 1e-9       # polynomial exactness, relative (see _exactness)
CONTROL_MIN = 1e-7     # degree-(nop+1) controls must exceed this (measured 2.4e-5 to
                       # 1.2e-4 on correct operators, gate mesh, both legs)
CONS_TOL = 1e-10       # conservation, absolute (conforming measured 5.2e-12)
CONSIST_TOL = 1e-9     # consistency, absolute


# ---------------------------------------------------------------------------
# geometry and topology
# ---------------------------------------------------------------------------

class Mesh2:
    """Element boxes, levels, and face contacts derived from the node dump."""

    def __init__(self, x, y, nop, domain, nex, ney, tol=GEOM_TOL):
        self.nop = nop
        self.ngl = ngl = nop + 1
        self.domain = domain
        self.nex, self.ney = nex, ney
        (self.x0, self.x1), (self.y0, self.y1) = domain
        self.Lx = self.x1 - self.x0
        self.Ly = self.y1 - self.y0
        self.tol = tol
        n = len(x)
        if n % (ngl * ngl):
            raise AssertionError(f"{n} DOFs is not a multiple of ngl^2 = {ngl * ngl}")
        self.nelem = n // (ngl * ngl)
        self.xgl, self.wgl = lgl_gen(ngl)
        self.x, self.y = x, y
        self._boxes()
        self._levels()
        self._tiling()
        self._contacts()

    def dof(self, e, i, j):
        return e * self.ngl * self.ngl + j * self.ngl + i

    def _boxes(self):
        ngl = self.ngl
        self.box = np.zeros((self.nelem, 4))   # xmin, xmax, ymin, ymax
        self.max_node_err = 0.0
        for e in range(self.nelem):
            xa, ya = self.x[self.dof(e, 0, 0)], self.y[self.dof(e, 0, 0)]
            xb, yb = self.x[self.dof(e, ngl - 1, ngl - 1)], self.y[self.dof(e, ngl - 1, ngl - 1)]
            if not (xb > xa and yb > ya):
                raise AssertionError(
                    f"element {e + 1}: corner (i=ngl,j=ngl) = ({xb}, {yb}) is not "
                    f"above-right of (i=1,j=1) = ({xa}, {ya}) -- lattice orientation broken")
            self.box[e] = (xa, xb, ya, yb)
            for j in range(ngl):
                for i in range(ngl):
                    ip = self.dof(e, i, j)
                    xe = xa + (self.xgl[i] + 1.0) / 2.0 * (xb - xa)
                    ye = ya + (self.xgl[j] + 1.0) / 2.0 * (yb - ya)
                    err = max(abs(self.x[ip] - xe), abs(self.y[ip] - ye))
                    self.max_node_err = max(self.max_node_err, err)
                    if err > self.tol:
                        raise AssertionError(
                            f"DOF ip={ip + 1} (iel={e + 1}, i={i + 1}, j={j + 1}) at "
                            f"({self.x[ip]}, {self.y[ip]}) is off its element's LGL "
                            f"lattice ({xe}, {ye}) by {err:.3e}")

    def _levels(self):
        dxb = self.Lx / self.nex
        dyb = self.Ly / self.ney
        self.level = np.zeros(self.nelem, dtype=int)
        for e in range(self.nelem):
            xa, xb, ya, yb = self.box[e]
            lx = np.log2(dxb / (xb - xa))
            ly = np.log2(dyb / (yb - ya))
            l = int(round(lx))
            if abs(lx - l) > 1e-6 or abs(ly - l) > 1e-6:
                raise AssertionError(
                    f"element {e + 1}: size {(xb - xa):.6g} x {(yb - ya):.6g} is not a "
                    f"power-of-two refinement of the {dxb:.6g} x {dyb:.6g} base cell "
                    f"(log2 ratios {lx:.4f}, {ly:.4f})")
            self.level[e] = l

    def _tiling(self):
        area = np.sum((self.box[:, 1] - self.box[:, 0]) * (self.box[:, 3] - self.box[:, 2]))
        if abs(area - self.Lx * self.Ly) > 1e-9 * self.Lx * self.Ly:
            raise AssertionError(
                f"element areas sum to {area}, domain area {self.Lx * self.Ly}")
        for a in range(self.nelem):
            for b in range(a + 1, self.nelem):
                ox = min(self.box[a, 1], self.box[b, 1]) - max(self.box[a, 0], self.box[b, 0])
                oy = min(self.box[a, 3], self.box[b, 3]) - max(self.box[a, 2], self.box[b, 2])
                if ox > self.tol and oy > self.tol:
                    raise AssertionError(f"elements {a + 1} and {b + 1} overlap")

    def _same(self, u, v, period):
        d = (u - v) % period
        return min(d, period - d) <= self.tol

    def _contacts(self):
        """Every positive-length contact between the high side of element a and
        the low side of element b, along x (axis 0) and y (axis 1), wrapping
        periodically. A conforming face is one contact; a 2:1 face is two."""
        self.contacts = []
        for axis in (0, 1):
            hi, lo = (1, 0) if axis == 0 else (3, 2)
            t0, t1 = (2, 3) if axis == 0 else (0, 1)
            period = self.Lx if axis == 0 else self.Ly
            bound = self.x1 if axis == 0 else self.y1
            for a in range(self.nelem):
                for b in range(self.nelem):
                    if not self._same(self.box[a, hi], self.box[b, lo], period):
                        continue
                    lo_t = max(self.box[a, t0], self.box[b, t0])
                    hi_t = min(self.box[a, t1], self.box[b, t1])
                    if hi_t - lo_t <= self.tol:
                        continue
                    periodic = abs(self.box[a, hi] - bound) <= self.tol
                    la = self.box[a, t1] - self.box[a, t0]
                    lb = self.box[b, t1] - self.box[b, t0]
                    if abs(la - lb) <= self.tol:
                        kind = "conforming"
                    elif abs(la - 2 * lb) <= self.tol or abs(2 * la - lb) <= self.tol:
                        kind = "mortar"
                    else:
                        raise AssertionError(
                            f"contact between elements {a + 1} and {b + 1} has side "
                            f"lengths {la}, {lb}: not conforming and not 2:1")
                    self.contacts.append(dict(axis=axis, a=a, b=b, periodic=periodic,
                                              kind=kind, t=(lo_t, hi_t)))
        # census: every element side covered exactly once by its contacts
        cover = np.zeros((self.nelem, 4))
        for c in self.contacts:
            length = c["t"][1] - c["t"][0]
            hi_side, lo_side = (1, 0) if c["axis"] == 0 else (3, 2)
            cover[c["a"], hi_side] += length
            cover[c["b"], lo_side] += length
        for e in range(self.nelem):
            sx = self.box[e, 3] - self.box[e, 2]
            sy = self.box[e, 1] - self.box[e, 0]
            want = (sx, sx, sy, sy)
            for s in range(4):
                if abs(cover[e, s] - want[s]) > self.tol:
                    raise AssertionError(
                        f"element {e + 1} side {('x-min', 'x-max', 'y-min', 'y-max')[s]}: "
                        f"contacts cover {cover[e, s]} of {want[s]}")

    def trace(self, e, side):
        """DOFs on a side, ascending along the tangential coordinate.
        side: 0 x-min, 1 x-max, 2 y-min, 3 y-max."""
        ngl = self.ngl
        if side in (0, 1):
            i = 0 if side == 0 else ngl - 1
            return [self.dof(e, i, k) for k in range(ngl)]
        j = 0 if side == 2 else ngl - 1
        return [self.dof(e, k, j) for k in range(ngl)]

    def mortar_sides(self):
        s = set()
        for c in self.contacts:
            if c["kind"] == "mortar":
                s.add(c["a"])
                s.add(c["b"])
        return s


# ---------------------------------------------------------------------------
# the checks
# ---------------------------------------------------------------------------

def _upwind(c, cx, cy):
    """(upwind element, its side, downwind element, its side) for a contact,
    or None when c.n == 0 (no coupling)."""
    cn = cx if c["axis"] == 0 else cy
    hi_side, lo_side = (1, 0) if c["axis"] == 0 else (3, 2)
    if cn > 0:
        return c["a"], hi_side, c["b"], lo_side
    if cn < 0:
        return c["b"], lo_side, c["a"], hi_side
    return None


def allowed_pattern(mesh, cx, cy):
    """Boolean n x n: own-element lines (the volume term) + downwind trace rows
    x upwind trace columns per contact (matched node for conforming contacts,
    the full trace block for mortar halves)."""
    n = mesh.nelem * mesh.ngl ** 2
    ngl = mesh.ngl
    A = np.zeros((n, n), dtype=bool)
    for e in range(mesh.nelem):
        for j in range(ngl):
            for i in range(ngl):
                r = mesh.dof(e, i, j)
                for m in range(ngl):
                    A[r, mesh.dof(e, m, j)] = True
                    A[r, mesh.dof(e, i, m)] = True
    for c in mesh.contacts:
        up = _upwind(c, cx, cy)
        if up is None:
            continue
        eu, su, ed, sd = up
        tu, td = mesh.trace(eu, su), mesh.trace(ed, sd)
        if c["kind"] == "conforming":
            for k in range(ngl):
                A[td[k], tu[k]] = True
        else:
            for r in td:
                for col in tu:
                    A[r, col] = True
    return A


def _support(L, mesh, cx, cy):
    scale = float(np.max(np.abs(L)))
    A = allowed_pattern(mesh, cx, cy)
    outside = np.abs(L) * (~A)
    n_out = int(np.sum(outside > SUPPORT_REL * scale))
    worst = float(outside.max()) if outside.size else 0.0
    halves = []
    for c in mesh.contacts:
        if c["kind"] != "mortar":
            continue
        up = _upwind(c, cx, cy)
        if up is None:
            continue
        eu, su, ed, sd = up
        tu, td = mesh.trace(eu, su), mesh.trace(ed, sd)
        fwd = float(np.max(np.abs(L[np.ix_(td, tu)])))
        rev = float(np.max(np.abs(L[np.ix_(tu, td)])))
        halves.append(dict(up=eu, down=ed, axis=c["axis"], fwd=fwd, rev=rev,
                           up_level=int(mesh.level[eu]), down_level=int(mesh.level[ed])))
    return dict(scale=scale, n_outside=n_out, worst_outside=worst, halves=halves)


def _poly(mesh, a, b):
    xh = (mesh.x - 0.5 * (mesh.x0 + mesh.x1)) / (0.5 * mesh.Lx)
    yh = (mesh.y - 0.5 * (mesh.y0 + mesh.y1)) / (0.5 * mesh.Ly)
    u = xh ** a * yh ** b
    ux = (a * xh ** (a - 1) * yh ** b * (2.0 / mesh.Lx)) if a > 0 else np.zeros_like(u)
    uy = (b * xh ** a * yh ** (b - 1) * (2.0 / mesh.Ly)) if b > 0 else np.zeros_like(u)
    return u, ux, uy


def safe_elements(mesh, cx, cy):
    """Elements none of whose inflow contacts cross the periodic wrap: a global
    polynomial is continuous everywhere such an element can see."""
    bad = set()
    for c in mesh.contacts:
        up = _upwind(c, cx, cy)
        if up is None:
            continue
        if c["periodic"]:
            bad.add(up[2])
    return [e for e in range(mesh.nelem) if e not in bad]


def _exactness(L, mesh, cx, cy):
    """Residual of L u + c.grad u on safe rows, relative to max|L|*max|u|
    over those rows -- the operator scale, so a coupling error of relative
    size eps shows up at ~eps and round-off at ~1e-15."""
    rows = [mesh.dof(e, i, j) for e in safe_elements(mesh, cx, cy)
            for j in range(mesh.ngl) for i in range(mesh.ngl)]
    rows = np.array(rows, dtype=int)
    scale_L = float(np.max(np.abs(L[rows])))
    res = {}
    for a in range(mesh.nop + 1):
        for b in range(mesh.nop + 1):
            u, ux, uy = _poly(mesh, a, b)
            r = (L @ u)[rows] + (cx * ux + cy * uy)[rows]
            res[(a, b)] = float(np.max(np.abs(r)) / (scale_L * max(np.max(np.abs(u)), 1e-300)))
    ctrl = {}
    for a, b in ((mesh.nop + 1, 0), (0, mesh.nop + 1)):
        u, ux, uy = _poly(mesh, a, b)
        r = (L @ u)[rows] + (cx * ux + cy * uy)[rows]
        ctrl[(a, b)] = float(np.max(np.abs(r)) / (scale_L * np.max(np.abs(u))))
    return dict(n_rows=len(rows), n_elems=len(rows) // mesh.ngl ** 2,
                res=res, ctrl=ctrl)


def _far_rows(L, mesh, cx, cy):
    """Rows of level-0 elements with no mortar contact, entrywise vs the
    analytic conforming operator on the base grid, columns mapped by
    geometry. Every column these rows reach belongs to a level-0 element
    (a mortar contact would have made the element non-far)."""
    ngl = mesh.ngl
    dxb, dyb = mesh.Lx / mesh.nex, mesh.Ly / mesh.ney
    cell_of = {}
    for e in range(mesh.nelem):
        if mesh.level[e] == 0:
            ex = int(round((mesh.box[e, 0] - mesh.x0) / dxb))
            ey = int(round((mesh.box[e, 2] - mesh.y0) / dyb))
            cell_of[e] = (ex, ey)
    elem_of_cell = {v: k for k, v in cell_of.items()}
    A = build_analytic_operator_2d(mesh.nop, mesh.nex, mesh.ney, cx, cy,
                                   domain=mesh.domain)
    nb = ngl * ngl

    def canon(cell):
        return cell[1] * mesh.nex + cell[0]

    msides = mesh.mortar_sides()
    far = [e for e in cell_of if e not in msides]
    worst = 0.0
    worst_where = None
    for e in far:
        ea = canon(cell_of[e])
        for j in range(ngl):
            for i in range(ngl):
                r = mesh.dof(e, i, j)
                ra = ea * nb + j * ngl + i
                expect = np.zeros(L.shape[1])
                for ca in np.nonzero(A[ra])[0]:
                    cell = ((ca // nb) % mesh.nex, (ca // nb) // mesh.nex)
                    if cell not in elem_of_cell:
                        raise AssertionError(
                            f"far element {e + 1}: analytic row couples to base cell "
                            f"{cell}, which is refined in this mesh -- far-row "
                            f"selection is wrong")
                    rem = ca % nb
                    expect[mesh.dof(elem_of_cell[cell], rem % ngl, rem // ngl)] = A[ra, ca]
                d = np.abs(L[r] - expect)
                if d.max() > worst:
                    worst = float(d.max())
                    worst_where = (e, i, j, int(np.argmax(d)))
    return dict(n_far=len(far), worst=worst, where=worst_where)


def run_gate(L, x, y, M=None, adlvl=None, nop=GATE_NOP, domain=GATE_DOMAIN,
             nex=GATE_NEX, ney=GATE_NEY, cx=GATE_CX, cy=GATE_CY,
             expect_nelem=None, expect_mortar_halves=None, verbose=True):
    """All checks; returns (failures, report). Usable on arrays (the tests)
    and from main() on the probe's files."""
    out = print if verbose else (lambda *a, **k: None)
    failures = []
    rep = {}

    def verdict(ok, name):
        if not ok:
            failures.append(name)
        return "PASS" if ok else "FAIL"

    ngl = nop + 1
    n = L.shape[0]
    # [1] counts
    ok = (L.shape == (n, n) and len(x) == n and n % (ngl * ngl) == 0
          and (M is None or len(M) == n))
    nelem = n // (ngl * ngl) if n % (ngl * ngl) == 0 else None
    if adlvl is not None:
        ok = ok and nelem is not None and len(adlvl) == nelem
    if expect_nelem is not None:
        ok = ok and nelem == expect_nelem
    out(f"[1] counts: n = {n}, nelem = {nelem}, mass {'dumped' if M is not None else 'absent'} "
        f"({len(M) if M is not None else '-'}), ad_lvl "
        f"{len(adlvl) if adlvl is not None else 'absent'}"
        f"{'' if expect_nelem is None else f', expected nelem {expect_nelem}'}  "
        f"{verdict(ok, 'counts')}")
    if not ok:
        return failures, rep

    # [2] geometry + topology
    try:
        mesh = Mesh2(x, y, nop, domain, nex, ney)
    except AssertionError as exc:
        out(f"[2] geometry: FAIL -- {exc}")
        failures.append("geometry")
        return failures, rep
    levels = {int(l): int(np.sum(mesh.level == l)) for l in np.unique(mesh.level)}
    lvl_ok = adlvl is None or bool(np.array_equal(np.asarray(adlvl, dtype=int), mesh.level))
    kinds = {}
    for c in mesh.contacts:
        key = (c["kind"], c["periodic"])
        kinds[key] = kinds.get(key, 0) + 1
    n_conf_int = kinds.get(("conforming", False), 0)
    n_conf_per = kinds.get(("conforming", True), 0)
    n_mortar = kinds.get(("mortar", False), 0) + kinds.get(("mortar", True), 0)
    n_mortar_per = kinds.get(("mortar", True), 0)
    ok = lvl_ok and n_mortar_per == 0
    if expect_mortar_halves is not None:
        ok = ok and n_mortar == expect_mortar_halves
    refined = mesh.box[mesh.level > 0]
    patch = (f"x [{refined[:, 0].min():g}, {refined[:, 1].max():g}], "
             f"y [{refined[:, 2].min():g}, {refined[:, 3].max():g}]") if len(refined) else "none"
    out(f"[2] geometry: max node-lattice err {mesh.max_node_err:.3e}; tiling OK; levels "
        f"{levels} (ad_lvl {'absent' if adlvl is None else ('match' if lvl_ok else 'MISMATCH')}); "
        f"refined patch {patch}")
    out(f"    contacts: {n_conf_int} conforming interior + {n_conf_per} conforming periodic "
        f"+ {n_mortar} mortar halves ({n_mortar_per} across the wrap)"
        f"{'' if expect_mortar_halves is None else f', expected {expect_mortar_halves} halves'}; "
        f"every side covered once  {verdict(ok, 'geometry')}")
    rep.update(mesh=mesh, levels=levels)

    # [3] mass
    wgl = mesh.wgl
    Mgeo = np.zeros(n)
    for e in range(mesh.nelem):
        jvol = (mesh.box[e, 1] - mesh.box[e, 0]) * (mesh.box[e, 3] - mesh.box[e, 2]) / 4.0
        for j in range(ngl):
            for i in range(ngl):
                Mgeo[mesh.dof(e, i, j)] = wgl[i] * wgl[j] * jvol
    if M is not None:
        rel = float(np.max(np.abs(M - Mgeo) / Mgeo))
        ok = rel <= MASS_TOL
        out(f"[3] mass: max rel |M - w_i w_j Jvol| = {rel:.3e}; sum M = {np.sum(M):.12g} "
            f"(domain area {mesh.Lx * mesh.Ly:g})  {verdict(ok, 'mass')}")
        if not ok:
            k = int(np.argmax(np.abs(M - Mgeo) / Mgeo))
            e = k // (ngl * ngl)
            out(f"    worst at ip={k + 1} (iel={e + 1}, level {mesh.level[e]}): "
                f"M = {M[k]:.12e}, geometric {Mgeo[k]:.12e}")
        w = np.asarray(M, dtype=float)
        rep["mass_rel"] = rel
    else:
        out("[3] mass: no dump -- using the geometric mass as the conservation weight "
            "(check skipped)")
        w = Mgeo

    # [4] far rows
    try:
        far = _far_rows(L, mesh, cx, cy)
        ok = far["worst"] <= FAR_TOL and far["n_far"] > 0
        where = ""
        if far["where"] is not None and not ok:
            e, i, j, col = far["where"]
            where = f" at row iel={e + 1} i={i + 1} j={j + 1}, col ip={col + 1}"
        out(f"[4] far rows: {far['n_far']} elements vs analytic; max|diff| = "
            f"{far['worst']:.3e}{where}  {verdict(ok, 'far_rows')}")
        rep["far"] = far
    except AssertionError as exc:
        out(f"[4] far rows: FAIL -- {exc}")
        failures.append("far_rows")

    # [5] support
    sup = _support(L, mesh, cx, cy)
    fwd_ok = all(h["fwd"] > FORWARD_REL * sup["scale"] for h in sup["halves"])
    rev_ok = all(h["rev"] <= REVERSE_REL * sup["scale"] for h in sup["halves"])
    ok = sup["n_outside"] == 0 and fwd_ok and rev_ok
    fmin = min((h["fwd"] for h in sup["halves"]), default=float("nan"))
    rmax = max((h["rev"] for h in sup["halves"]), default=0.0)
    out(f"[5] support: entries outside the upwind pattern: {sup['n_outside']} "
        f"(max {sup['worst_outside']:.3e}, scale {sup['scale']:.3e}); mortar halves "
        f"{len(sup['halves'])}: min forward {fmin:.3e}, max reverse {rmax:.3e}  "
        f"{verdict(ok, 'support')}")
    for h in sup["halves"]:
        who = "parent" if h["up_level"] < h["down_level"] else "child"
        out(f"    {'x' if h['axis'] == 0 else 'y'}-face  upwind iel={h['up'] + 1} ({who}, "
            f"lvl {h['up_level']}) -> iel={h['down'] + 1} (lvl {h['down_level']}): "
            f"forward {h['fwd']:.3e}, reverse {h['rev']:.3e}")
    rep["support"] = sup

    # [6] exactness
    ex = _exactness(L, mesh, cx, cy)
    worst = max(ex["res"].values())
    wkey = max(ex["res"], key=ex["res"].get)
    cmin = min(ex["ctrl"].values())
    ok = worst <= EXACT_TOL and cmin > CONTROL_MIN
    out(f"[6] exactness: {ex['n_elems']} inflow-safe elements; max rel residual over "
        f"x^a y^b, a,b <= {nop}: {worst:.3e} (at a={wkey[0]}, b={wkey[1]}); controls "
        + ", ".join(f"({a},{b}) {v:.3e}" for (a, b), v in ex["ctrl"].items())
        + f"  {verdict(ok, 'exactness')}")
    rep["exactness"] = ex

    # [7]-[9] structural
    s = structural_report(L, w)
    rep["structural"] = s
    ok = s["conservation_mTL"] <= CONS_TOL
    out(f"[7] conservation |w^T L|_inf = {s['conservation_mTL']:.3e}  "
        f"{verdict(ok, 'conservation')}")
    if not ok:
        wl = np.abs(w @ L)
        for c in np.argsort(wl)[::-1][:5]:
            e = c // (ngl * ngl)
            out(f"    col ip={c + 1} (iel={e + 1}, level {mesh.level[e]}): {wl[c]:.3e}")
    ok = s["consistency_L1"] <= CONSIST_TOL
    out(f"[8] consistency |L 1|_inf = {s['consistency_L1']:.3e}  {verdict(ok, 'consistency')}")
    ok = (s["max_re_lambda"] < 1e-8 * s["spectral_radius"] and s["n_zero_modes"] == 1)
    out(f"[9] spectrum: max Re(lambda) = {s['max_re_lambda']:.3e} at rho = "
        f"{s['spectral_radius']:.4f}; kernel modes (|lam| < 1e-10 rho) = "
        f"{s['n_zero_modes']} (expect 1); near-zero band (|lam| < 1e-6 rho) = "
        f"{s['n_near_zero_modes']}  {verdict(ok, 'spectrum')}")
    out("    smallest |lambda|: [" + ", ".join(f"{v:.3e}" for v in s["smallest_abs_lambda"]) + "]")

    # [10] informational: entrywise vs the spec-built operator
    S = build_spec_operator_2d(mesh, cx, cy)
    diff = np.abs(L - S)
    dmax = float(diff.max())
    supp_eq = bool(np.array_equal(np.abs(L) > SUPPORT_REL * sup["scale"],
                                  np.abs(S) > SUPPORT_REL * sup["scale"]))
    n_over = int(np.sum(diff > 1e-8))
    out(f"[10] spec operator (informational): max|L - S| = {dmax:.3e} "
        f"(rel {dmax / sup['scale']:.3e}); entries > 1e-8: {n_over}; support "
        f"{'identical' if supp_eq else 'DIFFERS'}")
    if n_over:
        for flat in np.argsort(diff, axis=None)[::-1][:5]:
            r, col = np.unravel_index(flat, diff.shape)
            er, ec = r // (ngl * ngl), col // (ngl * ngl)
            out(f"     row ip={r + 1} (iel={er + 1}, lvl {mesh.level[er]}, i={r % ngl + 1}, "
                f"j={(r // ngl) % ngl + 1})  col ip={col + 1} (iel={ec + 1}, lvl "
                f"{mesh.level[ec]}, i={col % ngl + 1}, j={(col // ngl) % ngl + 1})  "
                f"jex={L[r, col]:+.12e}  spec={S[r, col]:+.12e}")
    rep["spec"] = dict(max_diff=dmax, n_over=n_over, support_identical=supp_eq)
    return failures, rep


def _find(d, pattern, required=True):
    hits = sorted(glob.glob(os.path.join(d, pattern)))
    if len(hits) > 1:
        raise SystemExit(f"more than one {pattern} in {d}: {hits}")
    if not hits:
        if required:
            raise SystemExit(f"no {pattern} in {d}")
        return None
    return hits[0]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", required=True,
                    help="the probe's output directory (holds the GATE2D_*.txt dumps)")
    ap.add_argument("--nex", type=int, default=GATE_NEX, help="base-grid elements in x")
    ap.add_argument("--ney", type=int, default=GATE_NEY, help="base-grid elements in y")
    ap.add_argument("--nop", type=int, default=GATE_NOP)
    ap.add_argument("--xmin", type=float, default=GATE_DOMAIN[0][0])
    ap.add_argument("--xmax", type=float, default=GATE_DOMAIN[0][1])
    ap.add_argument("--ymin", type=float, default=GATE_DOMAIN[1][0])
    ap.add_argument("--ymax", type=float, default=GATE_DOMAIN[1][1])
    ap.add_argument("--cx", type=float, default=GATE_CX)
    ap.add_argument("--cy", type=float, default=GATE_CY)
    ap.add_argument("--expect-nelem", type=int, default=None)
    ap.add_argument("--expect-mortar-halves", type=int, default=None)
    args = ap.parse_args(argv)

    op = _find(args.dir, "GATE2D_operator_*.txt")
    nd = _find(args.dir, "GATE2D_nodes_*.txt")
    ms = _find(args.dir, "GATE2D_mass_*.txt", required=False)
    lv = _find(args.dir, "GATE2D_adlvl_*.txt", required=False)
    domain = ((args.xmin, args.xmax), (args.ymin, args.ymax))

    print("=" * 76)
    print(f"GATE 2D mortar (Phase-6 item 7): nop={args.nop} base {args.nex}x{args.ney} "
          f"c=({args.cx}, {args.cy}) domain={domain}")
    for p in (op, nd, ms, lv):
        if p is not None:
            print(f"  {p}")
    print("=" * 76)
    L = np.loadtxt(op)
    xy = np.loadtxt(nd)
    M = np.loadtxt(ms) if ms else None
    adlvl = np.loadtxt(lv, dtype=int, ndmin=1) if lv else None
    failures, _ = run_gate(L, xy[:, 0], xy[:, 1], M=M, adlvl=adlvl, nop=args.nop,
                           domain=domain, nex=args.nex, ney=args.ney,
                           cx=args.cx, cy=args.cy,
                           expect_nelem=args.expect_nelem,
                           expect_mortar_halves=args.expect_mortar_halves)
    print("=" * 76)
    if failures:
        print(f"VERDICT: FAIL ({', '.join(failures)})")
    else:
        print("VERDICT: PASS -- the extracted operator is conservative, consistent, "
              "polynomially exact, correctly upwinded at every mortar half, and dissipative.")
    print("=" * 76)
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
