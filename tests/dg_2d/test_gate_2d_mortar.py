"""Pins for tools/gate_2d_mortar.py -- the Phase-6 item-7 instrument -- and for
tools/analytic_operator_2d_mortar.py, the spec-built 2:1 operator it uses as
check [10] and these tests use as ground truth.

The instrument must be shown to (a) PASS correct operators and (b) FAIL each
defect class it claims to catch, before it judges Jexpresso's 2:1 operator.
No Jexpresso artifact is needed: build_spec_operator_2d assembles a 2:1
operator directly from the five-step mortar specification (DG roadmap
section 5.17, "Mortar flux specification"), with the face operators built the
way Jexpresso's build_projection_1d builds them, and its _mutate test hook
injects one defect class at a time.

What each group pins:
  1. The spec-built operator itself: on an all-conforming mesh it reduces to
     the analytic operator of tools/analytic_operator_2d.py (itself pinned by
     reduction to the F1-verified 1D operator), and the two projection
     identities hold.
  2. Topology derivation on the static 2:1 mesh: the counts the item-7 run
     must reproduce (19 elements, 24 + 8 conforming contacts, 8 mortar
     halves, 12 inflow-safe elements).
  3. PASS on correct operators: the conforming analytic operator (canonical
     and shuffled element order) and the spec-built 2:1 operator, both legs.
  4. FAIL on each defect class, as the measured detection matrix
     (DETECTION below; the explanation doc's section-7 table is written from
     it): a wrong half used consistently (only exactness sees it); a central
     mortar flux (only support sees it); a doubled 1/2; an uncoupled face; a
     stray coupling; a corrupted mass; a mismatched ad_lvl.
  5. Check [10] is informational: it reads ~0 on a correct operator, flags a
     defective one, and never changes the verdict.
  6. Mortars across the periodic wrap (item 8b step 5, D-079): the five
     wrap-refined meshes pass with their expected wrap-half counts and are
     refused by the default expectation of none; a wrong half on the wrap
     parent faces only is caught by the wrap rows of check [6] -- in both
     directions (children upwind: project; parent upwind: interp) -- while
     the inflow-safe rows and conservation stay blind to it; and the
     detection matrix on a wrap mesh, which differs from the interior one in
     exactly one row (WRAP_DETECTION: a central wrap flux is seen by
     exactness too, through the upwind side's outflow face).
  7. Walls (item 8e): the analytic and spec builders agree on the mirror-ghost
     wall term; the inflow-wall penalty has its hand value; the walled and
     channel meshes (conforming, the inflow corner, the outflow corner, the
     inflow wall x wrap corner) pass with no kernel mode and the measured
     spectral radii; the two wall defects (a transmissive wall; the textbook
     exterior state in place of the mirror) are caught, the mortar matrix
     carries over with its measured differences, and a wrong --walls
     declaration cannot produce a PASS in either direction.

The spec-built operator takes face contacts from the instrument's own Mesh2
topology; group 2 pins that topology against hand-counted values, which is
what keeps the shared piece honest.

Run with: pytest tests/dg_2d/test_gate_2d_mortar.py -v
"""

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
    default_elem_cells,
)
from tools.analytic_operator_2d_mortar import (
    build_spec_operator_2d,
    geometric_mass,
    projection_1d,
)
from tools.gate_2d_mortar import EXACT_TOL, Mesh2, run_gate, safe_elements, wrap_inflow_axes

NGL = GATE_NOP + 1
REFINED_CELL = (2, 2)   # x in [0, 2.5], y in [10, 15] -- the item-3 case


# ---------------------------------------------------------------------------
# fixture: meshes, nodes, and the spec-built 2:1 operator
# ---------------------------------------------------------------------------

def boxes_conforming(nex=GATE_NEX, ney=GATE_NEY, domain=GATE_DOMAIN, cells=None):
    (x0, x1), (y0, y1) = domain
    dx, dy = (x1 - x0) / nex, (y1 - y0) / ney
    cells = cells or default_elem_cells(nex, ney)
    return [(x0 + ex * dx, x0 + (ex + 1) * dx, y0 + ey * dy, y0 + (ey + 1) * dy)
            for ex, ey in cells]


def boxes_2to1(refined=REFINED_CELL, domain=GATE_DOMAIN):
    """15 base elements in canonical order, then the four children in the
    NW, SW, NE, SE order Jexpresso's p4est numbering produces."""
    base = boxes_conforming(domain=domain)
    cells = default_elem_cells(GATE_NEX, GATE_NEY)
    k = cells.index(refined)
    xa, xb, ya, yb = base.pop(k)
    xm, ym = 0.5 * (xa + xb), 0.5 * (ya + yb)
    children = [(xa, xm, ym, yb), (xa, xm, ya, ym), (xm, xb, ym, yb), (xm, xb, ya, ym)]
    return base + children


def nodes_from_boxes(boxes, nop=GATE_NOP):
    ngl = nop + 1
    xgl, _ = lgl_gen(ngl)
    x, y = [], []
    for xa, xb, ya, yb in boxes:
        for j in range(ngl):
            for i in range(ngl):
                x.append(xa + (xgl[i] + 1) / 2 * (xb - xa))
                y.append(ya + (xgl[j] + 1) / 2 * (yb - ya))
    return np.array(x), np.array(y)


def assemble(boxes, cx, cy, nop=GATE_NOP, domain=GATE_DOMAIN, mutate=None):
    """(L, x, y, M) for a mesh of boxes: the spec-built operator of
    tools/analytic_operator_2d_mortar.py, its node coordinates, and its mass.
    mutate injects one defect class (see build_spec_operator_2d's test hook)."""
    x, y = nodes_from_boxes(boxes, nop)
    mesh = Mesh2(x, y, nop, domain, GATE_NEX, GATE_NEY)
    L = build_spec_operator_2d(mesh, cx, cy, _mutate=mutate)
    return L, x, y, geometric_mass(mesh)


def gate(L, x, y, M=None, adlvl=None, cx=GATE_CX, cy=GATE_CY, **kw):
    failures, rep = run_gate(L, x, y, M=M, adlvl=adlvl, cx=cx, cy=cy, verbose=False, **kw)
    return failures, rep


LEGS = [(GATE_CX, GATE_CY), (0.7, 0.3)]


# ---------------------------------------------------------------------------
# 1. the fixture
# ---------------------------------------------------------------------------

def test_projection_identities():
    interp, project = projection_1d(GATE_NOP)
    _, w = lgl_gen(NGL)
    assert np.allclose(project[0] @ interp[0] + project[1] @ interp[1], np.eye(NGL), atol=1e-13)
    for h in (0, 1):
        assert np.allclose(w @ project[h], 0.5 * w, atol=1e-14)


def test_fixture_reduces_to_analytic_on_conforming_mesh():
    for cx, cy in LEGS:
        L, _, _, _ = assemble(boxes_conforming(), cx, cy)
        A = build_analytic_operator_2d(GATE_NOP, GATE_NEX, GATE_NEY, cx, cy)
        assert np.max(np.abs(L - A)) < 1e-12


# ---------------------------------------------------------------------------
# 2. topology on the static 2:1 mesh
# ---------------------------------------------------------------------------

def test_2to1_topology_counts():
    x, y = nodes_from_boxes(boxes_2to1())
    mesh = Mesh2(x, y, GATE_NOP, GATE_DOMAIN, GATE_NEX, GATE_NEY)
    assert mesh.nelem == 19
    assert sorted(np.bincount(mesh.level)) == [4, 15]
    kinds = [(c["kind"], c["periodic"]) for c in mesh.contacts]
    assert kinds.count(("conforming", False)) == 24
    assert kinds.count(("conforming", True)) == 8
    assert kinds.count(("mortar", False)) == 8
    assert kinds.count(("mortar", True)) == 0
    for cx, cy in LEGS:
        assert len(safe_elements(mesh, cx, cy)) == 12


def test_conforming_topology_counts():
    x, y = nodes_from_boxes(boxes_conforming())
    mesh = Mesh2(x, y, GATE_NOP, GATE_DOMAIN, GATE_NEX, GATE_NEY)
    kinds = [(c["kind"], c["periodic"]) for c in mesh.contacts]
    assert kinds.count(("conforming", False)) == 24
    assert kinds.count(("conforming", True)) == 8
    assert len(safe_elements(mesh, GATE_CX, GATE_CY)) == 9


# ---------------------------------------------------------------------------
# 3. PASS on correct operators
# ---------------------------------------------------------------------------

def test_conforming_analytic_passes():
    x, y = nodes_from_boxes(boxes_conforming())
    for cx, cy in LEGS:
        A = build_analytic_operator_2d(GATE_NOP, GATE_NEX, GATE_NEY, cx, cy)
        failures, rep = gate(A, x, y, cx=cx, cy=cy)
        assert failures == []
        assert rep["far"]["n_far"] == 16
        assert rep["support"]["halves"] == []


def test_conforming_shuffled_order_passes():
    rng = np.random.default_rng(7)
    cells = default_elem_cells(GATE_NEX, GATE_NEY)
    cells = [cells[k] for k in rng.permutation(len(cells))]
    A = build_analytic_operator_2d(GATE_NOP, GATE_NEX, GATE_NEY, GATE_CX, GATE_CY,
                                   elem_cells=cells)
    x, y = nodes_from_boxes(boxes_conforming(cells=cells))
    failures, _ = gate(A, x, y)
    assert failures == []


def test_2to1_spec_operator_passes_both_legs():
    boxes = boxes_2to1()
    adlvl = [0] * 15 + [1] * 4
    for cx, cy in LEGS:
        L, x, y, M = assemble(boxes, cx, cy)
        failures, rep = gate(L, x, y, M=M, adlvl=adlvl, cx=cx, cy=cy,
                             expect_nelem=19, expect_mortar_halves=8)
        assert failures == []
        halves = rep["support"]["halves"]
        assert len(halves) == 8
        # the upwind structure: two parents upwind (left/bottom faces), two
        # parents downwind (right/top faces) -- 4 halves each
        assert sum(h["up_level"] == 0 for h in halves) == 4
        assert rep["far"]["n_far"] == 11
        assert rep["exactness"]["n_elems"] == 12


def test_2to1_spectral_radius_in_predicted_range():
    for (cx, cy), rho_conf in zip(LEGS, (5.892, 5.008)):
        L, x, y, M = assemble(boxes_2to1(), cx, cy)
        _, rep = gate(L, x, y, M=M, cx=cx, cy=cy)
        rho = rep["structural"]["spectral_radius"]
        assert rho_conf < rho <= 2 * rho_conf + 1e-9


# ---------------------------------------------------------------------------
# 4. FAIL on each defect class
# ---------------------------------------------------------------------------

def test_wrong_half_caught_by_exactness_not_by_conservation():
    """The row of the explanation table that motivates check [6]."""
    L, x, y, M = assemble(boxes_2to1(), GATE_CX, GATE_CY, mutate="wrong_half")
    failures, rep = gate(L, x, y, M=M)
    assert "exactness" in failures
    assert "conservation" not in failures
    assert rep["structural"]["conservation_mTL"] < 1e-10


def test_doubled_half_factor_caught_by_conservation():
    L, x, y, M = assemble(boxes_2to1(), GATE_CX, GATE_CY, mutate="double_half")
    failures, rep = gate(L, x, y, M=M)
    assert "conservation" in failures
    assert rep["structural"]["conservation_mTL"] > 1e-3


def test_uncoupled_mortar_caught():
    L, x, y, M = assemble(boxes_2to1(), GATE_CX, GATE_CY, mutate="uncoupled")
    failures, rep = gate(L, x, y, M=M)
    assert "support" in failures        # forward blocks are zero
    assert "conservation" in failures
    assert "spectrum" in failures       # the item-6 blow-up: max Re(lambda) > 0
    # exactness CANNOT see an uncoupled face: a continuous polynomial has zero
    # jumps, so an element with no face term still computes -c.grad u exactly
    assert "exactness" not in failures


# The measured detection matrix (2026-09-28, both legs identical): which checks
# fire for which defect. The explanation doc's section-7 table is written from
# this, not from reasoning -- keep them in step.
DETECTION = {
    "wrong_half": {"exactness"},
    "double_half": {"exactness", "conservation", "consistency", "spectrum"},
    "uncoupled": {"support", "conservation", "spectrum"},
    # the defect only the directional check can see: conservative, consistent,
    # polynomially exact, and energy-stable -- but coupled both ways
    "central": {"support"},
}


def test_detection_matrix_both_legs():
    for cx, cy in LEGS:
        for mutate, expected in DETECTION.items():
            L, x, y, M = assemble(boxes_2to1(), cx, cy, mutate=mutate)
            failures, _ = gate(L, x, y, M=M, cx=cx, cy=cy)
            assert set(failures) == expected, (cx, cy, mutate, failures)


def test_stray_reverse_coupling_caught_by_support():
    L, x, y, M = assemble(boxes_2to1(), GATE_CX, GATE_CY)
    mesh = Mesh2(x, y, GATE_NOP, GATE_DOMAIN, GATE_NEX, GATE_NEY)
    c = next(c for c in mesh.contacts if c["kind"] == "mortar")
    hi_side, lo_side = (1, 0) if c["axis"] == 0 else (3, 2)
    up = mesh.trace(c["a"], hi_side)       # c > 0: the low-side element a is upwind
    down = mesh.trace(c["b"], lo_side)
    L = L.copy()
    L[up[2], down[2]] += 1e-3               # upwind row sees a downwind column
    failures, _ = gate(L, x, y, M=M)
    assert "support" in failures


def test_corrupted_mass_caught():
    L, x, y, M = assemble(boxes_2to1(), GATE_CX, GATE_CY)
    M = M.copy()
    M[15 * NGL * NGL] *= 1.25               # a child corner node, the DSS_nc_*_mass! site
    failures, _ = gate(L, x, y, M=M)
    assert "mass" in failures


def test_adlvl_mismatch_caught():
    L, x, y, M = assemble(boxes_2to1(), GATE_CX, GATE_CY)
    failures, _ = gate(L, x, y, M=M, adlvl=[0] * 19)
    assert "geometry" in failures


def test_expected_counts_enforced():
    L, x, y, M = assemble(boxes_conforming(), GATE_CX, GATE_CY)
    failures, _ = gate(L, x, y, M=M, expect_nelem=19)
    assert failures == ["counts"]


# ---------------------------------------------------------------------------
# 5. check [10] is informational
# ---------------------------------------------------------------------------

def test_spec_check_reads_zero_on_correct_operators():
    for cx, cy in LEGS:
        L, x, y, M = assemble(boxes_2to1(), cx, cy)
        _, rep = gate(L, x, y, M=M, cx=cx, cy=cy)
        assert rep["spec"]["max_diff"] == 0.0
        assert rep["spec"]["support_identical"]
    x, y = nodes_from_boxes(boxes_conforming())
    A = build_analytic_operator_2d(GATE_NOP, GATE_NEX, GATE_NEY, GATE_CX, GATE_CY)
    _, rep = gate(A, x, y)
    assert rep["spec"]["max_diff"] < 1e-12


def test_spec_check_flags_defects_without_gating():
    for mutate, expected in DETECTION.items():
        L, x, y, M = assemble(boxes_2to1(), GATE_CX, GATE_CY, mutate=mutate)
        failures, rep = gate(L, x, y, M=M)
        assert rep["spec"]["n_over"] > 0, mutate
        assert set(failures) == expected, (mutate, failures)


def test_2to1_in_jexpresso_element_order():
    """Jexpresso numbers the four children 7-10 (NW, SW, NE, SE), in the middle
    of the element list (item-4 probe, env doc 8.6) -- not appended. Every
    check and the spec operator must be order-independent."""
    boxes = boxes_2to1()
    order = list(range(6)) + [15, 16, 17, 18] + list(range(6, 15))
    boxes = [boxes[k] for k in order]
    adlvl = [0] * 6 + [1] * 4 + [0] * 9
    for cx, cy in LEGS:
        L, x, y, M = assemble(boxes, cx, cy)
        failures, rep = gate(L, x, y, M=M, adlvl=adlvl, cx=cx, cy=cy,
                             expect_nelem=19, expect_mortar_halves=8)
        assert failures == []
        assert rep["spec"]["max_diff"] == 0.0


# ---------------------------------------------------------------------------
# 6. mortars across the periodic wrap (item 8b step 5)
# ---------------------------------------------------------------------------

# Refined base cell -> mortar halves across the wrap. With c > 0 in both
# components (both legs) a refined element on the HIGH side of a wrap has its
# children upwind of the wrap mortar (the parent row tests project); on the
# LOW side the parent is upwind (the children rows test interp). The five
# meshes cover both directions on both axes; corner_mixed is the corner the
# item-8b step-5 run refines (AMR_NATIVE_PREADAPT=corner).
WRAP_MESHES = {
    "corner_hi": ((3, 3), 4),     # x in [2.5, 5], y in [15, 20]: x-project, y-project
    "corner_lo": ((0, 0), 4),     # x in [-5, -2.5], y in [0, 5]: x-interp, y-interp
    "corner_mixed": ((3, 0), 4),  # x in [2.5, 5], y in [0, 5]: x-project, y-interp
    "x_lo": ((0, 2), 2),          # x in [-5, -2.5], y in [10, 15]: x-interp
    "y_hi": ((1, 3), 2),          # x in [-2.5, 0], y in [15, 20]: y-project
}


def _wrap_rows_max(rep):
    return max(v for wset in rep["exactness"]["wrap"].values() for v in wset["res"].values())


def test_wrap_mortar_topology_and_pass():
    adlvl = [0] * 15 + [1] * 4
    for name, (cell, n_wrap) in WRAP_MESHES.items():
        boxes = boxes_2to1(refined=cell)
        x, y = nodes_from_boxes(boxes)
        mesh = Mesh2(x, y, GATE_NOP, GATE_DOMAIN, GATE_NEX, GATE_NEY)
        kinds = [(c["kind"], c["periodic"]) for c in mesh.contacts]
        assert kinds.count(("mortar", True)) == n_wrap, name
        assert kinds.count(("mortar", False)) == 8 - n_wrap, name
        assert kinds.count(("conforming", True)) == 8 - n_wrap // 2, name
        for cx, cy in LEGS:
            L, x, y, M = assemble(boxes, cx, cy)
            failures, rep = gate(L, x, y, M=M, adlvl=adlvl, cx=cx, cy=cy,
                                 expect_nelem=19, expect_mortar_halves=8,
                                 expect_wrap_mortar_halves=n_wrap)
            assert failures == [], (name, cx, cy, failures)
            assert rep["n_wrap_mortar_halves"] == n_wrap
            ex = rep["exactness"]
            w = ex["wrap"]
            assert ex["n_elems"] + w["x"]["n_elems"] + w["y"]["n_elems"] + w["xy"]["n_elems"] == 19
            assert w["x"]["n_elems"] > 0 and w["y"]["n_elems"] > 0 and w["xy"]["n_elems"] == 1
            assert _wrap_rows_max(rep) < EXACT_TOL
            # the default expectation (no mortar across the wrap) refuses the mesh
            failures, _ = gate(L, x, y, M=M, adlvl=adlvl, cx=cx, cy=cy)
            assert failures == ["geometry"], (name, cx, cy, failures)


def test_wrap_inflow_axes_partition():
    """Every element has exactly one row set: safe, x-wrap, y-wrap or both."""
    for name, (cell, _) in WRAP_MESHES.items():
        x, y = nodes_from_boxes(boxes_2to1(refined=cell))
        mesh = Mesh2(x, y, GATE_NOP, GATE_DOMAIN, GATE_NEX, GATE_NEY)
        for cx, cy in LEGS:
            axes = wrap_inflow_axes(mesh, cx, cy)
            counts = {}
            for s in axes.values():
                counts[frozenset(s)] = counts.get(frozenset(s), 0) + 1
            assert sum(counts.values()) == 19, name
            assert len(safe_elements(mesh, cx, cy)) == counts.get(frozenset(), 0), name


def test_wrong_half_across_the_wrap_caught_by_wrap_rows():
    """A wrong half on the parent faces across the wrap only. Interior mortars
    are correct, so the inflow-safe rows of [6] cannot see it; conservation is
    blind to a same-wrong-flux-both-sides defect; the wrap rows see it -- in
    both directions on both axes."""
    for name, (cell, n_wrap) in WRAP_MESHES.items():
        boxes = boxes_2to1(refined=cell)
        for cx, cy in LEGS:
            L, x, y, M = assemble(boxes, cx, cy, mutate="wrong_half_wrap")
            failures, rep = gate(L, x, y, M=M, cx=cx, cy=cy,
                                 expect_wrap_mortar_halves=n_wrap)
            assert "exactness" in failures, (name, cx, cy, failures)
            assert "conservation" not in failures, (name, cx, cy, failures)
            assert max(rep["exactness"]["res"].values()) < EXACT_TOL, (name, cx, cy)
            assert _wrap_rows_max(rep) > 1e-3, (name, cx, cy)
            assert rep["spec"]["n_over"] > 0


# The detection matrix on a mesh whose mortars include wrap ones (measured
# 2026-10-15, both legs). One row changes: a CENTRAL mortar flux couples both
# ways, so the element upwind of a wrap mortar sees its downwind neighbour's
# trace across the wrap -- and x^a at x = 5 is not x^a at x = -5. On the
# interior mesh that reverse coupling carries a continuous polynomial and is
# invisible to exactness; across the wrap it is a jump on the upwind side's
# OUTFLOW face, which the rows tested with the other axis's family see
# (corner_lo: both wrap sets), or the safe rows see when the upwind parent is
# interior-safe (x_lo). The upwind flux never couples through an outflow
# face, so a correct operator is unaffected.
WRAP_DETECTION = dict(DETECTION, central={"support", "exactness"})


def test_wrap_mesh_detection_matrix():
    for name in ("corner_lo", "x_lo"):
        cell, n_wrap = WRAP_MESHES[name]
        for cx, cy in LEGS:
            for mutate, expected in WRAP_DETECTION.items():
                L, x, y, M = assemble(boxes_2to1(refined=cell), cx, cy, mutate=mutate)
                failures, rep = gate(L, x, y, M=M, cx=cx, cy=cy,
                                     expect_wrap_mortar_halves=n_wrap)
                assert set(failures) == expected, (name, cx, cy, mutate, failures)
                if mutate == "central":
                    safe_max = max(rep["exactness"]["res"].values())
                    wrap_max = _wrap_rows_max(rep)
                    if name == "corner_lo":
                        assert safe_max < EXACT_TOL and wrap_max > 1e-3, (name, cx, cy)
                    else:
                        assert safe_max > 1e-3, (name, cx, cy)

# ---------------------------------------------------------------------------
# 7. walls (item 8e): a walled axis is not a wrap
# ---------------------------------------------------------------------------

# The walled meshes of the 8e operator gate. (walls, refined base cell or None,
# expected wall sides, mortar halves, mortar halves across the wrap). The
# bottom-left corner has two inflow walls on the walled mesh and an inflow wall
# meeting the x-wrap on the channel; the top-right corner has two outflow walls.
WALL_MESHES = {
    "walled_conf": ("xy", None, 16, 0, 0),
    "channel_conf": ("y", None, 8, 0, 0),
    "walled_corner_ll": ("xy", (0, 0), 18, 4, 0),
    "walled_corner_ur": ("xy", (3, 3), 18, 4, 0),
    "channel_corner_ll": ("y", (0, 0), 9, 6, 2),
}


def _wall_boxes(cell):
    return boxes_conforming() if cell is None else boxes_2to1(refined=cell)


def _wall_mesh(name):
    walls, cell, *_ = WALL_MESHES[name]
    x, y = nodes_from_boxes(_wall_boxes(cell))
    return Mesh2(x, y, GATE_NOP, GATE_DOMAIN, GATE_NEX, GATE_NEY, walls=walls), x, y


def _wall_gate(name, cx, cy, mutate=None, **kw):
    walls, cell, n_wall, n_mortar, n_wrap = WALL_MESHES[name]
    mesh, x, y = _wall_mesh(name)
    L = build_spec_operator_2d(mesh, cx, cy, _mutate=mutate)
    args = dict(walls=walls, expect_wall_sides=n_wall, expect_mortar_halves=n_mortar,
                expect_wrap_mortar_halves=n_wrap)
    args.update(kw)
    return gate(L, x, y, M=geometric_mass(mesh), cx=cx, cy=cy, **args)


def test_walled_analytic_equals_spec_on_conforming_meshes():
    """The two builders agree on the wall term (mirror ghost about q_bc = 0):
    the analytic operator with walls, and the spec operator on a Mesh2 that
    lists the wall sides, differ at round-off, both legs, both wall sets."""
    for name in ("walled_conf", "channel_conf"):
        walls = WALL_MESHES[name][0]
        mesh, _, _ = _wall_mesh(name)
        for cx, cy in LEGS:
            S = build_spec_operator_2d(mesh, cx, cy)
            A = build_analytic_operator_2d(GATE_NOP, GATE_NEX, GATE_NEY, cx, cy, walls=walls)
            assert np.max(np.abs(S - A)) < 1e-12, (name, cx, cy)


def test_wall_penalty_hand_value():
    """P on an inflow-wall node is 2 |c.n| / (w_0 J_perp): on the 4x4 walled
    mesh at leg 1 the bottom wall gives 2*1/(0.1*2.5) = 8, the left wall
    2*0.5/(0.1*1.25) = 8, and the bottom-left corner node 16; outflow walls
    (top, right) add nothing. The drain b is w_k Jf |c.n| on every wall node."""
    from tools.gate_2d_mortar import wall_terms
    mesh, _, _ = _wall_mesh("walled_conf")
    P, b, n_in = wall_terms(mesh, GATE_CX, GATE_CY)
    assert n_in == 8
    assert np.isclose(np.max(np.abs(P)), 16.0)
    corner = mesh.dof(0, 0, 0)                 # element (0,0), node (i=1, j=1)
    assert np.isclose(P[corner], -16.0)
    bottom_mid = mesh.dof(1, 2, 0)             # bottom wall, interior face node
    assert np.isclose(P[bottom_mid], -8.0)
    top_right = mesh.dof(15, NGL - 1, NGL - 1)   # two outflow walls: no penalty, drains
    assert P[top_right] == 0.0 and b[top_right] > 0.0
    assert np.count_nonzero(b) == 16 * NGL - 4   # every wall node once (corners shared)


def test_wall_topology_counts():
    for name, (walls, cell, n_wall, n_mortar, n_wrap) in WALL_MESHES.items():
        mesh, _, _ = _wall_mesh(name)
        kinds = [(c["kind"], c["periodic"]) for c in mesh.contacts]
        assert len(mesh.walls) == n_wall, name
        assert kinds.count(("mortar", False)) + kinds.count(("mortar", True)) == n_mortar, name
        assert kinds.count(("mortar", True)) == n_wrap, name
        if walls == "xy":
            assert kinds.count(("conforming", True)) == 0, name
        n_conf_int = 24 if cell is None else 26
        assert kinds.count(("conforming", False)) == n_conf_int, name
    mesh, _, _ = _wall_mesh("channel_conf")
    assert [(c["kind"], c["periodic"]) for c in mesh.contacts].count(("conforming", True)) == 4
    mesh, _, _ = _wall_mesh("channel_corner_ll")
    assert [(c["kind"], c["periodic"]) for c in mesh.contacts].count(("conforming", True)) == 3


# Spectral radii of the spec-built walled operators (measured 2026-10-18): the
# inflow-wall penalty sets rho, so the walled conforming mesh and the outflow
# corner preadapt share it, and refining the inflow corner doubles it (the
# children's J_perp is half). The 8e gate pre-declares these for Jexpresso.
WALL_RHO = {
    "walled_conf": (11.4454, 9.7286),
    "channel_conf": (8.6686, 5.8411),
    "walled_corner_ll": (22.8907, 19.4571),
    "walled_corner_ur": (11.4454, 9.7286),
    "channel_corner_ll": (13.5769, 8.0166),
}


def test_walled_spec_operators_pass_both_legs():
    for name, rhos in WALL_RHO.items():
        for (cx, cy), rho in zip(LEGS, rhos):
            failures, rep = _wall_gate(name, cx, cy)
            assert failures == [], (name, cx, cy, failures)
            s = rep["structural"]
            assert s["n_zero_modes"] == 0, name           # the constant is not in the kernel
            assert s["max_re_lambda"] < 0.0, name
            assert abs(s["spectral_radius"] - rho) < 5e-4, (name, cx, cy, s["spectral_radius"])
            assert rep["conservation"] < 1e-13 and rep["consistency"] < 1e-13, name
            assert rep["spec"]["max_diff"] == 0.0
            assert rep["n_wall_sides"] == WALL_MESHES[name][2]


# The measured detection matrices on the walled meshes (2026-10-18, both legs
# identical). The two wall defects: a transmissive wall (no term at all) is
# seen by everything but support; the textbook exterior state q_bc = 0 in
# place of the mirror (half the inflow penalty) is seen by the mass identity,
# consistency, exactness and the far rows -- so the gate certifies the
# operator as the mirror construction builds it, not merely "some" wall
# condition. The mortar rows carry over from DETECTION with one measured
# difference: an uncoupled mortar whose downwind neighbours have inflow walls
# of their own is damped by their penalty, so the spectrum stays in the left
# half-plane (walled_corner_ll), while at the outflow corner it does not
# (walled_corner_ur, as on the periodic mesh); and across the wrap a central
# flux is seen by exactness too (channel_corner_ll, as WRAP_DETECTION).
WALL_ONLY_DETECTION = {
    "transmissive_wall": {"conservation", "consistency", "exactness", "far_rows", "spectrum"},
    "standard_inflow": {"conservation", "consistency", "exactness", "far_rows"},
}
WALL_DETECTION = {
    "walled_corner_ll": dict(WALL_ONLY_DETECTION, **{
        "wrong_half": {"exactness"},
        "double_half": {"conservation", "consistency", "exactness"},
        "uncoupled": {"conservation", "support"},
        "central": {"support"},
    }),
    "walled_corner_ur": dict(WALL_ONLY_DETECTION, **{
        "wrong_half": {"exactness"},
        "double_half": {"conservation", "consistency", "exactness"},
        "uncoupled": {"conservation", "support", "spectrum"},
        "central": {"support"},
    }),
    "channel_corner_ll": dict(WALL_ONLY_DETECTION, **{
        "wrong_half": {"exactness"},
        "double_half": {"conservation", "consistency", "exactness"},
        "uncoupled": {"conservation", "support"},
        "central": {"support", "exactness"},
    }),
}


def test_wall_detection_matrices_both_legs():
    for name in ("walled_conf", "channel_conf"):
        for cx, cy in LEGS:
            for mutate, expected in WALL_ONLY_DETECTION.items():
                failures, _ = _wall_gate(name, cx, cy, mutate=mutate)
                assert set(failures) == expected, (name, cx, cy, mutate, failures)
    for name, matrix in WALL_DETECTION.items():
        for cx, cy in LEGS:
            for mutate, expected in matrix.items():
                failures, _ = _wall_gate(name, cx, cy, mutate=mutate)
                assert set(failures) == expected, (name, cx, cy, mutate, failures)


def test_walled_operator_judged_as_periodic_fails():
    """Declaring no walls on a walled operator: the boxes still tile the
    periodic domain, so geometry passes and every structural check fails --
    a mis-declared --walls cannot produce a PASS."""
    for name in ("walled_conf", "channel_corner_ll"):
        failures, _ = _wall_gate(name, GATE_CX, GATE_CY, walls=(), expect_wall_sides=None,
                                 expect_wrap_mortar_halves=None, expect_mortar_halves=None)
        assert {"conservation", "consistency", "exactness", "far_rows", "spectrum"} <= set(failures), (name, failures)


def test_periodic_operator_judged_as_walled_fails():
    """The converse: the periodic analytic operator carries wrap coupling and
    no wall term, so under --walls xy it fails support (coupling across a
    declared wall) and the wall checks."""
    x, y = nodes_from_boxes(boxes_conforming())
    A = build_analytic_operator_2d(GATE_NOP, GATE_NEX, GATE_NEY, GATE_CX, GATE_CY)
    failures, _ = gate(A, x, y, walls="xy", expect_wall_sides=16)
    assert "support" in failures and "conservation" in failures and "consistency" in failures
