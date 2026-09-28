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
from tools.gate_2d_mortar import Mesh2, run_gate, safe_elements

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
