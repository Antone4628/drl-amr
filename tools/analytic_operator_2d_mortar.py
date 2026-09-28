"""Spec-built 2D collocated DG operator on a mesh with 2:1 (mortar) faces.

The 2:1 sibling of tools/analytic_operator_2d.py. Constructs the exact
semi-discrete operator L (du/dt = L u) for

    du/dt + cx du/dx + cy du/dy = 0

with collocated (DGSEM / lumped-mass) nodal DG and an upwind flux, on any
mesh of axis-aligned boxes whose faces are conforming or 2:1, directly from
the five-step mortar specification of the DG roadmap (section 5.17, "Mortar
flux specification"):

  per parent face, for each child half h:
    1. scatter the parent trace:  qS = interp[h] qP
    2. upwind flux at the child nodes, L = child, R = scattered parent,
       normal n pointing child -> parent:  F*_h = aL qC + aR qS
    3. child correction:   rhs[C] += w_k Jfc (cn qC - F*_h) / M
  then once per parent face:
    4. gather:             F*_p = sum_h project[h] F*_h
    5. parent correction:  rhs[P] -= w_k Jfp (cn qP - F*_p) / M
       with the parent's OWN trace flux cn qP.

Conforming faces and the volume term are the construction of
analytic_operator_2d.py, per element with that element's own Jacobians.
The face operators are built as Jexpresso's build_projection_1d builds them:
interp[h] is barycentric Lagrange interpolation from the parent's LGL nodes
onto child half h (h = 0 is Jexpresso's half 1, the upper half of the parent
trace; h = 1 its half 2), and project[h] = 1/2 Mex^-1 interp[h]^T Mex with
Mex the exact LGL-nodal mass matrix (Gauss-Legendre, ngl points).

Uses: check [10] of tools/gate_2d_mortar.py (informational entrywise
comparison), and the fixture of tests/dg_2d/test_gate_2d_mortar.py.

Independence caveats -- read before trusting agreement:
  - This is one reading of the same specification Jexpresso's mortar loop
    implements. A misreading of the SPEC shared by both would agree here;
    the structural checks [5]-[9] of the gate do not depend on the spec's
    details and are the defence against that.
  - Face contacts come from gate_2d_mortar.Mesh2, the instrument's own
    topology derivation; its tests pin that topology to hand-counted values
    on the gate mesh.
  - The basis comes from backends.python_1d.dg.basis (see the caveat in
    analytic_operator_2d.py; F1-verified).
"""

import numpy as np

from backends.python_1d.dg.basis import Lagrange_basis, lgl_gen
from tools.analytic_operator_2d import lgl_and_diff


def projection_1d(nop):
    """(interp, project): lists indexed h = 0 (upper half, Jexpresso half 1)
    and h = 1 (lower half, Jexpresso half 2). interp[h] maps parent nodal
    values to child-half nodal values; project[h] maps child-half nodal
    values back to the parent (the L2 projection, carrying the 1/2)."""
    ngl = nop + 1
    xgl, _ = lgl_gen(ngl)
    xg, wg = np.polynomial.legendre.leggauss(ngl)
    psi_g, _ = Lagrange_basis(ngl, ngl, xgl, xg)
    mex = psi_g @ np.diag(wg) @ psi_g.T
    interp, project = [], []
    for shift in (+1.0, -1.0):
        psi, _ = Lagrange_basis(ngl, ngl, xgl, (xgl + shift) / 2.0)
        i_h = psi.T
        interp.append(i_h)
        project.append(np.linalg.solve(mex, i_h.T @ mex) / 2.0)
    return interp, project


def geometric_mass(mesh):
    """Diagonal DG mass w_i w_j Jvol(e), in the mesh's DOF order."""
    return np.concatenate([
        np.kron(mesh.wgl, mesh.wgl) * (b[1] - b[0]) * (b[3] - b[2]) / 4.0
        for b in mesh.box])


def build_spec_operator_2d(mesh, cx, cy, _mutate=None):
    """Assemble the operator on a gate_2d_mortar.Mesh2 (element boxes and
    contacts derived from a node dump), in that mesh's element and DOF order.

    _mutate is a TEST HOOK, never used by the gate: "wrong_half" (the other
    half's interp and project, consistently on both sides), "double_half"
    (project without its 1/2), "uncoupled" (mortar faces skipped), "central"
    (lambda = 0 in the mortar flux only)."""
    nop, ngl = mesh.nop, mesh.ngl
    _, wgl, D = lgl_and_diff(nop)
    interp, project = projection_1d(nop)
    n = mesh.nelem * ngl * ngl
    M = geometric_mass(mesh)
    L = np.zeros((n, n))

    for e in range(mesh.nelem):
        xa, xb, ya, yb = mesh.box[e]
        jx, jy = (xb - xa) / 2.0, (yb - ya) / 2.0
        for j in range(ngl):
            for i in range(ngl):
                r = mesh.dof(e, i, j)
                for m in range(ngl):
                    L[r, mesh.dof(e, m, j)] += -(cx / jx) * D[i, m]
                    L[r, mesh.dof(e, i, m)] += -(cy / jy) * D[j, m]

    def coeffs(cn):
        return 0.5 * cn + 0.5 * abs(cn), 0.5 * cn - 0.5 * abs(cn)

    parents = {}
    for c in mesh.contacts:
        ax = c["axis"]
        cn = cx if ax == 0 else cy
        hi_side, lo_side = (1, 0) if ax == 0 else (3, 2)
        a, b = c["a"], c["b"]
        if c["kind"] == "conforming":
            ta, tb = mesh.trace(a, hi_side), mesh.trace(b, lo_side)
            jf = (c["t"][1] - c["t"][0]) / 2.0
            al, ar = coeffs(cn)
            for k in range(ngl):
                ra, rb = ta[k], tb[k]
                # L side a: + w Jf (cn qa - F*);  R side b: - w Jf (cn qb - F*)
                L[ra, ra] += wgl[k] * jf * (cn - al) / M[ra]
                L[ra, rb] += wgl[k] * jf * (-ar) / M[ra]
                L[rb, rb] += -wgl[k] * jf * (cn - ar) / M[rb]
                L[rb, ra] += -wgl[k] * jf * (-al) / M[rb]
            continue
        if _mutate == "uncoupled":
            continue
        t0, t1 = (2, 3) if ax == 0 else (0, 1)
        la = mesh.box[a][t1] - mesh.box[a][t0]
        lb = mesh.box[b][t1] - mesh.box[b][t0]
        if la > lb:      # parent on the low side, child on the high side
            P, sP, C, sC, cn_cp = a, hi_side, b, lo_side, -cn
        else:            # child on the low side, parent on the high side
            P, sP, C, sC, cn_cp = b, lo_side, a, hi_side, cn
        upper = (0.5 * (mesh.box[C][t0] + mesh.box[C][t1])
                 > 0.5 * (mesh.box[P][t0] + mesh.box[P][t1]))
        parents.setdefault((P, sP), dict(cn_cp=cn_cp, t=(t0, t1), halves=[]))[
            "halves"].append((C, sC, 0 if upper else 1))

    for (P, sP), pf in parents.items():
        if len(pf["halves"]) != 2:
            raise AssertionError(
                f"parent element {P + 1}: {len(pf['halves'])} child halves on one "
                f"side, expected 2")
        cn_cp = pf["cn_cp"]
        if _mutate == "central":
            al, ar = 0.5 * cn_cp, 0.5 * cn_cp
        else:
            al, ar = coeffs(cn_cp)
        t0, t1 = pf["t"]
        tp = mesh.trace(P, sP)
        jfp = (mesh.box[P][t1] - mesh.box[P][t0]) / 2.0
        fstar_p = np.zeros((ngl, n))          # rows: parent face node; cols: DOFs
        for C, sC, h in pf["halves"]:
            hh = 1 - h if _mutate == "wrong_half" else h
            tc = mesh.trace(C, sC)
            jfc = (mesh.box[C][t1] - mesh.box[C][t0]) / 2.0
            fstar = np.zeros((ngl, n))
            for k in range(ngl):
                fstar[k, tc[k]] += al
                for m in range(ngl):
                    fstar[k, tp[m]] += ar * interp[hh][k, m]
            for k in range(ngl):
                r = tc[k]
                L[r, r] += wgl[k] * jfc * cn_cp / M[r]
                L[r] -= wgl[k] * jfc * fstar[k] / M[r]
            proj = project[hh] * (2.0 if _mutate == "double_half" else 1.0)
            fstar_p += proj @ fstar
        for k in range(ngl):
            r = tp[k]
            L[r, r] -= wgl[k] * jfp * cn_cp / M[r]
            L[r] += wgl[k] * jfp * fstar_p[k] / M[r]
    return L
