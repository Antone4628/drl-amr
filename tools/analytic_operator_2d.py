"""Analytic 2D collocated DG semi-discrete operator, built from first principles.

Step-7 gate artifact (DG roadmap Phase 5; D-064 one dimension up, 2026-08).
Constructs the exact semi-discrete operator L (du/dt = L u) for

    du/dt + cx du/dx + cy du/dy = 0

discretized by collocated (DGSEM / lumped-mass) nodal DG with an upwind
numerical flux on a uniform, doubly periodic, tensor-product quad mesh.
The extracted Jexpresso 2D operator is compared entrywise against this
construction by tools/gate_2d_compare.py.

Construction (strong form, per element, local nodes (i, j), i along x):

  volume:   du_{ij}/dt = -(cx/Jx) sum_m D[i,m] u_{mj}
                         -(cy/Jy) sum_m D[j,m] u_{im}

  faces:    at face node k, the element on the L side of a face with unit
            normal n (pointing L -> R) accumulates

                rhs_el += + w_k * J_face * (Fn_int,L - Fn*)

            and the R side accumulates

                rhs_el += - w_k * J_face * (Fn_int,R - Fn*)

            with Fn = (c . n) u, the upwind flux in Rusanov form
            Fn* = 0.5 (Fn_L + Fn_R) - 0.5 |c . n| (u_R - u_L),
            and everything divided by the lumped mass M_{ij} = w_i w_j Jvol.

  This is the Sec. 5.15(f)/(g) convention: the surface term carries the face
  quadrature weight and face Jacobian (omit the volume mass, keep the face
  measure); the volume mass enters only through the division by M. On a
  Cartesian element the x-face factor collapses to 1/(w_perp * Jx) per grid
  line -- the F1-verified 1D formula -- and the anisotropic gate mesh
  (dx != dy) keeps the two face families' factors distinct on purpose.

Geometry conventions: Jx = dx/2, Jy = dy/2, Jvol = Jx*Jy; an x-normal face
runs along y so J_face_x = Jy, and vice versa.

Element ordering: the operator is assembled in the caller-supplied element
order via `elem_cells` (element index -> (ex, ey) grid cell), so it can be
born directly in Jexpresso's element ordering and compared with no
permutation. DOF ordering within the operator matches the Jexpresso DG
numbering ip = (iel-1)*ngl^2 + (j-1)*ngl + i, zero-based here:
idx = e*ngl^2 + j*ngl + i.

Independence caveat (same as tests/python_1d/dg/test_operator.py): LGL
nodes/weights and the differentiation matrix come from
backends.python_1d.dg.basis, which is shared with the Python solver -- but
the basis itself was verified against the fully independent Julia-side
analytic construction in F1 (2026-07-27), and every defect class this gate
guards (assembly, flux correction, face weighting, periodic pairing, mass
lumping) lives above the basis level.

Self-checks: `python -m tools.analytic_operator_2d` prints the structural
report for the gate configuration. The pytest suite
(tests/dg_2d/test_analytic_operator_2d.py) additionally pins the reduction
of this construction to the F1-verified 1D operator along each axis.
"""

import numpy as np

from backends.python_1d.dg.basis import Lagrange_basis, lgl_gen

# The step-7 gate configuration: 4x4 elements of the conv-family domain.
GATE_DOMAIN = ((-5.0, 5.0), (0.0, 20.0))
GATE_NEX = 4
GATE_NEY = 4
GATE_NOP = 4
GATE_CX = 0.5   # advection_velocity() in problems/AdvDiff/advection2d_dg
GATE_CY = 1.0


def lgl_and_diff(nop):
    """LGL nodes, weights, and the differentiation matrix D[i, m] = psi_m'(xi_i).

    Orientation is self-checked exactly as in tests/python_1d/dg/test_operator.py:
    the basis evaluated at its own nodes must be the identity, and D applied to
    the node vector must differentiate xi to 1 (with the transpose fallback for
    the Lagrange_basis storage convention).
    """
    ngl = nop + 1
    xgl, wgl = lgl_gen(ngl)
    psi, dpsi = Lagrange_basis(ngl, ngl, xgl, xgl)

    assert np.allclose(psi, np.eye(ngl), atol=1e-12), (
        "Lagrange_basis at its own LGL nodes is not the identity -- "
        "orientation or node-generation assumptions are wrong"
    )
    D = dpsi.T
    if not np.allclose(D @ xgl, np.ones(ngl), atol=1e-10):
        D = dpsi
    assert np.allclose(D @ xgl, np.ones(ngl), atol=1e-10), (
        "neither dpsi nor dpsi.T differentiates xi to 1 -- Lagrange_basis "
        "conventions have changed"
    )
    return xgl, wgl, D


def default_elem_cells(nex, ney):
    """Canonical element ordering: ex fastest, e = ey*nex + ex."""
    return [(ex, ey) for ey in range(ney) for ex in range(nex)]


def lumped_mass_vector_2d(nop, nex, ney, domain=GATE_DOMAIN):
    """m[idx] = w_i * w_j * Jvol -- the diagonal of the lumped 2D mass matrix,
    in the idx = e*ngl^2 + j*ngl + i ordering (element-independent block)."""
    ngl = nop + 1
    _, wgl = lgl_gen(ngl)
    (xmin, xmax), (ymin, ymax) = domain
    jvol = ((xmax - xmin) / nex / 2.0) * ((ymax - ymin) / ney / 2.0)
    block = jvol * np.kron(wgl, wgl)   # block[j*ngl + i] = wgl[j] * wgl[i]
    return np.tile(block, nex * ney)


def build_analytic_operator_2d(nop, nex, ney, cx, cy,
                               domain=GATE_DOMAIN, elem_cells=None):
    """Assemble the analytic operator. See the module docstring for the scheme.

    Parameters
    ----------
    nop : polynomial order (ngl = nop + 1 LGL nodes per direction)
    nex, ney : elements per direction (uniform, doubly periodic)
    cx, cy : advection velocity (either sign; upwinding via the Rusanov-form
        flux with lambda = |c . n|, which is exact upwind for linear advection)
    domain : ((xmin, xmax), (ymin, ymax))
    elem_cells : element index -> (ex, ey) grid cell, len nex*ney; defaults to
        the canonical ordering. Pass Jexpresso's ordering (derived from the
        probe's coordinate dump) to assemble directly in its DOF order.

    Returns the dense (n, n) operator, n = nex*ney*ngl^2.

    Note: assembly is accumulation (+=) throughout, so meshes where an
    element is its own periodic neighbour (nex < 3 or ney < 3) are still
    assembled correctly -- but the gate deliberately uses 4x4 so that no
    self-coupling enters (roadmap: a terrible first thing to debug against).
    """
    ngl = nop + 1
    xgl, wgl, D = lgl_and_diff(nop)
    (xmin, xmax), (ymin, ymax) = domain
    dx = (xmax - xmin) / nex
    dy = (ymax - ymin) / ney
    jx = dx / 2.0
    jy = dy / 2.0
    jvol = jx * jy
    jface_x = jy   # x-normal faces run along y
    jface_y = jx   # y-normal faces run along x

    nelem = nex * ney
    if elem_cells is None:
        elem_cells = default_elem_cells(nex, ney)
    assert len(elem_cells) == nelem, "elem_cells must cover every element"
    cell_to_elem = {}
    for e, (ex, ey) in enumerate(elem_cells):
        assert 0 <= ex < nex and 0 <= ey < ney, f"cell out of range: {(ex, ey)}"
        assert (ex, ey) not in cell_to_elem, f"duplicate cell {(ex, ey)}"
        cell_to_elem[(ex, ey)] = e

    n = nelem * ngl * ngl
    L = np.zeros((n, n))

    def dof(e, i, j):
        return e * ngl * ngl + j * ngl + i

    # ---- volume term -------------------------------------------------------
    for e in range(nelem):
        for j in range(ngl):
            for i in range(ngl):
                r = dof(e, i, j)
                for m in range(ngl):
                    L[r, dof(e, m, j)] += -(cx / jx) * D[i, m]
                    L[r, dof(e, i, m)] += -(cy / jy) * D[j, m]

    # ---- face terms --------------------------------------------------------
    # Fn* = aL*uL + aR*uR with the coefficients of the Rusanov-form upwind
    # flux; cn = c . n for the face family's L->R normal.
    #
    # x-normal faces: L = cell (ex, ey), R = cell ((ex+1) % nex, ey), n = (+1, 0).
    cn = cx
    lam = abs(cx)
    a_l = 0.5 * cn + 0.5 * lam
    a_r = 0.5 * cn - 0.5 * lam
    for ey in range(ney):
        for ex in range(nex):
            e_l = cell_to_elem[(ex, ey)]
            e_r = cell_to_elem[((ex + 1) % nex, ey)]
            for k in range(ngl):                    # face node, along y
                r_l = dof(e_l, ngl - 1, k)          # L trace: i = ngl-1
                r_r = dof(e_r, 0, k)                # R trace: i = 0
                s_l = (wgl[k] * jface_x) / (wgl[ngl - 1] * wgl[k] * jvol)
                s_r = (wgl[k] * jface_x) / (wgl[0] * wgl[k] * jvol)
                # L side: + w_k Jf (Fn_L - Fn*) / M
                L[r_l, r_l] += +s_l * (cn - a_l)
                L[r_l, r_r] += +s_l * (-a_r)
                # R side: - w_k Jf (Fn_R - Fn*) / M
                L[r_r, r_r] += -s_r * (cn - a_r)
                L[r_r, r_l] += -s_r * (-a_l)

    # y-normal faces: L = cell (ex, ey), R = cell (ex, (ey+1) % ney), n = (0, +1).
    cn = cy
    lam = abs(cy)
    a_l = 0.5 * cn + 0.5 * lam
    a_r = 0.5 * cn - 0.5 * lam
    for ey in range(ney):
        for ex in range(nex):
            e_l = cell_to_elem[(ex, ey)]
            e_r = cell_to_elem[(ex, (ey + 1) % ney)]
            for k in range(ngl):                    # face node, along x
                r_l = dof(e_l, k, ngl - 1)          # L trace: j = ngl-1
                r_r = dof(e_r, k, 0)                # R trace: j = 0
                s_l = (wgl[k] * jface_y) / (wgl[ngl - 1] * wgl[k] * jvol)
                s_r = (wgl[k] * jface_y) / (wgl[0] * wgl[k] * jvol)
                L[r_l, r_l] += +s_l * (cn - a_l)
                L[r_l, r_r] += +s_l * (-a_r)
                L[r_r, r_r] += -s_r * (cn - a_r)
                L[r_r, r_l] += -s_r * (-a_l)

    return L


def structural_report(L, m):
    """Conservation, consistency, and spectrum of an operator L with lumped
    mass vector m. Returns a dict of floats/ints; assertion policy is the
    caller's (the gate tool and the pytest apply different subsets).

    Zero-mode counting at 2D (established 2026-08-31, sandbox-verified).
    The 1D pin's |lambda| < 1e-6*rho threshold is UNSOUND one dimension up:
    the 2D operator on a uniform tensor mesh is a Kronecker sum, so its
    spectrum is {lam_i + lam_j} over the two 1D sub-spectra -- and each
    conjugate pair contributes lam + conj(lam) = 2*Re(lam) with the
    imaginary parts cancelled exactly. High-order upwind's physical modes
    hug the imaginary axis (Re ~ -theta^(2p+2): ~ -2e-7 for the p=4,
    4-element chain), so those sums land ~4e-7 -- inside 1e-6*rho -- while
    the TRUE kernel (the constant) sits at machine zero (~1e-15). The gate
    configuration makes the cancellation exact because cx/dx = cy/dy makes
    the two 1D sub-operators the same matrix. Hence:
      n_zero_modes      |lambda| < 1e-10*rho  -- the genuine kernel (expect 1)
      n_near_zero_modes |lambda| < 1e-6*rho   -- kernel + conjugate-cancelled
                                                 physical pairs (informational)
    """
    n = L.shape[0]
    lam = np.linalg.eigvals(L)
    rho = float(np.max(np.abs(lam)))
    return {
        "conservation_mTL": float(np.max(np.abs(m @ L))),
        "consistency_L1": float(np.max(np.abs(L @ np.ones(n)))),
        "spectral_radius": rho,
        "max_re_lambda": float(np.max(lam.real)),
        "n_zero_modes": int(np.sum(np.abs(lam) < 1e-10 * rho)),
        "n_near_zero_modes": int(np.sum(np.abs(lam) < 1e-6 * rho)),
        "smallest_abs_lambda": [float(v) for v in np.sort(np.abs(lam))[:6]],
    }


def main():
    """Self-check: print the structural report for the gate configuration."""
    print(f"gate config: nop={GATE_NOP} mesh={GATE_NEX}x{GATE_NEY} "
          f"c=({GATE_CX}, {GATE_CY}) domain={GATE_DOMAIN}")
    L = build_analytic_operator_2d(GATE_NOP, GATE_NEX, GATE_NEY,
                                   GATE_CX, GATE_CY)
    m = lumped_mass_vector_2d(GATE_NOP, GATE_NEX, GATE_NEY)
    rep = structural_report(L, m)
    print(f"n = {L.shape[0]}")
    for k, v in rep.items():
        if isinstance(v, float):
            print(f"  {k} = {v:.6e}")
        elif isinstance(v, list):
            print(f"  {k} = [" + ", ".join(f"{u:.3e}" for u in v) + "]")
        else:
            print(f"  {k} = {v}")


if __name__ == "__main__":
    main()
