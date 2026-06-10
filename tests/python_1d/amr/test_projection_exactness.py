"""Scatter-projection exactness — assertion regression (RESTRUCTURE Phase 4).

Refining a degree-p element by L2 scatter projection must reproduce the parent
polynomial exactly at the child node locations (refinement adds resolution, not
error). This is the pass/fail core of tools/projection_exactness.py, with the
plotting stripped (2026-06-09 test-convention note: pytest for the invariant,
the diagnostic for the picture).

Run: pytest tests/python_1d/amr/test_projection_exactness.py -v
"""

import numpy as np

from backends.python_1d.dg.basis import Lagrange_basis
from backends.python_1d.solvers.dg_advection_solver_multiround import DGAdvectionSolver

BASE_XELEM = np.array([-1.0, -0.4, 0.0, 0.4, 1.0])


def test_scatter_projection_is_exact():
    solver = DGAdvectionSolver(
        nop=4, xelem=BASE_XELEM.copy(), max_elements=64, max_level=4,
        icase=1, periodic=True, balance=False,
    )
    ngl = solver.ngl
    xgl = solver.xgl.copy()

    # Capture parent (pre-refinement) per-element data.
    pre_nelem = solver.nelem
    pre_q = solver.q.copy()
    pre_intma = solver.intma.copy()
    pre_xelem = solver.xelem.copy()

    parents = []
    for p in range(pre_nelem):
        nodes = pre_intma[:, p]
        parents.append({
            "u": pre_q[nodes].copy(),
            "x_left": pre_xelem[p],
            "x_right": pre_xelem[p + 1],
        })

    # Refine ALL base elements via the single-element primitive, high index to
    # low so positions don't shift. Net layout: children of parent p land at
    # active positions 2p, 2p+1 (same as a simultaneous refine-all).
    for i in reversed(range(pre_nelem)):
        assert solver.refine_element(i)
    assert solver.nelem == 2 * pre_nelem

    post_coord = solver.coord
    post_q = solver.q
    post_intma = solver.intma

    max_diff = 0.0
    for p in range(pre_nelem):
        p_u = parents[p]["u"]
        p_left = parents[p]["x_left"]
        p_width = parents[p]["x_right"] - p_left
        for c_local in range(2):
            c_idx = 2 * p + c_local
            c_nodes = post_intma[:, c_idx]
            c_x_phys = post_coord[c_nodes]
            c_u_scatter = post_q[c_nodes]
            # Map child node locations into the parent reference [-1, 1] and
            # evaluate the parent's degree-p interpolant there.
            c_x_ref = 2.0 * (c_x_phys - p_left) / p_width - 1.0
            psi_at_child, _ = Lagrange_basis(ngl, ngl, xgl, c_x_ref)
            u_interp = psi_at_child.T @ p_u
            max_diff = max(max_diff, float(np.max(np.abs(c_u_scatter - u_interp))))

    assert max_diff < 1e-10, f"scatter projection not exact: max diff = {max_diff:.2e}"