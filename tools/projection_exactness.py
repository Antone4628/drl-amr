"""
Test: Is scatter projection exact for degree-p polynomials?

Initialize the Gaussian IC (icase=1) on the 4-element base mesh [-1, -0.4, 0, 0.4, 1].
Plot the solution (dashed). Refine all 4 elements via scatter projection (no time-stepping).
Plot the refined solution on top (solid). If scatter is exact, the curves overlap perfectly.

Also: evaluate the parent Lagrange interpolants at the children's physical node locations
and compare directly against the scattered values. This isolates the scatter operation from
any grid-rebuild or matrix-update effects.

Run from project root:
    python -m tests.multiround_buildout.test_projection_exactness
    
Or directly (with path setup):
    python tests/multiround_buildout/test_projection_exactness.py
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

from backends.python_1d.dg.basis import Lagrange_basis
from backends.python_1d.solvers.dg_advection_solver_multiround import DGAdvectionSolver


def main():
    # =========================================================================
    # 1. Initialize solver on 4-element base mesh
    # =========================================================================
    xelem = np.array([-1.0, -0.4, 0.0, 0.4, 1.0])
    solver = DGAdvectionSolver(
        nop=4,
        xelem=xelem,
        max_elements=64,
        max_level=4,
        courant_max=0.1,
        icase=1,          # Gaussian pulse
        periodic=True,
        verbose=False,
        balance=False
    )

    ngl = solver.ngl  # 5 LGL nodes per element

    # =========================================================================
    # 2. Capture pre-refinement state
    # =========================================================================
    pre_coord = solver.coord.copy()
    pre_q = solver.q.copy()
    pre_active = solver.active.copy()
    pre_nelem = solver.nelem
    pre_intma = solver.intma.copy()
    pre_xelem = solver.xelem.copy()
    xgl = solver.xgl.copy()

    print("=" * 70)
    print("SCATTER PROJECTION EXACTNESS TEST")
    print("=" * 70)
    print(f"\nPre-refinement: {pre_nelem} elements, {len(pre_q)} DOFs")
    print(f"Element boundaries: {pre_xelem}")
    print(f"Active element IDs: {pre_active}")

    # Store per-element data for later interpolation check
    parent_data = []
    for i in range(pre_nelem):
        nodes = pre_intma[:, i]
        parent_data.append({
            'x': pre_coord[nodes].copy(),
            'u': pre_q[nodes].copy(),
            'x_left': pre_xelem[i],
            'x_right': pre_xelem[i + 1],
        })

    # =========================================================================
    # 3. Refine all 4 elements (scatter projection, no time-stepping)
    # =========================================================================
    # Refine all base elements via the single-element primitive, high index to
    # low so positions don't shift; children of parent p land at 2p, 2p+1.
    for i in reversed(range(pre_nelem)):
        solver.refine_element(i)

    post_coord = solver.coord.copy()
    post_q = solver.q.copy()
    post_nelem = solver.nelem
    post_intma = solver.intma.copy()
    post_xelem = solver.xelem.copy()

    print(f"\nPost-refinement: {post_nelem} elements, {len(post_q)} DOFs")
    print(f"Element boundaries: {post_xelem}")

    # =========================================================================
    # 4. Direct interpolation check (bypasses grid rebuild entirely)
    #    For each parent, evaluate its Lagrange interpolant at child node
    #    physical locations, compare against scattered child values.
    # =========================================================================
    print("\n" + "-" * 70)
    print("DIRECT INTERPOLATION CHECK")
    print("(Evaluate parent polynomial at child node locations vs scattered values)")
    print("-" * 70)

    max_diff_overall = 0.0

    for p_idx in range(pre_nelem):
        p = parent_data[p_idx]
        p_u = p['u']        # parent's 5 nodal values
        p_left = p['x_left']
        p_right = p['x_right']
        p_width = p_right - p_left

        # The two children of this parent are at post-refinement indices 2*p_idx, 2*p_idx+1
        for c_local in range(2):
            c_idx = 2 * p_idx + c_local
            c_nodes = post_intma[:, c_idx]
            c_x_phys = post_coord[c_nodes]   # child's physical node locations
            c_u_scatter = post_q[c_nodes]     # child's scattered values

            # Map child physical locations into parent's reference [-1, 1]
            c_x_ref = 2.0 * (c_x_phys - p_left) / p_width - 1.0

            # Evaluate parent's Lagrange interpolant at these reference locations
            psi_at_child, _ = Lagrange_basis(ngl, ngl, xgl, c_x_ref)
            u_interp = psi_at_child.T @ p_u  # (5,) interpolated values

            diff = np.abs(c_u_scatter - u_interp)
            max_diff = np.max(diff)
            max_diff_overall = max(max_diff_overall, max_diff)

            label = "left" if c_local == 0 else "right"
            print(f"  Parent {p_idx} -> {label} child (elem {c_idx}): "
                  f"max|scatter - interp| = {max_diff:.2e}")

    print(f"\n  >>> OVERALL MAX DIFFERENCE: {max_diff_overall:.2e}")
    if max_diff_overall < 1e-13:
        print("  >>> VERDICT: Scatter projection is EXACT (to machine precision)")
    elif max_diff_overall < 1e-10:
        print("  >>> VERDICT: Scatter projection has SMALL errors (likely floating point)")
    else:
        print("  >>> VERDICT: Scatter projection has SIGNIFICANT errors — investigate!")

    # =========================================================================
    # 5. Plot: pre-refinement (dashed) vs post-refinement (solid)
    # =========================================================================
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})

    # --- Top panel: solution overlay ---
    ax = axes[0]

    # Plot pre-refinement solution element by element (dashed, thick)
    for i in range(pre_nelem):
        nodes = pre_intma[:, i]
        x_elem = pre_coord[nodes]
        u_elem = pre_q[nodes]
        label = 'Pre-refinement (4 elements)' if i == 0 else None
        ax.plot(x_elem, u_elem, 'b--', linewidth=2.5, label=label)

    # Plot post-refinement solution element by element (solid, thinner)
    for i in range(post_nelem):
        nodes = post_intma[:, i]
        x_elem = post_coord[nodes]
        u_elem = post_q[nodes]
        label = 'Post-refinement (8 elements, scatter only)' if i == 0 else None
        ax.plot(x_elem, u_elem, 'r-', linewidth=1.2, label=label, alpha=0.9)

    # Mark element boundaries
    for xb in pre_xelem:
        ax.axvline(xb, color='blue', linestyle=':', alpha=0.3, linewidth=0.8)
    for xb in post_xelem:
        ax.axvline(xb, color='red', linestyle=':', alpha=0.2, linewidth=0.5)

    # Mark pre-refinement LGL nodes
    ax.scatter(pre_coord, pre_q, color='blue', s=30, zorder=5, alpha=0.7)
    # Mark post-refinement LGL nodes
    ax.scatter(post_coord, post_q, color='red', s=15, zorder=5, marker='x', alpha=0.7)

    ax.set_ylabel('u(x)', fontsize=12)
    ax.set_title(f'Scatter Projection Exactness Test (icase=1, p=4)\n'
                 f'Max pointwise difference: {max_diff_overall:.2e}', fontsize=13)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3)

    # --- Bottom panel: pointwise difference at child nodes ---
    ax2 = axes[1]

    for p_idx in range(pre_nelem):
        p = parent_data[p_idx]
        p_u = p['u']
        p_left = p['x_left']
        p_right = p['x_right']
        p_width = p_right - p_left

        for c_local in range(2):
            c_idx = 2 * p_idx + c_local
            c_nodes = post_intma[:, c_idx]
            c_x_phys = post_coord[c_nodes]
            c_u_scatter = post_q[c_nodes]

            c_x_ref = 2.0 * (c_x_phys - p_left) / p_width - 1.0
            psi_at_child, _ = Lagrange_basis(ngl, ngl, xgl, c_x_ref)
            u_interp = psi_at_child.T @ p_u

            diff = np.abs(c_u_scatter - u_interp)
            ax2.semilogy(c_x_phys, diff, 'ko-', markersize=4)

    ax2.set_xlabel('x', fontsize=12)
    ax2.set_ylabel('|scatter - interp|', fontsize=12)
    ax2.set_title('Pointwise difference: scattered values vs parent interpolant', fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(1e-15, color='green', linestyle='--', alpha=0.5, label='machine epsilon')
    ax2.legend(fontsize=9)

    plt.tight_layout()

    # Save
    out_path = os.path.join(os.path.dirname(__file__), 'projection_exactness.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved to: {out_path}")
    plt.close()

    return max_diff_overall


if __name__ == '__main__':
    max_diff = main()
    sys.exit(0 if max_diff < 1e-10 else 1)
