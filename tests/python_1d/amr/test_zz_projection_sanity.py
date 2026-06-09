"""Phase Z1.2 sanity check for create_zz_patch_projection().

Three checks:
  1. Polynomials of degree ≤ p are projected exactly (residual ~ machine ε).
  2. Higher-degree functions produce nonzero residual.
  3. Smooth non-polynomials show roughly h^(p+1) convergence.

Run from repo root: python tests/multiround_buildout/test_zz_projection_sanity.py
"""

import numpy as np

from backends.python_1d.amr.projection import create_zz_patch_projection
from backends.python_1d.dg.basis import Lagrange_basis, lgl_gen


def patch_l2_residual(P, h_left, h_right, ngl, nq, wnq, xgl, xnq, f):
    """Project f onto the patch and return the L² residual integrated over the patch."""
    h_p = h_left + h_right
    
    # Element LGL physical coords for nodal sampling of f
    x_L_left  = -h_left
    x_L_right = 0.0
    x_left  = x_L_left  + (1.0 + xgl) * h_left  / 2.0
    x_right = x_L_right + (1.0 + xgl) * h_right / 2.0
    
    u_left  = f(x_left)
    u_right = f(x_right)
    
    # Apply projection
    v_coeffs = P @ np.concatenate([u_left, u_right])
    
    # Evaluate u_h (piecewise nodal interpolant) at quadrature points
    psi_elem, _ = Lagrange_basis(ngl, nq, xgl, xnq)
    u_h_left_q  = psi_elem.T @ u_left
    u_h_right_q = psi_elem.T @ u_right
    
    # Evaluate v at the same physical points (via patch ref)
    xi_left  = (1.0 + xnq) * h_left  / h_p - 1.0
    xi_right = (2.0 * h_left + (1.0 + xnq) * h_right) / h_p - 1.0
    psi_patch_left,  _ = Lagrange_basis(ngl, nq, xgl, xi_left)
    psi_patch_right, _ = Lagrange_basis(ngl, nq, xgl, xi_right)
    v_at_left_q  = psi_patch_left.T  @ v_coeffs
    v_at_right_q = psi_patch_right.T @ v_coeffs
    
    # L² residual on each element via higher-order LGL quadrature
    res_left  = np.sum(wnq * (u_h_left_q  - v_at_left_q )**2) * (h_left  / 2.0)
    res_right = np.sum(wnq * (u_h_right_q - v_at_right_q)**2) * (h_right / 2.0)
    
    return np.sqrt(res_left + res_right)


def main():
    ngl = 5
    xgl, wgl = lgl_gen(ngl)
    
    # Higher-order quadrature for cross-mass and residual integration.
    # nq = ngl + 2 = 7 gives LGL exactness to degree 11, comfortably above
    # the degree-8 cross-mass integrand for ngl=5.
    nq = ngl + 2
    xnq, wnq = lgl_gen(nq)
    
    # Test 1: degree-≤4 polynomials project exactly
    print("Test 1: degree ≤ 4 polynomials (expect residual ~ 1e-14)")
    print("-" * 70)
    for h_left, h_right in [(1.0, 1.0), (0.6, 0.4), (0.3, 0.6), (0.6, 0.3)]:
        P = create_zz_patch_projection(h_left, h_right, ngl, nq, wnq, xgl, xnq)
        for deg, f in [(0, lambda x: np.ones_like(x)),
                       (1, lambda x: x),
                       (2, lambda x: x**2),
                       (3, lambda x: x**3),
                       (4, lambda x: x**4)]:
            res = patch_l2_residual(P, h_left, h_right, ngl, nq, wnq, xgl, xnq, f)
            print(f"  h=({h_left},{h_right}), deg={deg}: residual = {res:.2e}")
    
    # Test 2: degree-5 monomial gives nonzero residual
    print("\nTest 2: degree 5 monomial x^5 (expect nonzero residual)")
    print("-" * 70)
    for h_left, h_right in [(1.0, 1.0), (0.6, 0.4)]:
        P = create_zz_patch_projection(h_left, h_right, ngl, nq, wnq, xgl, xnq)
        res = patch_l2_residual(P, h_left, h_right, ngl, nq, wnq, xgl, xnq,
                                 lambda x: x**5)
        print(f"  h=({h_left},{h_right}): residual = {res:.2e}")
    
    # Test 3: convergence with h for smooth Gaussian
    print("\nTest 3: Gaussian f(x) = exp(-50*x²), symmetric patches — expect ~h^5")
    print("-" * 70)
    prev_res = None
    for h in [0.4, 0.2, 0.1, 0.05]:
        P = create_zz_patch_projection(h, h, ngl, nq, wnq, xgl, xnq)
        res = patch_l2_residual(P, h, h, ngl, nq, wnq, xgl, xnq,
                                 lambda x: np.exp(-50.0 * x**2))
        rate = "" if prev_res is None else f"  rate = {np.log2(prev_res/res):.2f}"
        print(f"  h=({h},{h}): residual = {res:.2e}{rate}")
        prev_res = res


if __name__ == "__main__":
    main()