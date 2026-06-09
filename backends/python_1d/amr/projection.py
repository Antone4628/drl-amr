"""Projection operators for h-adaptive mesh refinement.

This module creates scatter and gather matrices for projecting DG solutions
between parent and child elements during mesh adaptation. The projection
preserves polynomial solutions exactly up to the DG polynomial order.

Key Functions:
    projections: Main entry point — creates all scatter/gather matrices.
    create_S_matrix: Build integration matrices for parent-child projection.
    create_scatters: Build scatter operators (parent → two children).
    create_gathers: Build gather operators (two children → parent).
    create_zz_patch_projection: ZZ-style patch projection for error estimation (D-032).

Mathematical Background:
    When refining an element, the parent solution must be "scattered" to two
    child elements. This is done via L2 projection:
    
        u_child = M⁻¹ S u_parent
    
    where M is the mass matrix and S is the projection integral matrix.
    
    For coarsening, two child solutions are "gathered" to form the parent:
    
        u_parent = 0.5 * M⁻¹ (S1ᵀ u_child1 + S2ᵀ u_child2)
    
    The 0.5 factor accounts for the domain size change (each child covers
    half the parent's domain in reference coordinates).

Note:
    All matrices operate on the reference element [-1, 1]. Child 1 maps to
    [-1, 0] and child 2 maps to [0, 1] in the parent's reference space.
"""

import numpy as np

from ..dg.basis import Lagrange_basis


def S_psi(P, Q, xlgl, xs, c):
    """Evaluate Lagrange basis functions mapped from child to parent reference space.
    
    Computes basis functions defined on the parent element [-1, 1] evaluated
    at quadrature points mapped from the child element. This is the core
    building block for parent-child projection integrals.
    
    The mapping from parent ξ to child ζ coordinates is:
        Child 1 (left):  ζ = 0.5*ξ - 0.5  (maps [-1,1] → [-1,0])
        Child 2 (right): ζ = 0.5*ξ + 0.5  (maps [-1,1] → [0,1])
    
    Args:
        P: Number of interpolation points (polynomial order + 1).
        Q: Number of quadrature points for integration.
        xlgl: LGL interpolation nodes on reference element [-1, 1], shape (P,).
        xs: Quadrature points on reference element [-1, 1], shape (Q,).
        c: Child number: 1 for left child, 2 for right child.
        
    Returns:
        tuple: (psi, dpsi)
            psi: Basis function values, shape (P, Q). psi[i, l] = Lᵢ(ζ(xs[l])).
            dpsi: Basis function derivatives, shape (P, Q). Included for API
                consistency with Lagrange_basis.
    
    See Also:
        Lagrange_basis: Standard basis evaluation without coordinate mapping.
    """

    psi = np.zeros([P,Q]) # PxQ matrix
    dpsi = np.zeros([P,Q])


    for l in range(Q):
        xl = xs[l]
        if c == 1:
            zl = 0.5*xl - 0.5
        elif c == 2:
            zl = 0.5*xl + 0.5

        for i in range(P):
            xi = xlgl[i]
            zi = xi

            psi[i][l]=1
            dpsi[i][l]=0

            for j in range(P):
                xj = xlgl[j]
                zj = xj

                if(i != j):
                    psi[i][l]=psi[i][l]*((zl-zj)/(zi-zj))

                ddpsi=1

                if(i!=j):
                    for k in range(P):
                        xk=xlgl[k]
                        zk = xk
                        
                        if(k!=i and k!=j):
                            ddpsi=ddpsi*((zl-zk)/(zi-zk))

                    dpsi[i][l]=dpsi[i][l]+(ddpsi/(zi-zj))

    return psi, dpsi



def create_S_matrix(ngl, nq, wnq, xgl, xnq):
    """Create projection integral matrices for parent-child element mapping.
    
    Computes the matrices S1 and S2 used in the L2 projection between parent
    and child elements. These matrices satisfy:
    
        S[i,j] = ∫₋₁¹ ψᵢ(ξ) φⱼ(ξ) dξ
    
    where ψ are basis functions on the parent and φ are basis functions on
    the child (mapped to parent coordinates).
    
    Args:
        ngl: Number of LGL points per element (polynomial order + 1).
        nq: Number of quadrature points for integration.
        wnq: Quadrature weights, shape (nq,).
        xgl: LGL node positions in reference element [-1, 1], shape (ngl,).
        xnq: Quadrature points in reference element [-1, 1], shape (nq,).
        
    Returns:
        tuple: (S1, S2)
            S1: Projection integral matrix for child 1 (left half), shape (ngl, ngl).
            S2: Projection integral matrix for child 2 (right half), shape (ngl, ngl).
    
    Note:
        The matrices are computed via numerical quadrature:
            S[i,j] = Σₖ wₖ ψᵢ(xₖ) φⱼ(xₖ)
        
        These matrices are combined with the mass matrix inverse to form
        the final scatter/gather operators.
    """

    #get psi1 and psi 2
    psi1, dpsi1 = S_psi(ngl, nq, xgl, xnq, 1)
    psi2, dpsi2 = S_psi(ngl, nq, xgl, xnq, 2)
    psi, dpsi = Lagrange_basis(ngl, nq, xgl, xnq)

    #initialize element S matrix
    S1 = np.zeros((ngl,ngl))
    S2 = np.zeros((ngl,ngl))

    #     #Do LGL integration
    for k in range(nq):
        wk=wnq[k]

        for i in range(ngl):
            h_i1=psi[i][k]
            h_i2=psi[i][k]
            for j in range(ngl):
                h_j1=psi1[j][k]
                h_j2=psi2[j][k]
                S1[i][j] = S1[i][j] + wk*h_i1*h_j1
                S2[i][j] = S2[i][j] + wk*h_i2*h_j2


    ''' returns two arrays with dimensions (ngl,ngl) '''
    return S1, S2


def create_scatters(M, S1, S2):
    """Create scatter operators for projecting parent solution to children.
    
    Scatter operators project a solution from a parent element onto its
    two child elements during mesh refinement. The projection is exact
    for polynomials up to the DG order.
    
    Args:
        M: Reference mass matrix on [-1, 1], shape (ngl, ngl).
        S1: Projection integral matrix for child 1, shape (ngl, ngl).
        S2: Projection integral matrix for child 2, shape (ngl, ngl).
        
    Returns:
        tuple: (PS1, PS2)
            PS1: Scatter operator for child 1. Usage: u_child1 = PS1 @ u_parent.
            PS2: Scatter operator for child 2. Usage: u_child2 = PS2 @ u_parent.
    
    Note:
        The scatter operator is: PS = M⁻¹ S
        
        This performs L2 projection of the parent solution onto the child's
        polynomial basis.
    """

    Minv = np.linalg.inv(M)
    PS1 = np.matmul(Minv, S1)
    PS2 = np.matmul(Minv, S2)

    return PS1, PS2

def create_gathers(M, S1, S2):
    """Create gather operators for projecting children solutions to parent.

    Gather operators combine solutions from two child elements into a
    single parent element during mesh coarsening. The projection is
    exact for polynomials up to the DG order.

    Args:
        M: Reference mass matrix on [-1, 1], shape (ngl, ngl).
        S1: Projection integral matrix for child 1, shape (ngl, ngl).
        S2: Projection integral matrix for child 2, shape (ngl, ngl).
        
    Returns:
        tuple: (PG1, PG2)
            PG1: Gather operator for child 1.
            PG2: Gather operator for child 2.

    Note:
        Usage: u_parent = PG1 @ u_child1 + PG2 @ u_child2
        
        The gather operator is: PG = 0.5 * M⁻¹ Sᵀ
        
        The 0.5 factor accounts for the Jacobian change: each child element
        has half the size of the parent in reference coordinates.
    """
    s = 0.5
    Minv = np.linalg.inv(M)
    PG1 = s*np.matmul(Minv, S1.T)
    PG2 = s*np.matmul(Minv, S2.T)

    return PG1, PG2


def projections(RM, ngl, nq, wnq, xgl, xnq):
    """Create all projection operators for h-adaptive mesh refinement.
    
    Main entry point for creating scatter and gather matrices used during
    mesh adaptation. Call this once during solver initialization; the
    returned matrices are mesh-independent and can be reused.
    
    Args:
        RM: Reference mass matrix on [-1, 1], shape (ngl, ngl).
        ngl: Number of LGL points per element (polynomial order + 1).
        nq: Number of quadrature points for integration.
        wnq: Quadrature weights, shape (nq,).
        xgl: LGL node positions in reference element [-1, 1], shape (ngl,).
        xnq: Quadrature points in reference element [-1, 1], shape (nq,).
        
    Returns:
        tuple: (PS1, PS2, PG1, PG2)
            PS1: Scatter matrix for child 1. Usage: u_child1 = PS1 @ u_parent.
            PS2: Scatter matrix for child 2. Usage: u_child2 = PS2 @ u_parent.
            PG1: Gather matrix for child 1.
            PG2: Gather matrix for child 2.
    
    Note:
        Scatter matrices project parent → children during refinement.
        Gather matrices project children → parent during coarsening.
        
        Usage in adapt_sol:
            Refinement: child1_vals = PS1 @ parent_vals
                        child2_vals = PS2 @ parent_vals
            Coarsening: parent_vals = PG1 @ child1_vals + PG2 @ child2_vals
    
    Example:
        >>> from numerical.dg.basis import lgl_gen, Lagrange_basis
        >>> from numerical.dg.matrices import create_RM_matrix
        >>> ngl = 5  # Polynomial order 4
        >>> xgl, wgl = lgl_gen(ngl)
        >>> psi, _ = Lagrange_basis(ngl, ngl, xgl, xgl)
        >>> RM = create_RM_matrix(ngl, ngl, wgl, psi)
        >>> PS1, PS2, PG1, PG2 = projections(RM, ngl, ngl, wgl, xgl, xgl)
    """

    S1, S2 = create_S_matrix(ngl, nq, wnq, xgl, xnq)
    #create scattters
    PS1, PS2 = create_scatters(RM, S1, S2)
    #create gathers
    PG1, PG2 = create_gathers(RM, S1, S2)


    return PS1, PS2, PG1, PG2


# =============================================================================
# ZZ-Style Patch Projection (D-032)
# =============================================================================

def create_zz_patch_projection(h_left, h_right, ngl, nq, wnq, xgl, xnq):
    """Create the L²-projection operator for a two-element ZZ patch.
    
    Given two adjacent elements of physical widths h_left and h_right, returns
    the matrix P that maps their stacked nodal values to the L²-best-fit
    single degree-(ngl-1) polynomial across the patch:
    
        v_coeffs = P @ np.concatenate([u_left, u_right])
    
    where v_coeffs are the patch polynomial's nodal values at the patch's
    own LGL nodes (on patch reference [-1, 1]). This is the projection
    operator at the heart of the ZZ-style error estimator (D-032,
    `zz_estimator_method.tex`).
    
    Mathematical derivation:
        Find v ∈ P_p(R) minimizing ∫_R (u_h − v)² dx, where R = Ω_left ∪ Ω_right
        and u_h is piecewise on the two elements. The variational equation is
        
            M_patch · v_coeffs = S_left · u_left + S_right · u_right
        
        with M_patch = (h_p/2)·∫₋₁¹ Lₐ^patch Lᵦ^patch dξ and
        (S_e)_{a,b} = ∫_{Ω_e} φ_a^patch ψ_b^(e) dx. Both integrands are
        degree 2(ngl−1) in their reference variable, so (xnq, wnq) must
        provide LGL exactness up to that degree: 2nq−3 ≥ 2(ngl−1), i.e.,
        nq ≥ ngl+1. For ngl=5 use nq=7.
    
    Args:
        h_left: Physical width of the left element of the patch.
        h_right: Physical width of the right element.
        ngl: Number of LGL points per element (polynomial order + 1).
        nq: Number of quadrature points. Must satisfy nq ≥ ngl + 1.
        wnq: Quadrature weights, shape (nq,).
        xgl: LGL interpolation nodes in reference [-1, 1], shape (ngl,).
        xnq: Quadrature points in reference [-1, 1], shape (nq,).
    
    Returns:
        P: Patch projection matrix, shape (ngl, 2*ngl). Apply via
           v_coeffs = P @ np.concatenate([u_left, u_right]).
    
    Note:
        Computed on-the-fly per patch since the base mesh is non-uniform
        ([-1, -0.4, 0, 0.4, 1]) and refinement produces many distinct
        width ratios. Cost is dominated by a single (ngl × ngl) inverse —
        microseconds per call, negligible relative to the DG advance.
    
    See Also:
        projections: Parent↔child projection operators (different geometry,
            same M⁻¹S construction pattern, same quadrature API).
        numerical.solvers.error_indicators.compute_element_errors_zz:
            Consumer of this projection.
    """
    h_p = h_left + h_right
    
    # Map element-ref quadrature points to patch-ref coords for each side
    xi_left  = (1.0 + xnq) * h_left  / h_p - 1.0
    xi_right = (2.0 * h_left + (1.0 + xnq) * h_right) / h_p - 1.0
    
    # Element basis at element-ref quadrature: psi_elem[b, q] = L_b(xnq[q]).
    # This also doubles as the patch basis at patch-ref quadrature, since
    # both bases are LGL Lagrange on [-1, 1] and we use xnq as the
    # quadrature in both frames.
    psi_elem, _ = Lagrange_basis(ngl, nq, xgl, xnq)
    
    # Patch basis at the corresponding patch-ref quadrature points (per side):
    # psi_patch_e[a, q] = L_a^patch(xi_e(xnq[q])).
    psi_patch_left,  _ = Lagrange_basis(ngl, nq, xgl, xi_left)
    psi_patch_right, _ = Lagrange_basis(ngl, nq, xgl, xi_right)
    
    Wnq = wnq[np.newaxis, :]  # broadcast helper
    
    # Patch mass matrix (consistent, integrated with the supplied quadrature):
    # M_patch[a, b] = (h_p/2) ∫₋₁¹ L_a^patch L_b^patch dξ
    M_patch = (h_p / 2.0) * ((psi_elem * Wnq) @ psi_elem.T)
    
    # Cross-mass matrices: integrate patch basis × element basis over each element.
    # S_e[a, b] = (h_e/2) Σ_q wnq[q] · L_a^patch(xi_e(xnq[q])) · L_b(xnq[q])
    S_left  = (h_left  / 2.0) * (psi_patch_left  * Wnq) @ psi_elem.T
    S_right = (h_right / 2.0) * (psi_patch_right * Wnq) @ psi_elem.T
    
    # Combined projection: v_coeffs = M_patch⁻¹ (S_left u_left + S_right u_right)
    M_inv = np.linalg.inv(M_patch)
    P = M_inv @ np.hstack([S_left, S_right])  # shape (ngl, 2*ngl)
    
    return P

