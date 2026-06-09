"""Basis functions for Discontinuous Galerkin discretization.

This module provides Legendre polynomial evaluation, Legendre-Gauss-Lobatto
(LGL) quadrature nodes and weights, and Lagrange interpolation basis functions.
These are the foundational building blocks for the DG spatial discretization.

Key Functions:
    leg_poly: Evaluate Legendre polynomial and derivatives at a point.
    lgl_gen: Generate LGL quadrature nodes and weights.
    Lagrange_basis: Compute Lagrange basis functions on LGL nodes.

Note:
    All functions operate on the reference element [-1, 1].
"""
import numpy as np


#Legendre-Poly
def leg_poly(p: int, x: float):
    """Evaluate Legendre polynomial and its derivatives at a point.
    
    Uses the three-term recurrence relation to compute P_p(x) and its
    first two derivatives.
    
    Args:
        p: Polynomial order (degree). Must be non-negative.
        x: Evaluation point in [-1, 1].
        
    Returns:
        L0: Legendre polynomial value P_p(x).
        L0_1: First derivative dP_p/dx at x.
        L0_2: Second derivative d²P_p/dx² at x.
    """
    
    L1, L1_1, L1_2 = 0, 0, 0
    L0, L0_1, L0_2 = 1, 0, 0

    for i in range(1,p+1):
        L2, L2_1, L2_2 = L1, L1_1, L1_2
        L1, L1_1, L1_2 = L0, L0_1, L0_2
        a  = (2*i-1)/i
        b  = (i-1)/i
        L0 = a*x*L1 - b*L2
        L0_1 = a*(L1+x*L1_1)-b*L2_1
        L0_2 = a*(2*L1_1+x*L1_2) - b*L2_2

    return L0, L0_1, L0_2
    

### Routine for generating Legendre-Gauss_Lobatto points
def lgl_gen(P: int):
    """Generate Legendre-Gauss-Lobatto quadrature nodes and weights.
    
    LGL nodes are the roots of (1-x²)P'_{P-1}(x), which always include
    the endpoints x = ±1. Used for both interpolation and integration.
    
    Args:
        P: Number of nodes (polynomial order + 1). Must be >= 2.
        
    Returns:
        lgl_nodes: LGL node locations in [-1, 1], shape (P,).
        lgl_weights: Corresponding quadrature weights, shape (P,).
    
    Note:
        Nodes are computed via Newton iteration on the Legendre polynomial.
        Weights satisfy: ∫₋₁¹ f(x)dx ≈ Σᵢ wᵢ f(xᵢ) (exact for polynomials
        of degree ≤ 2P-3).
    """
    # P is number of interpolation nodes. (P = order + 1)
    p = P-1 #Poly order
    ph = int(np.floor( (p+1)/2.0 ))

    lgl_nodes   = np.zeros(P)
    lgl_weights = np.zeros(P)

    for i in range(1,ph+1):
        x = np.cos((2*i-1)*np.pi/(2*p+1))

        for _k in range (1,21):
            L0,L0_1,L0_2 = leg_poly(p,x)

            dx = -((1-x**2)*L0_1)/(-2*x*L0_1 + (1-x**2)*L0_2)
            x = x+dx

            if(abs(dx)<1.0e-20):
                break

        lgl_nodes[p+1-i]=x
        lgl_weights[p+1-i]=2/(p*(p+1)*L0**2)

    #Check for Zero root
    if(p+1 != 2*ph):
        x = 0
        L0, dum, dumm = leg_poly(p,x)
        lgl_nodes[ph] = x
        lgl_weights[ph] = 2/(p*(p+1)*L0**2)

    #Find remainder of roots via symmetry
    for i in range(1,ph+1):
        lgl_nodes[i-1] = -lgl_nodes[p+1-i]
        lgl_weights[i-1] =  lgl_weights[p+1-i]

    return lgl_nodes,lgl_weights

#Lagrange basis
def Lagrange_basis(P: int, Q: int, xlgl, xs):
    """Compute Lagrange basis functions and derivatives at quadrature points.
    
    Evaluates the P Lagrange interpolating polynomials (defined on xlgl nodes)
    at Q quadrature points xs. Used for interpolation and differentiation
    in the DG formulation.
    
    Args:
        P: Number of interpolation points (basis functions).
        Q: Number of quadrature points to evaluate at.
        xlgl: Interpolation nodes (typically LGL), shape (P,).
        xs: Evaluation points (quadrature nodes), shape (Q,).
        
    Returns:
        psi: Basis function values, shape (P, Q). psi[i,l] = Lᵢ(xs[l]).
        dpsi: Basis function derivatives, shape (P, Q). dpsi[i,l] = dLᵢ/dx(xs[l]).
    
    Note:
        Lagrange basis Lᵢ(x) = ∏_{j≠i} (x - xⱼ)/(xᵢ - xⱼ) satisfies
        Lᵢ(xⱼ) = δᵢⱼ (Kronecker delta).
    """
    psi  = np.zeros([P,Q]) # PxQ matrix
    dpsi = np.zeros([P,Q])

    for l in range(Q):
        xl = xs[l]

        for i in range(P):
            xi = xlgl[i]
            psi[i][l]=1
            dpsi[i][l]=0

            for j in range(P):
                xj = xlgl[j]
                if(i != j):
                    psi[i][l]=psi[i][l]*((xl-xj)/(xi-xj))
                ddpsi=1
                if(i!=j):
                    for k in range(P):
                        xk=xlgl[k]
                        if(k!=i and k!=j):
                            ddpsi=ddpsi*((xl-xk)/(xi-xk))

                    dpsi[i][l]=dpsi[i][l]+(ddpsi/(xi-xj))

    return psi, dpsi