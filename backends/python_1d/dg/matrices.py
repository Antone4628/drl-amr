"""Discontinuous Galerkin matrix construction utilities.

This module provides functions for constructing the matrices needed in the
Discontinuous Galerkin (DG) discretization of the 1D advection equation:

    ∂u/∂t + c * ∂u/∂x = 0

where c is the wave speed (advection velocity).

The DG method represents the solution as piecewise polynomials within each
element, using Lagrange interpolation at Legendre-Gauss-Lobatto (LGL) points.
The discretization produces a system:

    M * du/dt = -D * u + F * u

where:
    M: Mass matrix (local integration of basis functions)
    D: Differentiation/stiffness matrix (volume integral of derivatives)
    F: Flux matrix (inter-element coupling via numerical flux)

Key Components:
    create_mass_matrix: Element-wise mass matrices for time integration.
    create_diff_matrix: Element differentiation matrix for spatial derivatives.
    Fmatrix_upwind_flux: Upwind numerical flux for advection stability.
    Matrix_DSS: Direct stiffness summation for global matrix assembly.

Mathematical Background:
    The weak form of the DG method on element e is:
    
    ∫_e φ_i * ∂u/∂t dx = -∫_e φ_i * c * ∂u/∂x dx + [φ_i * c * u*]|_∂e
    
    where φ_i are test functions and u* is the numerical flux at interfaces.
    The upwind flux for positive c is: u* = u^- (value from upwind element).

Note:
    All matrices use 0-based indexing consistent with NumPy conventions.
    Quadrature is performed using LGL points for diagonal mass matrices.

See Also:
    numerical.dg.basis: Lagrange basis function construction at LGL points.
    numerical.solvers.dg_advection_solver: Uses these matrices for time stepping.

References:
    Hesthaven, J.S. and Warburton, T. (2008). Nodal Discontinuous Galerkin
    Methods: Algorithms, Analysis, and Applications. Springer.
"""

import numpy as np

# =============================================================================
# Element Matrices (Local Operations)
# =============================================================================

def create_diff_matrix(
    ngl: int, 
    nq: int, 
    wnq: np.ndarray, 
    psi: np.ndarray, 
    dpsi: np.ndarray
) -> np.ndarray:
    """Create element-wise differentiation matrix using LGL quadrature.
    
    Constructs the local differentiation matrix D_e for a reference element,
    which approximates the spatial derivative integral:
    
        D_ij = ∫_{-1}^{1} (dφ_i/dξ) * φ_j dξ
    
    This matrix is used in the volume integral term of the DG formulation.
    The actual derivative is obtained by scaling with the Jacobian: D/J.
    
    Args:
        ngl: Number of LGL (Legendre-Gauss-Lobatto) points per element.
            Also equals the polynomial degree + 1.
        nq: Number of quadrature points for integration (typically = ngl).
        wnq: Quadrature weights at each quadrature point, shape (nq,).
        psi: Lagrange basis functions evaluated at quadrature points,
            shape (ngl, nq). psi[i, k] = φ_i(ξ_k).
        dpsi: Basis function derivatives at quadrature points,
            shape (ngl, nq). dpsi[i, k] = dφ_i/dξ(ξ_k).
    
    Returns:
        Differentiation matrix on reference element, shape (ngl, ngl).
        Entry [i, j] represents contribution of node j to derivative at node i.
    
    Note:
        For LGL quadrature with nq = ngl, the integration is exact for
        polynomials up to degree 2*ngl - 3. This is sufficient for the
        differentiation matrix since the integrand is degree 2*(ngl-1) - 1.
    
    Example:
        >>> ngl, nq = 4, 4
        >>> xgl, wgl = legendre_gauss_lobatto(ngl)
        >>> psi, dpsi = lagrange_basis(xgl, xgl)
        >>> D = create_diff_matrix(ngl, nq, wgl, psi, dpsi)
        >>> D.shape
        (4, 4)
    """
    # Initialize element derivative matrix
    e_diff = np.zeros((ngl, ngl))
    
    # Numerical quadrature: D_ij = Σ_k w_k * (dφ_i/dξ)_k * (φ_j)_k
    for k in range(nq):
        wk = wnq[k]  # Quadrature weight at point k
        
        for i in range(ngl):  # Row index (test function)
            dhdx_i = dpsi[i][k]  # Derivative of basis function i at point k
            
            for j in range(ngl):  # Column index (trial function)
                h_j = psi[j][k]  # Basis function j at point k
                e_diff[i][j] += wk * dhdx_i * h_j
    
    return e_diff


def create_mass_matrix(
    intma: np.ndarray,
    coord: np.ndarray,
    nelem: int,
    ngl: int,
    nq: int,
    wnq: np.ndarray,
    psi: np.ndarray
) -> np.ndarray:
    """Create element-wise mass matrices using LGL quadrature.
    
    Constructs the local mass matrix M_e for each element, which appears
    in the time derivative term of the DG formulation:
    
        M_ij^e = ∫_{Ω_e} φ_i * φ_j dx = J_e * ∫_{-1}^{1} φ_i * φ_j dξ
    
    where J_e = (x_R - x_L)/2 is the Jacobian of the coordinate mapping.
    
    The mass matrix is symmetric positive definite and diagonal for
    LGL collocation (when nq = ngl and quadrature points = interpolation points).
    
    Args:
        intma: Element-node connectivity matrix, shape (ngl, nelem).
            intma[i, e] gives global node index for local node i in element e.
        coord: Global node coordinates, shape (npoin,).
        nelem: Number of elements in the mesh.
        ngl: Number of LGL points per element.
        nq: Number of quadrature points (typically = ngl).
        wnq: Quadrature weights, shape (nq,).
        psi: Lagrange basis functions at quadrature points, shape (ngl, nq).
    
    Returns:
        Mass matrices for each element, shape (nelem, ngl, ngl).
        Me[e] is the mass matrix for element e.
    
    Note:
        With LGL quadrature where interpolation and quadrature points coincide,
        the mass matrix is diagonal (lumped mass). This significantly speeds
        up time integration since M^{-1} is trivially computed.
    
    Example:
        >>> Me = create_mass_matrix(intma, coord, nelem, ngl, nq, wnq, psi)
        >>> # Check symmetry
        >>> np.allclose(Me[0], Me[0].T)
        True
    """
    # Initialize mass matrices for all elements
    e_mass = np.zeros((nelem, ngl, ngl))
    x = np.zeros(ngl)  # Local coordinate storage
    
    for e in range(nelem):
        # Extract coordinates for this element
        for i in range(ngl):
            I = int(intma[i][e])  # Global node index
            x[i] = coord[I]
        
        # Compute Jacobian of mapping: physical element → reference [-1, 1]
        # For 1D: J = (x_right - x_left) / 2
        dx = x[-1] - x[0]  # Element length
        jac = dx / 2.0     # Jacobian
        
        # Numerical quadrature: M_ij = J * Σ_k w_k * φ_i(ξ_k) * φ_j(ξ_k)
        for k in range(nq):
            wk = wnq[k] * jac  # Scaled quadrature weight
            
            for i in range(ngl):  # Row index
                h_i = psi[i][k]   # φ_i at quadrature point k
                
                for j in range(ngl):  # Column index
                    h_j = psi[j][k]  # φ_j at quadrature point k
                    e_mass[e][i][j] += wk * h_i * h_j
    
    return e_mass


def create_RM_matrix(
    ngl: int,
    nq: int,
    wnq: np.ndarray,
    psi: np.ndarray
) -> np.ndarray:
    """Create reference mass matrix on the canonical element [-1, 1].
    
    Constructs the mass matrix without Jacobian scaling, useful for
    projection operations and basis transformations where the physical
    element size is handled separately.
    
        M_ij^{ref} = ∫_{-1}^{1} φ_i * φ_j dξ
    
    The actual element mass matrix is: M_e = J_e * M^{ref}
    
    Args:
        ngl: Number of LGL points per element.
        nq: Number of quadrature points.
        wnq: Quadrature weights on reference element, shape (nq,).
        psi: Lagrange basis functions at quadrature points, shape (ngl, nq).
    
    Returns:
        Reference mass matrix, shape (ngl, ngl).
    
    Note:
        "RM" stands for "Reference Mass". This matrix is element-independent
        and only needs to be computed once for a given polynomial degree.
    
    See Also:
        create_mass_matrix: Computes physical element mass matrices.
        numerical.amr.projection: Uses reference mass for L2 projection.
    """
    # Initialize reference mass matrix
    r_mass = np.zeros((ngl, ngl))
    
    # Numerical quadrature on reference element (no Jacobian)
    for k in range(nq):
        wk = wnq[k]
        
        for i in range(ngl):
            h_i = psi[i][k]
            
            for j in range(ngl):
                h_j = psi[j][k]
                r_mass[i][j] += wk * h_i * h_j
    
    return r_mass


def create_mass_matrix_vectorized(
    intma: np.ndarray,
    coord: np.ndarray,
    nelem: int,
    ngl: int,
    nq: int,
    wnq: np.ndarray,
    psi: np.ndarray
) -> np.ndarray:
    """Create element-wise mass matrices using vectorized operations.
    
    Functionally equivalent to create_mass_matrix but uses NumPy's
    vectorized operations for improved performance on large meshes.
    
    Args:
        intma: Element-node connectivity matrix, shape (ngl, nelem).
        coord: Global node coordinates, shape (npoin,).
        nelem: Number of elements in the mesh.
        ngl: Number of LGL points per element.
        nq: Number of quadrature points.
        wnq: Quadrature weights, shape (nq,).
        psi: Lagrange basis functions at quadrature points, shape (ngl, nq).
    
    Returns:
        Mass matrices for each element, shape (nelem, ngl, ngl).
    
    Implementation Notes:
        - Uses array slicing to extract coordinates instead of loops
        - Leverages np.outer for vectorized outer product computation
        - Approximately 5-10x faster than loop-based version for large meshes
    
    Example:
        >>> # Time comparison
        >>> %timeit create_mass_matrix(intma, coord, nelem, ngl, nq, wnq, psi)
        >>> %timeit create_mass_matrix_vectorized(intma, coord, nelem, ngl, nq, wnq, psi)
    """
    # Preallocate element mass matrix
    e_mass = np.zeros((nelem, ngl, ngl))

    for e in range(nelem):
        # Extract coordinates using vectorized indexing
        x = coord[intma[:, e]]

        # Compute Jacobian
        dx = x[-1] - x[0]
        jac = dx / 2.0

        # Vectorized integration using outer product
        for k in range(nq):
            wk = wnq[k] * jac
            h_i = psi[:, k]
            h_j = psi[:, k]
            
            # Outer product: M_ij += w_k * φ_i * φ_j
            e_mass[e] += wk * np.outer(h_i, h_j)

    return e_mass


# =============================================================================
# Flux Matrices (Inter-Element Coupling)
# =============================================================================

def Fmatrix_upwind_flux(
    intma: np.ndarray,
    nelem: int,
    npoin: int,
    ngl: int,
    u: float,
    periodic: bool = True
) -> np.ndarray:
    """Create upwind flux matrix for DG advection formulation.
    
    Constructs the flux matrix F that handles inter-element coupling through
    the upwind numerical flux. For the advection equation ∂u/∂t + c*∂u/∂x = 0
    with positive wave speed c, the upwind flux uses information from the
    left (upwind) side:
    
        u* = u^-  (value from element to the left)
    
    The flux contributes at element interfaces:
        - Left boundary of element: receives flux from left neighbor
        - Right boundary of element: sends flux to right neighbor
    
    Args:
        intma: Element-node connectivity, shape (ngl, nelem).
        nelem: Number of elements.
        npoin: Number of global points (nodes).
        ngl: Number of LGL points per element.
        u: Wave speed (advection velocity). Positive = rightward propagation.
        periodic: If True, use periodic boundary conditions. Default True.
    
    Returns:
        Flux matrix, shape (npoin, npoin). Sparse in practice but stored dense.
    
    Note:
        The upwind flux is chosen for stability. For positive wave speed:
        - Information propagates left→right
        - Each node's flux depends on the upwind (left) neighbor
        - This produces a stable, first-order accurate flux
        
        For negative wave speed, the roles reverse (downwind becomes upwind).
    
    Warning:
        This implementation assumes positive wave speed. For negative speeds,
        the flux direction logic may need adjustment.
    
    See Also:
        Fmatrix_upwind_flux_bc: Version with explicit boundary condition support.
        Fmatrix_rusanov_flux: Alternative flux with artificial dissipation.
    """
    # Initialize flux matrix (sparse pattern but stored dense)
    Fmat = np.zeros((npoin, npoin), dtype=float)

    for e in range(nelem):
        # === Left boundary of element (inflow face for u > 0) ===
        # The leftmost DOF receives flux from its left neighbor
        i = 0  # Local index of left boundary node
        I = intma[i][e]  # Global index
        
        # Index of upwind neighbor (one to the left)
        Im = I - 1
        
        # Handle boundary: wrap around for periodic, or stay at boundary
        if Im < 0:
            if periodic:
                Im = npoin - 1  # Wrap to last node
            else:
                Im = 0  # Non-periodic: use boundary value
        
        # Flux from upwind neighbor: F[I, Im] = -1 means "subtract upwind value"
        Fmat[I][Im] = -1

        # === Right boundary of element (outflow face for u > 0) ===
        # The rightmost DOF contributes its value to the flux
        i = ngl - 1  # Local index of right boundary node
        I = intma[i][e]  # Global index
        
        # For upwind with positive u, outflow face uses local value
        # F[I, I] = +1 means "add local value"
        Fmat[I][I] = 1

    # Scale by wave speed
    Fmat = Fmat * u
    
    return Fmat


def Fmatrix_upwind_flux_bc(
    intma: np.ndarray,
    nelem: int,
    npoin: int,
    ngl: int,
    u: float,
    periodic: bool = False
) -> np.ndarray:
    """Create upwind flux matrix with explicit boundary condition support.
    
    Similar to Fmatrix_upwind_flux but with clearer handling of non-periodic
    boundary conditions. Useful when imposing inflow/outflow conditions.
    
    Args:
        intma: Element-node connectivity, shape (ngl, nelem).
        nelem: Number of elements.
        npoin: Number of global points.
        ngl: Number of LGL points per element.
        u: Wave speed.
        periodic: If True, use periodic BCs. If False, use explicit boundaries.
    
    Returns:
        Flux matrix, shape (npoin, npoin).
    
    Boundary Treatment (non-periodic, u > 0):
        - Left boundary (x=0): Inflow, flux comes from boundary condition
        - Right boundary (x=1): Outflow, flux exits domain
    
    See Also:
        Fmatrix_upwind_flux: Simpler version with periodic BCs.
    """
    # Main flux matrix
    Fmat = np.zeros((npoin, npoin), dtype=float)
    
    # Identify boundary nodes
    left_boundary_idx = intma[0, 0]           # First node in first element
    
    for e in range(nelem):
        # === Left boundary of element ===
        i = 0
        I = intma[i][e]
        Im = I - 1
        
        if Im < 0 and not periodic:
            # Non-periodic: inflow boundary uses boundary node value
            Fmat[I][left_boundary_idx] = -1
        elif Im < 0 and periodic:
            # Periodic: wrap around
            Im = npoin - 1
            Fmat[I][Im] = -1
        else:
            # Interior: standard upwind
            Fmat[I][Im] = -1

        # === Right boundary of element ===
        i = ngl - 1
        I = intma[i][e]
        
        # For upwind with positive u, outflow face uses local value
        Fmat[I][I] = 1
    
    # Scale by wave speed
    Fmat = Fmat * u
    
    return Fmat


def Fmatrix_centered_flux(
    intma: np.ndarray,
    nelem: int,
    npoin: int,
    ngl: int,
    u: float
) -> np.ndarray:
    """Create centered flux matrix for DG formulation.
    
    Constructs the flux matrix using centered (average) flux:
    
        u* = (u^- + u^+) / 2
    
    where u^- and u^+ are values from left and right of the interface.
    
    Args:
        intma: Element-node connectivity, shape (ngl, nelem).
        nelem: Number of elements.
        npoin: Number of global points.
        ngl: Number of LGL points per element.
        u: Wave speed.
    
    Returns:
        Flux matrix, shape (npoin, npoin).
    
    Warning:
        Centered flux is energy-neutral and may lead to oscillations for
        advection-dominated problems. Consider upwind or Rusanov flux for
        stability. Boundaries still use upwind for stability.
    
    See Also:
        Fmatrix_upwind_flux: Stable upwind alternative.
        Fmatrix_rusanov_flux: Centered flux with dissipation.
    """
    Fmat = np.zeros((npoin, npoin), dtype=float)
    
    for e in range(nelem):
        # === Left boundary of element ===
        i = 0
        I = intma[i][e]
        Im = I - 1
        
        if Im < 0:
            # Boundary: use upwind for stability at inflow
            if u > 0:
                Fmat[I][0] = -u  # Inflow boundary influence
        else:
            # Interior: centered flux = average from both sides
            # Contribution: -u * (u_left + u_right) / 2
            Fmat[I][Im] = -u / 2  # From left neighbor
            Fmat[I][I] = -u / 2   # From local value
            
        # === Right boundary of element ===
        i = ngl - 1
        I = intma[i][e]
        Ip = I + 1
        
        if Ip >= npoin:
            # Boundary: outflow
            if u > 0:
                Fmat[I][I] = u
        else:
            # Interior: centered flux
            Fmat[I][I] = u / 2    # From local value
            Fmat[I][Ip] = u / 2   # From right neighbor
            
    return Fmat


def Fmatrix_rusanov_flux(
    intma: np.ndarray,
    nelem: int,
    npoin: int,
    ngl: int,
    wave_speed: float,
    periodic: bool = False
) -> np.ndarray:
    """Create Rusanov (Local Lax-Friedrichs) flux matrix for DG formulation.
    
    The Rusanov flux combines the centered flux with artificial dissipation
    proportional to the maximum wave speed:
    
        F* = (F^- + F^+)/2 - (λ_max/2) * (u^+ - u^-)
    
    where λ_max = |wave_speed| is the maximum characteristic speed.
    
    For the linear advection equation F = c*u, this simplifies to:
    
        u* = (u^- + u^+)/2 - (sign(c)/2) * (u^+ - u^-)
    
    which reduces to upwind flux when the dissipation coefficient equals |c|.
    
    Args:
        intma: Element-node connectivity matrix, shape (ngl, nelem).
        nelem: Number of elements.
        npoin: Number of global points.
        ngl: Number of LGL points per element.
        wave_speed: Wave speed (advection velocity).
        periodic: Whether to use periodic boundary conditions.
    
    Returns:
        Rusanov flux matrix, shape (npoin, npoin).
    
    Note:
        The Rusanov flux is:
        - More dissipative than pure upwind for linear problems
        - Useful for nonlinear problems where characteristics vary
        - Reduces to upwind for scalar linear advection
        
        For the linear advection equation, pure upwind is typically preferred
        as it provides optimal dissipation.
    
    See Also:
        Fmatrix_upwind_flux: Less dissipative for linear advection.
        Fmatrix_centered_flux: Zero dissipation (may be unstable).
    """
    # Main flux matrix
    Fmat = np.zeros((npoin, npoin), dtype=float)
    
    # Maximum eigenvalue (absolute wave speed)
    lambda_max = abs(wave_speed)
    
    # === Boundary treatment (non-periodic) ===
    if not periodic:
        # Left boundary (inflow for positive wave speed)
        left_idx = intma[0, 0]
        Fmat[left_idx, left_idx] = 0.5 * wave_speed + 0.5 * lambda_max
        
        # Right boundary (outflow for positive wave speed)
        right_idx = intma[ngl-1, nelem-1]
        Fmat[right_idx, right_idx] = 0.5 * wave_speed + 0.5 * lambda_max
    
    # === Internal interfaces ===
    for e in range(nelem):
        # Process right interface of current element (connects to element e+1)
        if e < nelem - 1:
            # Current element's right node
            I_right = intma[ngl - 1, e]
            
            # Next element's left node
            J_left = intma[0, e + 1]
            
            # Rusanov flux contributions:
            # F* = (c/2)(u^- + u^+) - (λ/2)(u^+ - u^-)
            #    = (c/2 + λ/2)u^- + (c/2 - λ/2)u^+
            
            # Contribution to right node of current element (uses u^-)
            Fmat[I_right, I_right] += 0.5 * wave_speed + 0.5 * lambda_max
            Fmat[I_right, J_left] += 0.5 * wave_speed - 0.5 * lambda_max
            
            # Contribution to left node of next element (uses u^+)
            Fmat[J_left, J_left] += -0.5 * wave_speed + 0.5 * lambda_max
            Fmat[J_left, I_right] += -0.5 * wave_speed - 0.5 * lambda_max
    
    # === Periodic boundary connection ===
    if periodic:
        # Last element's right node connects to first element's left node
        I_right = intma[ngl - 1, nelem - 1]
        J_left = intma[0, 0]
        
        # Same flux formula for periodic interface
        Fmat[I_right, I_right] += 0.5 * wave_speed + 0.5 * lambda_max
        Fmat[I_right, J_left] += 0.5 * wave_speed - 0.5 * lambda_max
        
        Fmat[J_left, J_left] += -0.5 * wave_speed + 0.5 * lambda_max
        Fmat[J_left, I_right] += -0.5 * wave_speed - 0.5 * lambda_max
    
    return Fmat


# =============================================================================
# Global Matrix Assembly
# =============================================================================

def Matrix_DSS(
    Me: np.ndarray,
    De: np.ndarray,
    u: float,
    intma: np.ndarray,
    periodicity: np.ndarray,
    ngl: int,
    nelem: int,
    npoin: int
) -> tuple[np.ndarray, np.ndarray]:
    """Assemble global matrices via Direct Stiffness Summation.
    
    Combines element-level mass and differentiation matrices into global
    system matrices by summing contributions at shared nodes. This is the
    standard finite element assembly process.
    
    For continuous Galerkin (CG), shared nodes are truly shared and
    contributions add directly. For DG, the periodicity array handles
    the coupling at periodic boundaries.
    
    The assembled system is:
    
        M * du/dt = -D * u + F * u
    
    where M and D come from this function and F from the flux matrices.
    
    Args:
        Me: Element mass matrices, shape (nelem, ngl, ngl).
            Me[e] is the mass matrix for element e.
        De: Element differentiation matrix, shape (ngl, ngl).
            Same for all elements (reference element matrix).
        u: Wave speed (for scaling the differentiation matrix).
        intma: Element-node connectivity, shape (ngl, nelem).
            intma[i, e] = global index of local node i in element e.
        periodicity: Periodic boundary mapping, shape (npoin,).
            periodicity[I] = J means node I maps to node J for assembly.
            For interior nodes: periodicity[I] = I (identity).
            For periodic pairs: periodicity[I_right] = I_left.
        ngl: Number of LGL points per element.
        nelem: Number of elements.
        npoin: Number of global points (after applying periodicity).
    
    Returns:
        M: Global mass matrix, shape (npoin, npoin).
        D: Global differentiation matrix (scaled by u), shape (npoin, npoin).
    
    Note:
        The periodicity array implements periodic boundary conditions by
        mapping the rightmost node to the leftmost node, effectively making
        them the same degree of freedom.
    
    Example:
        >>> M, D = Matrix_DSS(Me, De, wave_speed, intma, periodicity, ngl, nelem, npoin)
        >>> # Verify mass matrix properties
        >>> np.allclose(M, M.T)  # Symmetric
        True
        >>> np.all(np.linalg.eigvals(M) > 0)  # Positive definite
        True
    """
    # Initialize global matrices
    M = np.zeros((npoin, npoin))
    D = np.zeros((npoin, npoin))
    
    # Loop over elements and assemble
    for e in range(nelem):
        for i in range(ngl):
            # Get global index (with periodic mapping)
            ip = periodicity[intma[i][e]]
            
            for j in range(ngl):
                # Get global index (with periodic mapping)
                jp = periodicity[intma[j][e]]
                
                # Add element contributions to global matrices
                M[ip][jp] += Me[e][i][j]
                D[ip][jp] += u * De[i][j]  # Scale by wave speed
    
    return M, D
