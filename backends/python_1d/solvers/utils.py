import numpy as np
from scipy.special import erf

"""Utility functions for exact solutions and error calculations.

This module provides exact analytical solutions for various initial conditions
(wave equation test cases) and error metrics for comparing numerical solutions
against exact solutions.

Key Functions:
    exact_solution: Compute exact solution for any icase at position (x, t).
    eff: Compute forcing/effective function for AMR refinement indicators.
    calculate_grid_normalized_l2_error: Compute mesh-independent L2 error metric.

Note:
    The exact solutions assume periodic boundary conditions on [-1, 1] with
    wave speed u = 2.0. Solutions are computed as q(x, t) = f(x - u*t).
"""



def exact_solution(coord, npoin, time, icase, k=5, omega=np.pi):
    """Compute exact solution of the wave equation for a given initial condition.
    
    Evaluates q(x, t) = f((x - u*t) mod L) where f is determined by icase
    and L = 2 is the domain length.
    
    Args:
        coord: Spatial coordinates, shape (npoin,). Values in [-1, 1].
        npoin: Number of points. Must match len(coord).
        time: Simulation time. Non-negative.
        icase: Initial condition case number:
            - 1: Gaussian pulse (always positive)
            - 2: Square pulse
            - 3: Diffusing Gaussian
            - 7: Tanh profile
            - 8: Sine wave
            - 10: Tanh smooth square (has negative values)
            - 11: Erf smooth square
            - 12: Sigmoid smooth square (has negative values)
            - 13: Multi-Gaussian
            - 14: Bump function (compact support)
            - 15: Sech² soliton
            - 16: Mexican hat / Ricker wavelet (has negative values)
        k: Steepness parameter for icase 10-12. Defaults to 5.
        omega: Frequency parameter for icase 10-12. Defaults to π.
        
    Returns:
        qe: Exact solution values, shape (npoin,).
        u: Wave speed (= 2.0 for all cases).
    
    Note:
        Domain is [-1, 1] with periodic BCs. Wave speed u = 2.0.
        
        Models trained only on icase=1 may learn spurious correlations
        (refining where q > 0) since Gaussian is always positive. Test
        on cases with negative values (10, 12, 16) to detect this.
    """
    # ================================================================
    # EXISTING CONSTANTS (unchanged)
    # ================================================================
    w = 1
    visc = 0
    xc = 0
    xmin = -1
    xmax = 1
    x1 = xmax - xmin  # Domain length = 2
    sigma0 = 0.125
    rc = 0.125
    sigma = np.sqrt(sigma0**2 + 2*visc*time)
    u = w * x1  # Wave speed = 2.0
    alph = 1.0
    beta = 256.0
    
    # Initialize solution array
    qe = np.zeros(npoin)
    
    # Time wrapping for periodic domain (wave returns after t=1 with c=2, L=2)
    timec = time - np.floor(time)
    
    # ================================================================
    # EXISTING CASES (1-9) - UNCHANGED
    # ================================================================
    for i in range(npoin):
        x = coord[i]
        xbar = xc + u * timec
        if xbar >= xmax:
            xbar = xmin + (xbar - xmax)
        r = x - xbar
        domain_length = xmax - xmin
        r = r - domain_length * np.round(r / domain_length)
        
        if icase == 1:
            # Gaussian pulse with periodic images
            qe[i] = np.exp(-beta * (x - xbar)**2)
            # Add periodic images for continuity at boundaries
            domain_length = x1
            xbar_left = xbar - domain_length
            qe[i] += np.exp(-beta * (x - xbar_left)**2)
            xbar_right = xbar + domain_length
            qe[i] += np.exp(-beta * (x - xbar_right)**2)
            
        elif icase == 2:
            if abs(r) <= rc:
                qe[i] = 1
                
        elif icase == 3:
            qe[i] = sigma0 / sigma * np.exp(-(x - xbar)**2 / (2 * sigma**2))
            
        elif icase == 4:
            if abs(r) <= rc:
                qe[i] = 1
                
        elif icase == 5:
            if x <= xc:
                qe[i] = 1
                
        elif icase == 6:
            qe[i] = np.sin(((x + 1) * np.pi) / 2.0)
            
        elif icase == 7:
            qe[i] = 1 - np.tanh(alph * (1 - 4 * ((x) - 1/4)))
            
        elif icase == 8:
            qe[i] = np.sin(np.pi * x)
            
        elif icase == 9:
            # Large domain Gaussian pulse (domain [-4, 4])
            mu = -4
            sigma_sq = 0.25
            c = 1.0
            x_shifted = x - c * time
            domain_length = 8
            main_pulse = np.exp(-1 / (2 * sigma_sq) * ((x_shifted - mu)**2))
            left_image = np.exp(-1 / (2 * sigma_sq) * ((x_shifted - mu + domain_length)**2))
            right_image = np.exp(-1 / (2 * sigma_sq) * ((x_shifted - mu - domain_length)**2))
            qe[i] = main_pulse + left_image + right_image
            u = 1.0
            
        # ================================================================
        # NEW CASES (10-12) - GENERALIZATION TEST FUNCTIONS
        # ================================================================
        elif icase == 10:
            # Hyperbolic tangent smooth square wave
            # u_tanh(x,t) = tanh(k * sin(omega * (x - c*t)))
            #
            # Features:
            # - Sharp transitions at x ≈ 0 and x ≈ ±1 (at t=0)
            # - Flat plateaus at ±1 between transitions
            # - Smooth and differentiable everywhere
            # - Automatically periodic since sin(omega*(-1)) = sin(omega*(1)) = 0
            #
            # Parameters: k controls steepness, omega controls frequency
            
            c_wave = u  # Use standard wave speed (2.0)
            xi = x - c_wave * timec  # Advected coordinate
            qe[i] = np.tanh(k * np.sin(omega * xi))
            
        elif icase == 11:
            # Error function smooth square wave
            # u_erf(x,t) = erf(k * sin(omega * (x - c*t)))
            #
            # Features:
            # - Similar to tanh but slightly different saturation profile
            # - Based on Gaussian integral (related to training IC)
            # - Smooth and differentiable everywhere
            
            c_wave = u
            xi = x - c_wave * timec
            qe[i] = erf(k * np.sin(omega * xi))
            
        elif icase == 12:
            # Sigmoid smooth square wave (shifted to [-1, 1] range)
            # u_sigmoid(x,t) = 2 / (1 + exp(-k * sin(omega * (x - c*t)))) - 1
            #
            # Features:
            # - Gentlest transitions of the three
            # - Tests lower bound of model sensitivity to gradients
            # - May challenge models trained on sharper Gaussian
            
            c_wave = u
            xi = x - c_wave * timec
            qe[i] = 2.0 / (1.0 + np.exp(-k * np.sin(omega * xi))) - 1.0

        elif icase == 13:
            # Multi-Gaussian: Two separated pulses
            # Tests whether model can split budget between multiple features
            #
            # Features:
            #   - Two localized features requiring refinement
            #   - Large flat region between pulses
            #   - Similar shape to training but different spatial distribution
            
            c_wave = u  # Wave speed = 2.0
            beta_pulse = 256.0  # Match training Gaussian sharpness
            x1_init = -0.5  # Left pulse initial center
            x2_init = 0.5   # Right pulse initial center
            
            # Advected pulse centers with periodic wrapping
            timec_local = time - np.floor(time)
            x1_center = x1_init + c_wave * timec_local
            x2_center = x2_init + c_wave * timec_local
            
            # Wrap centers to domain [-1, 1]
            domain_length = 2.0
            if x1_center > 1.0:
                x1_center = x1_center - domain_length
            if x2_center > 1.0:
                x2_center = x2_center - domain_length
            
            # Two Gaussian pulses with periodic images
            pulse1 = np.exp(-beta_pulse * (x - x1_center)**2)
            pulse1 += np.exp(-beta_pulse * (x - x1_center - domain_length)**2)
            pulse1 += np.exp(-beta_pulse * (x - x1_center + domain_length)**2)
            
            pulse2 = np.exp(-beta_pulse * (x - x2_center)**2)
            pulse2 += np.exp(-beta_pulse * (x - x2_center - domain_length)**2)
            pulse2 += np.exp(-beta_pulse * (x - x2_center + domain_length)**2)
            
            qe[i] = pulse1 + pulse2

        elif icase == 14:
            # Bump function with compact support
            # Exactly zero outside support region - tests "nothing to refine"
            #
            # Features:
            #   - Smooth (C^infinity) and infinitely differentiable
            #   - EXACTLY zero outside support (not just approximately)
            #   - Sharp but smooth transition at edges
            
            c_wave = u  # Wave speed = 2.0
            a = 0.3  # Support radius
            
            # Advected center with periodic wrapping
            timec_local = time - np.floor(time)
            center = c_wave * timec_local
            
            # Wrap center to domain [-1, 1]
            if center > 1.0:
                center = center - 2.0
            
            # Distance from center (considering periodicity)
            dist = x - center
            domain_length = 2.0
            
            # Wrap distance for periodic domain
            if dist > 1.0:
                dist = dist - domain_length
            elif dist < -1.0:
                dist = dist + domain_length
            
            # Bump function: exp(-1/(1-r^2)) for |r| < 1, else 0
            r = dist / a
            if abs(r) < 1.0:
                qe[i] = np.exp(-1.0 / (1.0 - r**2))
            else:
                qe[i] = 0.0

        elif icase == 15:
            # Sech² (Soliton profile)
            # Classic soliton shape from KdV equation
            #
            # Features:
            #   - Algebraic tail decay (slower than Gaussian)
            #   - Single localized feature
            #   - Common in nonlinear wave physics
            
            c_wave = u  # Wave speed = 2.0
            k_sech = 10.0  # Width parameter
            
            # Advected center with periodic wrapping
            timec_local = time - np.floor(time)
            center = c_wave * timec_local
            
            # Wrap center to domain [-1, 1]
            if center > 1.0:
                center = center - 2.0
            
            # Distance from center (considering periodicity)
            dist = x - center
            domain_length = 2.0
            
            # Use closest periodic image
            if dist > 1.0:
                dist = dist - domain_length
            elif dist < -1.0:
                dist = dist + domain_length
            
            # sech²(k*x) = 1/cosh²(k*x)
            qe[i] = 1.0 / np.cosh(k_sech * dist)**2

        elif icase == 16:
            # Mexican Hat (Ricker wavelet)
            # Central positive peak with negative side lobes
            #
            # Features:
            #   - Non-monotonic profile (has negative values)
            #   - Multiple gradient regions at different scales
            #   - Common in seismology and signal processing
            
            c_wave = u  # Wave speed = 2.0
            sigma = 8.0  # Width parameter (larger = narrower)
            
            # Advected center with periodic wrapping
            timec_local = time - np.floor(time)
            center = c_wave * timec_local
            
            # Wrap center to domain [-1, 1]
            if center > 1.0:
                center = center - 2.0
            
            # Distance from center (considering periodicity)
            dist = x - center
            domain_length = 2.0
            
            # Use closest periodic image
            if dist > 1.0:
                dist = dist - domain_length
            elif dist < -1.0:
                dist = dist + domain_length
            
            # Mexican hat: (1 - 2(πσξ)²) * exp(-(πσξ)²)
            pi_sigma_xi = np.pi * sigma * dist
            qe[i] = (1.0 - 2.0 * pi_sigma_xi**2) * np.exp(-pi_sigma_xi**2)
    
    return qe, u




def eff(coord, npoin, fcase, wave_speed, time=0.0):
    """Compute forcing/effective function for AMR refinement indicators.
    
    The effective function is the spatial derivative of the exact solution,
    used to identify regions where mesh refinement improves accuracy.
    Large absolute values indicate steep gradients.
    
    Args:
        coord: Spatial coordinates, shape (npoin,).
        npoin: Number of points.
        fcase: Test case number (matches icase in exact_solution).
        wave_speed: Wave propagation speed.
        time: Simulation time. Defaults to 0.0.
        
    Returns:
        Forcing function values, shape (npoin,).
    
    Note:
        Verify derivatives match before training on new fcase:
            python analysis/verification/verify_eff_derivatives.py
    """
    # Constants (matching those in exact_solution)
    xc = 0
    xmin = -1
    xmax = 1
    x1 = xmax-xmin
    alph = 1.0
    beta = 256.0
    
    # Initialize forcing vector
    f = np.zeros(npoin)
    
    # Time computation (same as in exact_solution)
    timec = time - np.floor(time)
    
    for i in range(npoin):
        x = coord[i]
        
        # Calculate the center position (moves with time)
        xbar = xc + wave_speed*timec
        if xbar > xmax:
            xbar = xmin + (xbar-xmax)
        
        if fcase == 1:
            # Correct forcing for moving Gaussian: u * d/dx[exp(-beta*(x-xbar)^2)]
            f[i] = -2*wave_speed*beta*(x-xbar)*np.exp(-beta*(x-xbar)**2)
        
        elif fcase == 7:
            # Forcing for tanh solution
            def sech(x):
                return 1 / np.cosh(x)
            inner = alph * (1 - 4 * (x - 1/4))
            f[i] = 4 * alph * sech(inner)**2
        
        elif fcase == 8:
            # Forcing for sine solution
            f[i] = wave_speed*np.pi * np.cos(np.pi * x)
        
        elif fcase == 9:
            # Forcing for Gaussian pulse (should be zero for pure advection)
            f[i] = 0

        elif fcase == 10:
            # Tanh smooth square wave: u = tanh(k*sin(omega*(x - c*t)))
            k = 5
            omega = np.pi
            xi = x - wave_speed * timec
            # Periodic wrapping
            domain_length = 2.0
            xi = xi - domain_length * np.round(xi / domain_length)
            arg = k * np.sin(omega * xi)
            sech_sq = 1.0 / np.cosh(arg)**2
            f[i] = wave_speed * k * omega * np.cos(omega * xi) * sech_sq
            
        elif fcase == 11:
            # Erf smooth square wave: u = erf(k*sin(omega*(x - c*t)))
            k = 5
            omega = np.pi
            xi = x - wave_speed * timec
            domain_length = 2.0
            xi = xi - domain_length * np.round(xi / domain_length)
            arg = k * np.sin(omega * xi)
            f[i] = wave_speed * (2.0 / np.sqrt(np.pi)) * k * omega * np.cos(omega * xi) * np.exp(-arg**2)
            
        elif fcase == 12:
            # Sigmoid smooth square wave: u = 2/(1 + exp(-k*sin(omega*(x-ct)))) - 1
            k = 5
            omega = np.pi
            xi = x - wave_speed * timec
            domain_length = 2.0
            xi = xi - domain_length * np.round(xi / domain_length)
            y = k * np.sin(omega * xi)
            sigmoid_y = 1.0 / (1.0 + np.exp(-y))
            f[i] = wave_speed * 2.0 * k * omega * np.cos(omega * xi) * sigmoid_y * (1.0 - sigmoid_y)
            
        elif fcase == 13:
            # Multi-Gaussian: u = exp(-β(x-x₁-ct)²) + exp(-β(x-x₂-ct)²)
            beta_mg = 256.0
            x1, x2 = -0.5, 0.5
            xi1 = x - x1 - wave_speed * timec
            xi2 = x - x2 - wave_speed * timec
            domain_length = 2.0
            xi1 = xi1 - domain_length * np.round(xi1 / domain_length)
            xi2 = xi2 - domain_length * np.round(xi2 / domain_length)
            f[i] = -2.0 * wave_speed * beta_mg * (
                xi1 * np.exp(-beta_mg * xi1**2) + 
                xi2 * np.exp(-beta_mg * xi2**2)
            )
            
        elif fcase == 14:
            # Bump function (compact support)
            a = 0.3
            xi = x - wave_speed * timec
            domain_length = 2.0
            xi = xi - domain_length * np.round(xi / domain_length)
            if np.abs(xi) < a:
                xi_norm = xi / a
                denom = 1.0 - xi_norm**2
                u_val = np.exp(-1.0 / denom)
                f[i] = wave_speed * u_val * (-2.0 * xi / a**2) / denom**2
            else:
                f[i] = 0.0
                
        elif fcase == 15:
            # Sech² soliton: u = sech²(k(x - ct))
            k = 10
            xi = x - wave_speed * timec
            domain_length = 2.0
            xi = xi - domain_length * np.round(xi / domain_length)
            sech_val = 1.0 / np.cosh(k * xi)
            tanh_val = np.tanh(k * xi)
            f[i] = -2.0 * wave_speed * k * sech_val**2 * tanh_val
            
        elif fcase == 16:
            # Mexican hat (Ricker wavelet)
            sigma = 8.0
            xi = x - wave_speed * timec
            domain_length = 2.0
            xi = xi - domain_length * np.round(xi / domain_length)
            y = np.pi * sigma * xi
            f[i] = wave_speed * (-2.0) * (np.pi * sigma)**2 * xi * np.exp(-y**2) * (3.0 - 2.0 * y**2)
    
    return f

# def L2_err_norm(nop, nelem, q0, qe):
#     """
#     Computes L2 error norm between numerical and exact solutions.
    
#     Args:
#         nop (int): Polynomial order
#         nelem (int): Number of elements
#         q0 (array): Numerical solution
#         qe (array): Exact solution
        
#     Returns:
#         float: L2 error norm
#     """
#     Np = nelem*nop + 1
#     num = 0
#     den = 0
#     for i in range(Np):
#         num = num + (qe[i]-q0[i])**2
#         den = den + qe[i]**2
#     err = np.sqrt(num/den)

#     return err

# def L2normerr(q0, qe):
#     num = np.norm(q0, qe)
#     den = np.norm(qe)
#     err = num/den
#     return err

def calculate_grid_normalized_l2_error(final_solution, final_coord, initial_coord, time_final, icase):
    """Compute grid-normalized L2 error for mesh-independent comparison.
    
    Interpolates the final numerical solution (on adapted mesh) back onto
    the initial uniform grid, then computes relative L2 error against the
    exact solution. This enables fair comparison across different meshes.
    
    Args:
        final_solution: Numerical solution on adapted mesh, shape (n_final,).
        final_coord: Coordinates of adapted mesh, shape (n_final,).
        initial_coord: Coordinates of initial uniform mesh, shape (n_initial,).
        time_final: Final simulation time.
        icase: Test case identifier.
        
    Returns:
        Relative L2 error (dimensionless). Lower is better.
        Computed as ||q - q_exact||_2 / ||q_exact||_2.
    
    Note:
        This is the primary accuracy metric for model comparison and
        flagship selection (distance-to-ideal calculations).
    """
    # Interpolate final solution onto initial grid coordinates
    solution_on_initial_grid = np.interp(initial_coord, final_coord, final_solution)
    
    # Calculate exact solution on initial grid
    exact_on_initial_grid, _ = exact_solution(initial_coord, len(initial_coord), time_final, icase)
    
    # Calculate L2 norm on consistent grid
    grid_normalized_l2_error = np.sqrt(
        np.sum((solution_on_initial_grid - exact_on_initial_grid)**2) / 
        np.sum(exact_on_initial_grid**2)
    )
    
    return grid_normalized_l2_error


def compute_total_mass(q, Me, intma):
    """Compute total mass (integral of solution) using element mass matrices.
    
    Args:
        q: Solution vector, shape (npoin,).
        Me: Element mass matrices, shape (nelem, ngl, ngl).
        intma: Element-node connectivity, shape (ngl, nelem).
        
    Returns:
        Total integrated mass (should be conserved for periodic wave equation).
    
    Note:
        Computes ∫ q² dx via Gaussian quadrature using precomputed mass matrices.
    """
    # print(f'intma:\n {intma[0][:]}')
    mass = 0
    for e in range(len(intma[0][:])):  # loop over elements
        # Get local solution and mass matrix
        # print(f'intma[e]: {intma[:,e]}')
        q_local = q[intma[:,e]]
        Me_local = Me[e]
        # Add contribution from this element
        mass += np.dot(np.dot(q_local, Me_local), q_local)
    return mass


