"""
Discontinuous Galerkin Advection Solver with Adaptive Mesh Refinement.

This module implements a high-order Discontinuous Galerkin (DG) solver for the 
1D linear advection equation with h-adaptation capabilities using hierarchical 
mesh refinement.

The solver implements:
    - Legendre-Gauss-Lobatto (LGL) nodal basis functions
    - Upwind numerical fluxes for interface treatment  
    - Low-storage 5-stage Runge-Kutta time integration
    - Hierarchical binary tree mesh refinement with 2:1 balance
    - L2 projection for solution transfer between meshes

The advection equation solved is:
    ∂q/∂t + c * ∂q/∂x = f(x,t)
    
where 'c' is the wave speed and f is an optional forcing term.

Example:
    >>> import numpy as np
    >>> xelem = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    >>> solver = DGAdvectionSolver(
    ...     nop=4, 
    ...     xelem=xelem, 
    ...     max_elements=64, 
    ...     max_level=4,
    ...     icase=1
    ... )
    >>> times, solutions, grids, coords = solver.solve(time_final=1.0)

References:
    - Foucart et al. (2023) for DRL-AMR methodology
"""

import numpy as np

from ..amr.adapt import (
    adapt_mesh,
    adapt_sol,
    check_balance,
    coarsen_pair,
    enforce_balance,
    mark,
    project_coarsen_single,
    project_refine_single,
    refine_single,
)
from ..amr.forest import forest
from ..amr.projection import projections
from ..dg.basis import Lagrange_basis, lgl_gen
from ..dg.matrices import (
    Fmatrix_centered_flux,
    Fmatrix_upwind_flux_bc,
    Matrix_DSS,
    create_diff_matrix,
    create_mass_matrix,
    create_RM_matrix,
)
from ..grid.mesh import create_grid_us
from .utils import eff, exact_solution


class DGAdvectionSolver:
    """
    Discontinuous Galerkin solver for 1D advection equation with AMR.
    
    This solver combines high-order DG spatial discretization with hierarchical
    h-refinement for adaptive mesh control. It is designed to work with the
    DGAMREnv reinforcement learning environment for learned refinement strategies.
    
    The solver maintains:
        - A hierarchical forest structure tracking element relationships
        - Projection operators for solution transfer during adaptation
        - DG operators (mass, differentiation, flux matrices) updated after adaptation
    
    Attributes:
        nop (int): Polynomial order for DG basis functions.
        ngl (int): Number of LGL points per element (nop + 1).
        nelem (int): Current number of active elements.
        xelem (ndarray): Element boundary coordinates, shape (nelem+1,).
        max_level (int): Maximum allowed refinement level.
        max_elements (int): Maximum allowed number of elements.
        dt (float): Current time step size (CFL-limited).
        time (float): Current simulation time.
        icase (int): Test case identifier for initial/exact solutions.
        wave_speed (float): Advection velocity.
        q (ndarray): Current solution vector at DG nodes.
        coord (ndarray): Physical coordinates of all DG nodes.
        active (list): Indices of currently active elements in forest.
        label_mat (ndarray): Forest structure tracking element hierarchy.
        periodic (bool): Whether to use periodic boundary conditions.
        balance (bool): Whether to enforce 2:1 mesh balance.
        verbose (bool): Whether to print diagnostic information.
    """
    
    def __init__(self, nop, xelem, max_elements, max_level, courant_max=0.1, 
                 icase=1, periodic=True, verbose=False, balance=False):
        """
        Initialize the DG advection solver with AMR capabilities.
        
        Args:
            nop: Polynomial order for basis functions. Determines accuracy 
                (convergence rate is O(h^{nop+1}) for smooth solutions).
            xelem: Initial element boundary coordinates as 1D array.
                Must be monotonically increasing.
            max_elements: Maximum number of elements allowed. Refinement
                requests that would exceed this are rejected.
            max_level: Maximum refinement level allowed. Level 0 is the
                initial mesh; each level halves element size.
            courant_max: Maximum Courant number for time step calculation.
                Default 0.1 is conservative for RK time stepping.
            icase: Test case identifier selecting initial condition and
                exact solution from utils.exact_solution().
            periodic: If True, use periodic boundary conditions.
                If False, use inflow/outflow boundaries.
            verbose: If True, print diagnostic information during
                mesh adaptation and time stepping.
            balance: If True, enforce 2:1 balance constraint on mesh
                (no element can be more than one level different from neighbors).
        
        Note:
            The solver initializes to a steady-state solution of the forced
            advection equation. The initial condition from icase determines
            the forcing function that maintains this steady state.
        """
        # Store polynomial order and compute derived quantities
        self.nop = nop
        self.ngl = nop + 1  # Number of LGL points per element
        self.nq = nop + 2   # Number of quadrature points (over-integration)
        
        # Store mesh limits
        self.max_elements = max_elements
        self.max_level = max_level
        
        # Store initial grid
        self.xelem = xelem
        self.nelem = len(xelem) - 1
        
        # Store solver options
        self.icase = icase
        self.courant_max = courant_max
        self.periodic = periodic
        self.balance = balance
        self.verbose = verbose
        
        # Initialize simulation time
        self.time = 0.0
        
        # Compute minimum element size at max refinement
        self.dx_min = np.min(np.diff(xelem)) / (2**max_level)
        
        # Generate LGL nodes and weights for interpolation
        self.xgl, self.wgl = lgl_gen(self.ngl)
        
        # Generate quadrature nodes and weights for integration
        self.xnq, self.wnq = lgl_gen(self.nq)
        
        # Compute Lagrange basis functions and derivatives at quadrature points
        self.psi, self.dpsi = Lagrange_basis(self.ngl, self.nq, self.xgl, self.xnq)
        
        # Initialize mesh, solution, and operators
        self._initialize_mesh()
        self.q = self._initialize_solution()
        self.qe = self.get_exact_solution()
        
        # Compute stable time step
        self._compute_timestep(use_actual_max_level=True)
        
        # Initialize projection operators for AMR
        self._initialize_projections()
        
        # Initialize forcing and DG operators
        self.f = self._initialize_forcing()
        self._update_matrices()

    # =========================================================================
    # Initialization Methods
    # =========================================================================
    
    def _initialize_mesh(self):
        """
        Initialize mesh data structures and AMR forest.
        
        Creates the DG grid with element-to-node connectivity and initializes
        the hierarchical forest structure for tracking refinement relationships.
        
        Sets:
            npoin_cg: Number of continuous Galerkin nodes (for reference).
            npoin_dg: Total number of DG nodes across all elements.
            label_mat: Forest label matrix tracking element hierarchy.
            info_mat: Forest info matrix with element metadata.
            active: List of currently active element indices.
            coord: Physical coordinates of all DG nodes.
            intma: Element-to-node connectivity matrix.
            periodicity: Node mapping for periodic boundaries.
        """
        # Compute node counts
        self.npoin_cg = self.nop * self.nelem + 1
        self.npoin_dg = self.ngl * self.nelem
        
        # Create hierarchical forest structure
        self.label_mat, self.info_mat, self.active = forest(self.xelem, self.max_level)
        
        # Create DG grid with connectivity
        self.coord, self.intma, self.periodicity = create_grid_us(
            self.ngl, self.nelem, self.npoin_cg, self.npoin_dg, 
            self.xgl, self.xelem
        )
    
    def _initialize_solution(self):
        """
        Initialize solution vector from test case.
        
        Evaluates the exact solution at all DG nodes for the specified
        test case (icase) at time t=0.
        
        Returns:
            Solution vector at all DG nodes.
            
        Note:
            Also sets self.wave_speed from the test case definition.
        """
        q, self.wave_speed = exact_solution(
            self.coord, self.npoin_dg, self.time, self.icase
        )
        return q
    
    def _initialize_forcing(self):
        """
        Initialize forcing function for steady-state problem.
        
        Computes the forcing term f(x) that makes the initial condition
        a steady-state solution of the forced advection equation.
        
        Returns:
            Forcing vector at all DG nodes.
        """
        self.f = eff(self.coord, self.npoin_dg, self.icase, self.wave_speed, self.time)
        return self.f
    
    def _update_forcing(self):
        """
        Update forcing function for current mesh and time.
        
        Called after mesh adaptation to recompute forcing at new node locations.
        
        Returns:
            Updated forcing vector at all DG nodes.
        """
        self.f = eff(self.coord, self.npoin_dg, self.icase, self.wave_speed, self.time)
        return self.f
    
    def _initialize_projections(self):
        """
        Initialize projection operators for AMR solution transfer.
        
        Creates the projection matrices needed to transfer solution data
        when elements are refined (split) or coarsened (merged).
        
        Sets:
            PS1, PS2: Projection matrices for splitting (parent to children).
            PG1, PG2: Projection matrices for gathering (children to parent).
        """
        RM = create_RM_matrix(self.ngl, self.nq, self.wnq, self.psi)
        self.PS1, self.PS2, self.PG1, self.PG2 = projections(
            RM, self.ngl, self.nq, self.wnq, self.xgl, self.xnq
        )
    
    def _compute_timestep(self, use_actual_max_level=False):
        """
        Compute stable time step based on CFL condition.
        
        The time step is computed as dt = CFL * dx_min / wave_speed,
        where dx_min is the smallest element size in the mesh.
        
        Args:
            use_actual_max_level: If True, compute dx_min from actual mesh.
                If False, use theoretical minimum at max_level.
        """
        if use_actual_max_level:
            current_max_level = self.get_current_max_refinement_level()
            # Use half the minimum element size for stability margin
            dx_min = np.min(np.diff(self.xelem)) / 2
            
            if self.verbose:
                print(f"Using max refinement level: {current_max_level}/{self.max_level}")
        else:
            # Theoretical minimum at maximum refinement
            dx_min = np.min(np.diff(self.xelem)) / (2**self.max_level)
            print(f'dt: {self.courant_max * dx_min / self.wave_speed}')
        
        old_dt = getattr(self, 'dt', None)
        self.dt = self.courant_max * dx_min / self.wave_speed
        
        if self.verbose and old_dt is not None and abs(old_dt - self.dt) > 1e-10:
            print(f"Time step updated: {old_dt:.6e} -> {self.dt:.6e}")
    
    def _update_matrices(self):
        """
        Update DG operators for current mesh configuration.
        
        Recomputes mass matrix, differentiation matrix, and flux matrices
        after mesh adaptation. Also forms the combined operator Dhat used
        for time stepping.
        
        Raises:
            ValueError: If mass matrix condition number exceeds threshold.
            np.linalg.LinAlgError: If matrix solve fails.
        """
        # Create element mass matrices
        self.Me = create_mass_matrix(
            self.intma, self.coord, self.nelem, self.ngl, 
            self.nq, self.wnq, self.psi
        )
        
        # Create element differentiation matrix
        self.De = create_diff_matrix(self.ngl, self.nq, self.wnq, self.psi, self.dpsi)
        
        # Assemble global mass and differentiation matrices
        self.M, self.D = Matrix_DSS(
            self.Me, self.De, self.wave_speed, self.intma, 
            self.periodicity, self.ngl, self.nelem, self.npoin_dg
        )
        
        # Check conditioning
        cond_num = np.linalg.cond(self.M)
        if self.verbose:
            print(f"Mass matrix condition number: {cond_num}")
        
        if cond_num > 1e10:
            raise ValueError(f"Mass matrix condition number too high: {cond_num}")
        
        # Create flux matrices
        self.F = Fmatrix_upwind_flux_bc(
            self.intma, self.nelem, self.npoin_dg, self.ngl, 
            self.wave_speed, self.periodic
        )
        self.Fcent = Fmatrix_centered_flux(
            self.intma, self.nelem, self.npoin_dg, self.ngl, self.wave_speed
        )
        
        # Form RHS operator: D - F (differentiation minus flux)
        R = self.D - self.F

        # Solve M * Dhat = R for the combined operator
        # Use lstsq instead of solve: on macOS Accelerate, solve() hangs
        # on singular matrices instead of raising LinAlgError.
        self.Dhat, residuals, rank, sv = np.linalg.lstsq(self.M, R, rcond=None)
        if rank < self.M.shape[0]:
            print(f"WARNING: Mass matrix rank-deficient ({rank}/{self.M.shape[0]})")
            print(f"  nelem={self.nelem}, npoin_dg={self.npoin_dg}")
            print(f"  Element sizes: {np.diff(self.xelem)}")
            print(f"  Active elements: {self.active}")
        
        # # Solve M * Dhat = R for the combined operator
        # try:
        #     self.Dhat = np.linalg.solve(self.M, R)
        # except np.linalg.LinAlgError:
        #     if self.verbose:
        #         print("Matrix solve failed. Current mesh configuration:")
        #         print(f"Number of elements: {self.nelem}")
        #         print(f"Element sizes: {np.diff(self.xelem)}")
        #     raise

    # =========================================================================
    # Query Methods
    # =========================================================================
    
    def get_current_max_refinement_level(self):
        """
        Get the maximum refinement level present in active mesh.
        
        Returns:
            Maximum refinement level among all active elements.
            Returns 0 if no elements are active.
        """
        if len(self.active) == 0:
            return 0
            
        active_levels = np.zeros(len(self.active), dtype=int)
        for i, elem in enumerate(self.active):
            # label_mat is 1-indexed; level is in column 4
            active_levels[i] = self.label_mat[elem - 1][4]
            
        return np.max(active_levels)
    
    def get_active_levels(self):
        """
        Get refinement level for each active element.
        
        Returns:
            Array of refinement levels for all active elements.
        """
        active_levels = np.zeros(len(self.active), dtype=int)
        for i, elem in enumerate(self.active):
            active_levels[i] = self.label_mat[elem - 1][4]
        return active_levels
    
    def get_exact_solution(self):
        """
        Evaluate exact solution at current time.
        
        Returns:
            Exact solution values at all DG nodes.
        """
        qe, _ = exact_solution(self.coord, self.npoin_dg, self.time, self.icase)
        return qe
    
    def get_forcing(self):
        """
        Evaluate forcing function at current node locations.
        
        Returns:
            Forcing values at all DG nodes.
        """
        return eff(self.coord, self.npoin_dg, self.icase, self.wave_speed, self.time)

    # =========================================================================
    # Mesh Quality and Validation
    # =========================================================================
    
    def check_mesh_quality(self, grid):
        """
        Check if mesh satisfies quality constraints.
        
        Validates that element sizes and size ratios are within acceptable
        bounds for numerical stability.
        
        Args:
            grid: Element boundary coordinates to check.
            
        Returns:
            Tuple of (is_valid, issues_string) where is_valid is True if
            mesh passes all checks, and issues_string describes any problems.
        """
        element_sizes = np.diff(grid)
        size_ratio = np.max(element_sizes) / np.min(element_sizes)
        domain_size = grid[-1] - grid[0]
        
        issues = []
        
        # Check global size ratio
        if size_ratio > 500:
            issues.append(f"Element size ratio too large: {size_ratio:.2f}")
        
        # Check for extremely small elements
        min_size = np.min(element_sizes)
        if min_size < domain_size * 1e-6:
            issues.append(f"Elements too small: {min_size:.2e}")
        
        # Check neighbor size ratios (only if more than one element)
        if len(element_sizes) > 1:
            neighbor_ratios = element_sizes[1:] / element_sizes[:-1]
            max_neighbor_ratio = max(np.max(neighbor_ratios), np.max(1/neighbor_ratios))
            if max_neighbor_ratio > 512:
                issues.append(f"Rapid size change between neighbors: ratio {max_neighbor_ratio:.2f}")
        
        return len(issues) == 0, "; ".join(issues)
    
    def verify_state(self):
        """
        Verify solver state is internally consistent.
        
        Checks that element count, sizes, mesh quality, and solution values
        are all valid.
        
        Raises:
            ValueError: If any validation check fails.
        """
        # Check element count
        if len(self.active) > self.max_elements:
            raise ValueError(
                f"Element count {len(self.active)} exceeds maximum {self.max_elements}"
            )
        
        # Check element sizes
        element_sizes = np.diff(self.xelem)
        if np.any(element_sizes <= 0):
            raise ValueError("Invalid element sizes detected")
        
        # Check mesh quality
        quality_ok, issues = self.check_mesh_quality(self.xelem)
        if not quality_ok:
            raise ValueError(f"Mesh quality issues: {issues}")
        
        # Check solution values
        if np.any(~np.isfinite(self.q)):
            raise ValueError("Invalid solution values detected")

    # =========================================================================
    # Mesh Adaptation Methods
    # =========================================================================
    
    def balance_mesh(self, balance=None):
        """
        Enforce 2:1 balance constraint on mesh if needed.
        
        The 2:1 balance constraint ensures no element is more than one
        refinement level different from its neighbors, which is important
        for numerical stability and projection accuracy.
        
        Args:
            balance: Whether to enforce balance. If None, uses instance setting.
            
        Returns:
            True if balancing was performed, False otherwise.
        """
        use_balance = self.balance if balance is None else balance
        
        if use_balance and not check_balance(self.active, self.label_mat):
            if self.verbose:
                print("Enforcing mesh balance...")
                print(f'Pre-balance active elements: {len(self.active)}')
            
            # Enforce balance through refinement
            (bal_q, bal_active, bal_nelem, bal_intma, bal_coord, 
             bal_grid, bal_npoin_dg, bal_periodicity) = enforce_balance(
                self.active, self.label_mat, self.xelem, self.info_mat,
                self.nop, self.coord, 
                self.PS1, self.PS2, self.PG1, self.PG2,
                self.ngl, self.xgl, self.q, self.max_level
            )
            
            # Update solver state
            self.q = bal_q
            self.active = bal_active
            self.nelem = bal_nelem
            self.intma = bal_intma
            self.coord = bal_coord
            self.xelem = bal_grid
            self.npoin_dg = bal_npoin_dg
            self.periodicity = bal_periodicity
            
            if self.verbose:
                print(f'Post-balance active elements: {len(self.active)}')
            
            return True
        
        return False
    
    def refine_element(self, active_idx):
        """Refine a single element by splitting it into two children.

        Calls the stateless primitives from adapt.py to update mesh topology
        and project the solution. Rebuilds grid connectivity afterward.

        Does NOT rebuild DG matrices (_update_matrices), forcing
        (_update_forcing), or recompute timestep. The caller is responsible
        for these after all mesh modifications (including balance enforcement)
        are complete. This avoids redundant matrix rebuilds when balance
        cascades follow the action.

        Typical caller pattern (env._execute_action):
            solver.refine_element(idx)
            solver.balance_mesh(balance=True)   # may refine more elements
            solver._update_matrices()
            solver._update_forcing()

        Args:
            active_idx: Index into self.active (0-based) of the element
                to refine.

        Returns:
            True if refinement was performed, False if element is already
            at max_level.
        """
        # === Mesh topology via primitive ===
        new_grid, new_active, success = refine_single(
            self.nop, self.xelem, self.active, self.label_mat,
            self.info_mat, active_idx, self.max_level,
        )
        if not success:
            return False

        # === Solution projection via primitive ===
        ngl = self.ngl
        parent_vals = self.q[active_idx * ngl : (active_idx + 1) * ngl]
        child1_vals, child2_vals = project_refine_single(
            parent_vals, self.PS1, self.PS2,
        )
        new_q = np.concatenate([
            self.q[: active_idx * ngl],
            child1_vals,
            child2_vals,
            self.q[(active_idx + 1) * ngl :],
        ])

        # === Update solver state ===
        self.xelem = new_grid
        self.active = new_active
        self.nelem = len(new_active)
        self.q = new_q
        self.npoin_dg = ngl * self.nelem
        self.npoin_cg = self.nop * self.nelem + 1

        # === Rebuild grid connectivity ===
        self.coord, self.intma, self.periodicity = create_grid_us(
            self.ngl, self.nelem, self.npoin_cg, self.npoin_dg,
            self.xgl, self.xelem,
        )

        return True
    
    def coarsen_element(self, active_idx):
        """Coarsen an element by merging it with its sibling into their parent.

        Finds the sibling in the active array, calls the stateless primitives
        from adapt.py to update mesh topology and project the solution, and
        rebuilds grid connectivity.

        Does NOT rebuild DG matrices (_update_matrices), forcing
        (_update_forcing), or recompute timestep. The caller is responsible
        for these after all mesh modifications (including balance enforcement)
        are complete. See refine_element() docstring for the typical caller
        pattern.

        Args:
            active_idx: Index into self.active (0-based) of one of the two
                siblings to coarsen. The method finds the other sibling
                automatically.

        Returns:
            True if coarsening was performed, False if:
            - Element is at level 0 (no parent)
            - Sibling is not adjacent in the active array
        """
        elem = self.active[active_idx]
        parent = self.label_mat[elem - 1][1]

        # Level 0 elements have no parent — cannot coarsen
        if parent == 0:
            return False

        # === Find sibling in adjacent active-array positions ===
        sib_idx = None

        # Check left neighbor
        if active_idx > 0:
            left_elem = self.active[active_idx - 1]
            if self.label_mat[left_elem - 1][1] == parent:
                sib_idx = active_idx - 1

        # Check right neighbor
        if sib_idx is None and active_idx < len(self.active) - 1:
            right_elem = self.active[active_idx + 1]
            if self.label_mat[right_elem - 1][1] == parent:
                sib_idx = active_idx + 1

        if sib_idx is None:
            return False

        # === Determine left/right ordering ===
        left_idx = min(active_idx, sib_idx)
        right_idx = max(active_idx, sib_idx)

        # === Mesh topology via primitive ===
        new_grid, new_active, success = coarsen_pair(
            self.nop, self.xelem, self.active, self.label_mat,
            left_idx, right_idx,
        )
        if not success:
            return False

        # === Solution projection via primitive ===
        ngl = self.ngl
        child1_vals = self.q[left_idx * ngl : (left_idx + 1) * ngl]
        child2_vals = self.q[right_idx * ngl : (right_idx + 1) * ngl]
        parent_vals = project_coarsen_single(
            child1_vals, child2_vals, self.PG1, self.PG2,
        )
        new_q = np.concatenate([
            self.q[: left_idx * ngl],
            parent_vals,
            self.q[(right_idx + 1) * ngl :],
        ])

        # === Update solver state ===
        self.xelem = new_grid
        self.active = new_active
        self.nelem = len(new_active)
        self.q = new_q
        self.npoin_dg = ngl * self.nelem
        self.npoin_cg = self.nop * self.nelem + 1

        # === Rebuild grid connectivity ===
        self.coord, self.intma, self.periodicity = create_grid_us(
            self.ngl, self.nelem, self.npoin_cg, self.npoin_dg,
            self.xgl, self.xelem,
        )

        return True
    
    def adapt_mesh(self, criterion=1, update_dt=True, balance=None):
        """
        Perform mesh adaptation based on marking criterion or explicit marks.
        
        This is the main entry point for mesh adaptation. It can either use
        an automatic marking criterion or explicit refinement/coarsening marks
        provided by an external controller (e.g., RL agent).
        
        Args:
            criterion: Marking criterion ID for automatic marking.
            marks_override: Dict mapping element index to mark value
                (-1=coarsen, 0=no change, 1=refine). Overrides criterion.
            element_budget: Maximum elements allowed. Refinement requests
                that would exceed budget are rejected.
            update_dt: Whether to recompute time step after adaptation.
            balance: Whether to enforce 2:1 balance after adaptation.
        
        Raises:
            ValueError: If marks_override contains invalid element indices.
        """

        marks = mark(self.active, self.label_mat, self.intma, self.q, criterion)
        
        # Store pre-adaptation state
        pre_coord = self.coord
        pre_active = self.active
        
        # Perform mesh adaptation
        new_grid, new_active, _, new_nelem, npoin_cg, new_npoin_dg = adapt_mesh(
            self.nop, self.xelem, self.active, self.label_mat, 
            self.info_mat, marks, self.max_level
        )
        
        # Create new grid connectivity
        new_coord, new_intma, new_periodicity = create_grid_us(
            self.ngl, new_nelem, npoin_cg, new_npoin_dg, 
            self.xgl, new_grid
        )
        
        # Project solution to new mesh
        q_new = adapt_sol(
            self.q, pre_coord, marks, pre_active, self.label_mat,
            self.PS1, self.PS2, self.PG1, self.PG2, self.ngl
        )
        
        # Update solver state
        self.q = q_new
        self.active = new_active
        self.nelem = new_nelem
        self.intma = new_intma
        self.coord = new_coord
        self.xelem = new_grid
        self.npoin_dg = new_npoin_dg
        self.periodicity = new_periodicity
        
        # Apply balancing if requested
        self.balance_mesh(balance)
        
        # Update operators for new mesh
        self._update_matrices()
        self._update_forcing()
        self.verify_state()
        
        if update_dt:
            self._compute_timestep(use_actual_max_level=True)
    
    
    def initialize_with_refinement(self, refinement_mode='none', refinement_level=2, 
                                   refinement_probability=0.5):
        """
        Reinitialize solver with specified initial mesh refinement.
        
        Args:
            refinement_mode: One of 'none', 'fixed', or 'random'.
            refinement_level: Target level for 'fixed' mode, or max level for 'random'.
            refinement_probability: Probability of refining each element in 'random' mode.
        """
        # Reset to base mesh
        self._initialize_mesh()
        
        if refinement_mode == 'none':
            return
        
        # Cap at max level
        if refinement_level > self.max_level:
            refinement_level = self.max_level
        
        # Apply refinement
        if refinement_mode == 'fixed':
            self._perform_fixed_refinement(refinement_level)
        elif refinement_mode == 'random':
            self._perform_random_refinement(refinement_level, refinement_probability)
        
        # Reinitialize solution and operators
        self.q = self._initialize_solution()
        self._update_matrices()
        self._compute_timestep(use_actual_max_level=True)
        self.verify_state()
    
    def _perform_fixed_refinement(self, target_level):
        """
        Refine all elements to specified level.
        
        Args:
            target_level: Target refinement level (all elements refined to this level).
        """
        for _level in range(target_level):
            # Mark all active elements for refinement
            marks = np.ones(len(self.active), dtype=int)
            
            # Apply refinement
            new_grid, new_active, _, new_nelem, npoin_cg, new_npoin_dg = adapt_mesh(
                self.nop, self.xelem, self.active, self.label_mat, 
                self.info_mat, marks, self.max_level
            )
            
            # Create new grid
            new_coord, new_intma, new_periodicity = create_grid_us(
                self.ngl, new_nelem, npoin_cg, new_npoin_dg, 
                self.xgl, new_grid
            )
            
            # Project solution (zeros since we reinitialize after)
            q_new = adapt_sol(
                np.zeros_like(self.q), self.coord, marks, self.active, self.label_mat,
                self.PS1, self.PS2, self.PG1, self.PG2, self.ngl
            )
            
            # Update state
            self.q = q_new
            self.active = new_active
            self.nelem = new_nelem
            self.intma = new_intma
            self.coord = new_coord
            self.xelem = new_grid
            self.npoin_dg = new_npoin_dg
            self.periodicity = new_periodicity
    
    def _perform_random_refinement(self, max_initial_level, probability):
        """
        Randomly refine elements up to specified level.
        
        Args:
            max_initial_level: Maximum number of refinement passes.
            probability: Probability of refining each element per pass.
        """
        for _level in range(max_initial_level):
            # Randomly mark elements
            n_active = len(self.active)
            marks = (np.random.random(n_active) < probability).astype(int)
            
            if not np.any(marks):
                continue
            
            # Apply refinement
            new_grid, new_active, _, new_nelem, npoin_cg, new_npoin_dg = adapt_mesh(
                self.nop, self.xelem, self.active, self.label_mat, 
                self.info_mat, marks, self.max_level
            )
            
            new_coord, new_intma, new_periodicity = create_grid_us(
                self.ngl, new_nelem, npoin_cg, new_npoin_dg, 
                self.xgl, new_grid
            )
            
            q_new = adapt_sol(
                np.zeros_like(self.q), self.coord, marks, self.active, self.label_mat,
                self.PS1, self.PS2, self.PG1, self.PG2, self.ngl
            )
            
            # Update state
            self.q = q_new
            self.active = new_active
            self.nelem = new_nelem
            self.intma = new_intma
            self.coord = new_coord
            self.xelem = new_grid
            self.npoin_dg = new_npoin_dg
            self.periodicity = new_periodicity
            
            # Enforce balance if enabled
            self.balance_mesh(self.balance)
    
    def reset(self, refinement_mode='none', refinement_level=0, 
              refinement_probability=0.5, refinement_max_level=3,
              icase=None):
        """
        Reset solver to initial state with optional refinement.
        
        Used by RL environment to reset episodes with varied initial meshes.
        
        Args:
            refinement_mode: One of 'none', 'fixed', or 'random'.
            refinement_level: Target level for 'fixed' mode.
            refinement_probability: Probability for 'random' mode.
            refinement_max_level: Max level for 'random' mode.
            icase: Test case identifier. If provided, switches the IC type
                for this episode. If None, keeps the current icase.
            
        Returns:
            Initial solution vector.
        """
        # Switch IC type if requested (multi-IC training support)
        if icase is not None:
            self.icase = icase
        
        # Reset to original 4-element grid
        self.xelem = np.array([-1, -0.4, 0, 0.4, 1])
        self.nelem = len(self.xelem) - 1
        
        # Recompute grid parameters
        self.npoin_cg = self.nop * self.nelem + 1
        self.npoin_dg = self.ngl * self.nelem
        
        # Reset forest
        self.label_mat, self.info_mat, self.active = forest(self.xelem, self.max_level)
        
        # Create grid
        self.coord, self.intma, self.periodicity = create_grid_us(
            self.ngl, self.nelem, self.npoin_cg, self.npoin_dg, 
            self.xgl, self.xelem
        )
        
        # Apply initial refinement
        if refinement_mode == 'fixed' and refinement_level > 0:
            self._perform_fixed_refinement(refinement_level)
        elif refinement_mode == 'random' and refinement_max_level > 0:
            self._perform_random_refinement(refinement_max_level, refinement_probability)
        
        # Reset solution
        self.q, _ = exact_solution(self.coord, self.npoin_dg, 0.0, self.icase)
        self.time = 0.0
        
        # Validate mesh
        quality_ok, issues = self.check_mesh_quality(self.xelem)
        if not quality_ok:
            raise ValueError(f"Initial mesh quality issues: {issues}")
        
        # Update operators
        self._update_matrices()
        self._compute_timestep(use_actual_max_level=True)
        self.verify_state()
        
        return self.q

    # =========================================================================
    # Time Stepping Methods
    # =========================================================================
    
    def step(self, dt=None):
        """
        Advance solution by one time step using low-storage RK method.
        
        Uses a 5-stage, 4th-order low-storage Runge-Kutta scheme that requires
        only 2N storage (solution + increment) rather than the usual 6N for
        a standard RK4 implementation.
        
        Args:
            dt: Time step size. If None, uses self.dt.
        """
        if dt is None:
            dt = self.dt
        
        # Low-storage RK coefficients (Carpenter-Kennedy 4th order)
        RKA = np.array([
            0.0,
            -567301805773.0/1357537059087.0,
            -2404267990393.0/2016746695238.0,
            -3550918686646.0/2091501179385.0,
            -1275806237668.0/842570457699.0
        ])
        
        RKB = np.array([
            1432997174477.0/9575080441755.0,
            5161836677717.0/13612068292357.0,
            1720146321549.0/2090206949498.0,
            3134564353537.0/4481467310338.0,
            2277821191437.0/14882151754819.0
        ])
        
        dq = np.zeros(self.npoin_dg)
        qp = self.q.copy()
        
        # RK stages
        for s in range(len(RKA)):
            R = self.Dhat @ qp
            
            for i in range(self.npoin_dg):
                dq[i] = RKA[s] * dq[i] + dt * R[i]
                qp[i] = qp[i] + RKB[s] * dq[i]
            
            # Enforce periodicity
            if self.periodicity[-1] == self.periodicity[0]:
                qp[-1] = qp[0]
        
        self.q = qp
        self.time += dt
    
