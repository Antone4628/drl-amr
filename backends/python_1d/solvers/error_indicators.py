"""Error indicators for DG adaptive mesh refinement.

Standalone utility functions for computing per-element error indicators,
alpha-based thresholds, and normalized error observations. Used by both the
multiround RL environment and the threshold AMR baseline.

The primary error indicator is the boundary jump magnitude — the absolute
solution discontinuity at element interfaces. For DG methods, the exact
solution is continuous, so interface jumps measure discretization error.
This is in the same family as DynAMO's h-refinement estimator (ZZ-type
interface non-conformity), simplified to raw jump magnitude.

Architecture Decisions:
    D-032: ZZ-style patch-projection estimator (primary; raw boundary jump
           retained for ablation)

alpha-normalization / thresholds (D-004) moved to agent/normalization.py
(RESTRUCTURE Phase 5) — pure math on the error vector, not indicator
computation. Max-over-interval accumulation (D-008) lives in the
solver-advance / reward path, not here.

References:
    - Dzanic et al. (2024), DynAMO — Eq. 15, 16, 21, 22
    - Foucart et al. (2023) — boundary jump observation (Component A)
    - Cockburn (2003) — relationship between DG interface jumps and error

Usage:
   >>> from backends.python_1d.solvers.error_indicators import (
    ...     compute_element_errors,
    ...     compute_element_errors_zz,
    ...     compute_errors,
    ... )
    >>> errors = compute_element_errors(solver)
    >>> errors_zz = compute_element_errors_zz(solver)
    >>> errors_dispatched = compute_errors(solver, indicator='zz_style')
"""

import numpy as np

from ..amr.mesh_utils import find_neighbor_index
from ..amr.projection import create_zz_patch_projection
from ..dg.basis import Lagrange_basis

# =========================================================================
# Core: Per-element error indicators
# =========================================================================

# compute_element_errors(solver) -> np.ndarray
# Compute boundary jump magnitudes for all active elements.
# Returns array of shape (n_active,) aligned with solver.active.

def compute_element_errors(solver):
    """Compute per-element error indicators for all active elements.
    
    Uses boundary jump magnitude as the error indicator: for each element,
    computes the absolute solution discontinuity at both interfaces with
    neighbors and returns the average.
    
    For DG methods, the exact solution is continuous, so interface jumps
    measure discretization error. This is in the same family as DynAMO's
    h-refinement estimator (ZZ-type interface non-conformity), simplified
    to raw jump magnitude.
    
    Args:
        solver: DGAdvectionSolver instance with current mesh and solution.
            Must have: q, intma, active, label_mat attributes.
            Assumes periodic boundary conditions.
    
    Returns:
        errors: np.ndarray of shape (n_active,) containing the average
            boundary jump magnitude for each active element. Elements
            are ordered consistently with solver.active.
    """
    n_active = len(solver.active)
    errors = np.zeros(n_active)
    
    for i in range(n_active):
        # Get current element's solution at boundary nodes
        elem_nodes = solver.intma[:, i]
        elem_sol = solver.q[elem_nodes]
        elem_left = elem_sol[0]      # leftmost LGL node value
        elem_right = elem_sol[-1]    # rightmost LGL node value
        
        jumps = []
        
        # Left neighbor
        left_idx = find_neighbor_index(solver, i, direction='left')
        if left_idx >= 0:
            left_nodes = solver.intma[:, left_idx]
            left_sol = solver.q[left_nodes]
            jumps.append(abs(elem_left - left_sol[-1]))
        
        # Right neighbor
        right_idx = find_neighbor_index(solver, i, direction='right')
        if right_idx >= 0:
            right_nodes = solver.intma[:, right_idx]
            right_sol = solver.q[right_nodes]
            jumps.append(abs(elem_right - right_sol[0]))
        
        errors[i] = np.mean(jumps) if jumps else 0.0
    
    return errors

# =========================================================================
# ZZ-style error indicator (D-032)
# =========================================================================

def _zz_patch_residual_squared(
    u_left, u_right, h_left, h_right, target_side, u_target_at_q,
    ngl, nq, wnq, xgl, xnq,
):
    """Squared L² residual of a ZZ patch projection on one element of the patch.

    Builds the two-element patch projection from the supplied nodal data and
    integrates (u_h - v_patch)² over the *target* element only — the one
    being assigned an error indicator. The patch is the pair (left, right)
    of physical elements; `target_side` selects which is Ω_k.

    The geometric subtlety this resolves: an element Ω_k contributes to two
    patches, the left-edge patch (Ω_{k-1}, Ω_k) and the right-edge patch
    (Ω_k, Ω_{k+1}). Ω_k is the RIGHT half of the left-edge patch and the
    LEFT half of the right-edge patch, with different element-ref → patch-ref
    mappings in each case. `target_side` makes that distinction explicit so
    the caller does not need to thread the right ξ-mapping through.

    Args:
        u_left:  Nodal values on the left element of the patch, shape (ngl,).
        u_right: Nodal values on the right element of the patch, shape (ngl,).
        h_left:  Physical width of the left element.
        h_right: Physical width of the right element.
        target_side: 'left' or 'right' — which element of the patch is Ω_k.
        u_target_at_q: u_h on Ω_k evaluated at element-ref quadrature points,
            shape (nq,). Passed in rather than recomputed so the caller can
            evaluate it once per element and reuse it across both patches.
        ngl, nq, wnq, xgl, xnq: DG basis/quadrature parameters (typically
            solver.ngl, solver.nq, solver.wnq, solver.xgl, solver.xnq).
            nq must satisfy nq >= ngl + 1 — see
            create_zz_patch_projection docstring for the derivation.

    Returns:
        Squared L² residual ∫_{Ω_target} (u_h - v_patch)² dx
        (a non-negative float).
    """
    P = create_zz_patch_projection(h_left, h_right, ngl, nq, wnq, xgl, xnq)
    v_coeffs = P @ np.concatenate([u_left, u_right])

    h_p = h_left + h_right
    if target_side == 'right':
        # Ω_target = right element. Element-ref ζ → patch-ref ξ:
        #     ξ = (2 h_left + (1+ζ) h_right) / h_p - 1
        xi_target = (2.0 * h_left + (1.0 + xnq) * h_right) / h_p - 1.0
        h_target = h_right
    else:  # 'left'
        # Ω_target = left element. Element-ref ζ → patch-ref ξ:
        #     ξ = (1+ζ) h_left / h_p - 1
        xi_target = (1.0 + xnq) * h_left / h_p - 1.0
        h_target = h_left

    # Patch basis at the target's element-ref quadrature locations
    # (mapped into patch-ref): psi_patch[a, q] = L_a^patch(xi_target[q]).
    psi_patch, _ = Lagrange_basis(ngl, nq, xgl, xi_target)
    v_at_q = psi_patch.T @ v_coeffs

    diff = u_target_at_q - v_at_q
    return (h_target / 2.0) * np.sum(wnq * diff * diff)


def compute_element_errors_zz(solver):
    """Compute per-element ZZ-style error indicators for all active elements.

    For each element Ω_k, builds two two-element patch projections —
    (Ω_{k-1}, Ω_k) on the left and (Ω_k, Ω_{k+1}) on the right — and L²-
    projects each patch's piecewise DG data to a single degree-(ngl-1)
    polynomial across the patch. The L² residual is then measured on Ω_k
    itself for each patch. The two edge contributions are combined via RMS:

        e_k = sqrt(0.5 * (e_k^{(L)2} + e_k^{(R)2}))

    Unlike the raw boundary jump (`compute_element_errors`), this estimator
    is generically nonzero at t=0: it compares full polynomial shapes across
    a patch rather than just nodal values at the shared interface, so it
    sees mismatched-shapes-with-matched-endpoints cases that raw jumps miss.
    This is the t=0 deployment-initialization fix motivating D-032.

    Architecture Decisions:
        D-032: Adopt ZZ-style estimator as primary error indicator
               (replaces raw jump as primary; raw jump retained for ablation).

    Method reference:
        strategy/architecture_description/zz_estimator_method.tex

    Args:
        solver: DGAdvectionSolver instance with current mesh and solution.
            Must expose: q, intma, active, xelem, ngl, nq, wnq, xgl, xnq.
            Assumes periodic boundary conditions (every active element has
            both neighbors via find_neighbor_index's modular wrap).

    Returns:
        errors: np.ndarray of shape (n_active,) containing the ZZ-style
            error indicator for each active element. Ordered consistently
            with solver.active.

    See Also:
        numerical.amr.projection.create_zz_patch_projection: builds the
            per-patch L² projection operator consumed here.
        compute_element_errors: raw boundary-jump variant (legacy / ablation).
    """
    n_active = len(solver.active)
    errors = np.zeros(n_active)

    ngl = solver.ngl
    nq = solver.nq
    wnq = solver.wnq
    xgl = solver.xgl
    xnq = solver.xnq

    # Element basis at element-ref quadrature; shape (ngl, nq) and identical
    # for every element. Hoisted out of the loop since it depends only on
    # (ngl, xgl, xnq) — all solver-fixed.
    psi_elem, _ = Lagrange_basis(ngl, nq, xgl, xnq)

    for i in range(n_active):
        # Element k state
        u_k = solver.q[solver.intma[:, i]]
        h_k = solver.xelem[i + 1] - solver.xelem[i]
        u_k_at_q = psi_elem.T @ u_k  # u_h on Ω_k at quadrature pts, shape (nq,)

        # Left-edge patch (Ω_left, Ω_k) — Ω_k is the RIGHT element
        left_idx = find_neighbor_index(solver, i, direction='left')
        u_left = solver.q[solver.intma[:, left_idx]]
        h_left = solver.xelem[left_idx + 1] - solver.xelem[left_idx]
        e_L_sq = _zz_patch_residual_squared(
            u_left, u_k, h_left, h_k, 'right', u_k_at_q,
            ngl, nq, wnq, xgl, xnq,
        )

        # Right-edge patch (Ω_k, Ω_right) — Ω_k is the LEFT element
        right_idx = find_neighbor_index(solver, i, direction='right')
        u_right = solver.q[solver.intma[:, right_idx]]
        h_right = solver.xelem[right_idx + 1] - solver.xelem[right_idx]
        e_R_sq = _zz_patch_residual_squared(
            u_k, u_right, h_k, h_right, 'left', u_k_at_q,
            ngl, nq, wnq, xgl, xnq,
        )

        errors[i] = np.sqrt(0.5 * (e_L_sq + e_R_sq))

    return errors


# =========================================================================
# Indicator Registry and Dispatcher
# =========================================================================
# Extensible registry mapping string keys to indicator functions.
# Each registered function has signature: f(solver) -> np.ndarray
# returning per-element error values for all active elements.
#
# To add a new indicator:
#   1. Implement compute_element_errors_foo(solver) above
#   2. Add 'foo': compute_element_errors_foo to INDICATOR_REGISTRY
#   3. Use error_indicator: "foo" in YAML config
#
# Used by DGAMREnvMultiround via compute_errors() dispatcher.
# =========================================================================

INDICATOR_REGISTRY = {
    'raw_jump': compute_element_errors,
    'zz_style': compute_element_errors_zz,
}


def compute_errors(solver, indicator='zz_style'):
    """Dispatch to the named error indicator function.

    Central entry point for all error indicator computation. The
    environment calls this instead of individual indicator functions,
    allowing indicator selection via config string.

    Args:
        solver: DGAdvectionSolver instance with an active mesh and
            current solution.
        indicator: String key into INDICATOR_REGISTRY. Must match a
            registered indicator name.

    Returns:
        np.ndarray of shape (n_active,) with per-element error values.

    Raises:
        ValueError: If ``indicator`` is not in INDICATOR_REGISTRY.
    """
    if indicator not in INDICATOR_REGISTRY:
        raise ValueError(
            f"Unknown error indicator '{indicator}'. "
            f"Available: {sorted(INDICATOR_REGISTRY.keys())}"
        )
    return INDICATOR_REGISTRY[indicator](solver)