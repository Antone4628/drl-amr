"""Tests for numerical/dg/matrices.py

Tests cover all DG matrix construction functions:
    - create_diff_matrix: Element differentiation matrix
    - create_mass_matrix: Element mass matrices
    - create_RM_matrix: Reference mass matrix
    - create_mass_matrix_vectorized: Vectorized mass matrix construction
    - Fmatrix_upwind_flux: Upwind flux matrix
    - Fmatrix_upwind_flux_bc: Upwind flux with boundary conditions
    - Fmatrix_centered_flux: Centered flux matrix
    - Fmatrix_rusanov_flux: Rusanov (Lax-Friedrichs) flux matrix
    - Matrix_DSS: Direct stiffness summation for global assembly

Run with: pytest tests/python_1d/dg/test_matrices.py -v
"""


import numpy as np
import pytest

from backends.python_1d.dg.basis import Lagrange_basis, lgl_gen
from backends.python_1d.dg.matrices import (
    Fmatrix_centered_flux,
    Fmatrix_rusanov_flux,
    Fmatrix_upwind_flux,
    Fmatrix_upwind_flux_bc,
    Matrix_DSS,
    create_diff_matrix,
    create_mass_matrix,
    create_mass_matrix_vectorized,
    create_RM_matrix,
)
from backends.python_1d.grid.mesh import create_grid_us

# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def basis_p2():
    """LGL nodes, weights, and basis functions for P=2 (3 points)."""
    ngl = 3
    xgl, wgl = lgl_gen(ngl)
    nq = ngl
    xnq, wnq = xgl, wgl
    psi, dpsi = Lagrange_basis(ngl, nq, xgl, xnq)
    
    return {
        'ngl': ngl,
        'nq': nq,
        'xgl': xgl,
        'wgl': wgl,
        'wnq': wnq,
        'psi': psi,
        'dpsi': dpsi
    }


@pytest.fixture
def basis_p4():
    """LGL nodes, weights, and basis functions for P=4 (5 points)."""
    ngl = 5
    xgl, wgl = lgl_gen(ngl)
    nq = ngl
    xnq, wnq = xgl, wgl
    psi, dpsi = Lagrange_basis(ngl, nq, xgl, xnq)
    
    return {
        'ngl': ngl,
        'nq': nq,
        'xgl': xgl,
        'wgl': wgl,
        'wnq': wnq,
        'psi': psi,
        'dpsi': dpsi
    }


@pytest.fixture
def mesh_4elem(basis_p4):
    """4-element uniform mesh on [0, 1] with P=4."""
    nelem = 4
    ngl = basis_p4['ngl']
    nop = ngl - 1
    xgl = basis_p4['xgl']
    
    xelem = np.linspace(0, 1, nelem + 1)
    npoin_cg = nop * nelem + 1
    npoin_dg = ngl * nelem
    
    coord, intma, periodicity = create_grid_us(
        ngl, nelem, npoin_cg, npoin_dg, xgl, xelem
    )
    
    return {
        'nelem': nelem,
        'ngl': ngl,
        'npoin_cg': npoin_cg,
        'npoin_dg': npoin_dg,
        'coord': coord,
        'intma': intma,
        'periodicity': periodicity,
        'xelem': xelem
    }


@pytest.fixture
def mesh_single_elem(basis_p4):
    """Single element mesh on [0, 1] with P=4."""
    nelem = 1
    ngl = basis_p4['ngl']
    nop = ngl - 1
    xgl = basis_p4['xgl']
    
    xelem = np.array([0.0, 1.0])
    npoin_cg = nop * nelem + 1
    npoin_dg = ngl * nelem
    
    coord, intma, periodicity = create_grid_us(
        ngl, nelem, npoin_cg, npoin_dg, xgl, xelem
    )
    
    return {
        'nelem': nelem,
        'ngl': ngl,
        'npoin_cg': npoin_cg,
        'npoin_dg': npoin_dg,
        'coord': coord,
        'intma': intma,
        'periodicity': periodicity,
        'xelem': xelem
    }


# =============================================================================
# Tests: create_diff_matrix
# =============================================================================

class TestCreateDiffMatrix:
    """Tests for element differentiation matrix construction."""
    
    def test_shape(self, basis_p4):
        """Diff matrix should be ngl x ngl."""
        b = basis_p4
        D = create_diff_matrix(b['ngl'], b['nq'], b['wnq'], b['psi'], b['dpsi'])
        
        assert D.shape == (b['ngl'], b['ngl'])
    
    def test_shape_p2(self, basis_p2):
        """Diff matrix shape for lower order."""
        b = basis_p2
        D = create_diff_matrix(b['ngl'], b['nq'], b['wnq'], b['psi'], b['dpsi'])
        
        assert D.shape == (b['ngl'], b['ngl'])
    
    def test_not_symmetric(self, basis_p4):
        """Diff matrix should not be symmetric."""
        b = basis_p4
        D = create_diff_matrix(b['ngl'], b['nq'], b['wnq'], b['psi'], b['dpsi'])
        
        # Differentiation is not a symmetric operation
        assert not np.allclose(D, D.T)
    
    def test_acts_on_linear(self, basis_p4):
        """D applied to linear function should give nonzero result."""
        b = basis_p4
        D = create_diff_matrix(b['ngl'], b['nq'], b['wnq'], b['psi'], b['dpsi'])
        
        # Linear function on reference element
        f_nodes = b['xgl'].copy()
        Df = D @ f_nodes
        
        assert not np.allclose(Df, 0)


# =============================================================================
# Tests: create_mass_matrix
# =============================================================================

class TestCreateMassMatrix:
    """Tests for element mass matrix construction."""
    
    def test_shape(self, basis_p4, mesh_4elem):
        """Mass matrix should be (nelem, ngl, ngl)."""
        b = basis_p4
        m = mesh_4elem
        
        Me = create_mass_matrix(
            m['intma'], m['coord'], m['nelem'],
            b['ngl'], b['nq'], b['wnq'], b['psi']
        )
        
        assert Me.shape == (m['nelem'], b['ngl'], b['ngl'])
    
    def test_symmetric(self, basis_p4, mesh_4elem):
        """Each element mass matrix should be symmetric."""
        b = basis_p4
        m = mesh_4elem
        
        Me = create_mass_matrix(
            m['intma'], m['coord'], m['nelem'],
            b['ngl'], b['nq'], b['wnq'], b['psi']
        )
        
        for e in range(m['nelem']):
            assert np.allclose(Me[e], Me[e].T), f"Element {e} not symmetric"
    
    def test_positive_definite(self, basis_p4, mesh_4elem):
        """Each element mass matrix should be positive definite."""
        b = basis_p4
        m = mesh_4elem
        
        Me = create_mass_matrix(
            m['intma'], m['coord'], m['nelem'],
            b['ngl'], b['nq'], b['wnq'], b['psi']
        )
        
        for e in range(m['nelem']):
            eigenvalues = np.linalg.eigvalsh(Me[e])
            assert np.all(eigenvalues > 0), f"Element {e} not positive definite"
    
    def test_scales_with_element_size(self, basis_p4):
        """Mass matrix scales linearly with element size."""
        b = basis_p4
        ngl = b['ngl']
        nop = ngl - 1
        
        # Single element [0, 1]
        xelem1 = np.array([0.0, 1.0])
        npoin_cg1 = nop + 1
        npoin_dg1 = ngl
        coord1, intma1, _ = create_grid_us(ngl, 1, npoin_cg1, npoin_dg1, b['xgl'], xelem1)
        Me1 = create_mass_matrix(intma1, coord1, 1, ngl, b['nq'], b['wnq'], b['psi'])
        
        # Single element [0, 2] (double size)
        xelem2 = np.array([0.0, 2.0])
        coord2, intma2, _ = create_grid_us(ngl, 1, npoin_cg1, npoin_dg1, b['xgl'], xelem2)
        Me2 = create_mass_matrix(intma2, coord2, 1, ngl, b['nq'], b['wnq'], b['psi'])
        
        # Me2 should be 2x Me1
        assert np.allclose(Me2[0], 2 * Me1[0])
    
    def test_single_element(self, basis_p4, mesh_single_elem):
        """Mass matrix works for single element mesh."""
        b = basis_p4
        m = mesh_single_elem
        
        Me = create_mass_matrix(
            m['intma'], m['coord'], m['nelem'],
            b['ngl'], b['nq'], b['wnq'], b['psi']
        )
        
        assert Me.shape == (1, b['ngl'], b['ngl'])
        assert np.all(np.linalg.eigvalsh(Me[0]) > 0)


# =============================================================================
# Tests: create_RM_matrix
# =============================================================================

class TestCreateRMMatrix:
    """Tests for reference mass matrix construction."""
    
    def test_shape(self, basis_p4):
        """Reference mass matrix should be ngl x ngl."""
        b = basis_p4
        RM = create_RM_matrix(b['ngl'], b['nq'], b['wnq'], b['psi'])
        
        assert RM.shape == (b['ngl'], b['ngl'])
    
    def test_symmetric(self, basis_p4):
        """Reference mass matrix should be symmetric."""
        b = basis_p4
        RM = create_RM_matrix(b['ngl'], b['nq'], b['wnq'], b['psi'])
        
        assert np.allclose(RM, RM.T)
    
    def test_positive_definite(self, basis_p4):
        """Reference mass matrix should be positive definite."""
        b = basis_p4
        RM = create_RM_matrix(b['ngl'], b['nq'], b['wnq'], b['psi'])
        
        eigenvalues = np.linalg.eigvalsh(RM)
        assert np.all(eigenvalues > 0)
    
    def test_integrates_to_two(self, basis_p4):
        """Total integral on [-1, 1] should be 2."""
        b = basis_p4
        RM = create_RM_matrix(b['ngl'], b['nq'], b['wnq'], b['psi'])
        
        # Sum of all entries ≈ integral of 1 over [-1, 1] = 2
        total = np.sum(RM)
        assert np.isclose(total, 2.0, rtol=0.01)
    
    def test_invertible(self, basis_p4):
        """Reference mass matrix should be invertible."""
        b = basis_p4
        RM = create_RM_matrix(b['ngl'], b['nq'], b['wnq'], b['psi'])
        
        RM_inv = np.linalg.inv(RM)
        assert np.allclose(RM @ RM_inv, np.eye(b['ngl']))


# =============================================================================
# Tests: create_mass_matrix_vectorized
# =============================================================================

class TestCreateMassMatrixVectorized:
    """Tests for vectorized mass matrix construction."""
    
    def test_matches_standard(self, basis_p4, mesh_4elem):
        """Vectorized version should match standard version."""
        b = basis_p4
        m = mesh_4elem
        
        Me_standard = create_mass_matrix(
            m['intma'], m['coord'], m['nelem'],
            b['ngl'], b['nq'], b['wnq'], b['psi']
        )
        
        Me_vectorized = create_mass_matrix_vectorized(
            m['intma'], m['coord'], m['nelem'],
            b['ngl'], b['nq'], b['wnq'], b['psi']
        )
        
        assert np.allclose(Me_standard, Me_vectorized)
    
    def test_shape(self, basis_p4, mesh_4elem):
        """Vectorized version should have correct shape."""
        b = basis_p4
        m = mesh_4elem
        
        Me = create_mass_matrix_vectorized(
            m['intma'], m['coord'], m['nelem'],
            b['ngl'], b['nq'], b['wnq'], b['psi']
        )
        
        assert Me.shape == (m['nelem'], b['ngl'], b['ngl'])


# =============================================================================
# Tests: Fmatrix_upwind_flux
# =============================================================================

class TestFmatrixUpwindFlux:
    """Tests for upwind flux matrix construction."""
    
    def test_shape(self, mesh_4elem):
        """Flux matrix should be npoin x npoin."""
        m = mesh_4elem
        
        F = Fmatrix_upwind_flux(
            m['intma'], m['nelem'], m['npoin_dg'],
            m['ngl'], 1.0, periodic=True
        )
        
        assert F.shape == (m['npoin_dg'], m['npoin_dg'])
    
    def test_sparse_pattern(self, mesh_4elem):
        """Flux matrix should be sparse."""
        m = mesh_4elem
        
        F = Fmatrix_upwind_flux(
            m['intma'], m['nelem'], m['npoin_dg'],
            m['ngl'], 1.0, periodic=True
        )
        
        nnz = np.count_nonzero(F)
        total = m['npoin_dg'] ** 2
        
        # Flux matrix should be very sparse (< 20% nonzero)
        assert nnz < 0.2 * total
    
    def test_scales_with_wave_speed(self, mesh_4elem):
        """Flux matrix should scale linearly with wave speed."""
        m = mesh_4elem
        
        F1 = Fmatrix_upwind_flux(
            m['intma'], m['nelem'], m['npoin_dg'],
            m['ngl'], 1.0, periodic=True
        )
        
        F2 = Fmatrix_upwind_flux(
            m['intma'], m['nelem'], m['npoin_dg'],
            m['ngl'], 2.0, periodic=True
        )
        
        assert np.allclose(F2, 2 * F1)
    
    def test_zero_wave_speed(self, mesh_4elem):
        """Zero wave speed should give zero flux matrix."""
        m = mesh_4elem
        
        F = Fmatrix_upwind_flux(
            m['intma'], m['nelem'], m['npoin_dg'],
            m['ngl'], 0.0, periodic=True
        )
        
        assert np.allclose(F, 0)
    
    def test_periodic_vs_nonperiodic(self, mesh_4elem):
        """Periodic and non-periodic should differ."""
        m = mesh_4elem
        
        F_periodic = Fmatrix_upwind_flux(
            m['intma'], m['nelem'], m['npoin_dg'],
            m['ngl'], 1.0, periodic=True
        )
        
        F_nonperiodic = Fmatrix_upwind_flux(
            m['intma'], m['nelem'], m['npoin_dg'],
            m['ngl'], 1.0, periodic=False
        )
        
        assert not np.allclose(F_periodic, F_nonperiodic)


# =============================================================================
# Tests: Fmatrix_upwind_flux_bc
# =============================================================================

class TestFmatrixUpwindFluxBC:
    """Tests for upwind flux matrix with boundary conditions."""
    
    def test_shape(self, mesh_4elem):
        """Flux matrix should be npoin x npoin."""
        m = mesh_4elem
        
        F = Fmatrix_upwind_flux_bc(
            m['intma'], m['nelem'], m['npoin_dg'],
            m['ngl'], 1.0, periodic=False
        )
        
        assert F.shape == (m['npoin_dg'], m['npoin_dg'])
    
    def test_periodic_matches_standard(self, mesh_4elem):
        """Periodic BC version should match standard upwind flux."""
        m = mesh_4elem
        
        F_standard = Fmatrix_upwind_flux(
            m['intma'], m['nelem'], m['npoin_dg'],
            m['ngl'], 1.0, periodic=True
        )
        
        F_bc = Fmatrix_upwind_flux_bc(
            m['intma'], m['nelem'], m['npoin_dg'],
            m['ngl'], 1.0, periodic=True
        )
        
        assert np.allclose(F_standard, F_bc)


# =============================================================================
# Tests: Fmatrix_centered_flux
# =============================================================================

class TestFmatrixCenteredFlux:
    """Tests for centered flux matrix construction."""
    
    def test_shape(self, mesh_4elem):
        """Flux matrix should be npoin x npoin."""
        m = mesh_4elem
        
        F = Fmatrix_centered_flux(
            m['intma'], m['nelem'], m['npoin_dg'],
            m['ngl'], 1.0
        )
        
        assert F.shape == (m['npoin_dg'], m['npoin_dg'])
    
    def test_scales_with_wave_speed(self, mesh_4elem):
        """Centered flux should scale with wave speed."""
        m = mesh_4elem
        
        F1 = Fmatrix_centered_flux(
            m['intma'], m['nelem'], m['npoin_dg'],
            m['ngl'], 1.0
        )
        
        F2 = Fmatrix_centered_flux(
            m['intma'], m['nelem'], m['npoin_dg'],
            m['ngl'], 2.0
        )
        
        assert np.allclose(F2, 2 * F1)
    
    def test_different_from_upwind(self, mesh_4elem):
        """Centered flux should differ from upwind flux."""
        m = mesh_4elem
        
        F_upwind = Fmatrix_upwind_flux(
            m['intma'], m['nelem'], m['npoin_dg'],
            m['ngl'], 1.0, periodic=True
        )
        
        F_centered = Fmatrix_centered_flux(
            m['intma'], m['nelem'], m['npoin_dg'],
            m['ngl'], 1.0
        )
        
        assert not np.allclose(F_upwind, F_centered)


# =============================================================================
# Tests: Fmatrix_rusanov_flux
# =============================================================================

class TestFmatrixRusanovFlux:
    """Tests for Rusanov flux matrix construction."""
    
    def test_shape(self, mesh_4elem):
        """Flux matrix should be npoin x npoin."""
        m = mesh_4elem
        
        F = Fmatrix_rusanov_flux(
            m['intma'], m['nelem'], m['npoin_dg'],
            m['ngl'], 1.0, periodic=True
        )
        
        assert F.shape == (m['npoin_dg'], m['npoin_dg'])
    
    def test_periodic_vs_nonperiodic(self, mesh_4elem):
        """Periodic and non-periodic should differ."""
        m = mesh_4elem
        
        F_periodic = Fmatrix_rusanov_flux(
            m['intma'], m['nelem'], m['npoin_dg'],
            m['ngl'], 1.0, periodic=True
        )
        
        F_nonperiodic = Fmatrix_rusanov_flux(
            m['intma'], m['nelem'], m['npoin_dg'],
            m['ngl'], 1.0, periodic=False
        )
        
        assert not np.allclose(F_periodic, F_nonperiodic)
    
    def test_includes_dissipation(self, mesh_4elem):
        """Rusanov flux should have off-diagonal elements."""
        m = mesh_4elem
        
        F = Fmatrix_rusanov_flux(
            m['intma'], m['nelem'], m['npoin_dg'],
            m['ngl'], 1.0, periodic=True
        )
        
        # Check it's not purely diagonal
        off_diag = F - np.diag(np.diag(F))
        assert not np.allclose(off_diag, 0)


# =============================================================================
# Tests: Matrix_DSS
# =============================================================================

class TestMatrixDSS:
    """Tests for global matrix assembly via DSS."""
    
    def test_output_shapes(self, basis_p4, mesh_4elem):
        """DSS should produce global matrices of correct shape."""
        b = basis_p4
        m = mesh_4elem
        
        Me = create_mass_matrix(
            m['intma'], m['coord'], m['nelem'],
            b['ngl'], b['nq'], b['wnq'], b['psi']
        )
        De = create_diff_matrix(b['ngl'], b['nq'], b['wnq'], b['psi'], b['dpsi'])
        
        M, D = Matrix_DSS(
            Me, De, 1.0, m['intma'], m['periodicity'],
            b['ngl'], m['nelem'], m['npoin_dg']
        )
        
        assert M.shape == (m['npoin_dg'], m['npoin_dg'])
        assert D.shape == (m['npoin_dg'], m['npoin_dg'])
    
    def test_mass_matrix_symmetric(self, basis_p4, mesh_4elem):
        """Global mass matrix should be symmetric."""
        b = basis_p4
        m = mesh_4elem
        
        Me = create_mass_matrix(
            m['intma'], m['coord'], m['nelem'],
            b['ngl'], b['nq'], b['wnq'], b['psi']
        )
        De = create_diff_matrix(b['ngl'], b['nq'], b['wnq'], b['psi'], b['dpsi'])
        
        M, D = Matrix_DSS(
            Me, De, 1.0, m['intma'], m['periodicity'],
            b['ngl'], m['nelem'], m['npoin_dg']
        )
        
        assert np.allclose(M, M.T)
    
    def test_mass_matrix_positive_semidefinite(self, basis_p4, mesh_4elem):
        """Global mass matrix should be positive semidefinite."""
        b = basis_p4
        m = mesh_4elem
        
        Me = create_mass_matrix(
            m['intma'], m['coord'], m['nelem'],
            b['ngl'], b['nq'], b['wnq'], b['psi']
        )
        De = create_diff_matrix(b['ngl'], b['nq'], b['wnq'], b['psi'], b['dpsi'])
        
        M, D = Matrix_DSS(
            Me, De, 1.0, m['intma'], m['periodicity'],
            b['ngl'], m['nelem'], m['npoin_dg']
        )
        
        eigenvalues = np.linalg.eigvalsh(M)
        assert np.all(eigenvalues >= -1e-10)
    
    def test_diff_matrix_scales_with_wave_speed(self, basis_p4, mesh_4elem):
        """Global diff matrix should scale with wave speed."""
        b = basis_p4
        m = mesh_4elem
        
        Me = create_mass_matrix(
            m['intma'], m['coord'], m['nelem'],
            b['ngl'], b['nq'], b['wnq'], b['psi']
        )
        De = create_diff_matrix(b['ngl'], b['nq'], b['wnq'], b['psi'], b['dpsi'])
        
        M1, D1 = Matrix_DSS(
            Me, De, 1.0, m['intma'], m['periodicity'],
            b['ngl'], m['nelem'], m['npoin_dg']
        )
        
        M2, D2 = Matrix_DSS(
            Me, De, 2.0, m['intma'], m['periodicity'],
            b['ngl'], m['nelem'], m['npoin_dg']
        )
        
        # Mass matrices should be identical
        assert np.allclose(M1, M2)
        
        # Diff matrix should scale with wave speed
        assert np.allclose(D2, 2 * D1)


# =============================================================================
# Tests: Edge Cases
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""
    
    def test_single_element_flux(self, mesh_single_elem):
        """Flux matrix works for single element."""
        m = mesh_single_elem
        
        F = Fmatrix_upwind_flux(
            m['intma'], m['nelem'], m['npoin_dg'],
            m['ngl'], 1.0, periodic=True
        )
        
        assert F.shape == (m['npoin_dg'], m['npoin_dg'])
    
    def test_negative_wave_speed(self, mesh_4elem):
        """Negative wave speed produces negative flux matrix."""
        m = mesh_4elem
        
        F_pos = Fmatrix_upwind_flux(
            m['intma'], m['nelem'], m['npoin_dg'],
            m['ngl'], 1.0, periodic=True
        )
        
        F_neg = Fmatrix_upwind_flux(
            m['intma'], m['nelem'], m['npoin_dg'],
            m['ngl'], -1.0, periodic=True
        )
        
        assert np.allclose(F_neg, -F_pos)
    
    def test_mass_matrix_well_conditioned(self, basis_p4, mesh_4elem):
        """Mass matrices should be well-conditioned."""
        b = basis_p4
        m = mesh_4elem
        
        Me = create_mass_matrix(
            m['intma'], m['coord'], m['nelem'],
            b['ngl'], b['nq'], b['wnq'], b['psi']
        )
        
        for e in range(m['nelem']):
            cond = np.linalg.cond(Me[e])
            assert cond < 100, f"Element {e} poorly conditioned: {cond}"
    
    def test_flux_matrix_sparsity(self, mesh_4elem):
        """Each row of flux matrix should have few nonzeros."""
        m = mesh_4elem
        
        F = Fmatrix_upwind_flux(
            m['intma'], m['nelem'], m['npoin_dg'],
            m['ngl'], 1.0, periodic=True
        )
        
        for i in range(m['npoin_dg']):
            nnz_row = np.count_nonzero(F[i, :])
            # Each row should have at most 2 nonzeros
            assert nnz_row <= 2, f"Row {i} has {nnz_row} nonzeros"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
