"""Tests for adapt.py mesh adaptation routines.

Tests cover:
    - mark: Element marking based on solution threshold
    - check_balance: 2:1 balance verification
    - balance_mark: Identifying elements needing refinement for balance
    - adapt_mesh: Mesh refinement and coarsening
    - adapt_sol: Solution projection during adaptation

Setup uses a 4-element base mesh with max_level=2, giving 28 possible elements.
"""

import numpy as np
import pytest

from backends.python_1d.amr.adapt import adapt_mesh, adapt_sol, balance_mark, check_balance, mark

# Import modules under test
from backends.python_1d.amr.forest import forest
from backends.python_1d.amr.projection import projections
from backends.python_1d.dg.basis import Lagrange_basis, lgl_gen
from backends.python_1d.dg.matrices import create_RM_matrix
from backends.python_1d.grid.mesh import create_grid_us

# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def base_setup():
    """Create minimal forest and mesh for testing.
    
    4 base elements, max_level=2 → 28 total elements in forest.
    Uses polynomial order nop=4 (ngl=5 nodes per element).
    """
    # Domain and discretization
    xmin, xmax = 0.0, 1.0
    nelem0 = 4  # Base elements
    nop = 4     # Polynomial order
    ngl = nop + 1
    max_level = 2
    
    # Build forest (all possible elements)
    xelem0 = np.linspace(xmin, xmax, nelem0 + 1)
    label_mat, info_mat, active = forest(xelem0, max_level)
    
    # Initial uniform mesh (base elements only)
    grid = xelem0.copy()
    nelem = len(active)
    
    # Get LGL nodes and weights
    xgl, wgl = lgl_gen(ngl)
    nq = ngl
    xnq, wnq = xgl, wgl
    
    # Compute mesh dimensions
    npoin_cg = nop * nelem + 1
    npoin_dg = ngl * nelem
    
    # Create DG mesh
    coord, intma, periodicity = create_grid_us(ngl, nelem, npoin_cg, npoin_dg, xgl, grid)
    
    # Create projection matrices
    psi, dpsi = Lagrange_basis(ngl, nq, xgl, xnq)
    RM = create_RM_matrix(ngl, nq, wnq, psi)
    PS1, PS2, PG1, PG2 = projections(RM, ngl, nq, wnq, xgl, xnq)
    
    return {
        'label_mat': label_mat,
        'info_mat': info_mat,
        'active': active,
        'grid': grid,
        'coord': coord,
        'intma': intma,
        'periodicity': periodicity,
        'nop': nop,
        'ngl': ngl,
        'nelem': nelem,
        'npoin_cg': npoin_cg,
        'npoin_dg': npoin_dg,
        'max_level': max_level,
        'xgl': xgl,
        'wgl': wgl,
        'PS1': PS1,
        'PS2': PS2,
        'PG1': PG1,
        'PG2': PG2,
    }


@pytest.fixture
def refined_setup(base_setup):
    """Create a mesh with element 2 refined (for coarsening tests).
    
    Starting from base 4-element mesh, refine element 2.
    Result: elements [1, 7, 8, 3, 4] with 5 active elements.
    """
    s = base_setup
    
    # Marks: refine element 2 only
    marks = np.array([0, 1, 0, 0])
    
    # Adapt mesh
    new_grid, new_active, _, new_nelem, new_npoin_cg, new_npoin_dg = adapt_mesh(
        s['nop'], s['grid'], s['active'], s['label_mat'], 
        s['info_mat'], marks, s['max_level']
    )
    
    # Create new DG mesh
    new_coord, new_intma, new_periodicity = create_grid_us(
        s['ngl'], new_nelem, new_npoin_cg, new_npoin_dg, s['xgl'], new_grid
    )
    
    return {
        **s,
        'active': new_active,
        'grid': new_grid,
        'coord': new_coord,
        'intma': new_intma,
        'periodicity': new_periodicity,
        'nelem': new_nelem,
        'npoin_cg': new_npoin_cg,
        'npoin_dg': new_npoin_dg,
    }


# ============================================================================
# Test: mark()
# ============================================================================

class TestMark:
    """Tests for mark() function."""
    
    def test_no_marks_when_uniform_below_threshold(self, base_setup):
        """All elements below threshold → no marks."""
        s = base_setup
        # Solution below threshold everywhere
        q = np.zeros(s['npoin_dg'])
        
        marks = mark(s['active'], s['label_mat'], s['intma'], q, 
                     criterion=1, threshold=0.5)
        
        assert np.all(marks == 0), "Should have no marks when solution < threshold"
    
    def test_refinement_when_above_threshold(self, base_setup):
        """Element above threshold → marked for refinement."""
        s = base_setup
        ngl = s['ngl']
        
        # Put high value on element 2 only (index 1)
        q = np.zeros(s['npoin_dg'])
        q[ngl:2*ngl] = 1.0  # Element 2 nodes
        
        marks = mark(s['active'], s['label_mat'], s['intma'], q,
                     criterion=1, threshold=0.5)
        
        assert marks[1] == 1, "Element 2 should be marked for refinement"
        assert marks[0] == 0, "Element 1 should not be marked"
        assert marks[2] == 0, "Element 3 should not be marked"
        assert marks[3] == 0, "Element 4 should not be marked"
    
    def test_coarsening_requires_both_siblings(self, refined_setup):
        """Coarsening only occurs when both siblings qualify."""
        s = refined_setup
        ngl = s['ngl']
        
        # Active is [1, 5, 6, 3, 4] — elements 5, 6 are siblings
        # Put low values on both siblings (indices 1 and 2)
        q = np.ones(s['npoin_dg'])  # All above threshold
        q[ngl:3*ngl] = 0.0  # Elements 5 and 6 below threshold
        
        marks = mark(s['active'], s['label_mat'], s['intma'], q,
                     criterion=1, threshold=0.5)
        
        # Both siblings should be marked for coarsening
        assert marks[1] == -1, "Element 5 should be marked for coarsening"
        assert marks[2] == -1, "Element 6 should be marked for coarsening"
    
    def test_no_coarsening_when_one_sibling_above_threshold(self, refined_setup):
        """If only one sibling qualifies, neither is coarsened."""
        s = refined_setup
        ngl = s['ngl']

        # Active is [1, 7, 8, 3, 4] — elements 7, 8 are siblings (children of element 2)
        # Put low value on element 7, high on element 8
        q = np.ones(s['npoin_dg'])
        q[ngl:2*ngl] = 0.0  # Element 7 below threshold
        # Element 8 stays above threshold

        marks = mark(s['active'], s['label_mat'], s['intma'], q,
                    criterion=1, threshold=0.5)

        # Neither sibling should be marked for coarsening (-1)
        # Element 7 qualifies for coarsening but sibling doesn't, so it stays 0
        # Element 8 is above threshold so it may be marked for refinement (+1)
        assert marks[1] != -1, "Element 7 should not be coarsened (sibling doesn't qualify)"
        assert marks[2] != -1, "Element 8 should not be coarsened (above threshold)"


# ============================================================================
# Test: check_balance()
# ============================================================================

class TestCheckBalance:
    """Tests for check_balance() function."""
    
    def test_uniform_mesh_is_balanced(self, base_setup):
        """Uniform mesh (all same level) is balanced."""
        s = base_setup
        assert check_balance(s['active'], s['label_mat']) 
    
    def test_one_refinement_is_balanced(self, refined_setup):
        """Single refinement maintains balance (level diff = 1)."""
        s = refined_setup
        # Active is [1, 5, 6, 3, 4] — levels are [0, 1, 1, 0, 0]
        assert check_balance(s['active'], s['label_mat']) 
    
    def test_two_level_jump_is_unbalanced(self, base_setup):
        """Two refinements on same element creates imbalance."""
        s = base_setup
        
        # Refine element 2 twice
        active = s['active'].copy()
        grid = s['grid'].copy()
        
        # First refinement
        marks = np.array([0, 1, 0, 0])
        grid, active, _, nelem, npoin_cg, npoin_dg = adapt_mesh(
            s['nop'], grid, active, s['label_mat'],
            s['info_mat'], marks, s['max_level']
        )
        
        # Second refinement of first child (element 5 at index 1)
        marks = np.array([0, 1, 0, 0, 0])
        grid, active, _, nelem, npoin_cg, npoin_dg = adapt_mesh(
            s['nop'], grid, active, s['label_mat'],
            s['info_mat'], marks, s['max_level']
        )
        
        # Active now has level-2 elements adjacent to level-0 elements
        assert not check_balance(active, s['label_mat']) 


# ============================================================================
# Test: balance_mark()
# ============================================================================

class TestBalanceMark:
    """Tests for balance_mark() function."""
    
    def test_no_marks_for_balanced_mesh(self, base_setup):
        """Balanced mesh produces no marks."""
        s = base_setup
        marks = balance_mark(s['active'], s['label_mat'])
        assert np.all(marks == 0)
    
    def test_marks_coarser_neighbor(self, base_setup):
        """Marks the coarser element when level difference > 1."""
        s = base_setup
        
        # Create imbalanced mesh: refine element 2 twice
        active = s['active'].copy()
        grid = s['grid'].copy()
        
        # First refinement
        marks = np.array([0, 1, 0, 0])
        grid, active, _, nelem, npoin_cg, npoin_dg = adapt_mesh(
            s['nop'], grid, active, s['label_mat'],
            s['info_mat'], marks, s['max_level']
        )
        
        # Second refinement
        marks = np.array([0, 1, 0, 0, 0])
        grid, active, _, nelem, npoin_cg, npoin_dg = adapt_mesh(
            s['nop'], grid, active, s['label_mat'],
            s['info_mat'], marks, s['max_level']
        )
        
        # Get balance marks
        bal_marks = balance_mark(active, s['label_mat'])
        
        # Should mark coarser neighbors for refinement
        assert np.sum(bal_marks) > 0, "Should mark at least one element"
        assert np.all(bal_marks >= 0), "Balance marks should be >= 0 (refine only)"


# ============================================================================
# Test: adapt_mesh()
# ============================================================================

class TestAdaptMesh:
    """Tests for adapt_mesh() function."""
    
    def test_no_change_with_zero_marks(self, base_setup):
        """Zero marks → no mesh changes."""
        s = base_setup
        marks = np.zeros(len(s['active']), dtype=int)
        
        new_grid, new_active, new_marks, new_nelem, _, _ = adapt_mesh(
            s['nop'], s['grid'], s['active'], s['label_mat'],
            s['info_mat'], marks, s['max_level']
        )
        
        assert new_nelem == len(s['active'])
        assert np.array_equal(new_active, s['active'])
        assert np.array_equal(new_grid, s['grid'])
    
    def test_single_refinement(self, base_setup):
        """Refinement splits one element into two children."""
        s = base_setup
        marks = np.array([0, 1, 0, 0])  # Refine element 2

        new_grid, new_active, _, new_nelem, _, new_npoin_dg = adapt_mesh(
            s['nop'], s['grid'], s['active'], s['label_mat'],
            s['info_mat'], marks, s['max_level']
        )

        # Should have 5 elements now
        assert new_nelem == 5
        assert len(new_active) == 5

        # Element 2's children are 7, 8 (from label_mat)
        assert 7 in new_active
        assert 8 in new_active
        assert 2 not in new_active

        # Grid should have one more point
        assert len(new_grid) == len(s['grid']) + 1
    
    def test_single_coarsening(self, refined_setup):
        """Coarsening merges two siblings into parent."""
        s = refined_setup
        # Active is [1, 5, 6, 3, 4]
        marks = np.array([0, -1, -1, 0, 0])  # Coarsen elements 5, 6
        
        new_grid, new_active, _, new_nelem, _, _ = adapt_mesh(
            s['nop'], s['grid'], s['active'], s['label_mat'],
            s['info_mat'], marks, s['max_level']
        )
        
        # Should be back to 4 elements
        assert new_nelem == 4
        assert len(new_active) == 4
        
        # Children removed, parent restored
        assert 5 not in new_active
        assert 6 not in new_active
        assert 2 in new_active
    
    def test_multiple_refinements(self, base_setup):
        """Multiple elements can be refined in one call."""
        s = base_setup
        marks = np.array([1, 1, 0, 0])  # Refine elements 1 and 2
        
        new_grid, new_active, _, new_nelem, _, _ = adapt_mesh(
            s['nop'], s['grid'], s['active'], s['label_mat'],
            s['info_mat'], marks, s['max_level']
        )
        
        # 4 - 2 + 4 = 6 elements
        assert new_nelem == 6
        
        # Children of element 1: 5, 6
        # Children of element 2: 7, 8
        assert 5 in new_active
        assert 6 in new_active
        assert 7 in new_active
        assert 8 in new_active


# ============================================================================
# Test: adapt_sol()
# ============================================================================

class TestAdaptSol:
    """Tests for adapt_sol() function."""
    
    def test_no_change_preserves_solution(self, base_setup):
        """Zero marks → solution unchanged."""
        s = base_setup
        ngl = s['ngl']
        
        # Linear solution: q = x
        q = s['coord'].copy()
        marks = np.zeros(s['nelem'], dtype=int)
        
        new_q = adapt_sol(q, s['coord'], marks, s['active'], s['label_mat'],
                          s['PS1'], s['PS2'], s['PG1'], s['PG2'], ngl)
        
        assert np.allclose(new_q, q)
    
    def test_refinement_increases_dof_count(self, base_setup):
        """Refinement should increase number of DOFs."""
        s = base_setup
        ngl = s['ngl']
        
        q = s['coord'].copy()  # q = x
        marks = np.array([0, 1, 0, 0])  # Refine element 2
        
        new_q = adapt_sol(q, s['coord'], marks, s['active'], s['label_mat'],
                          s['PS1'], s['PS2'], s['PG1'], s['PG2'], ngl)
        
        # New solution should have 5*ngl points (5 elements)
        assert len(new_q) == 5 * ngl
    
    def test_coarsening_decreases_dof_count(self, refined_setup):
        """Coarsening should decrease number of DOFs."""
        s = refined_setup
        ngl = s['ngl']
        
        q = s['coord'].copy()  # q = x
        marks = np.array([0, -1, -1, 0, 0])  # Coarsen 5, 6 back to 2
        
        new_q = adapt_sol(q, s['coord'], marks, s['active'], s['label_mat'],
                          s['PS1'], s['PS2'], s['PG1'], s['PG2'], ngl)
        
        # Should have 4*ngl points (back to 4 elements)
        assert len(new_q) == 4 * ngl
    
    def test_refine_then_coarsen_recovers_original(self, base_setup):
        """Scatter then gather should recover original solution (for polynomials)."""
        s = base_setup
        ngl = s['ngl']
        
        # Original solution (linear - exactly representable)
        q_orig = s['coord'].copy()
        
        # Refine element 2
        marks_refine = np.array([0, 1, 0, 0])
        q_refined = adapt_sol(q_orig, s['coord'], marks_refine, s['active'],
                              s['label_mat'], s['PS1'], s['PS2'], 
                              s['PG1'], s['PG2'], ngl)
        
        # Get refined mesh info
        grid_refined, active_refined, _, nelem_ref, npoin_cg_ref, npoin_dg_ref = adapt_mesh(
            s['nop'], s['grid'], s['active'], s['label_mat'],
            s['info_mat'], marks_refine, s['max_level']
        )
        coord_refined, _, _ = create_grid_us(
            ngl, nelem_ref, npoin_cg_ref, npoin_dg_ref, s['xgl'], grid_refined
        )
        
        # Coarsen back
        marks_coarsen = np.array([0, -1, -1, 0, 0])
        q_recovered = adapt_sol(q_refined, coord_refined, marks_coarsen,
                                active_refined, s['label_mat'],
                                s['PS1'], s['PS2'], s['PG1'], s['PG2'], ngl)
        
        # Should recover original solution
        assert np.allclose(q_recovered, q_orig, atol=1e-10)


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests combining multiple adapt functions."""
    
    def test_mark_adapt_cycle(self, base_setup):
        """Full cycle: mark → adapt_mesh → adapt_sol."""
        s = base_setup
        ngl = s['ngl']
        
        # Create solution with peak in element 2
        q = np.zeros(s['npoin_dg'])
        q[ngl:2*ngl] = 1.0
        
        # Mark based on solution
        marks = mark(s['active'], s['label_mat'], s['intma'], q,
                     criterion=1, threshold=0.5)
        
        # Adapt mesh
        new_grid, new_active, _, new_nelem, new_npoin_cg, new_npoin_dg = adapt_mesh(
            s['nop'], s['grid'], s['active'], s['label_mat'],
            s['info_mat'], marks, s['max_level']
        )
        
        # Adapt solution
        new_q = adapt_sol(q, s['coord'], marks, s['active'], s['label_mat'],
                          s['PS1'], s['PS2'], s['PG1'], s['PG2'], ngl)
        
        # Verify consistency
        assert len(new_q) == ngl * new_nelem
        assert len(new_active) == new_nelem
        assert len(new_grid) == new_nelem + 1


# ============================================================================
# Run tests
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v'])