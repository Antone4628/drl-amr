"""Unit tests for AMR single-element primitives.

Tests the four primitive functions extracted from adapt_mesh() and adapt_sol():
    - refine_single()
    - coarsen_pair()
    - project_refine_single()
    - project_coarsen_single()

Also tests equivalence between the primitive path and the batch marks path
to verify that the refactoring preserved identical behavior.

Run: python -m tests.amr.test_amr_primitives
"""

import sys

import numpy as np

from backends.python_1d.amr.adapt import (
    adapt_mesh,
    adapt_sol,
    coarsen_pair,
    project_coarsen_single,
    project_refine_single,
    refine_single,
)
from backends.python_1d.amr.forest import forest
from backends.python_1d.amr.projection import projections
from backends.python_1d.dg.basis import Lagrange_basis, lgl_gen
from backends.python_1d.dg.matrices import create_RM_matrix
from backends.python_1d.grid.mesh import create_grid_us


def setup_test_mesh(nop=4, max_level=3):
    """Create a test mesh with forest structure and projection matrices.

    Returns a dict with all the data structures needed for testing.
    Uses the standard 4-element base grid [-1, -0.4, 0, 0.4, 1].
    """
    xelem = np.array([-1.0, -0.4, 0.0, 0.4, 1.0])
    ngl = nop + 1
    nq = nop + 2

    # Create forest
    label_mat, info_mat, active = forest(xelem, max_level)

    # Create LGL nodes and basis
    xgl, wgl = lgl_gen(ngl)
    xnq, wnq = lgl_gen(nq)
    psi, dpsi = Lagrange_basis(ngl, nq, xgl, xnq)

    # Create projection matrices
    RM = create_RM_matrix(ngl, nq, wnq, psi)
    PS1, PS2, PG1, PG2 = projections(RM, ngl, nq, wnq, xgl, xnq)

    # Create grid connectivity
    nelem = len(active)
    npoin_cg = nop * nelem + 1
    npoin_dg = ngl * nelem
    coord, intma, periodicity = create_grid_us(
        ngl, nelem, npoin_cg, npoin_dg, xgl, xelem
    )

    # Known solution: linear function q(x) = 2x + 1
    # Linear polynomials project exactly through scatter/gather
    q = 2.0 * coord + 1.0

    return {
        'nop': nop,
        'ngl': ngl,
        'max_level': max_level,
        'xelem': xelem,
        'label_mat': label_mat,
        'info_mat': info_mat,
        'active': active,
        'coord': coord,
        'intma': intma,
        'q': q,
        'PS1': PS1, 'PS2': PS2,
        'PG1': PG1, 'PG2': PG2,
        'xgl': xgl,
    }


# =====================================================================
# Task 1: refine_single — basic
# =====================================================================

def test_refine_single():
    """Refine a known element, verify grid and active arrays."""
    d = setup_test_mesh()

    grid = d['xelem'].copy()
    active = d['active'].copy()
    original_n = len(active)

    # Refine element at index 1 (second element, covers [-0.4, 0.0])
    new_grid, new_active, success = refine_single(
        d['nop'], grid, active, d['label_mat'], d['info_mat'],
        active_idx=1, max_level=d['max_level']
    )

    assert success, "Refinement should succeed for level-0 element"
    assert len(new_active) == original_n + 1, \
        f"Expected {original_n + 1} elements, got {len(new_active)}"
    assert len(new_grid) == len(d['xelem']) + 1, \
        "Grid should have one additional boundary point"

    # Midpoint of [-0.4, 0.0] should be -0.2
    assert np.isclose(new_grid[2], -0.2), \
        f"Expected midpoint at -0.2, got {new_grid[2]}"

    # Parent replaced by its two children
    parent_elem = d['active'][1]
    c1 = d['label_mat'][parent_elem - 1][2]
    c2 = d['label_mat'][parent_elem - 1][3]
    assert new_active[1] == c1, "Child 1 should be at index 1"
    assert new_active[2] == c2, "Child 2 should be at index 2"

    # Other elements shifted but unchanged
    assert new_active[0] == active[0], "Element 0 unchanged"
    assert new_active[3] == active[2], "Original element 2 shifted to index 3"
    assert new_active[4] == active[3], "Original element 3 shifted to index 4"

    print("  PASS: refine_single produces correct grid and active arrays")


# =====================================================================
# Task 2: refine_single — max level guard
# =====================================================================

def test_refine_single_max_level():
    """Refinement fails at max_level."""
    d = setup_test_mesh(max_level=3)

    grid = d['xelem'].copy()
    active = d['active'].copy()

    # Refine element 0 repeatedly up to max_level
    for level in range(d['max_level']):
        grid, active, success = refine_single(
            d['nop'], grid, active, d['label_mat'], d['info_mat'],
            active_idx=0, max_level=d['max_level']
        )
        assert success, f"Refinement at level {level} should succeed"

    # One more should fail
    grid_before = grid.copy()
    active_before = active.copy()
    grid, active, success = refine_single(
        d['nop'], grid, active, d['label_mat'], d['info_mat'],
        active_idx=0, max_level=d['max_level']
    )

    assert not success, "Refinement at max_level should fail"
    assert np.array_equal(grid, grid_before), "Grid unchanged on failure"
    assert np.array_equal(active, active_before), "Active unchanged on failure"

    print("  PASS: refine_single correctly blocked at max_level")


# =====================================================================
# Task 3: coarsen_pair — basic
# =====================================================================

def test_coarsen_pair():
    """Coarsen a sibling pair created by refinement."""
    d = setup_test_mesh()

    # Refine element 1 to create a coarsenable pair
    grid, active, success = refine_single(
        d['nop'], d['xelem'].copy(), d['active'].copy(),
        d['label_mat'], d['info_mat'],
        active_idx=1, max_level=d['max_level']
    )
    assert success, "Setup refinement should succeed"
    n_after_refine = len(active)

    # Coarsen the pair at indices 1 and 2
    new_grid, new_active, success = coarsen_pair(
        d['nop'], grid, active, d['label_mat'],
        left_idx=1, right_idx=2
    )

    assert success, "Coarsening sibling pair should succeed"
    assert len(new_active) == n_after_refine - 1, \
        f"Expected {n_after_refine - 1} elements, got {len(new_active)}"

    # Should be back to original state
    assert np.array_equal(new_grid, d['xelem']), \
        "Grid should return to original after refine + coarsen"
    assert np.array_equal(new_active, d['active']), \
        "Active should return to original after refine + coarsen"

    print("  PASS: coarsen_pair produces correct grid and active arrays")


# =====================================================================
# Task 4: coarsen_pair — invalid (non-siblings)
# =====================================================================

def test_coarsen_pair_invalid():
    """Coarsening fails for elements that don't share a parent."""
    d = setup_test_mesh()

    grid = d['xelem'].copy()
    active = d['active'].copy()

    # Base-level elements have parent=0, so coarsening should fail
    new_grid, new_active, success = coarsen_pair(
        d['nop'], grid, active, d['label_mat'],
        left_idx=0, right_idx=1
    )

    assert not success, "Coarsening non-siblings should fail"
    assert np.array_equal(new_grid, grid), "Grid unchanged on failure"
    assert np.array_equal(new_active, active), "Active unchanged on failure"

    print("  PASS: coarsen_pair correctly rejects non-siblings")


# =====================================================================
# Task 5: project_refine_single — polynomial exactness
# =====================================================================

def test_project_refine_single():
    """Scatter projection preserves a linear solution exactly."""
    d = setup_test_mesh()
    ngl = d['ngl']

    # Parent solution on element 1 (covers [-0.4, 0.0])
    parent_vals = d['q'][1 * ngl:2 * ngl]

    c1_vals, c2_vals = project_refine_single(parent_vals, d['PS1'], d['PS2'])

    assert c1_vals.shape == (ngl,), f"Child 1 should have {ngl} values"
    assert c2_vals.shape == (ngl,), f"Child 2 should have {ngl} values"

    # Refine to get children's physical coordinates
    grid, active, _ = refine_single(
        d['nop'], d['xelem'].copy(), d['active'].copy(),
        d['label_mat'], d['info_mat'],
        active_idx=1, max_level=d['max_level']
    )
    new_nelem = len(active)
    new_npoin_cg = d['nop'] * new_nelem + 1
    new_npoin_dg = ngl * new_nelem
    new_coord, _, _ = create_grid_us(
        ngl, new_nelem, new_npoin_cg, new_npoin_dg, d['xgl'], grid
    )

    # Evaluate exact linear function on children's coordinates
    expected_c1 = 2.0 * new_coord[1 * ngl:2 * ngl] + 1.0
    expected_c2 = 2.0 * new_coord[2 * ngl:3 * ngl] + 1.0

    assert np.allclose(c1_vals, expected_c1, atol=1e-12), \
        f"Child 1 error: {np.max(np.abs(c1_vals - expected_c1)):.2e}"
    assert np.allclose(c2_vals, expected_c2, atol=1e-12), \
        f"Child 2 error: {np.max(np.abs(c2_vals - expected_c2)):.2e}"

    print("  PASS: project_refine_single preserves linear solution exactly")


# =====================================================================
# Task 6: project_coarsen_single — roundtrip exactness
# =====================================================================

def test_project_coarsen_single():
    """Scatter then gather roundtrip preserves a linear solution."""
    d = setup_test_mesh()
    ngl = d['ngl']

    # Parent solution on element 1
    parent_vals_original = d['q'][1 * ngl:2 * ngl]

    # Scatter to children
    c1_vals, c2_vals = project_refine_single(
        parent_vals_original, d['PS1'], d['PS2']
    )

    # Gather back to parent
    parent_vals_recovered = project_coarsen_single(
        c1_vals, c2_vals, d['PG1'], d['PG2']
    )

    assert np.allclose(parent_vals_recovered, parent_vals_original, atol=1e-12), \
        f"Roundtrip error: {np.max(np.abs(parent_vals_recovered - parent_vals_original)):.2e}"

    print("  PASS: project_coarsen_single roundtrip preserves linear solution")


# =====================================================================
# Task 7: Roundtrip topology (refine + coarsen = identity)
# =====================================================================

def test_roundtrip_topology():
    """Refine then coarsen returns to original mesh state."""
    d = setup_test_mesh()

    original_grid = d['xelem'].copy()
    original_active = d['active'].copy()

    # Refine element 2
    grid, active, _ = refine_single(
        d['nop'], d['xelem'].copy(), d['active'].copy(),
        d['label_mat'], d['info_mat'],
        active_idx=2, max_level=d['max_level']
    )

    # Coarsen the pair back
    grid, active, success = coarsen_pair(
        d['nop'], grid, active, d['label_mat'],
        left_idx=2, right_idx=3
    )

    assert success
    assert np.array_equal(grid, original_grid), "Grid should match original"
    assert np.array_equal(active, original_active), "Active should match original"

    print("  PASS: refine + coarsen roundtrip restores original topology")


# =====================================================================
# Task 8: Equivalence — refine (primitive vs batch)
# =====================================================================

def test_equivalence_refine():
    """Primitive refine path produces identical results to batch marks path."""
    d = setup_test_mesh()
    ngl = d['ngl']
    nop = d['nop']

    # --- Batch path ---
    marks_batch = np.array([0, 1, 0, 0])
    grid_batch, active_batch, _, _, _, _ = adapt_mesh(
        nop, d['xelem'].copy(), d['active'].copy(),
        d['label_mat'], d['info_mat'],
        marks_batch, d['max_level']
    )
    q_batch = adapt_sol(
        d['q'].copy(), d['coord'], np.array([0, 1, 0, 0]),
        d['active'].copy(), d['label_mat'],
        d['PS1'], d['PS2'], d['PG1'], d['PG2'], ngl
    )

    # --- Primitive path ---
    grid_prim, active_prim, success = refine_single(
        nop, d['xelem'].copy(), d['active'].copy(),
        d['label_mat'], d['info_mat'],
        active_idx=1, max_level=d['max_level']
    )

    parent_vals = d['q'][1 * ngl:2 * ngl]
    c1_vals, c2_vals = project_refine_single(parent_vals, d['PS1'], d['PS2'])

    # Reconstruct full q: [elem0 | child1 | child2 | elem2 | elem3]
    q_prim = np.concatenate([
        d['q'][0 * ngl:1 * ngl],
        c1_vals,
        c2_vals,
        d['q'][2 * ngl:3 * ngl],
        d['q'][3 * ngl:4 * ngl],
    ])

    # --- Compare ---
    assert np.array_equal(grid_prim, grid_batch), \
        "Grids differ between primitive and batch paths"
    assert np.array_equal(active_prim, active_batch), \
        "Active arrays differ between primitive and batch paths"
    assert np.allclose(q_prim, q_batch, atol=1e-15), \
        f"Solutions differ: max = {np.max(np.abs(q_prim - q_batch)):.2e}"

    print("  PASS: refine equivalence — primitive matches batch")


# =====================================================================
# Task 9: Equivalence — coarsen (primitive vs batch)
# =====================================================================

def test_equivalence_coarsen():
    """Primitive coarsen path produces identical results to batch marks path."""
    d = setup_test_mesh()
    ngl = d['ngl']
    nop = d['nop']

    # Setup: refine element 1 to create a coarsenable pair
    grid_setup, active_setup, _ = refine_single(
        nop, d['xelem'].copy(), d['active'].copy(),
        d['label_mat'], d['info_mat'],
        active_idx=1, max_level=d['max_level']
    )
    q_setup = adapt_sol(
        d['q'].copy(), d['coord'], np.array([0, 1, 0, 0]),
        d['active'].copy(), d['label_mat'],
        d['PS1'], d['PS2'], d['PG1'], d['PG2'], ngl
    )
    # Now: 5 elements [elem0 | child1 | child2 | elem2 | elem3]

    # --- Batch path ---
    marks_batch = np.array([0, -1, -1, 0, 0])
    grid_batch, active_batch, _, _, _, _ = adapt_mesh(
        nop, grid_setup.copy(), active_setup.copy(),
        d['label_mat'], d['info_mat'],
        marks_batch, d['max_level']
    )
    q_batch = adapt_sol(
        q_setup.copy(), None, np.array([0, -1, -1, 0, 0]),
        active_setup.copy(), d['label_mat'],
        d['PS1'], d['PS2'], d['PG1'], d['PG2'], ngl
    )

    # --- Primitive path ---
    grid_prim, active_prim, success = coarsen_pair(
        nop, grid_setup.copy(), active_setup.copy(),
        d['label_mat'], left_idx=1, right_idx=2
    )

    child1_vals = q_setup[1 * ngl:2 * ngl]
    child2_vals = q_setup[2 * ngl:3 * ngl]
    parent_vals = project_coarsen_single(
        child1_vals, child2_vals, d['PG1'], d['PG2']
    )

    q_prim = np.concatenate([
        q_setup[0 * ngl:1 * ngl],
        parent_vals,
        q_setup[3 * ngl:4 * ngl],
        q_setup[4 * ngl:5 * ngl],
    ])

    # --- Compare ---
    assert np.array_equal(grid_prim, grid_batch), \
        "Grids differ between primitive and batch paths"
    assert np.array_equal(active_prim, active_batch), \
        "Active arrays differ between primitive and batch paths"
    assert np.allclose(q_prim, q_batch, atol=1e-15), \
        f"Solutions differ: max = {np.max(np.abs(q_prim - q_batch)):.2e}"

    print("  PASS: coarsen equivalence — primitive matches batch")


# =====================================================================
# Runner
# =====================================================================

def main():
    print("\n" + "=" * 60)
    print("AMR Primitives Unit Tests")
    print("=" * 60)

    tests = [
        ("refine_single — basic", test_refine_single),
        ("refine_single — max level guard", test_refine_single_max_level),
        ("coarsen_pair — basic", test_coarsen_pair),
        ("coarsen_pair — invalid (non-siblings)", test_coarsen_pair_invalid),
        ("project_refine_single — polynomial exactness", test_project_refine_single),
        ("project_coarsen_single — roundtrip exactness", test_project_coarsen_single),
        ("roundtrip topology — refine + coarsen", test_roundtrip_topology),
        ("equivalence — refine (primitive vs batch)", test_equivalence_refine),
        ("equivalence — coarsen (primitive vs batch)", test_equivalence_coarsen),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        try:
            print(f"\nTask: {name}")
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  FAIL: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    print(f"{'=' * 60}")

    if failed > 0:
        sys.exit(1)
    print("\nAll tests passed!")


if __name__ == "__main__":
    main()