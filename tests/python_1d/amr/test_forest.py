"""Tests for numerical/amr/forest.py"""
import numpy as np

from backends.python_1d.amr.forest import forest, get_active_levels, level_arrays, next_level


def test_next_level():
    """next_level should insert midpoints between coordinates."""
    xelem = np.array([-1.0, 0.0, 1.0])
    result = next_level(xelem)
    expected = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"
    print("test_next_level PASSED")


def test_level_arrays():
    """level_arrays should generate correct number of levels."""
    xelem = np.array([-1.0, 0.0, 1.0])
    levels = level_arrays(xelem, max_level=2)
    assert len(levels) == 3, f"Expected 3 levels, got {len(levels)}"
    assert len(levels[0]) == 3   # Level 0: 3 points
    assert len(levels[1]) == 5   # Level 1: 5 points
    assert len(levels[2]) == 9   # Level 2: 9 points
    print("test_level_arrays PASSED")


def test_forest_shapes():
    """forest should return arrays with correct shapes."""
    xelem0 = np.array([-1.0, -0.4, 0.0, 0.4, 1.0])
    max_level = 3
    nelem0 = len(xelem0) - 1  # 4 elements
    
    label_mat, info_mat, active = forest(xelem0, max_level)
    
    # Total elements = nelem0 * (2^(max_level+1) - 1) = 4 * 15 = 60
    total_expected = nelem0 * (2**(max_level + 1) - 1)
    
    assert label_mat.shape == (total_expected, 5), f"label_mat shape: {label_mat.shape}"
    assert info_mat.shape == (total_expected, 5), f"info_mat shape: {info_mat.shape}"
    assert len(active) == nelem0, f"active length: {len(active)}"
    print("test_forest_shapes PASSED")


def test_forest_base_elements():
    """Base elements should have no parent and correct children."""
    xelem0 = np.array([-1.0, 0.0, 1.0])  # 2 elements
    label_mat, info_mat, active = forest(xelem0, max_level=2)
    
    # Active should be base elements [1, 2]
    assert np.array_equal(active, np.array([1, 2])), f"active: {active}"
    
    # Element 1: no parent (0), children 3 and 4
    assert label_mat[0, 1] == 0, "Element 1 should have no parent"
    assert label_mat[0, 2] == 3, "Element 1 left child should be 3"
    assert label_mat[0, 3] == 4, "Element 1 right child should be 4"
    assert label_mat[0, 4] == 0, "Element 1 should be level 0"
    
    # Element 2: no parent (0), children 5 and 6
    assert label_mat[1, 1] == 0, "Element 2 should have no parent"
    assert label_mat[1, 2] == 5, "Element 2 left child should be 5"
    assert label_mat[1, 3] == 6, "Element 2 right child should be 6"
    assert label_mat[1, 4] == 0, "Element 2 should be level 0"
    
    print("test_forest_base_elements PASSED")


def test_forest_child_parent_consistency():
    """Children should point back to correct parent."""
    xelem0 = np.array([-1.0, 0.0, 1.0])  # 2 elements
    label_mat, info_mat, active = forest(xelem0, max_level=2)
    
    # Element 3 (left child of 1) should have parent 1
    assert label_mat[2, 1] == 1, f"Element 3 parent: {label_mat[2, 1]}"
    # Element 4 (right child of 1) should have parent 1
    assert label_mat[3, 1] == 1, f"Element 4 parent: {label_mat[3, 1]}"
    # Element 5 (left child of 2) should have parent 2
    assert label_mat[4, 1] == 2, f"Element 5 parent: {label_mat[4, 1]}"
    
    print("test_forest_child_parent_consistency PASSED")


def test_forest_coordinates():
    """Element coordinates should match expected values."""
    xelem0 = np.array([-1.0, 0.0, 1.0])  # 2 elements
    label_mat, info_mat, active = forest(xelem0, max_level=1)
    
    # Element 1: [-1.0, 0.0]
    assert np.isclose(info_mat[0, 3], -1.0), f"Element 1 left: {info_mat[0, 3]}"
    assert np.isclose(info_mat[0, 4], 0.0), f"Element 1 right: {info_mat[0, 4]}"
    
    # Element 3 (left child of 1): [-1.0, -0.5]
    assert np.isclose(info_mat[2, 3], -1.0), f"Element 3 left: {info_mat[2, 3]}"
    assert np.isclose(info_mat[2, 4], -0.5), f"Element 3 right: {info_mat[2, 4]}"
    
    print("test_forest_coordinates PASSED")


def test_get_active_levels():
    """get_active_levels should return correct levels for active elements."""
    xelem0 = np.array([-1.0, 0.0, 1.0])
    label_mat, info_mat, active = forest(xelem0, max_level=2)
    
    # Initially active = [1, 2], both at level 0
    levels = get_active_levels(active, label_mat)
    assert np.array_equal(levels, np.array([0, 0])), f"levels: {levels}"
    
    # Simulate refinement: active = [3, 4, 2] (element 1 refined)
    active_refined = np.array([3, 4, 2])
    levels_refined = get_active_levels(active_refined, label_mat)
    expected = np.array([1, 1, 0])  # children at level 1, element 2 still at level 0
    assert np.array_equal(levels_refined, expected), f"levels: {levels_refined}"
    
    print("test_get_active_levels PASSED")


if __name__ == "__main__":
    test_next_level()
    test_level_arrays()
    test_forest_shapes()
    test_forest_base_elements()
    test_forest_child_parent_consistency()
    test_forest_coordinates()
    test_get_active_levels()
    print("\n=== All tests PASSED ===")