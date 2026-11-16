"""Tests for the mock backend's synthetic fields (backends/mock_quad/fields.py).

AGENT_2D_ROADMAP item 1.2, sitting 3, signals F1, F6, F7 and F10 at the
function level. One Gaussian feature of width w moves at a fixed velocity;
with d the distance from an element's centroid to the feature's center,
g = exp(-d**2 / (2 * w**2)) and rho = max(h) / w:

    ind_a = rho * (d / w) * g      a ring, zero at the center
    ind_b = rho**2 * g             a bump, largest at the center
    true_error = rho**3 * g        the auxiliary field

The instance is the explanation doc's (section 1.2.14): two base cells A and
B on [0,2]x[0,1], periodic in x and walled in y, at level 0. The feature
starts on A's centroid with width 0.25 and velocity (1, 0), so rho = 4 and
at time 1 it sits on B's centroid.

The verbs built on these functions are tested in
test_mock_backend_interval.py.

Run: pytest tests/mock_quad/test_mock_fields.py -v
"""
import numpy as np
import pytest

from backends.mock_quad.fields import (
    AUX_FIELDS,
    INDICATORS,
    REDUCTIONS,
    check_names,
    evaluate,
    mass,
    offsets,
    raw_fields,
)
from backends.mock_quad.quadtree import QuadGrid, order_leaves, topology, uniform_leaves

IC = {"x0": 0.5, "y0": 0.5, "width": 0.25}
VELOCITY = (1.0, 0.0)
E = np.exp

A_CHILDREN = [(1, 0, 0), (1, 1, 0), (1, 0, 1), (1, 1, 1)]
B = (0, 1, 0)


def two_cell_grid():
    """Two base cells side by side: periodic in x, walled in y."""
    return QuadGrid(nx=2, ny=1, periodic=(True, False), lo=(0.0, 0.0), hi=(2.0, 1.0), max_level=2)


def walled_grid():
    """A walled 2x2 grid on the unit square."""
    return QuadGrid(nx=2, ny=2, periodic=(False, False), lo=(0.0, 0.0), hi=(1.0, 1.0), max_level=2)


def geometry(grid, leaves):
    """The centroids and sizes of a mesh, as the backend hands them over."""
    arrays = topology(grid, order_leaves(grid, leaves))
    for name in ("centroid", "h"):
        arrays[name].flags.writeable = False
    return arrays["centroid"], arrays["h"]


def two_cells():
    grid = two_cell_grid()
    return (grid, *geometry(grid, uniform_leaves(grid, 0)))


def points(*xy):
    """Centroids given directly, each with a unit-size element."""
    centroid = np.array(xy, dtype=np.float64)
    return centroid, np.ones_like(centroid)


# --- The names -------------------------------------------------------------

def test_registered_names():
    assert INDICATORS == ("ind_a", "ind_b")
    assert AUX_FIELDS == ("true_error",)
    assert REDUCTIONS == ("scalar", "per_variable")
    assert not set(INDICATORS) & set(AUX_FIELDS)


# --- offsets ---------------------------------------------------------------

def test_offsets_at_the_start():
    grid, centroid, _ = two_cells()
    delta = offsets(grid, centroid, IC, VELOCITY, 0.0)
    # B is exactly half the domain away: either image, one unit.
    assert np.abs(delta).tolist() == [[0.0, 0.0], [1.0, 0.0]]


def test_offsets_are_a_new_float64_array():
    grid, centroid, _ = two_cells()
    delta = offsets(grid, centroid, IC, VELOCITY, 0.0)
    assert delta.dtype == np.float64 and delta.shape == (2, 2)
    assert delta.flags.writeable
    assert delta is not centroid


def test_the_feature_moves_with_the_velocity():
    grid, centroid, _ = two_cells()
    delta = offsets(grid, centroid, IC, VELOCITY, 0.25)
    assert delta.tolist() == [[-0.25, 0.0], [0.75, 0.0]]
    delta = offsets(grid, centroid, IC, (0.0, 0.5), 0.5)
    assert delta[0].tolist() == [0.0, -0.25]


def test_a_periodic_axis_takes_the_nearest_image():
    # The center at x = 0.125 and a point at x = 1.875: 0.25 apart through
    # the wrap, not 1.75 across the domain.
    centroid, _ = points((1.875, 0.5), (0.375, 0.5))
    delta = offsets(two_cell_grid(), centroid, {**IC, "x0": 0.125}, VELOCITY, 0.0)
    assert delta[:, 0].tolist() == [-0.25, 0.25]


def test_a_walled_axis_takes_the_plain_difference():
    centroid, _ = points((0.125, 0.875))
    delta = offsets(walled_grid(), centroid, {"x0": 0.875, "y0": 0.125, "width": 0.1}, VELOCITY, 0.0)
    assert delta.tolist() == [[-0.75, 0.75]]


def test_offsets_stay_within_half_a_period():
    rng = np.random.default_rng(0)
    grid = two_cell_grid()
    centroid = rng.uniform((0.0, 0.0), (2.0, 1.0), size=(200, 2))
    for time in (0.0, 0.3, 7.9, 123.4):
        delta = offsets(grid, centroid, IC, VELOCITY, time)
        assert (np.abs(delta[:, 0]) <= 1.0).all()


# --- F1: the three fields on the instance ----------------------------------

def test_fields_at_the_start():
    grid, centroid, h = two_cells()
    raw = raw_fields(grid, centroid, h, IC, VELOCITY, 0.0)
    assert set(raw) == {"ind_a", "ind_b", "true_error"}
    assert raw["ind_a"].tolist() == pytest.approx([0.0, 16 * E(-8)])
    assert raw["ind_b"].tolist() == pytest.approx([16.0, 16 * E(-8)])
    assert raw["true_error"].tolist() == pytest.approx([64.0, 64 * E(-8)])
    # On the feature's center the values are exact.
    assert (raw["ind_a"][0], raw["ind_b"][0], raw["true_error"][0]) == (0.0, 16.0, 64.0)
    assert all(arr.shape == (2,) and arr.dtype == np.float64 for arr in raw.values())


# The explanation doc's table: (sample, A ind_a, A ind_b, B ind_a, B ind_b).
TABLE = [
    (0, 0.0, 16.0, 16 * E(-8), 16 * E(-8)),
    (1, 4 * E(-0.5), 16 * E(-0.5), 12 * E(-4.5), 16 * E(-4.5)),
    (2, 8 * E(-2), 16 * E(-2), 8 * E(-2), 16 * E(-2)),
    (3, 12 * E(-4.5), 16 * E(-4.5), 4 * E(-0.5), 16 * E(-0.5)),
    (4, 16 * E(-8), 16 * E(-8), 0.0, 16.0),
]


@pytest.mark.parametrize("sample, a_ring, a_bump, b_ring, b_bump", TABLE)
def test_fields_along_the_interval(sample, a_ring, a_bump, b_ring, b_bump):
    grid, centroid, h = two_cells()
    raw = raw_fields(grid, centroid, h, IC, VELOCITY, 0.25 * sample)
    assert raw["ind_a"].tolist() == pytest.approx([a_ring, b_ring])
    assert raw["ind_b"].tolist() == pytest.approx([a_bump, b_bump])


def test_the_table_rounds_to_the_explanation_docs_numbers():
    assert [round(4 * E(-0.5), 3), round(8 * E(-2), 3), round(12 * E(-4.5), 4)] == [2.426, 1.083, 0.1333]
    assert [round(16 * E(-0.5), 2), round(16 * E(-2), 3), round(16 * E(-4.5), 3)] == [9.70, 2.165, 0.178]
    assert round(16 * E(-8), 4) == 0.0054


# --- The shapes of the three fields ----------------------------------------

def test_the_ring_peaks_one_width_from_the_center():
    centroid, h = points((0.5, 0.5), (0.625, 0.5), (0.75, 0.5), (1.0, 0.5))
    ring = raw_fields(two_cell_grid(), centroid, h, IC, VELOCITY, 0.0)["ind_a"]
    assert ring[0] == 0.0
    assert ring[2] == pytest.approx(4 * E(-0.5))
    assert ring[2] > ring[1] > ring[0]
    assert ring[2] > ring[3]


def test_the_bump_falls_away_from_the_center():
    centroid, h = points((0.5, 0.5), (0.625, 0.5), (0.75, 0.5), (1.0, 0.5))
    bump = raw_fields(two_cell_grid(), centroid, h, IC, VELOCITY, 0.0)["ind_b"]
    assert bump[0] == 16.0
    assert (np.diff(bump) < 0).all()


def test_the_fields_depend_on_direction_only_through_distance():
    centroid, h = points((0.75, 0.5), (0.5, 0.75), (0.25, 0.5), (0.5, 0.25))
    raw = raw_fields(two_cell_grid(), centroid, h, IC, VELOCITY, 0.0)
    for arr in raw.values():
        assert arr.tolist() == pytest.approx([arr[0]] * 4)


def test_each_field_has_its_own_power_of_the_element_size():
    centroid = np.array([[0.75, 0.5], [0.75, 0.5]])
    h = np.array([[1.0, 1.0], [0.5, 0.5]])
    raw = raw_fields(two_cell_grid(), centroid, h, IC, VELOCITY, 0.0)
    assert raw["ind_a"][0] / raw["ind_a"][1] == pytest.approx(2.0)
    assert raw["ind_b"][0] / raw["ind_b"][1] == pytest.approx(4.0)
    assert raw["true_error"][0] / raw["true_error"][1] == pytest.approx(8.0)


def test_a_non_square_element_is_measured_by_its_longer_side():
    centroid = np.array([[0.5, 0.5], [0.5, 0.5]])
    h = np.array([[1.0, 0.5], [0.25, 1.0]])
    assert raw_fields(two_cell_grid(), centroid, h, IC, VELOCITY, 0.0)["ind_b"].tolist() == [16.0, 16.0]


def test_a_narrower_feature_gives_larger_values_at_its_center():
    centroid, h = points((0.5, 0.5))
    wide = raw_fields(two_cell_grid(), centroid, h, IC, VELOCITY, 0.0)["ind_b"][0]
    narrow = raw_fields(two_cell_grid(), centroid, h, {**IC, "width": 0.125}, VELOCITY, 0.0)["ind_b"][0]
    assert (wide, narrow) == (16.0, 64.0)


# --- F6: the same functions on a refined mesh ------------------------------

def test_refining_under_the_feature_lowers_the_bump_and_raises_the_ring():
    grid = two_cell_grid()
    centroid, h = geometry(grid, A_CHILDREN + [B])
    raw = raw_fields(grid, centroid, h, IC, VELOCITY, 0.0)
    # A's children: rho = 2, d / w = sqrt(2). A itself had ind_b = 16, ind_a = 0.
    assert raw["ind_b"][:4].tolist() == pytest.approx([4 * E(-1)] * 4)
    assert raw["ind_a"][:4].tolist() == pytest.approx([2 * np.sqrt(2) * E(-1)] * 4)
    # B is as it was on the two-cell mesh.
    assert raw["ind_b"][4] == pytest.approx(16 * E(-8))
    assert raw["ind_a"][4] == pytest.approx(16 * E(-8))


# --- F5 and F10: around the wrap, and out through a wall -------------------

def test_on_a_periodic_axis_the_feature_comes_back():
    grid, centroid, h = two_cells()
    start = raw_fields(grid, centroid, h, IC, VELOCITY, 0.0)
    for laps in (1, 2, 5):
        later = raw_fields(grid, centroid, h, IC, VELOCITY, 2.0 * laps)
        for name in start:
            assert later[name].tolist() == pytest.approx(start[name].tolist())


def test_through_a_wall_the_feature_is_gone():
    grid = walled_grid()
    centroid, h = geometry(grid, uniform_leaves(grid, 1))
    # The feature starts on a centroid: rho = 0.25 / 0.05 = 5 there.
    ic = {"x0": 0.375, "y0": 0.375, "width": 0.05}
    assert raw_fields(grid, centroid, h, ic, VELOCITY, 0.0)["ind_b"].max() == pytest.approx(25.0)
    for time in (3.0, 4.0, 50.0):
        raw = raw_fields(grid, centroid, h, ic, VELOCITY, time)
        assert all((arr < 1e-12).all() for arr in raw.values())


# --- evaluate: the names and the reduction (F7) ----------------------------

def evaluate_on_two_cells(names, reduction, nvar, time=0.0):
    grid, centroid, h = two_cells()
    return evaluate(names, reduction, nvar, grid=grid, centroid=centroid, h=h, ic=IC, velocity=VELOCITY, time=time)


def test_scalar_is_one_column():
    out = evaluate_on_two_cells(["ind_a", "ind_b", "true_error"], "scalar", nvar=1)
    assert list(out) == ["ind_a", "ind_b", "true_error"]
    assert all(arr.shape == (2, 1) and arr.dtype == np.float64 for arr in out.values())
    assert out["ind_b"][:, 0].tolist() == pytest.approx([16.0, 16 * E(-8)])
    assert out["true_error"][0, 0] == 64.0


def test_per_variable_scales_each_variable():
    out = evaluate_on_two_cells(["ind_a", "ind_b"], "per_variable", nvar=3, time=0.25)
    for arr in out.values():
        assert arr.shape == (2, 3) and arr.dtype == np.float64
        assert arr[:, 1].tolist() == pytest.approx((arr[:, 0] / 2).tolist())
        assert arr[:, 2].tolist() == pytest.approx((arr[:, 0] / 3).tolist())


@pytest.mark.parametrize("nvar", [1, 3])
def test_scalar_is_the_maximum_over_the_variables(nvar):
    scalar = evaluate_on_two_cells(INDICATORS, "scalar", nvar, time=0.25)
    per_variable = evaluate_on_two_cells(INDICATORS, "per_variable", nvar, time=0.25)
    for name in INDICATORS:
        assert scalar[name].shape == (2, 1)
        assert np.array_equal(scalar[name], per_variable[name].max(axis=1, keepdims=True))
        assert np.array_equal(scalar[name][:, 0], per_variable[name][:, 0])


def test_only_the_names_asked_for_come_back():
    assert list(evaluate_on_two_cells(["ind_b"], "scalar", 1)) == ["ind_b"]
    assert evaluate_on_two_cells([], "scalar", 1) == {}
    assert list(evaluate_on_two_cells(["ind_b", "ind_b"], "scalar", 1)) == ["ind_b"]


def test_evaluate_returns_new_writable_arrays():
    first = evaluate_on_two_cells(INDICATORS, "per_variable", 3)
    second = evaluate_on_two_cells(INDICATORS, "per_variable", 3)
    for name in INDICATORS:
        assert np.array_equal(first[name], second[name])
        assert first[name] is not second[name]
        assert first[name].flags.writeable


@pytest.mark.parametrize("reduction", ["mean", "", "SCALAR"])
def test_an_unknown_reduction_is_rejected(reduction):
    with pytest.raises(ValueError, match=rf"^reduction: '{reduction}', expected one of \['scalar', 'per_variable'\]"):
        evaluate_on_two_cells(["ind_a"], reduction, 1)


# --- check_names -----------------------------------------------------------

@pytest.mark.parametrize("names", [[], ["ind_a"], ["ind_b", "ind_a"], ("ind_a", "ind_a")])
def test_registered_names_pass(names):
    check_names("indicators", names, INDICATORS)


@pytest.mark.parametrize("label", ["indicators", "accumulate", "aux"])
def test_an_unknown_name_is_reported_under_the_arguments_label(label):
    message = rf"^{label}: unknown names \['nope', 'true_error'\], expected names from \['ind_a', 'ind_b'\]"
    with pytest.raises(ValueError, match=message):
        check_names(label, ["ind_a", "nope", "true_error"], INDICATORS)


# --- mass ------------------------------------------------------------------

def test_mass_is_the_gaussians_integral():
    total = mass(IC, 1)
    assert total.dtype == np.float64 and total.shape == (1,)
    assert total[0] == pytest.approx(2 * np.pi * 0.0625)
    assert round(float(total[0]), 4) == 0.3927


def test_mass_scales_like_the_fields():
    total = mass(IC, 3)
    assert total.shape == (3,)
    assert total.tolist() == pytest.approx([2 * np.pi * 0.0625 / v for v in (1, 2, 3)])


# --- Everywhere and always -------------------------------------------------

@pytest.mark.parametrize("seed", range(6))
def test_fields_are_finite_non_negative_and_bounded(seed):
    rng = np.random.default_rng(seed)
    periodic = (bool(rng.integers(2)), bool(rng.integers(2)))
    grid = QuadGrid(nx=3, ny=2, periodic=periodic, lo=(-1.0, 0.0), hi=(2.0, 2.0), max_level=3)
    centroid, h = geometry(grid, uniform_leaves(grid, int(rng.integers(0, 4))))
    ic = {"x0": rng.uniform(-1.0, 2.0), "y0": rng.uniform(0.0, 2.0), "width": rng.uniform(0.1, 0.5)}
    velocity = tuple(rng.uniform(-2.0, 2.0, size=2))
    rho = h.max(axis=1) / ic["width"]

    for time in rng.uniform(0.0, 20.0, size=5):
        raw = raw_fields(grid, centroid, h, ic, velocity, float(time))
        assert all(np.isfinite(arr).all() and (arr >= 0).all() for arr in raw.values())
        # g is at most 1, and (d / w) * g at most exp(-1/2).
        assert (raw["ind_b"] <= rho**2).all()
        assert (raw["true_error"] <= rho**3).all()
        assert (raw["ind_a"] <= rho * E(-0.5) * (1 + 1e-12)).all()
