"""The mock backend's synthetic fields: one moving feature, three functions.

The mock has no solution. Its indicators are closed-form functions of where
an element is and when: a Gaussian feature of width w starts at (x0, y0) and
moves at a fixed velocity, and each field is evaluated at an element's
centroid. With d the distance from the centroid to the feature's center,
g = exp(-d**2 / (2 * w**2)) and rho = max(h) / w:

    ind_a       rho * (d / w) * g    a ring: zero at the center
    ind_b       rho**2 * g           a bump: largest at the center
    true_error  rho**3 * g           the auxiliary, training-only field

None of them estimates anything. They follow the feature, depend on the
element's size, and differ from each other. Everything here is a pure
function: nothing is stored, so an adapt transfers nothing.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from backends.mock_quad.quadtree import QuadGrid

# Synthetic and named neutrally: nothing here is an error estimator.
INDICATORS = ("ind_a", "ind_b")
# Training-only: reachable through step_interval's aux argument alone.
AUX_FIELDS = ("true_error",)
REDUCTIONS = ("scalar", "per_variable")


def offsets(
    grid: QuadGrid,
    centroid: np.ndarray,
    ic: Mapping[str, float],
    velocity: tuple[float, float],
    time: float,
) -> np.ndarray:
    """Displacement from the feature's center to each centroid.

    On a periodic axis it is the displacement to the nearest image of the
    center, so the feature leaves one side and enters the other. On a walled
    axis it is the plain difference, and a feature that leaves is gone.

    Args:
        grid: The base grid; its domain and which axes wrap.
        centroid: Element centroids, (n, 2).
        ic: The feature's start, x0 and y0 (width is not read here).
        velocity: The feature's velocity, (vx, vy).
        time: Physical time.

    Returns:
        (n, 2) float64, a new array.
    """
    delta = np.empty(centroid.shape, dtype=np.float64)
    for axis, name in enumerate(("x0", "y0")):
        center = ic[name] + velocity[axis] * time
        delta[:, axis] = centroid[:, axis] - center
        if grid.periodic[axis]:
            length = grid.hi[axis] - grid.lo[axis]
            delta[:, axis] = (delta[:, axis] + length / 2) % length - length / 2
    return delta


def raw_fields(
    grid: QuadGrid,
    centroid: np.ndarray,
    h: np.ndarray,
    ic: Mapping[str, float],
    velocity: tuple[float, float],
    time: float,
) -> dict[str, np.ndarray]:
    """The three fields for one variable, each (n,) float64.

    Args:
        grid: The base grid.
        centroid: Element centroids, (n, 2).
        h: Element sizes per axis, (n, 2).
        ic: The feature's parameters: x0, y0, width.
        velocity: The feature's velocity, (vx, vy).
        time: Physical time.

    Returns:
        ind_a, ind_b and true_error by name, as new arrays.
    """
    width = ic["width"]
    delta = offsets(grid, centroid, ic, velocity, time)
    s = np.hypot(delta[:, 0], delta[:, 1]) / width
    g = np.exp(-0.5 * s**2)
    rho = h.max(axis=1) / width
    return {"ind_a": rho * s * g, "ind_b": rho**2 * g, "true_error": rho**3 * g}


def check_names(label: str, names: Sequence[str], allowed: Sequence[str]) -> None:
    """Raise if any name is not allowed, starting the message with label."""
    unknown = [name for name in names if name not in allowed]
    if unknown:
        raise ValueError(f"{label}: unknown names {unknown}, expected names from {list(allowed)}")


def evaluate(
    names: Sequence[str],
    reduction: str,
    nvar: int,
    *,
    grid: QuadGrid,
    centroid: np.ndarray,
    h: np.ndarray,
    ic: Mapping[str, float],
    velocity: tuple[float, float],
    time: float,
) -> dict[str, np.ndarray]:
    """Evaluate named fields on a mesh at a time, reduced to n_comp columns.

    Variable v carries a field scaled by 1 / (v + 1). per_variable returns
    every variable, n_comp = nvar; scalar returns variable 0, n_comp = 1,
    which is also the maximum over the variables.

    Args:
        names: Field names, from INDICATORS and AUX_FIELDS. The caller
            checks them (check_names), since which names it accepts depends
            on the verb.
        reduction: scalar or per_variable.
        nvar: Number of solution variables.
        grid: The base grid.
        centroid: Element centroids, (n, 2).
        h: Element sizes per axis, (n, 2).
        ic: The feature's parameters: x0, y0, width.
        velocity: The feature's velocity, (vx, vy).
        time: Physical time.

    Returns:
        One new (n, n_comp) float64 array per name.

    Raises:
        ValueError: If reduction is not a registered name.
    """
    if reduction not in REDUCTIONS:
        raise ValueError(f"reduction: {reduction!r}, expected one of {list(REDUCTIONS)}")
    raw = raw_fields(grid, centroid, h, ic, velocity, time)
    n_comp = 1 if reduction == "scalar" else nvar
    scale = 1.0 / np.arange(1, n_comp + 1)
    return {name: raw[name][:, None] * scale for name in names}


def mass(ic: Mapping[str, float], nvar: int) -> np.ndarray:
    """The feature's integral over the plane, per variable. (nvar,) float64.

    2 * pi * width**2, scaled like the fields. It is constant in time and
    independent of the mesh: the mock has nothing to lose.
    """
    return 2.0 * np.pi * ic["width"] ** 2 / np.arange(1, nvar + 1)