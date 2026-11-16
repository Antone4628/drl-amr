"""The mock quad backend: a quadtree behind the v2 solver contract's verbs.

MockQuadBackend implements SolverBackend with a real quadtree and no PDE. It
exists to build and test the 2D agent logic without Jexpresso, and to stay
the cheapest instrument for telling an agent bug from a solver or bridge bug.

The mesh is a list of leaves (backends/mock_quad/quadtree.py). The backend
holds one mesh at a time, builds its topology arrays once per mesh_epoch, and
hands the same read-only arrays to every SolverState of that epoch.

All nine verbs are in place (AGENT_2D_ROADMAP item 1.2); get_snapshot and
reinitialize_ic are declared absent and raise. The realization the two adapt
verbs share is backends/mock_quad/realize.py; the synthetic fields the two
field verbs share are backends/mock_quad/fields.py.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from backends.mock_quad.fields import AUX_FIELDS, INDICATORS, REDUCTIONS, check_names, evaluate, mass
from backends.mock_quad.quadtree import Leaf, QuadGrid, order_leaves, topology, uniform_leaves
from backends.mock_quad.realize import correspondence, realize
from contract.backend import BackendInfo
from contract.reports import AdaptReport, AuxFields, IntervalReport, PathSnapshot
from contract.state import SolverState

# The feature's width, as fractions of the domain's shorter side. The lower
# bound keeps a feature visible at the centroids of a coarse mesh: on a 4x4
# grid at level 1 the worst-placed centroid sees 21 % of its peak.
_WIDTH_BOUNDS = (0.05, 0.25)


class MockQuadBackend:
    """A uniform nx-by-ny base grid refined as a quadtree, with no PDE.

    Args:
        nx: Base cells along x.
        ny: Base cells along y.
        periodic: Whether each axis wraps, (x, y); an axis that does not
            wrap ends in walls.
        lo: The domain's lower corner.
        hi: The domain's upper corner.
        max_level: The finest level an element may reach.
        nvar: Number of solution variables the synthetic fields carry.
        dt_base: The base time step; one remesh interval is amr_freq of them.
        velocity: The synthetic feature's velocity, (vx, vy).
        rotate_cells: If True, the base cell that comes first in the element
            order moves with mesh_epoch, so the order changes at every adapt
            and reset even when the mesh does not.
    """

    def __init__(
        self,
        *,
        nx: int,
        ny: int,
        periodic: tuple[bool, bool] = (True, True),
        lo: tuple[float, float] = (0.0, 0.0),
        hi: tuple[float, float] = (1.0, 1.0),
        max_level: int = 2,
        nvar: int = 1,
        dt_base: float = 0.01,
        velocity: tuple[float, float] = (1.0, 0.0),
        rotate_cells: bool = True,
    ) -> None:
        if nvar < 1:
            raise ValueError(f"nvar: {nvar}, expected at least 1")
        if not dt_base > 0:
            raise ValueError(f"dt_base: {dt_base}, expected a positive step")
        self._grid = QuadGrid(nx=nx, ny=ny, periodic=periodic, lo=lo, hi=hi, max_level=max_level)
        self._nvar = nvar
        self._dt_base = float(dt_base)
        self._velocity = (float(velocity[0]), float(velocity[1]))
        self._rotate_cells = rotate_cells

        # The mesh and the episode; set by reset.
        self._epoch = 0
        self._leaves: list[Leaf] | None = None
        self._arrays: dict[str, np.ndarray] | None = None
        self._ic: dict[str, float] = {}
        self._amr_freq = 0
        self._intervals = 0
        self._time = 0.0

    # --- What this backend is ----------------------------------------------

    def describe(self) -> BackendInfo:
        """Return what this backend is and offers. No side effects."""
        grid = self._grid
        shorter = min(grid.hi[0] - grid.lo[0], grid.hi[1] - grid.lo[1])
        return BackendInfo(
            ndim=2,
            nvar=self._nvar,
            indicators=INDICATORS,
            reductions=REDUCTIONS,
            ic_params={
                "x0": (grid.lo[0], grid.hi[0]),
                "y0": (grid.lo[1], grid.hi[1]),
                "width": (_WIDTH_BOUNDS[0] * shorter, _WIDTH_BOUNDS[1] * shorter),
            },
            optional_verbs=frozenset({"dry_run"}),
        )

    # --- The episode and the state -----------------------------------------

    def reset(self, ic: Mapping[str, float], amr_freq: int, initial_level: int = 1) -> SolverState:
        """Begin an episode on a uniform mesh and return its first state.

        The mesh is the one initial_level all-refine adapts of the base grid
        would give. mesh_epoch rises by one, whatever the level.

        Args:
            ic: One float per name in describe().ic_params, within its bounds.
            amr_freq: Base steps per remesh interval.
            initial_level: The uniform level the episode starts at.

        Raises:
            ValueError: If ic has an unknown or a missing name or a value
                outside its bounds, or amr_freq or initial_level is out of
                range.
        """
        params = self.describe().ic_params
        if set(ic) != set(params):
            unknown, missing = sorted(set(ic) - set(params)), sorted(set(params) - set(ic))
            raise ValueError(f"ic: unknown names {unknown}, missing names {missing}")
        for name, (low, high) in params.items():
            if not low <= ic[name] <= high:
                raise ValueError(f"ic[{name!r}]: {ic[name]}, expected within [{low}, {high}]")
        if amr_freq < 1:
            raise ValueError(f"amr_freq: {amr_freq}, expected at least 1")
        if not 0 <= initial_level <= self._grid.max_level:
            raise ValueError(f"initial_level: {initial_level}, expected 0 to max_level = {self._grid.max_level}")

        self._ic = {name: float(ic[name]) for name in params}
        self._amr_freq = int(amr_freq)
        self._intervals = 0
        self._time = 0.0
        self._set_mesh(uniform_leaves(self._grid, initial_level))
        return self.get_state()

    def get_state(self) -> SolverState:
        """Return the current state. No side effects.

        Every state of one mesh_epoch shares the same read-only topology
        arrays; they are built once, in _set_mesh, and never written again.
        """
        self._require_reset()
        grid = self._grid
        finest = int(self._arrays["level"].max())
        return SolverState(
            **self._arrays,
            attributes=np.zeros((len(self._leaves), 0)),
            n_active=len(self._leaves),
            ndim=2,
            nvar=self._nvar,
            max_level=grid.max_level,
            mesh_epoch=self._epoch,
            time=self._time,
            interval_T=self._amr_freq * self._dt_base,
            dt_base=self._dt_base,
            amr_freq=self._amr_freq,
            dt_stable=self._dt_base / 2**finest,
            domain_lo=np.array(grid.lo, dtype=np.float64),
            domain_hi=np.array(grid.hi, dtype=np.float64),
            periodic=np.array(grid.periodic, dtype=bool),
        )

    def _set_mesh(self, leaves: list[Leaf]) -> None:
        """Install a new mesh: raise the epoch, order the leaves, build the arrays."""
        self._epoch += 1
        start_cell = self._epoch % (self._grid.nx * self._grid.ny) if self._rotate_cells else 0
        self._leaves = order_leaves(self._grid, leaves, start_cell)
        self._arrays = topology(self._grid, self._leaves)

    def _require_reset(self) -> None:
        """Raise if no episode has begun."""
        if self._leaves is None:
            raise RuntimeError("no mesh yet: call reset first")

    # --- The adapt ---------------------------------------------------------

    def dry_run(self, marks: np.ndarray) -> tuple[int, np.ndarray]:
        """Predict what apply_marks(marks) would realize, changing nothing.

        Returns:
            The predicted n_active, and the predicted realized flags,
            (n_active,) int8.

        Raises:
            ValueError: If marks has the wrong shape, a non-integer dtype or
                a value outside the vocabulary.
        """
        self._require_reset()
        leaves, realized = realize(self._grid, self._leaves, marks)
        return len(leaves), realized

    def apply_marks(self, marks: np.ndarray) -> tuple[SolverState, AdaptReport]:
        """Apply one round of marks and return the new state and the report.

        The realization is realize's: refine, then coarsen complete families,
        then corner balance, one level per call. mesh_epoch rises by one, for
        a round of all HOLD too; time does not advance.

        Raises:
            ValueError: If marks has the wrong shape, a non-integer dtype or
                a value outside the vocabulary. Nothing is changed.
        """
        self._require_reset()
        new_leaves, realized = realize(self._grid, self._leaves, marks)
        old_leaves, old_epoch = self._leaves, self._epoch
        self._set_mesh(new_leaves)
        new_to_old, new_kind = correspondence(old_leaves, self._leaves)
        report = AdaptReport(
            realized=realized,
            new_to_old=new_to_old,
            new_kind=new_kind,
            epoch_old=old_epoch,
            epoch_new=self._epoch,
        )
        return self.get_state(), report

    # --- The fields --------------------------------------------------------

    def compute_error(self, indicators: Sequence[str], reduction: str) -> dict[str, np.ndarray]:
        """Return one (n_active, n_comp) float64 array per indicator name.

        The synthetic fields of backends/mock_quad/fields.py, evaluated at
        the current time on the current mesh. Stateless: every call returns
        new arrays, which the caller owns, and changes nothing.

        Raises:
            ValueError: If an indicator or the reduction is not a registered
                name. true_error is not an indicator: it is reachable
                through step_interval's aux argument only.
        """
        self._require_reset()
        check_names("indicators", indicators, INDICATORS)
        return self._evaluate(indicators, reduction, self._time)

    def _evaluate(self, names: Sequence[str], reduction: str, time: float) -> dict[str, np.ndarray]:
        """Evaluate named fields on the current mesh at a given time."""
        return evaluate(
            names,
            reduction,
            self._nvar,
            grid=self._grid,
            centroid=self._arrays["centroid"],
            h=self._arrays["h"],
            ic=self._ic,
            velocity=self._velocity,
            time=time,
        )

    # --- The interval ------------------------------------------------------

    def step_interval(
        self, accumulate: Sequence[str], reduction: str, cadence: int, aux: Sequence[str] = ()
    ) -> IntervalReport:
        """Advance exactly one remesh interval. No mesh change.

        The feature moves; the mesh, mesh_epoch and the state arrays do not.
        The interval is amr_freq * 2**(finest level present) substeps. The
        fields are sampled at its start, after every cadence-th substep and
        after the last, and the elementwise maximum is returned.

        Args:
            accumulate: Indicator names to return in the report's acc.
            reduction: scalar or per_variable.
            cadence: Sample every this many substeps; at least 1.
            aux: Training-only names to return in the report's aux:
                true_error, or an indicator. Accumulated like acc.

        Raises:
            ValueError: If a name or the reduction is not registered, or
                cadence is below 1. Nothing is changed.
        """
        self._require_reset()
        check_names("accumulate", accumulate, INDICATORS)
        check_names("aux", aux, INDICATORS + AUX_FIELDS)
        if reduction not in REDUCTIONS:
            raise ValueError(f"reduction: {reduction!r}, expected one of {list(REDUCTIONS)}")
        if cadence < 1:
            raise ValueError(f"cadence: {cadence}, expected at least 1")

        interval_T = self._amr_freq * self._dt_base
        n_substeps = self._amr_freq * 2 ** int(self._arrays["level"].max())
        start = self._intervals * interval_T
        names = list(dict.fromkeys([*accumulate, *aux]))
        best: dict[str, np.ndarray] = {}
        for j in range(n_substeps + 1):
            if j % cadence == 0 or j == n_substeps:
                sample = self._evaluate(names, reduction, start + j * interval_T / n_substeps)
                for name, values in sample.items():
                    best[name] = np.maximum(best[name], values) if name in best else values

        self._intervals += 1
        self._time = self._intervals * interval_T
        return IntervalReport(
            time=self._time,
            n_substeps=n_substeps,
            acc={name: best[name] for name in accumulate},
            mass=mass(self._ic, self._nvar),
            aux=AuxFields(fields={name: best[name] for name in aux}) if aux else None,
        )

    # --- Optional verbs this backend does not offer ------------------------

    def get_snapshot(self, write: bool = False) -> PathSnapshot:
        """Absent: the mock writes no frames."""
        raise NotImplementedError("get_snapshot: the mock backend writes no frames")

    def reinitialize_ic(self) -> None:
        """Absent: training does not call it."""
        raise NotImplementedError("reinitialize_ic: not offered by the mock backend")