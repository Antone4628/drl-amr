"""The v2 SolverBackend over the 1D DG advection solver: the 1D adapter.

Python1DBackend speaks the nine verbs of contract.backend.SolverBackend
(SOLVER_CONTRACT_V2 section 6) by composing over Python1DSolverContract, which
it builds itself and does not change. The 1D state's left, right and sibling
become neighbors[:, 0, 0], neighbors[:, 1, 0] and family; can_coarsen keeps
the 1D post-coarsen balance check and element_id stays present (spec section
5.3).

The sequential stack (agent/, drivers/, envs/) still drives
Python1DSolverContract directly. Nothing here is on that path.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from contract.backend import BackendFailure, BackendInfo
from contract.reports import KIND_CHILD, KIND_MERGED, KIND_UNCHANGED, AdaptReport, AuxFields, IntervalReport
from contract.solver_contract import COARSEN, HOLD, REFINE
from contract.solver_snapshot import SolverSnapshot
from contract.state import SolverState

from .contract_impl import Python1DSolverContract
from .solvers.dg_advection_solver_multiround import DGAdvectionSolver
from .solvers.error_indicators import INDICATOR_REGISTRY

INDICATORS = tuple(sorted(INDICATOR_REGISTRY))
REDUCTIONS = ("scalar", "per_variable")
# The Stage 1A pool: the icase values reset accepts unless told otherwise.
DEFAULT_IC_CASES = (1, 10, 12, 13, 14, 15, 16)


class Python1DBackend:
    """SolverBackend over a DGAdvectionSolver.

    Construct it with a balance=False solver, as Python1DSolverContract
    requires. The backend owns the contract it wraps: driving that contract
    from outside would change the mesh without raising mesh_epoch.

    Not offered: dry_run. get_snapshot returns the 1D array SolverSnapshot,
    not a PathSnapshot (spec section 5.5).
    """

    def __init__(self, solver: DGAdvectionSolver, *, ic_cases: Sequence[int] = DEFAULT_IC_CASES) -> None:
        """Wrap a solver.

        Args:
            solver: The 1D solver, constructed with balance=False.
            ic_cases: The icase values reset accepts.
        """
        self._ic_cases = tuple(int(k) for k in ic_cases)
        if not self._ic_cases:
            raise ValueError("ic_cases: expected at least one icase")
        self._solver = solver
        self._contract = Python1DSolverContract(solver)
        self._mesh_epoch = 0
        self._amr_freq: int | None = None  # None until the first reset

    def _require_reset(self) -> None:
        """Raise if no episode has begun."""
        if self._amr_freq is None:
            raise RuntimeError("no episode: call reset first")

    # --- Lifecycle ---------------------------------------------------------
    def describe(self) -> BackendInfo:
        """Return what this backend is and offers. No side effects."""
        return BackendInfo(
            ndim=1,
            nvar=1,
            indicators=INDICATORS,
            reductions=REDUCTIONS,
            ic_params={"icase": (float(min(self._ic_cases)), float(max(self._ic_cases)))},
            optional_verbs=frozenset({"get_snapshot", "reinitialize_ic"}),
        )

    def reset(self, ic: Mapping[str, float], amr_freq: int, initial_level: int = 1) -> SolverState:
        """Begin an episode and return its first state.

        Everything is checked before anything changes, so a rejected call
        leaves the previous episode intact.

        Args:
            ic: {"icase": k}, k one of this backend's ic_cases. The bounds
                describe() publishes are the set's extremes; membership is
                checked here.
            amr_freq: Base steps per remesh interval, at least 1.
            initial_level: Uniform refinement level, 0 to max_level. The
                initial condition is sampled on the refined mesh.

        Raises:
            ValueError: Naming the offending argument.
        """
        if set(ic) != {"icase"}:
            raise ValueError(f"ic: expected the one name 'icase', got {sorted(ic)}")
        icase = ic["icase"]
        if icase not in self._ic_cases:
            raise ValueError(f"ic['icase']: {icase}, expected one of {self._ic_cases}")
        if not amr_freq >= 1:
            raise ValueError(f"amr_freq: {amr_freq}, expected at least 1")
        if not 0 <= initial_level <= self._solver.max_level:
            raise ValueError(f"initial_level: {initial_level}, expected 0 to {self._solver.max_level}")

        mode = "fixed" if initial_level > 0 else "none"
        self._contract.reset(icase=int(icase), refinement_mode=mode, refinement_level=int(initial_level))
        self._amr_freq = int(amr_freq)
        self._mesh_epoch += 1
        return self.get_state()

    # --- State -------------------------------------------------------------
    def get_state(self) -> SolverState:
        """Return the current state in the v2 schema. No side effects.

        Topology and eligibility are the wrapped contract's, translated:
        left and right fill the one slot of faces 0 and 1, and sibling
        becomes a two-member family row in child order (the lower-x child
        first, which in 1D is the lower position). can_coarsen keeps the 1D
        post-coarsen balance check, so a complete family row does not imply
        it. Geometry is read from the solver's element boundaries.
        """
        self._require_reset()
        old = self._contract.get_state()
        n = old.n_active
        position = np.arange(n)

        neighbors = np.stack([old.left, old.right], axis=1).astype(np.int32).reshape(n, 2, 1)
        # The old contract's neighbors are periodic: element 0's left and the
        # last element's right are the two faces that cross the wrap.
        neighbor_wrap = np.stack([position == 0, position == n - 1], axis=1)

        has = old.sibling >= 0
        family = np.full((n, 2), -1, dtype=np.int32)
        family[has, 0] = np.minimum(position, old.sibling)[has]
        family[has, 1] = np.maximum(position, old.sibling)[has]

        xelem = np.asarray(self._solver.xelem, dtype=np.float64)
        dt_base = float(old.stable_dt)

        return SolverState(
            level=old.level.astype(np.int32),
            neighbors=neighbors,
            neighbor_wrap=neighbor_wrap,
            family=family,
            can_refine=old.can_refine,
            can_coarsen=old.can_coarsen,
            centroid=((xelem[:-1] + xelem[1:]) / 2).reshape(n, 1),
            h=np.diff(xelem).reshape(n, 1),
            attributes=np.zeros((n, 0)),
            n_active=n,
            ndim=1,
            nvar=1,
            max_level=old.max_level,
            mesh_epoch=self._mesh_epoch,
            time=float(self._solver.time),
            interval_T=self._amr_freq * dt_base,
            dt_base=dt_base,
            amr_freq=self._amr_freq,
            dt_stable=dt_base,
            domain_lo=xelem[:1].copy(),
            domain_hi=xelem[-1:].copy(),
            periodic=np.array([True]),
            element_id=old.element_id.astype(np.int64),
        )

    # --- Error -------------------------------------------------------------
    @staticmethod
    def _check_names(argument: str, names: Sequence[str]) -> list[str]:
        """Return names without repeats, in order; reject an unknown one.

        Raises:
            ValueError: Starting with the argument's name.
        """
        if isinstance(names, str):
            raise ValueError(f"{argument}: expected a sequence of names, got the string {names!r}")
        unique = list(dict.fromkeys(names))
        unknown = [name for name in unique if name not in INDICATORS]
        if unknown:
            raise ValueError(f"{argument}: unknown names {unknown}, expected among {INDICATORS}")
        return unique

    @staticmethod
    def _check_reduction(reduction: str) -> None:
        """Reject a reduction this backend does not offer."""
        if reduction not in REDUCTIONS:
            raise ValueError(f"reduction: {reduction!r}, expected one of {REDUCTIONS}")

    def _errors(self, name: str) -> np.ndarray:
        """One indicator as an (n_active, 1) float64 array the caller owns."""
        return np.array(self._contract.compute_error(name), dtype=np.float64).reshape(-1, 1)

    def compute_error(self, indicators: Sequence[str], reduction: str) -> dict[str, np.ndarray]:
        """Return one (n_active, 1) float64 array per indicator name.

        With one variable both reductions give one column: the wrapped
        contract's vector, unchanged. Changes nothing.
        """
        self._require_reset()
        names = self._check_names("indicators", indicators)
        self._check_reduction(reduction)
        return {name: self._errors(name) for name in names}

    # --- Snapshot, burn-in, dry run ------------------------------------------
    def get_snapshot(self, write: bool = False) -> SolverSnapshot:
        """Return the 1D array snapshot (spec section 5.5), never a path.

        Raises:
            NotImplementedError: If write is True; this backend writes no
                frames.
        """
        self._require_reset()
        if write:
            raise NotImplementedError("get_snapshot(write=True): the 1D backend writes no frames")
        return self._contract.get_snapshot()

    def reinitialize_ic(self) -> None:
        """Re-sample the initial condition on the current mesh, time to zero.

        The mesh does not change, so mesh_epoch does not rise.
        """
        self._require_reset()
        self._contract.reinitialize_ic()

    def dry_run(self, marks: np.ndarray) -> tuple[int, np.ndarray]:
        """Not offered: the 1D backend has no topology-only predictor."""
        raise NotImplementedError("dry_run: not offered by the 1D backend")

    # --- The adapt -----------------------------------------------------------
    @staticmethod
    def _check_marks(marks: np.ndarray, n: int) -> np.ndarray:
        """Return an int64 copy of marks, or reject them.

        Raises:
            ValueError: Starting with "marks".
        """
        if not isinstance(marks, np.ndarray) or not np.issubdtype(marks.dtype, np.integer):
            raise ValueError("marks: expected an integer numpy array")
        if marks.shape != (n,):
            raise ValueError(f"marks: shape {marks.shape}, expected ({n},)")
        if not np.isin(marks, (COARSEN, HOLD, REFINE)).all():
            raise ValueError("marks: values must be COARSEN, HOLD or REFINE")
        return marks.astype(np.int64)

    def _correspondence(self, old_ids: list[int], new_ids: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Relate two active lists through the forest's fixed element ids.

        A forest row is [id, parent, left child, right child, level], ids
        from 1 and 0 for none. realized is read from what covers each old
        element afterward, not from its mark (spec section 5.4).

        Returns:
            realized (n_old,) int8, new_to_old (n_new, 2) int32 and new_kind
            (n_new,) int8, as AdaptReport holds them.

        Raises:
            RuntimeError: If an element moved more than one level.
        """
        label = self._solver.label_mat
        old_pos = {e: p for p, e in enumerate(old_ids)}
        new_set = set(new_ids)

        realized = np.empty(len(old_ids), dtype=np.int8)
        for p, e in enumerate(old_ids):
            parent, left, right = (int(v) for v in label[e - 1][1:4])
            if e in new_set:
                realized[p] = HOLD
            elif left in new_set and right in new_set:
                realized[p] = REFINE
            elif parent in new_set:
                realized[p] = COARSEN
            else:
                raise RuntimeError(f"element id {e} at position {p} moved more than one level in one adapt")

        new_to_old = np.full((len(new_ids), 2), -1, dtype=np.int32)
        new_kind = np.empty(len(new_ids), dtype=np.int8)
        for q, e in enumerate(new_ids):
            parent, left, right = (int(v) for v in label[e - 1][1:4])
            if e in old_pos:
                new_kind[q] = KIND_UNCHANGED
                new_to_old[q, 0] = old_pos[e]
            elif parent in old_pos:
                new_kind[q] = KIND_CHILD
                new_to_old[q, 0] = old_pos[parent]
            elif left in old_pos and right in old_pos:
                new_kind[q] = KIND_MERGED
                new_to_old[q] = (old_pos[left], old_pos[right])
            else:
                raise RuntimeError(f"element id {e} at new position {q} has no source one level away")
        return realized, new_to_old, new_kind

    def apply_marks(self, marks: np.ndarray) -> tuple[SolverState, AdaptReport]:
        """Apply one round of marks and return the new state and the report.

        The realization is the wrapped contract's batch verb: refine, then
        coarsen a family when both siblings are marked and can_coarsen holds
        on the pre-round levels, then one refine-only balance. That is more
        conservative than p4est, which merges first and lets balance decide.
        Time does not advance; mesh_epoch rises by one, on an all-HOLD round
        too.

        Raises:
            ValueError: Starting with "marks", before anything changes.
        """
        self._require_reset()
        old_ids = [int(e) for e in self._solver.active]
        checked = self._check_marks(marks, len(old_ids))

        self._contract.apply_marks(checked)
        self._mesh_epoch += 1
        new_ids = [int(e) for e in self._solver.active]
        realized, new_to_old, new_kind = self._correspondence(old_ids, new_ids)
        report = AdaptReport(
            realized=realized,
            new_to_old=new_to_old,
            new_kind=new_kind,
            epoch_old=self._mesh_epoch - 1,
            epoch_new=self._mesh_epoch,
        )
        return self.get_state(), report

    # --- The interval --------------------------------------------------------
    def _mass(self) -> np.ndarray:
        """The solution's integral over the domain, by LGL quadrature. (1,)."""
        solver = self._solver
        nodal = solver.q[solver.intma]  # (ngl, n_active): column e is element e
        widths = np.diff(solver.xelem)
        return np.array([float(np.sum((solver.wgl @ nodal) * widths / 2))])

    def step_interval(
        self, accumulate: Sequence[str], reduction: str, cadence: int, aux: Sequence[str] = ()
    ) -> IntervalReport:
        """Advance exactly one remesh interval, interval_T. No mesh change.

        The loop is the sequential driver's exact-landing advance, repeated
        step for step: the fixed step dt_base, the last step clamped so the
        total lands on interval_T. Samples are taken at the start, after
        every cadence-th step and after the last.

        Args:
            accumulate: Indicator names whose maximum over the samples is
                returned in acc.
            reduction: As compute_error takes it.
            cadence: Sample every this many steps, at least 1.
            aux: Indicator names to return in aux, accumulated like acc. This
                backend offers no true_error.

        Raises:
            ValueError: Starting with the argument's name, before anything
                changes.
            BackendFailure: If the solution is not finite afterward.
        """
        self._require_reset()
        acc_names = self._check_names("accumulate", accumulate)
        self._check_reduction(reduction)
        if not cadence >= 1:
            raise ValueError(f"cadence: {cadence}, expected at least 1")
        aux_names = self._check_names("aux", aux)
        names = list(dict.fromkeys(acc_names + aux_names))

        running: dict[str, np.ndarray] = {}

        def sample() -> None:
            for name in names:
                now = self._errors(name)
                running[name] = np.maximum(running[name], now) if name in running else now

        dt = self._contract.get_state().stable_dt
        duration = self._amr_freq * float(dt)
        n_steps = max(1, int(np.ceil(duration / dt)))

        sample()
        advanced, taken, fresh = 0.0, 0, True
        for _ in range(n_steps):
            step_dt = min(dt, duration - advanced)
            if step_dt <= 1e-15:
                break
            self._contract.step(dt=step_dt)
            advanced += step_dt
            taken += 1
            fresh = taken % cadence == 0
            if fresh:
                sample()
        if not fresh:
            sample()

        if not np.all(np.isfinite(self._solver.q)):
            raise BackendFailure(f"the solution is not finite after {taken} steps")
        return IntervalReport(
            time=float(self._solver.time),
            n_substeps=taken,
            acc={name: running[name] for name in acc_names},
            mass=self._mass(),
            aux=AuxFields({name: running[name].copy() for name in aux_names}) if aux_names else None,
        )