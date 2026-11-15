"""Report types returned by the v2 solver contract's verbs.

AdaptReport     returned by apply_marks: what the adapt realized, and how
                the new positions correspond to the old ones.
PathSnapshot    a frame on disk, for the visualization tools.
AuxFields       named training-only arrays (the auxiliary channel).
IntervalReport  returned by step_interval.

AdaptReport, AuxFields and IntervalReport are training-only: none of them
exists at deployment, so the observation builder accepts none of them
(contract/checks.py asserts it).
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from contract._arrays import check_and_freeze
from contract.solver_contract import COARSEN, HOLD, REFINE

# What a post-adapt element is, relative to the pre-adapt mesh (new_kind).
KIND_UNCHANGED = 0
KIND_CHILD = 1
KIND_MERGED = 2


@dataclass(frozen=True, slots=True, kw_only=True)
class AdaptReport:
    """What one apply_marks realized, and the map across it.

    Positions do not survive an adapt. This report is the only link between
    a position on the pre-adapt mesh (epoch_old) and one on the post-adapt
    mesh (epoch_new).

    Fields:
        realized: What happened to each pre-adapt element, in the mark
            vocabulary (COARSEN, HOLD, REFINE). (n_old,) int8. It differs
            from the requested marks where balance added a refine, where a
            coarsen was dropped (a partly marked family) or undone by
            balance, and where a mark was ineligible.
        new_to_old: For each post-adapt element, the pre-adapt positions it
            came from. (n_new, 2**ndim) int32. An unchanged element and a
            child carry one old position in slot 0, the rest -1. A merged
            parent carries all its children, in child order.
        new_kind: KIND_UNCHANGED, KIND_CHILD or KIND_MERGED for each
            post-adapt element. (n_new,) int8.
        epoch_old: mesh_epoch before the adapt.
        epoch_new: mesh_epoch after it; always epoch_old + 1.
    """

    realized: np.ndarray
    new_to_old: np.ndarray
    new_kind: np.ndarray
    epoch_old: int
    epoch_new: int

    def __post_init__(self) -> None:
        """Check ranks, then dtypes and shapes, then the codes and epochs."""
        if not isinstance(self.realized, np.ndarray) or self.realized.ndim != 1:
            raise ValueError("realized: expected a 1-D numpy array")
        if not isinstance(self.new_kind, np.ndarray) or self.new_kind.ndim != 1:
            raise ValueError("new_kind: expected a 1-D numpy array")
        if not isinstance(self.new_to_old, np.ndarray) or self.new_to_old.ndim != 2:
            raise ValueError("new_to_old: expected a 2-D numpy array")

        n_old, n_new = len(self.realized), len(self.new_kind)
        width = self.new_to_old.shape[1]
        if width not in (2, 4, 8):
            raise ValueError(f"new_to_old: width {width}, expected 2, 4 or 8")

        check_and_freeze("realized", self.realized, np.int8, (n_old,))
        check_and_freeze("new_to_old", self.new_to_old, np.int32, (n_new, width))
        check_and_freeze("new_kind", self.new_kind, np.int8, (n_new,))

        if not np.isin(self.realized, (COARSEN, HOLD, REFINE)).all():
            raise ValueError("realized: values must be COARSEN, HOLD or REFINE")
        if not np.isin(self.new_kind, (KIND_UNCHANGED, KIND_CHILD, KIND_MERGED)).all():
            raise ValueError("new_kind: values must be 0, 1 or 2")
        if self.epoch_new != self.epoch_old + 1:
            raise ValueError(
                f"epoch_new: {self.epoch_new}, expected epoch_old + 1 = {self.epoch_old + 1}"
            )

    @property
    def n_old(self) -> int:
        """Number of elements before the adapt."""
        return len(self.realized)

    @property
    def n_new(self) -> int:
        """Number of elements after the adapt."""
        return len(self.new_kind)


@dataclass(frozen=True, slots=True, kw_only=True)
class PathSnapshot:
    """A frame on disk. A 2D backend's snapshot is a path, not arrays.

    Fields:
        path: The frame's file (a .pvtu for Jexpresso).
        output_dir: The run's output directory.
        frame_index: The frame's index within the run.
        time: Physical time of the frame.
        mesh_epoch: The mesh the frame shows.
        n_active: Element count of that mesh.
        pre_adapt: True when the frame was written at an adapt time by the
            solver's own output and so shows the mesh before the adapt.
        checkpoint_path: A restart file written with the frame, if any.
    """

    path: str
    output_dir: str
    frame_index: int
    time: float
    mesh_epoch: int
    n_active: int
    pre_adapt: bool
    checkpoint_path: str | None = None


def _freeze_named(owner: str, arrays: Mapping[str, np.ndarray]) -> int | None:
    """Check a name -> array mapping and freeze its arrays.

    Every array must be float64 and 2-D, (n_active, n_comp), and all must
    share one row count.

    Returns:
        The common row count, or None when the mapping is empty.
    """
    rows = None
    for name, arr in arrays.items():
        label = f"{owner}[{name!r}]"
        if not isinstance(arr, np.ndarray) or arr.ndim != 2:
            raise ValueError(f"{label}: expected a 2-D numpy array")
        if rows is None:
            rows = arr.shape[0]
        check_and_freeze(label, arr, np.float64, (rows, arr.shape[1]))
    return rows


@dataclass(frozen=True, slots=True)
class AuxFields:
    """Named training-only arrays: the auxiliary channel.

    Holds what training may read and a deployed policy never sees, such as
    true_error where the case has an exact solution. Deliberately its own
    class, not a dict: a function that accepts it says so in its signature.

    Fields:
        fields: Name -> array, each (n_active, n_comp) float64.
    """

    fields: Mapping[str, np.ndarray]

    def __post_init__(self) -> None:
        """Check and freeze every array."""
        _freeze_named("aux", self.fields)


@dataclass(frozen=True, slots=True, kw_only=True)
class IntervalReport:
    """What one step_interval returns: one remesh interval, advanced.

    Fields:
        time: Physical time at the end of the interval.
        n_substeps: Solver steps taken; it varies with the finest level.
        acc: For each indicator named in accumulate, its maximum over the
            interval. Name -> (n_active, n_comp) float64.
        mass: The conserved total of each variable. (nvar,) float64.
        aux: The auxiliary fields asked for, or None.
        snapshot: The frame written during the interval, or None.
    """

    time: float
    n_substeps: int
    acc: Mapping[str, np.ndarray]
    mass: np.ndarray
    aux: AuxFields | None = None
    snapshot: PathSnapshot | None = None

    def __post_init__(self) -> None:
        """Check acc and mass, and that aux has acc's row count."""
        rows = _freeze_named("acc", self.acc)
        if not isinstance(self.mass, np.ndarray) or self.mass.ndim != 1:
            raise ValueError("mass: expected a 1-D numpy array")
        check_and_freeze("mass", self.mass, np.float64, (len(self.mass),))
        if self.aux is not None and rows is not None:
            aux_rows = _freeze_named("aux", self.aux.fields)
            if aux_rows is not None and aux_rows != rows:
                raise ValueError(f"aux: {aux_rows} rows, expected acc's {rows}")