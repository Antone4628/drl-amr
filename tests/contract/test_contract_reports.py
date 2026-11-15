"""Tests for the v2 report types (contract/reports.py).

AGENT_2D_ROADMAP item 1.1. The two adapt reports are those of the phase's
explanation doc: two base cells A and B; marks [REFINE, HOLD] give five
elements (A's four children at positions 0-3, B at 4); marks
[COARSEN x4, HOLD] on that mesh give two again.

These tests cover form only. Whether a report agrees with the states on
either side of the adapt is the scripted-action harness's job (item 4.2).

Run: pytest tests/contract/test_contract_reports.py -v
"""
import dataclasses
from collections.abc import Mapping

import numpy as np
import pytest

from contract.reports import (
    KIND_CHILD,
    KIND_MERGED,
    KIND_UNCHANGED,
    AdaptReport,
    AuxFields,
    IntervalReport,
    PathSnapshot,
)
from contract.solver_contract import COARSEN, HOLD, REFINE


def refine_report_fields():
    """A splits into four children; B is unchanged."""
    return dict(
        realized=np.array([REFINE, HOLD], dtype=np.int8),
        new_to_old=np.array([[0, -1, -1, -1]] * 4 + [[1, -1, -1, -1]], dtype=np.int32),
        new_kind=np.array([KIND_CHILD] * 4 + [KIND_UNCHANGED], dtype=np.int8),
        epoch_old=0,
        epoch_new=1,
    )


def coarsen_report_fields():
    """A's four children merge back; B is unchanged."""
    return dict(
        realized=np.array([COARSEN] * 4 + [HOLD], dtype=np.int8),
        new_to_old=np.array([[0, 1, 2, 3], [4, -1, -1, -1]], dtype=np.int32),
        new_kind=np.array([KIND_MERGED, KIND_UNCHANGED], dtype=np.int8),
        epoch_old=1,
        epoch_new=2,
    )


def snapshot():
    return PathSnapshot(
        path="out/iter_3.pvtu",
        output_dir="out",
        frame_index=3,
        time=0.3,
        mesh_epoch=7,
        n_active=5,
        pre_adapt=False,
    )


# --- S6: AdaptReport -------------------------------------------------------

def test_kind_codes():
    assert (KIND_UNCHANGED, KIND_CHILD, KIND_MERGED) == (0, 1, 2)


def test_refine_report_sizes():
    report = AdaptReport(**refine_report_fields())
    assert (report.n_old, report.n_new) == (2, 5)
    assert report.new_to_old.shape == (5, 4)


def test_coarsen_report_sizes():
    report = AdaptReport(**coarsen_report_fields())
    assert (report.n_old, report.n_new) == (5, 2)
    assert list(report.new_to_old[0]) == [0, 1, 2, 3]


def test_one_dimensional_report_has_width_two():
    report = AdaptReport(
        realized=np.array([REFINE], dtype=np.int8),
        new_to_old=np.array([[0, -1], [0, -1]], dtype=np.int32),
        new_kind=np.array([KIND_CHILD, KIND_CHILD], dtype=np.int8),
        epoch_old=0,
        epoch_new=1,
    )
    assert report.new_to_old.shape == (2, 2)


@pytest.mark.parametrize("name", ["realized", "new_to_old", "new_kind"])
def test_adapt_report_arrays_are_read_only(name):
    report = AdaptReport(**refine_report_fields())
    with pytest.raises(ValueError, match="read-only"):
        getattr(report, name)[...] = 0


def test_adapt_report_fields_cannot_be_rebound():
    report = AdaptReport(**refine_report_fields())
    with pytest.raises(dataclasses.FrozenInstanceError):
        report.epoch_new = 5


@pytest.mark.parametrize("epoch_new", [0, 2])
def test_epochs_must_be_consecutive(epoch_new):
    fields = refine_report_fields()
    fields["epoch_new"] = epoch_new
    with pytest.raises(ValueError, match=r"^epoch_new: "):
        AdaptReport(**fields)


def test_unknown_kind_code_rejected():
    fields = refine_report_fields()
    fields["new_kind"] = np.array([1, 1, 1, 1, 3], dtype=np.int8)
    with pytest.raises(ValueError, match=r"^new_kind: values"):
        AdaptReport(**fields)


def test_unknown_realized_code_rejected():
    fields = refine_report_fields()
    fields["realized"] = np.array([2, 0], dtype=np.int8)
    with pytest.raises(ValueError, match=r"^realized: values"):
        AdaptReport(**fields)


def test_row_count_mismatch_rejected():
    fields = refine_report_fields()
    fields["new_to_old"] = fields["new_to_old"][:4]
    with pytest.raises(ValueError, match=r"^new_to_old: shape"):
        AdaptReport(**fields)


@pytest.mark.parametrize(
    "name, bad_dtype",
    [("realized", np.int32), ("new_to_old", np.int64), ("new_kind", np.int32)],
)
def test_adapt_report_wrong_dtype_rejected(name, bad_dtype):
    fields = refine_report_fields()
    fields[name] = fields[name].astype(bad_dtype)
    with pytest.raises(ValueError, match=rf"^{name}: dtype"):
        AdaptReport(**fields)


def test_family_width_must_be_a_power_of_two():
    fields = refine_report_fields()
    fields["new_to_old"] = np.full((5, 3), -1, dtype=np.int32)
    with pytest.raises(ValueError, match=r"^new_to_old: width 3"):
        AdaptReport(**fields)


def test_new_to_old_must_be_two_dimensional():
    fields = refine_report_fields()
    fields["new_to_old"] = np.zeros(5, dtype=np.int32)
    with pytest.raises(ValueError, match=r"^new_to_old: expected a 2-D"):
        AdaptReport(**fields)


# --- PathSnapshot ----------------------------------------------------------

def test_path_snapshot_defaults_and_frozen():
    snap = snapshot()
    assert snap.checkpoint_path is None
    assert isinstance(snap.path, str)
    with pytest.raises(dataclasses.FrozenInstanceError):
        snap.path = "elsewhere"


# --- AuxFields -------------------------------------------------------------

def test_aux_fields_hold_named_arrays():
    aux = AuxFields({"true_error": np.zeros((5, 1))})
    assert aux.fields["true_error"].shape == (5, 1)


def test_aux_fields_is_not_a_mapping():
    aux = AuxFields({"true_error": np.zeros((5, 1))})
    assert not isinstance(aux, Mapping)


def test_aux_arrays_are_read_only():
    aux = AuxFields({"true_error": np.zeros((5, 1))})
    with pytest.raises(ValueError, match="read-only"):
        aux.fields["true_error"][...] = 1.0


def test_aux_array_must_be_two_dimensional():
    with pytest.raises(ValueError, match=r"^aux\['true_error'\]: expected a 2-D"):
        AuxFields({"true_error": np.zeros(5)})


def test_aux_array_must_be_float64():
    with pytest.raises(ValueError, match=r"^aux\['true_error'\]: dtype"):
        AuxFields({"true_error": np.zeros((5, 1), dtype=np.float32)})


def test_aux_arrays_must_share_a_row_count():
    with pytest.raises(ValueError, match=r"^aux\['b'\]: shape"):
        AuxFields({"a": np.zeros((5, 1)), "b": np.zeros((4, 1))})


# --- IntervalReport --------------------------------------------------------

def test_interval_report_minimal():
    report = IntervalReport(
        time=0.1, n_substeps=20, acc={"ind_a": np.zeros((5, 1))}, mass=np.zeros(1)
    )
    assert report.aux is None
    assert report.snapshot is None
    assert report.acc["ind_a"].shape == (5, 1)


def test_interval_report_full_at_nvar_3():
    report = IntervalReport(
        time=0.1,
        n_substeps=20,
        acc={"ind_a": np.zeros((5, 3)), "ind_b": np.zeros((5, 1))},
        mass=np.zeros(3),
        aux=AuxFields({"true_error": np.zeros((5, 3))}),
        snapshot=snapshot(),
    )
    assert report.mass.shape == (3,)
    assert report.aux.fields["true_error"].shape == (5, 3)
    assert report.snapshot.frame_index == 3


def test_interval_report_arrays_are_read_only():
    report = IntervalReport(
        time=0.1, n_substeps=20, acc={"ind_a": np.zeros((5, 1))}, mass=np.zeros(1)
    )
    with pytest.raises(ValueError, match="read-only"):
        report.acc["ind_a"][...] = 1.0
    with pytest.raises(ValueError, match="read-only"):
        report.mass[...] = 1.0


def test_interval_report_empty_acc_allowed():
    report = IntervalReport(time=0.1, n_substeps=20, acc={}, mass=np.zeros(1))
    assert dict(report.acc) == {}


def test_acc_arrays_must_share_a_row_count():
    with pytest.raises(ValueError, match=r"^acc\['ind_b'\]: shape"):
        IntervalReport(
            time=0.1,
            n_substeps=20,
            acc={"ind_a": np.zeros((5, 1)), "ind_b": np.zeros((6, 1))},
            mass=np.zeros(1),
        )


def test_mass_must_be_one_dimensional():
    with pytest.raises(ValueError, match=r"^mass: expected a 1-D"):
        IntervalReport(time=0.1, n_substeps=20, acc={}, mass=np.zeros((1, 1)))


def test_mass_must_be_float64():
    with pytest.raises(ValueError, match=r"^mass: dtype"):
        IntervalReport(
            time=0.1, n_substeps=20, acc={}, mass=np.zeros(1, dtype=np.float32)
        )


def test_aux_must_match_acc_row_count():
    with pytest.raises(ValueError, match=r"^aux: 4 rows"):
        IntervalReport(
            time=0.1,
            n_substeps=20,
            acc={"ind_a": np.zeros((5, 1))},
            mass=np.zeros(1),
            aux=AuxFields({"true_error": np.zeros((4, 1))}),
        )
