"""Tests for the v2 SolverState (contract/state.py).

AGENT_2D_ROADMAP item 1.1. The instance is the five-element mesh of the
phase's explanation doc: two base cells A and B side by side, periodic in x
and walled in y, with A refined once. Positions 0-3 are A's children in child
order (c = ix + 2*iy); position 4 is B.

    y=1  +-----+-----+-----------+
         |  2  |  3  |           |
         +-----+-----+     4     |
         |  0  |  1  |           |
    y=0  +-----+-----+-----------+
        x=0         x=1         x=2

These tests cover form only: shapes, dtypes, read-only arrays, the globals.
Whether a state describes a valid mesh (neighbor symmetry, family geometry,
2:1 balance) is the topology assertions' job (item 1.3).

Run: pytest tests/contract/test_contract_state.py -v
"""
import dataclasses

import numpy as np
import pytest

from contract.state import SolverState

# Every array field that is always present, with a dtype it must reject.
ARRAY_FIELDS = {
    "level": np.int64,
    "neighbors": np.int64,
    "neighbor_wrap": np.int8,
    "family": np.int64,
    "can_refine": np.int8,
    "can_coarsen": np.int8,
    "centroid": np.float32,
    "h": np.float32,
    "attributes": np.float32,
    "domain_lo": np.float32,
    "domain_hi": np.float32,
    "periodic": np.int8,
}


def five_element_fields(nvar=1):
    """Keyword arguments for the five-element instance (fresh arrays)."""
    neighbors = np.array(
        [
            [[4, -1], [1, -1], [-1, -1], [2, -1]],
            [[0, -1], [4, -1], [-1, -1], [3, -1]],
            [[4, -1], [3, -1], [0, -1], [-1, -1]],
            [[2, -1], [4, -1], [1, -1], [-1, -1]],
            [[1, 3], [0, 2], [-1, -1], [-1, -1]],
        ],
        dtype=np.int32,
    )
    neighbor_wrap = np.zeros((5, 4), dtype=bool)
    neighbor_wrap[0, 0] = neighbor_wrap[2, 0] = neighbor_wrap[4, 1] = True
    family = np.array([[0, 1, 2, 3]] * 4 + [[-1, -1, -1, -1]], dtype=np.int32)
    return dict(
        level=np.array([1, 1, 1, 1, 0], dtype=np.int32),
        neighbors=neighbors,
        neighbor_wrap=neighbor_wrap,
        family=family,
        can_refine=np.ones(5, dtype=bool),
        can_coarsen=np.array([True, True, True, True, False]),
        centroid=np.array(
            [[0.25, 0.25], [0.75, 0.25], [0.25, 0.75], [0.75, 0.75], [1.5, 0.5]]
        ),
        h=np.array([[0.5, 0.5]] * 4 + [[1.0, 1.0]]),
        attributes=np.zeros((5, 0)),
        n_active=5,
        ndim=2,
        nvar=nvar,
        max_level=2,
        mesh_epoch=1,
        time=0.0,
        interval_T=0.1,
        dt_base=0.01,
        amr_freq=10,
        dt_stable=0.005,
        domain_lo=np.array([0.0, 0.0]),
        domain_hi=np.array([2.0, 1.0]),
        periodic=np.array([True, False]),
    )


def blank_fields(ndim, n=3):
    """Keyword arguments for a zero-filled state of any dimension: right
    shapes and dtypes, no meaningful mesh."""
    faces, slots, children = 2 * ndim, 2 ** (ndim - 1), 2**ndim
    return dict(
        level=np.zeros(n, dtype=np.int32),
        neighbors=np.full((n, faces, slots), -1, dtype=np.int32),
        neighbor_wrap=np.zeros((n, faces), dtype=bool),
        family=np.full((n, children), -1, dtype=np.int32),
        can_refine=np.ones(n, dtype=bool),
        can_coarsen=np.zeros(n, dtype=bool),
        centroid=np.zeros((n, ndim)),
        h=np.ones((n, ndim)),
        attributes=np.zeros((n, 0)),
        n_active=n,
        ndim=ndim,
        nvar=1,
        max_level=2,
        mesh_epoch=0,
        time=0.0,
        interval_T=0.1,
        dt_base=0.01,
        amr_freq=10,
        dt_stable=0.01,
        domain_lo=np.zeros(ndim),
        domain_hi=np.ones(ndim),
        periodic=np.zeros(ndim, dtype=bool),
    )


# --- S1: the five-element instance -----------------------------------------

def test_five_element_instance_shapes():
    st = SolverState(**five_element_fields())
    assert st.neighbors.shape == (5, 4, 2)
    assert st.neighbor_wrap.shape == (5, 4)
    assert st.family.shape == (5, 4)
    assert st.centroid.shape == (5, 2)
    assert st.h.shape == (5, 2)
    assert st.attributes.shape == (5, 0)
    assert st.state_mean is None
    assert st.element_id is None


# --- S2: one schema, the dimension in the shapes ---------------------------

@pytest.mark.parametrize(
    "ndim, neighbors_shape, family_shape",
    [(1, (3, 2, 1), (3, 2)), (2, (3, 4, 2), (3, 4)), (3, (3, 6, 4), (3, 8))],
)
def test_schema_is_dimension_agnostic_by_shape(ndim, neighbors_shape, family_shape):
    st = SolverState(**blank_fields(ndim))
    assert st.neighbors.shape == neighbors_shape
    assert st.family.shape == family_shape


# --- S3: frozen means frozen -----------------------------------------------

@pytest.mark.parametrize("name", ARRAY_FIELDS)
def test_array_fields_are_read_only(name):
    st = SolverState(**five_element_fields())
    with pytest.raises(ValueError, match="read-only"):
        getattr(st, name)[...] = 0


def test_optional_arrays_are_read_only():
    fields = five_element_fields()
    fields["state_mean"] = np.zeros((5, 1))
    fields["element_id"] = np.arange(5, dtype=np.int64)
    st = SolverState(**fields)
    with pytest.raises(ValueError, match="read-only"):
        st.state_mean[...] = 0
    with pytest.raises(ValueError, match="read-only"):
        st.element_id[...] = 0


def test_fields_cannot_be_rebound():
    st = SolverState(**five_element_fields())
    with pytest.raises(dataclasses.FrozenInstanceError):
        st.level = np.zeros(5, dtype=np.int32)
    with pytest.raises(dataclasses.FrozenInstanceError):
        st.n_active = 6


def test_fields_are_keyword_only():
    fields = five_element_fields()
    with pytest.raises(TypeError):
        SolverState(*fields.values())


# --- S4: wrong dtype, wrong shape, named field -----------------------------

@pytest.mark.parametrize("name, bad_dtype", ARRAY_FIELDS.items())
def test_wrong_dtype_rejected(name, bad_dtype):
    fields = five_element_fields()
    fields[name] = fields[name].astype(bad_dtype)
    with pytest.raises(ValueError, match=rf"^{name}: dtype"):
        SolverState(**fields)


@pytest.mark.parametrize("name", ARRAY_FIELDS)
def test_wrong_shape_rejected(name):
    fields = five_element_fields()
    good = fields[name]
    # One extra entry along the first axis: a sixth element, or a third axis.
    fields[name] = np.concatenate([good, good[:1]], axis=0)
    with pytest.raises(ValueError, match=rf"^{name}: shape"):
        SolverState(**fields)


def test_list_rejected_with_field_name():
    fields = five_element_fields()
    fields["level"] = [1, 1, 1, 1, 0]
    with pytest.raises(ValueError, match=r"^level: expected a numpy array"):
        SolverState(**fields)


def test_attributes_must_be_two_dimensional():
    fields = five_element_fields()
    fields["attributes"] = np.zeros(5)
    with pytest.raises(ValueError, match=r"^attributes: expected a 2-D"):
        SolverState(**fields)


def test_attributes_width_is_free():
    fields = five_element_fields()
    fields["attributes"] = np.zeros((5, 3))
    assert SolverState(**fields).attributes.shape == (5, 3)


# --- S5: the optional arrays -----------------------------------------------

def test_state_mean_accepted_at_nvar_3():
    fields = five_element_fields(nvar=3)
    fields["state_mean"] = np.zeros((5, 3))
    assert SolverState(**fields).state_mean.shape == (5, 3)


def test_state_mean_must_have_nvar_columns():
    fields = five_element_fields(nvar=3)
    fields["state_mean"] = np.zeros((5, 1))
    with pytest.raises(ValueError, match=r"^state_mean: shape"):
        SolverState(**fields)


def test_element_id_accepted_as_int64():
    fields = five_element_fields()
    fields["element_id"] = np.arange(5, dtype=np.int64)
    assert SolverState(**fields).element_id.dtype == np.int64


def test_element_id_rejected_as_int32():
    fields = five_element_fields()
    fields["element_id"] = np.arange(5, dtype=np.int32)
    with pytest.raises(ValueError, match=r"^element_id: dtype"):
        SolverState(**fields)


# --- the globals -----------------------------------------------------------

@pytest.mark.parametrize(
    "name, bad_value", [("ndim", 4), ("ndim", 0), ("n_active", 0), ("nvar", 0)]
)
def test_bad_globals_rejected(name, bad_value):
    fields = five_element_fields()
    fields[name] = bad_value
    with pytest.raises(ValueError, match=rf"^{name}: "):
        SolverState(**fields)
