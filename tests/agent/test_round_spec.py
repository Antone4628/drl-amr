"""Tests for the observation spec of the 2D agent core: the noun, the dict,
the file beside a checkpoint, and the comparison of a spec with a round.

AGENT_2D_ROADMAP item 2.8, signals S2-S12. The module under test is
agent/round_spec.py. Eight groups:

  - the noun: what it holds, what it derives and what it refuses;
  - to_dict: the eleven keys, plain values, numpy scalars (S2, S5, S7);
  - from_dict: the round trip and every refusal (S3, S8, S9);
  - a file written by code that has since changed is refused (S8);
  - the paths (S6);
  - the file: its exact text on the instance of the explanation file's
    section 2.8.2, the round trip through disk, and what loading refuses
    (S2, S3, S8, S9);
  - on the mock, a loaded spec builds what the saved one builds, bit for
    bit (S4, S5);
  - spec_violations (S12); and the signatures (S10).

The instance is the configuration of the mock's five-element mesh: ind_b
under scalar, alpha 0.1, budget 64, relative with a floor of 8, beta 2,
first_decider, ndim 2, trained at max_level 2, 2 rounds, one column.

Run: pytest tests/agent/test_round_spec.py -v
"""
from __future__ import annotations

import ast
import copy
import itertools
import json
from pathlib import Path

import numpy as np
import pytest

import agent.round_spec as round_spec
from agent.round_cache import RoundCache, read_round
from agent.round_commit import start_round
from agent.round_mask import COARSEN_RULES
from agent.round_observation import (
    NORMALIZATIONS,
    ObservationConfig,
    actor_observation,
    critic_feature_names,
    feature_names,
)
from agent.round_queue import round_order, round_thresholds, zone_distance
from agent.round_spec import (
    FIXED_RULES,
    SPEC_SUFFIX,
    SPEC_VERSION,
    ObservationSpec,
    from_dict,
    load_spec,
    save_spec,
    spec_path,
    spec_violations,
    to_dict,
)
from backends.mock_quad.backend import MockQuadBackend
from contract.checks import training_only_params
from contract.solver_contract import HOLD, REFINE
from contract.state import SolverState

INDICATORS = ("ind_a", "ind_b")
IC = {"x0": 0.3, "y0": 0.6, "width": 0.2}
KEYS = [
    "spec_version", "ndim", "max_level", "n_rounds", "n_comp", "observation", "beta", "coarsen_rule",
    "fixed_rules", "actor_features", "critic_features",
]
NAMES_2D = [
    "error", "error_x_min", "error_x_max", "error_y_min", "error_y_max", "error_family_max",
    "level", "level_x_min", "level_x_max", "level_y_min", "level_y_max",
    "wall_x_min", "wall_x_max", "wall_y_min", "wall_y_max", "resource", "round_progress",
]
RULES = {
    "queue": "priority_magnitude_stable",
    "error_feature": "max_over_columns",
    "zone_distance": "max_signed_over_columns",
    "thresholds": "interval_first",
    "normalization_state": "none",
}
# The instance as the file holds it, written out by hand.
INSTANCE_DICT = {
    "spec_version": 1,
    "ndim": 2,
    "max_level": 2,
    "n_rounds": 2,
    "n_comp": 1,
    "observation": {
        "indicator": "ind_b", "reduction": "scalar", "alpha": 0.1, "element_budget": 64,
        "normalization": "relative", "floor_decades": 8.0,
    },
    "beta": 2,
    "coarsen_rule": "first_decider",
    "fixed_rules": RULES,
    "actor_features": NAMES_2D,
    "critic_features": NAMES_2D,
}
# Section 2.8.2's file, every line: sorted keys, two spaces of indent.
_LIST = "".join(f'    "{name}"{"," if name != NAMES_2D[-1] else ""}\n' for name in NAMES_2D)
INSTANCE_TEXT = (
    "{\n"
    '  "actor_features": [\n' + _LIST + "  ],\n"
    '  "beta": 2,\n'
    '  "coarsen_rule": "first_decider",\n'
    '  "critic_features": [\n' + _LIST + "  ],\n"
    '  "fixed_rules": {\n'
    '    "error_feature": "max_over_columns",\n'
    '    "normalization_state": "none",\n'
    '    "queue": "priority_magnitude_stable",\n'
    '    "thresholds": "interval_first",\n'
    '    "zone_distance": "max_signed_over_columns"\n'
    "  },\n"
    '  "max_level": 2,\n'
    '  "n_comp": 1,\n'
    '  "n_rounds": 2,\n'
    '  "ndim": 2,\n'
    '  "observation": {\n'
    '    "alpha": 0.1,\n'
    '    "element_budget": 64,\n'
    '    "floor_decades": 8.0,\n'
    '    "indicator": "ind_b",\n'
    '    "normalization": "relative",\n'
    '    "reduction": "scalar"\n'
    "  },\n"
    '  "spec_version": 1\n'
    "}\n"
)


# --- Specs, meshes and edits --------------------------------------------------------

def _config(**changes) -> ObservationConfig:
    fields = dict(
        indicator="ind_b", reduction="scalar", alpha=0.1, element_budget=64, normalization="relative",
        floor_decades=8.0,
    )
    return ObservationConfig(**{**fields, **changes})


def _fields(**changes) -> dict:
    fields = dict(
        observation=_config(), beta=2, coarsen_rule="first_decider", ndim=2, max_level=2, n_rounds=2, n_comp=1
    )
    return {**fields, **changes}


def _spec(**changes) -> ObservationSpec:
    return ObservationSpec(**_fields(**changes))


def _edited(key: str, value, inside: str | None = None) -> dict:
    """The instance's dict with one value replaced, at the top or inside
    one of its two inner dicts."""
    data = copy.deepcopy(INSTANCE_DICT)
    (data[inside] if inside else data)[key] = value
    return data


def _without(key: str, inside: str | None = None) -> dict:
    data = copy.deepcopy(INSTANCE_DICT)
    del (data[inside] if inside else data)[key]
    return data


def _two_cell_mesh(nvar: int = 1, max_level: int = 2) -> tuple[MockQuadBackend, SolverState]:
    """The mock's five-element mesh: cells A and B on [0,2]x[0,1], periodic
    in x and walled in y, A refined. Positions 0-3 are A's children."""
    mock = MockQuadBackend(
        nx=2, ny=1, periodic=(True, False), lo=(0.0, 0.0), hi=(2.0, 1.0), max_level=max_level, nvar=nvar,
        dt_base=0.25, rotate_cells=False,
    )
    state = mock.reset(IC, amr_freq=4, initial_level=0)
    state, _ = mock.apply_marks(np.where(state.centroid[:, 0] < 1.0, REFINE, HOLD).astype(np.int8))
    return mock, state


def _graded_mesh(periodic: tuple[bool, bool], nvar: int) -> tuple[MockQuadBackend, SolverState]:
    """A 4x4 grid at level 1 with the cells near the feature refined once
    more, the rotation on: levels 1 and 2, with 2:1 faces between them."""
    mock = MockQuadBackend(nx=4, ny=4, periodic=periodic, max_level=2, nvar=nvar, dt_base=0.25, rotate_cells=True)
    state = mock.reset(IC, amr_freq=4, initial_level=1)
    near = np.hypot(state.centroid[:, 0] - IC["x0"], state.centroid[:, 1] - IC["y0"]) < 0.15
    assert near.any()
    state, _ = mock.apply_marks(np.where(near, REFINE, HOLD).astype(np.int8))
    return mock, state


MESHES = {
    "two cells": lambda nvar: _two_cell_mesh(nvar),
    "4x4 periodic": lambda nvar: _graded_mesh((True, True), nvar),
    "4x4 walled": lambda nvar: _graded_mesh((False, False), nvar),
    "4x4 mixed": lambda nvar: _graded_mesh((True, False), nvar),
}
# The cells that run, as (nvar, reduction, normalization).
CELLS = [(1, "scalar", "relative"), (1, "scalar", "dynamo"), (3, "per_variable", "relative")]
CELL_IDS = ["scalar, relative", "scalar, dynamo", "per_variable, relative"]


def _round(mesh: str, nvar: int, reduction: str) -> tuple[SolverState, RoundCache]:
    mock, state = MESHES[mesh](nvar)
    return state, read_round(mock, state, INDICATORS, reduction)


def _spec_of(cache: RoundCache, n_rounds: int, **config) -> ObservationSpec:
    """The spec of a policy trained on this round."""
    return _spec(
        observation=_config(reduction=cache.reduction, **config), ndim=cache.state.ndim,
        max_level=cache.state.max_level, n_rounds=n_rounds, n_comp=cache.n_comp,
    )


def _built(spec: ObservationSpec, state: SolverState, cache: RoundCache) -> tuple[np.ndarray, ...]:
    """Everything the within-round logic builds from a spec on one round:
    the pair, the order, the distances and every element's observation."""
    config = spec.observation
    e_max, e_min = round_thresholds(cache, config, spec.beta)
    order = round_order(cache, config, e_max, e_min)
    distance = zone_distance(cache.errors[config.indicator], config, e_max, e_min)
    round_state = start_round(state, 1, spec.n_rounds)
    seen = np.stack([actor_observation(cache, round_state, p, config) for p in range(state.n_active)])
    return e_max, e_min, order, distance, seen


def _same_bits(first: tuple[np.ndarray, ...], second: tuple[np.ndarray, ...]) -> bool:
    return all(a.dtype == b.dtype and a.tobytes() == b.tobytes() for a, b in zip(first, second, strict=True))


# --- The noun ----------------------------------------------------------------------

def test_the_spec_holds_what_it_was_given():
    config = _config()
    spec = _spec(observation=config)
    assert spec.observation is config
    assert (spec.beta, spec.coarsen_rule, spec.ndim) == (2, "first_decider", 2)
    assert (spec.max_level, spec.n_rounds, spec.n_comp) == (2, 2, 1)


def test_the_constants():
    assert SPEC_VERSION == 1 and type(SPEC_VERSION) is int
    assert SPEC_SUFFIX == ".spec.json"
    assert FIXED_RULES == RULES


@pytest.mark.parametrize("ndim, count", [(1, 11), (2, 17), (3, 23)])
def test_the_feature_lists_are_the_builders(ndim, count):
    spec = _spec(ndim=ndim)
    assert spec.actor_features == feature_names(ndim) and len(spec.actor_features) == count
    assert spec.critic_features == critic_feature_names(ndim) and len(spec.critic_features) == count


def test_the_instances_feature_list_by_name():
    assert list(_spec().actor_features) == NAMES_2D


def test_the_spec_is_frozen():
    spec = _spec()
    for name in ("observation", "beta", "coarsen_rule", "ndim", "max_level", "n_rounds", "n_comp"):
        with pytest.raises(AttributeError):
            setattr(spec, name, getattr(spec, name))
    with pytest.raises((AttributeError, TypeError)):
        spec.p_ur = 10.0


def test_the_fields_are_keyword_only():
    with pytest.raises(TypeError):
        ObservationSpec(_config(), 2, "first_decider", 2, 2, 2, 1)


def test_beta_as_an_integer_or_a_float_is_one_spec():
    assert _spec(beta=2) == _spec(beta=2.0)
    assert hash(_spec(beta=2)) == hash(_spec(beta=2.0))


@pytest.mark.parametrize(
    "change",
    [
        dict(observation=_config(alpha=0.2)), dict(observation=_config(indicator="ind_a")), dict(beta=3),
        dict(coarsen_rule="unanimous"), dict(ndim=3), dict(max_level=3), dict(n_rounds=3), dict(n_comp=2),
    ],
    ids=["alpha", "indicator", "beta", "coarsen_rule", "ndim", "max_level", "n_rounds", "n_comp"],
)
def test_a_spec_that_differs_in_one_field_is_another_spec(change):
    assert _spec(**change) != _spec()


NOUN_REFUSALS = [
    ("observation", "a config", "observation: 'a config', expected an ObservationConfig"),
    ("observation", None, "observation: None, expected an ObservationConfig"),
    ("observation", INSTANCE_DICT["observation"], "observation: {"),
    ("beta", 1, "beta: 1, expected a finite number above 1"),
    ("beta", 0.5, "beta: 0.5, expected a finite number above 1"),
    ("beta", True, "beta: True, expected a finite number above 1"),
    ("beta", float("nan"), "beta: nan, expected a finite number above 1"),
    ("beta", float("inf"), "beta: inf, expected a finite number above 1"),
    ("beta", "2", "beta: '2', expected a finite number above 1"),
    ("coarsen_rule", "majority", "coarsen_rule: 'majority', expected one of ['first_decider', 'any_coarsen', "),
    ("coarsen_rule", None, "coarsen_rule: None, expected one of"),
    ("ndim", 4, "ndim: 4, expected 1, 2 or 3"),
    ("ndim", 0, "ndim: 0, expected 1, 2 or 3"),
    ("ndim", True, "ndim: True, expected 1, 2 or 3"),
    ("ndim", 2.0, "ndim: 2.0, expected 1, 2 or 3"),
    ("max_level", 0, "max_level: 0, expected an integer of at least 1"),
    ("max_level", True, "max_level: True, expected an integer of at least 1"),
    ("max_level", 2.0, "max_level: 2.0, expected an integer of at least 1"),
    ("n_rounds", 0, "n_rounds: 0, expected an integer of at least 1"),
    ("n_rounds", True, "n_rounds: True, expected an integer of at least 1"),
    ("n_comp", 0, "n_comp: 0, expected an integer of at least 1"),
    ("n_comp", 1.0, "n_comp: 1.0, expected an integer of at least 1"),
]


@pytest.mark.parametrize("name, value, message", NOUN_REFUSALS, ids=[f"{n}={v!r}"[:40] for n, v, _ in NOUN_REFUSALS])
def test_the_spec_refuses_a_bad_field(name, value, message):
    with pytest.raises(ValueError) as info:
        _spec(**{name: value})
    assert str(info.value).startswith(message)


def test_beta_is_refused_in_round_thresholds_words():
    """One beta, checked the same way where it is saved and where it is used."""
    _, cache = _round("two cells", 1, "scalar")
    for beta in (1, True, float("nan")):
        with pytest.raises(ValueError) as used:
            round_thresholds(cache, _config(), beta)
        with pytest.raises(ValueError) as saved:
            _spec(beta=beta)
        assert str(saved.value) == str(used.value)


def test_dynamo_is_refused_above_one_column():
    dynamo = _config(normalization="dynamo", reduction="per_variable")
    for n_comp in (2, 3):
        with pytest.raises(ValueError) as info:
            _spec(observation=dynamo, n_comp=n_comp)
        assert str(info.value) == (
            f"normalization: 'dynamo' takes one column per indicator, and the spec has n_comp = {n_comp}"
        )
    assert _spec(observation=dynamo, n_comp=1).n_comp == 1
    assert _spec(observation=_config(reduction="per_variable"), n_comp=3).n_comp == 3


# --- to_dict -----------------------------------------------------------------------

def test_the_instance_as_a_dict():
    """Signal S2: eleven keys, and every value the one written by hand."""
    data = to_dict(_spec())
    assert list(data) == KEYS and len(data) == 11
    assert data == INSTANCE_DICT
    assert len(data["actor_features"]) == len(data["critic_features"]) == 17


def _plain(value) -> bool:
    if isinstance(value, dict):
        return all(type(key) is str and _plain(item) for key, item in value.items())
    if isinstance(value, list):
        return all(_plain(item) for item in value)
    return type(value) in (str, int, float)


def test_the_dict_holds_plain_values_only():
    data = to_dict(_spec())
    assert _plain(data)
    assert type(data["actor_features"]) is list and type(data["fixed_rules"]) is dict


def test_the_dict_is_new_each_time():
    spec = _spec()
    first = to_dict(spec)
    first["fixed_rules"]["queue"] = "changed"
    first["actor_features"].append("changed")
    first["observation"]["alpha"] = 0.5
    assert FIXED_RULES == RULES
    assert to_dict(spec) == INSTANCE_DICT
    assert spec.observation.alpha == 0.1


def test_no_reward_weight_is_in_the_dict():
    """The reward is training-only and is not saved here."""
    text = json.dumps(to_dict(_spec()))
    assert not any(name in text for name in ("p_ur", "p_or", "p_cr", "lambda"))


def test_beta_is_written_as_it_was_given():
    """Signal S5: an integer stays an integer."""
    assert json.dumps(to_dict(_spec(beta=2))["beta"]) == "2"
    assert json.dumps(to_dict(_spec(beta=2.0))["beta"]) == "2.0"
    assert json.dumps(to_dict(_spec(beta=1.2))["beta"]) == "1.2"
    assert json.dumps(to_dict(_spec(observation=_config(floor_decades=8)))["observation"]["floor_decades"]) == "8"


def test_numpy_scalars_do_not_reach_the_dict():
    """Signal S7."""
    config = _config(alpha=np.float64(0.1), element_budget=np.int64(64), floor_decades=np.float32(8.0))
    spec = _spec(
        observation=config, beta=np.int64(2), ndim=np.int64(2), max_level=np.int32(2), n_rounds=np.int64(2),
        n_comp=np.int8(1),
    )
    data = to_dict(spec)
    assert _plain(data) and data == INSTANCE_DICT
    assert json.dumps(data, sort_keys=True) == json.dumps(INSTANCE_DICT, sort_keys=True)
    assert _plain(to_dict(_spec(beta=np.float64(2.5)))) and to_dict(_spec(beta=np.float32(1.5)))["beta"] == 1.5


# --- from_dict ---------------------------------------------------------------------

def test_the_instance_from_its_dict():
    spec = from_dict(INSTANCE_DICT)
    assert spec == _spec()
    assert type(spec.beta) is int and type(spec.observation.element_budget) is int
    assert type(spec.observation) is ObservationConfig


def test_from_dict_changes_nothing_it_is_handed():
    data = copy.deepcopy(INSTANCE_DICT)
    from_dict(data)
    assert data == INSTANCE_DICT


GRID = [
    (normalization, reduction, rule, ndim, beta)
    for normalization, reduction, rule, ndim, beta in itertools.product(
        NORMALIZATIONS, ("scalar", "per_variable"), COARSEN_RULES, (1, 2, 3), (2, 2.0, 1.2)
    )
]


@pytest.mark.parametrize("normalization, reduction, rule, ndim, beta", GRID)
def test_round_trips(tmp_path, normalization, reduction, rule, ndim, beta):
    """Signal S3: through the dict, through JSON text and through disk."""
    n_comp = 3 if (reduction, normalization) == ("per_variable", "relative") else 1
    spec = _spec(
        observation=_config(normalization=normalization, reduction=reduction, indicator="ind_a"), beta=beta,
        coarsen_rule=rule, ndim=ndim, max_level=ndim + 1, n_rounds=ndim + 2, n_comp=n_comp,
    )
    data = to_dict(spec)
    assert to_dict(from_dict(data)) == data
    back = from_dict(json.loads(json.dumps(data)))
    assert back == spec and type(back.beta) is type(beta)
    written = save_spec(spec, tmp_path / "model.zip")
    loaded = load_spec(tmp_path / "model.zip")
    assert loaded == spec and type(loaded.beta) is type(beta)
    assert save_spec(loaded, tmp_path / "again.zip").read_bytes() == written.read_bytes()


def test_floats_come_back_exactly():
    """Every float through JSON text returns bit for bit."""
    rng = np.random.default_rng(28)
    for _ in range(300):
        alpha = float(rng.uniform(1e-6, 0.999999))
        beta = float(1.0 + 1e-9 + rng.exponential(1.0))
        floor = float(rng.uniform(0.1, 30.0))
        spec = _spec(observation=_config(alpha=alpha, floor_decades=floor), beta=beta)
        back = from_dict(json.loads(json.dumps(to_dict(spec))))
        assert back == spec
        assert (back.beta.hex(), back.observation.alpha.hex(), back.observation.floor_decades.hex()) == (
            beta.hex(), alpha.hex(), floor.hex()
        )


_SWAPPED = [NAMES_2D[6], *NAMES_2D[1:6], NAMES_2D[0], *NAMES_2D[7:]]
DICT_REFUSALS = [
    ("version 2", _edited("spec_version", 2), "spec_version: 2, this code reads version 1"),
    ("version 0", _edited("spec_version", 0), "spec_version: 0, this code reads version 1"),
    ("version True", _edited("spec_version", True), "spec_version: True, this code reads version 1"),
    ("version 1.0", _edited("spec_version", 1.0), "spec_version: 1.0, this code reads version 1"),
    ("version '1'", _edited("spec_version", "1"), "spec_version: '1', this code reads version 1"),
    ("no beta", _without("beta"), "spec: missing keys ['beta'], unknown keys []"),
    ("no max_level", _without("max_level"), "spec: missing keys ['max_level'], unknown keys []"),
    ("no version", _without("spec_version"), "spec: missing keys ['spec_version'], unknown keys []"),
    ("no fixed rules", _without("fixed_rules"), "spec: missing keys ['fixed_rules'], unknown keys []"),
    ("no actor list", _without("actor_features"), "spec: missing keys ['actor_features'], unknown keys []"),
    ("a reward weight", _edited("p_ur", 10.0), "spec: missing keys [], unknown keys ['p_ur']"),
    ("a list", [1, 2], "spec: expected a mapping, got list"),
    ("a string", "spec", "spec: expected a mapping, got str"),
    ("None", None, "spec: expected a mapping, got NoneType"),
    ("empty", {}, "spec: missing keys ['spec_version', 'ndim', 'max_level', 'n_rounds', 'n_comp', 'observation'"),
    ("names swapped", _edited("actor_features", _SWAPPED), "actor_features: ['level', 'error_x_min'"),
    ("a name short", _edited("actor_features", NAMES_2D[:-1]), "actor_features: ['error', 'error_x_min'"),
    ("a name more", _edited("actor_features", [*NAMES_2D, "pending"]), "actor_features: ['error', 'error_x_min'"),
    ("a name changed", _edited("actor_features", [*NAMES_2D[:-1], "progress"]), "actor_features: ['error'"),
    ("names as a tuple", _edited("actor_features", tuple(NAMES_2D)), "actor_features: ('error'"),
    ("critic short", _edited("critic_features", NAMES_2D[:-1]), "critic_features: ['error', 'error_x_min'"),
    ("critic swapped", _edited("critic_features", _SWAPPED), "critic_features: ['level', 'error_x_min'"),
    ("ndim 3, 17 names", _edited("ndim", 3), "actor_features: ['error', 'error_x_min'"),
    ("ndim 1, 17 names", _edited("ndim", 1), "actor_features: ['error', 'error_x_min'"),
    ("ndim 4", _edited("ndim", 4), "ndim: 4, expected 1, 2 or 3"),
    ("ndim True", _edited("ndim", True), "ndim: True, expected 1, 2 or 3"),
    ("thresholds per round", _edited("thresholds", "per_round", "fixed_rules"), "fixed_rules: {'queue'"),
    ("error feature", _edited("error_feature", "mean_over_columns", "fixed_rules"), "fixed_rules: {'queue'"),
    ("zone distance", _edited("zone_distance", "first_column", "fixed_rules"), "fixed_rules: {'queue'"),
    ("queue", _edited("queue", "position", "fixed_rules"), "fixed_rules: {'queue': 'position'"),
    ("state", _edited("normalization_state", "running_max", "fixed_rules"), "fixed_rules: {'queue'"),
    ("a rule missing", _without("queue", "fixed_rules"), "fixed_rules: {'error_feature'"),
    ("a rule more", _edited("budget", "masked", "fixed_rules"), "fixed_rules: {'queue'"),
    ("rules a list", _edited("fixed_rules", ["queue"]), "fixed_rules: ['queue']"),
    ("beta 1", _edited("beta", 1), "beta: 1, expected a finite number above 1"),
    ("beta True", _edited("beta", True), "beta: True, expected a finite number above 1"),
    ("beta nan", _edited("beta", float("nan")), "beta: nan, expected a finite number above 1"),
    ("beta a string", _edited("beta", "2"), "beta: '2', expected a finite number above 1"),
    ("rule", _edited("coarsen_rule", "majority"), "coarsen_rule: 'majority', expected one of"),
    ("max_level 0", _edited("max_level", 0), "max_level: 0, expected an integer of at least 1"),
    ("n_rounds True", _edited("n_rounds", True), "n_rounds: True, expected an integer of at least 1"),
    ("n_comp 1.0", _edited("n_comp", 1.0), "n_comp: 1.0, expected an integer of at least 1"),
    ("alpha 1.5", _edited("alpha", 1.5, "observation"), "alpha: 1.5, expected between 0 and 1"),
    ("alpha a string", _edited("alpha", "0.1", "observation"), "alpha: '0.1', expected a number"),
    ("budget 0", _edited("element_budget", 0, "observation"), "element_budget: 0, expected an integer"),
    ("normalization", _edited("normalization", "zscore", "observation"), "normalization: 'zscore', expected one of"),
    ("indicator empty", _edited("indicator", "", "observation"), "indicator: '', expected a name"),
    ("beta in observation", _edited("beta", 2, "observation"), "observation: missing keys [], unknown keys ['beta']"),
    ("no alpha", _without("alpha", "observation"), "observation: missing keys ['alpha'], unknown keys []"),
    ("observation a string", _edited("observation", "x"), "observation: expected a mapping, got str"),
    ("observation a list", _edited("observation", [0.1]), "observation: expected a mapping, got list"),
    (
        "dynamo, three columns",
        {**_edited("normalization", "dynamo", "observation"), "n_comp": 3},
        "normalization: 'dynamo' takes one column per indicator, and the spec has n_comp = 3",
    ),
]


@pytest.mark.parametrize("data, message", [c[1:] for c in DICT_REFUSALS], ids=[c[0] for c in DICT_REFUSALS])
def test_from_dict_refuses(data, message):
    """Signals S8 and S9: each refusal names the key at fault."""
    with pytest.raises(ValueError) as info:
        from_dict(data)
    assert str(info.value).startswith(message)


def test_a_feature_refusal_shows_both_lists():
    with pytest.raises(ValueError) as info:
        from_dict(_edited("actor_features", _SWAPPED))
    assert str(info.value) == f"actor_features: {_SWAPPED!r}, this code builds {NAMES_2D!r}"


def test_a_changed_value_loads():
    """Signal S8's last row: the dict cannot vouch for its values. A value
    that is valid is taken, whatever the policy was trained with."""
    assert from_dict(_edited("alpha", 0.2, "observation")).observation.alpha == 0.2
    assert from_dict(_edited("beta", 3)).beta == 3
    assert from_dict(_edited("coarsen_rule", "unanimous")).coarsen_rule == "unanimous"
    assert from_dict(_edited("max_level", 5)).max_level == 5
    dynamo = from_dict(_edited("normalization", "dynamo", "observation"))
    assert dynamo.observation.normalization == "dynamo" and dynamo.n_comp == 1


def test_from_dict_takes_any_mapping():
    from types import MappingProxyType

    assert from_dict(MappingProxyType(INSTANCE_DICT)) == _spec()


# --- Code that has changed since the file was written ------------------------------

def _renamed(ndim):
    return (*feature_names(ndim)[:-1], "round_fraction")


def _longer(ndim):
    return (*feature_names(ndim), "pending_neighbor")


def _reordered(ndim):
    names = feature_names(ndim)
    return (names[1], names[0], *names[2:])


@pytest.mark.parametrize("builder", [_renamed, _longer, _reordered], ids=["renamed", "added", "reordered"])
def test_a_file_from_before_the_actor_list_changed_is_refused(tmp_path, monkeypatch, builder):
    """Signal S8: the saved list is compared with the code as it is now."""
    save_spec(_spec(), tmp_path / "old.zip")
    monkeypatch.setattr(round_spec, "feature_names", builder)
    with pytest.raises(ValueError) as info:
        load_spec(tmp_path / "old.zip")
    assert str(info.value).startswith("actor_features: ['error', 'error_x_min'")
    assert str(info.value).endswith(f"this code builds {list(builder(2))!r}")


def test_a_file_from_before_the_critic_list_changed_is_refused(tmp_path, monkeypatch):
    save_spec(_spec(), tmp_path / "old.zip")
    monkeypatch.setattr(round_spec, "critic_feature_names", lambda ndim: (*feature_names(ndim), "true_error"))
    with pytest.raises(ValueError) as info:
        load_spec(tmp_path / "old.zip")
    assert str(info.value).startswith("critic_features: ['error', 'error_x_min'")


@pytest.mark.parametrize("name", list(RULES))
def test_a_file_from_before_a_fixed_rule_changed_is_refused(tmp_path, monkeypatch, name):
    save_spec(_spec(), tmp_path / "old.zip")
    monkeypatch.setattr(round_spec, "FIXED_RULES", {**RULES, name: "another_way"})
    with pytest.raises(ValueError) as info:
        load_spec(tmp_path / "old.zip")
    assert str(info.value).startswith("fixed_rules: {")


def test_a_file_of_another_layout_version_is_refused(tmp_path, monkeypatch):
    save_spec(_spec(), tmp_path / "old.zip")
    monkeypatch.setattr(round_spec, "SPEC_VERSION", 2)
    with pytest.raises(ValueError) as info:
        load_spec(tmp_path / "old.zip")
    assert str(info.value) == "spec_version: 1, this code reads version 2"
    assert to_dict(_spec())["spec_version"] == 2


def test_the_written_lists_are_the_codes_at_writing(monkeypatch):
    monkeypatch.setattr(round_spec, "critic_feature_names", lambda ndim: (*feature_names(ndim), "true_error"))
    data = to_dict(_spec())
    assert data["critic_features"] == [*NAMES_2D, "true_error"] and data["actor_features"] == NAMES_2D
    assert from_dict(data) == _spec()


# --- The paths (S6) ----------------------------------------------------------------

PATHS = [
    ("final_model", "final_model.spec.json"),
    ("final_model.zip", "final_model.spec.json"),
    ("ckpt/mock_10000_steps.zip", "ckpt/mock_10000_steps.spec.json"),
    ("ckpt/mock_10000_steps", "ckpt/mock_10000_steps.spec.json"),
    ("run.v2", "run.v2.spec.json"),
    ("run.v2.zip", "run.v2.spec.json"),
    ("a/b.spec.json", "a/b.spec.json.spec.json"),
    ("model.ZIP", "model.ZIP.spec.json"),
    ("archive.zip.zip", "archive.zip.spec.json"),
    ("/abs/dir/model.zip", "/abs/dir/model.spec.json"),
]


@pytest.mark.parametrize("checkpoint, expected", PATHS)
def test_spec_path(checkpoint, expected):
    assert spec_path(checkpoint) == Path(expected)
    assert spec_path(Path(checkpoint)) == Path(expected)
    assert isinstance(spec_path(checkpoint), Path)
    assert str(spec_path(checkpoint)).endswith(SPEC_SUFFIX)


def test_a_checkpoint_and_its_zip_share_one_spec(tmp_path):
    assert spec_path(tmp_path / "m") == spec_path(tmp_path / "m.zip")
    assert spec_path(tmp_path / "m").parent == tmp_path


# --- The file ----------------------------------------------------------------------

def test_the_instances_file(tmp_path):
    """Signal S2: section 2.8.2's file, byte for byte."""
    written = save_spec(_spec(), tmp_path / "checkpoints" / "mock_10000_steps.zip")
    assert written == tmp_path / "checkpoints" / "mock_10000_steps.spec.json"
    text = written.read_text(encoding="utf-8")
    assert text == INSTANCE_TEXT
    assert len(written.read_bytes()) == 1216 and text.count("\n") == 62


def test_the_file_is_sorted_json_with_a_final_newline(tmp_path):
    text = save_spec(_spec(), tmp_path / "m").read_text(encoding="utf-8")
    data = json.loads(text)
    assert data == INSTANCE_DICT
    assert list(data) == sorted(data) and list(data["observation"]) == sorted(data["observation"])
    assert list(data["fixed_rules"]) == sorted(data["fixed_rules"])
    assert text.endswith("}\n") and not text.endswith("\n\n")
    assert text == json.dumps(INSTANCE_DICT, indent=2, sort_keys=True) + "\n"


def test_save_returns_the_path_and_makes_the_directory(tmp_path):
    target = tmp_path / "results" / "run_1" / "checkpoints" / "mock_500_steps.zip"
    assert not target.parent.exists()
    written = save_spec(_spec(), target)
    assert written == spec_path(target) and written.is_file()
    assert save_spec(_spec(), str(target)) == written
    assert sorted(p.name for p in target.parent.iterdir()) == ["mock_500_steps.spec.json"]


def test_save_replaces_a_spec_already_there(tmp_path):
    save_spec(_spec(beta=3), tmp_path / "m.zip")
    save_spec(_spec(beta=2), tmp_path / "m")
    assert load_spec(tmp_path / "m.zip").beta == 2
    assert len(list(tmp_path.iterdir())) == 1


def test_the_same_spec_gives_the_same_bytes(tmp_path):
    first = save_spec(_spec(), tmp_path / "a.zip").read_bytes()
    assert save_spec(_spec(beta=2.0, n_comp=np.int64(1)), tmp_path / "b.zip").read_bytes() != first
    assert save_spec(_spec(n_comp=np.int64(1)), tmp_path / "c.zip").read_bytes() == first
    assert save_spec(load_spec(tmp_path / "a"), tmp_path / "d.zip").read_bytes() == first


def test_load_reads_beside_the_checkpoint_with_or_without_zip(tmp_path):
    save_spec(_spec(), tmp_path / "m.zip")
    assert load_spec(tmp_path / "m.zip") == load_spec(tmp_path / "m") == load_spec(str(tmp_path / "m")) == _spec()


def test_a_checkpoint_with_no_spec(tmp_path):
    with pytest.raises(FileNotFoundError) as info:
        load_spec(tmp_path / "none.zip")
    assert "none.spec.json" in str(info.value)
    assert not isinstance(info.value, ValueError)


FILE_REFUSALS = [
    ("not json", "{not json", "Expecting property name"),
    ("empty", "", "Expecting value"),
    ("a list", "[1, 2]", "spec: expected a mapping, got list"),
    ("a number", "3", "spec: expected a mapping, got int"),
    ("beta NaN", INSTANCE_TEXT.replace('"beta": 2', '"beta": NaN'), "beta: nan, expected a finite number above 1"),
    ("beta Infinity", INSTANCE_TEXT.replace('"beta": 2', '"beta": Infinity'), "beta: inf, expected a finite"),
    ("version 2", INSTANCE_TEXT.replace('"spec_version": 1', '"spec_version": 2'), "spec_version: 2, this code"),
    ("per round", INSTANCE_TEXT.replace('"interval_first"', '"per_round"'), "fixed_rules: {"),
    ("a name edited", INSTANCE_TEXT.replace('"resource"', '"budget_used"', 1), "actor_features: ['error'"),
    ("a critic name edited", INSTANCE_TEXT.replace('"resource"', '"budget_used"').replace(
        '"budget_used"', '"resource"', 1), "critic_features: ['error'"),
    ("a key removed", INSTANCE_TEXT.replace('  "n_comp": 1,\n', ""), "spec: missing keys ['n_comp']"),
    ("a key added", INSTANCE_TEXT.replace('  "ndim": 2,\n', '  "ndim": 2,\n  "p_ur": 1.0,\n'), "spec: missing keys []"),
]


@pytest.mark.parametrize("text, message", [c[1:] for c in FILE_REFUSALS], ids=[c[0] for c in FILE_REFUSALS])
def test_load_refuses(tmp_path, text, message):
    """Signals S8 and S9, on the file: one ValueError family for a file
    that is not JSON and a file from_dict refuses."""
    spec_path(tmp_path / "m").write_text(text, encoding="utf-8")
    with pytest.raises(ValueError) as info:
        load_spec(tmp_path / "m.zip")
    assert str(info.value).startswith(message)


def test_a_hand_edited_value_loads(tmp_path):
    """Signal S8's last row, on the file."""
    written = save_spec(_spec(), tmp_path / "m.zip")
    written.write_text(INSTANCE_TEXT.replace('"alpha": 0.1', '"alpha": 0.2'), encoding="utf-8")
    assert load_spec(tmp_path / "m").observation.alpha == 0.2
    written.write_text(INSTANCE_TEXT.replace('"beta": 2,', '"beta": 2.0,'), encoding="utf-8")
    assert load_spec(tmp_path / "m") == _spec() and type(load_spec(tmp_path / "m").beta) is float


def test_save_writes_no_nan(tmp_path, monkeypatch):
    """The noun refuses a NaN, so none reaches the writer; if one did, the
    writer would raise and leave no file, not write text that is not JSON."""
    monkeypatch.setattr(round_spec, "to_dict", lambda spec: {"beta": float("nan")})
    with pytest.raises(ValueError):
        save_spec(_spec(), tmp_path / "m.zip")
    assert list(tmp_path.iterdir()) == []


def test_the_key_order_in_a_file_does_not_matter(tmp_path):
    spec_path(tmp_path / "m").write_text(json.dumps(INSTANCE_DICT), encoding="utf-8")
    assert load_spec(tmp_path / "m") == _spec()


# --- On the mock: a loaded spec builds what the saved one builds (S4, S5) ----------

@pytest.mark.parametrize("nvar, reduction, normalization", CELLS, ids=CELL_IDS)
@pytest.mark.parametrize("indicator", INDICATORS)
@pytest.mark.parametrize("mesh", list(MESHES))
def test_a_loaded_spec_builds_the_same_round(tmp_path, mesh, indicator, nvar, reduction, normalization):
    state, cache = _round(mesh, nvar, reduction)
    spec = _spec_of(cache, 2, indicator=indicator, normalization=normalization)
    loaded = load_spec(save_spec(spec, tmp_path / "m.zip").with_name("m.zip"))
    assert loaded == spec and loaded is not spec
    first, second = _built(spec, state, cache), _built(loaded, state, cache)
    assert first[4].shape == (state.n_active, 17)
    assert _same_bits(first, second)


def test_the_instances_round_from_a_loaded_spec(tmp_path):
    """Section 2.8.2: the numbers of section 2.7.2, from the file."""
    state, cache = _round("two cells", 1, "scalar")
    loaded = load_spec(save_spec(_spec(), tmp_path / "m.zip").with_name("m"))
    e_max, e_min, order, distance, seen = _built(loaded, state, cache)
    assert e_max.tolist() == pytest.approx([0.457260], abs=1e-6)
    assert e_min.tolist() == pytest.approx([0.045726], abs=1e-6)
    assert order.tolist() == [2, 4, 0, 1, 3]
    assert distance.tolist() == pytest.approx([0.457132, 0.0, 1.0, 0.0, -0.790866], abs=1e-6)
    assert seen.shape == (5, 17)


@pytest.mark.parametrize("normalization", NORMALIZATIONS)
@pytest.mark.parametrize("indicator", INDICATORS)
@pytest.mark.parametrize("mesh", list(MESHES))
def test_beta_as_an_integer_or_a_float_gives_the_same_thresholds(tmp_path, mesh, indicator, normalization):
    """Signal S5: the file keeps whichever was given, and it does not matter."""
    state, cache = _round(mesh, 1, "scalar")
    whole = _spec_of(cache, 2, indicator=indicator, normalization=normalization)
    written = save_spec(ObservationSpec(**{**_as_fields(whole), "beta": 2.0}), tmp_path / "m.zip")
    real = load_spec(written.with_name("m"))
    assert type(whole.beta) is int and type(real.beta) is float and whole == real
    assert _same_bits(_built(whole, state, cache), _built(real, state, cache))


def _as_fields(spec: ObservationSpec) -> dict:
    names = ("observation", "beta", "coarsen_rule", "ndim", "max_level", "n_rounds", "n_comp")
    return {name: getattr(spec, name) for name in names}


def test_a_different_beta_gives_different_thresholds():
    """The comparisons above are not of things that cannot differ."""
    state, cache = _round("two cells", 1, "scalar")
    assert not _same_bits(_built(_spec(), state, cache), _built(_spec(beta=3), state, cache))
    other = _spec(observation=_config(alpha=0.2))
    assert not _same_bits(_built(_spec(), state, cache), _built(other, state, cache))


# --- spec_violations (S12) ---------------------------------------------------------

@pytest.mark.parametrize("nvar, reduction, normalization", CELLS, ids=CELL_IDS)
@pytest.mark.parametrize("mesh", list(MESHES))
def test_a_spec_built_from_a_round_has_no_finding_there(mesh, nvar, reduction, normalization):
    state, cache = _round(mesh, nvar, reduction)
    for n_rounds in (1, 2, 3):
        spec = _spec_of(cache, n_rounds, normalization=normalization)
        for round_index in range(1, n_rounds + 1):
            assert spec_violations(spec, cache, start_round(state, round_index, n_rounds)) == []


def test_the_instances_own_round_has_no_finding():
    state, cache = _round("two cells", 1, "scalar")
    found = spec_violations(_spec(), cache, start_round(state, 1, 2))
    assert found == [] and type(found) is list


def test_the_column_count_is_the_caches_not_the_states():
    """Three variables read under scalar are one column."""
    state, cache = _round("two cells", 3, "scalar")
    assert (state.nvar, cache.n_comp) == (3, 1)
    assert spec_violations(_spec(), cache, start_round(state, 1, 2)) == []
    assert spec_violations(_spec(n_comp=3), cache, start_round(state, 1, 2)) == [
        "n_comp: the spec says 3, the round has 1"
    ]


def test_another_max_level_is_found():
    mock, state = _two_cell_mesh(max_level=3)
    cache = read_round(mock, state, INDICATORS, "scalar")
    assert spec_violations(_spec(), cache, start_round(state, 1, 2)) == [
        "max_level: the spec says 2, the round has 3"
    ]


def test_another_round_count_is_found():
    state, cache = _round("two cells", 1, "scalar")
    assert spec_violations(_spec(), cache, start_round(state, 1, 3)) == ["n_rounds: the spec says 2, the round has 3"]
    assert spec_violations(_spec(), cache, start_round(state, 3, 3)) == ["n_rounds: the spec says 2, the round has 3"]


def test_five_findings_in_order():
    """Section 2.8.2's last row: three variables under per_variable, another
    max_level, three rounds, and a cache without the spec's indicator."""
    mock, state = _two_cell_mesh(nvar=3, max_level=3)
    cache = read_round(mock, state, ["ind_a"], "per_variable")
    assert spec_violations(_spec(), cache, start_round(state, 2, 3)) == [
        "max_level: the spec says 2, the round has 3",
        "n_comp: the spec says 1, the round has 3",
        "reduction: the spec says 'scalar', the cache was read under 'per_variable'",
        "indicator: 'ind_b' is not in the cache, which holds ['ind_a']",
        "n_rounds: the spec says 2, the round has 3",
    ]


ONE_RULE = [
    (dict(ndim=3), "ndim: the spec says 3, the round has 2"),
    (dict(ndim=1), "ndim: the spec says 1, the round has 2"),
    (dict(max_level=1), "max_level: the spec says 1, the round has 2"),
    (dict(max_level=7), "max_level: the spec says 7, the round has 2"),
    (dict(n_comp=2), "n_comp: the spec says 2, the round has 1"),
    (
        dict(observation=_config(reduction="per_variable")),
        "reduction: the spec says 'per_variable', the cache was read under 'scalar'",
    ),
    (
        dict(observation=_config(indicator="ind_c")),
        "indicator: 'ind_c' is not in the cache, which holds ['ind_a', 'ind_b']",
    ),
    (dict(n_rounds=1), "n_rounds: the spec says 1, the round has 2"),
    (dict(n_rounds=4), "n_rounds: the spec says 4, the round has 2"),
]


@pytest.mark.parametrize("change, finding", ONE_RULE, ids=[f.split(":")[0] + f[-2:] for _, f in ONE_RULE])
def test_each_rule_alone(change, finding):
    """One field of the spec off the round gives that rule's finding and
    no other."""
    state, cache = _round("two cells", 1, "scalar")
    assert spec_violations(_spec(**change), cache, start_round(state, 1, 2)) == [finding]


def test_all_six_rules_at_once():
    state, cache = _round("two cells", 1, "scalar")
    spec = _spec(
        observation=_config(reduction="per_variable", indicator="ind_c"), ndim=3, max_level=4, n_rounds=5, n_comp=3
    )
    found = spec_violations(spec, cache, start_round(state, 1, 2))
    assert [finding.split(":")[0] for finding in found] == [
        "ndim", "max_level", "n_comp", "reduction", "indicator", "n_rounds"
    ]


def test_what_does_not_move_within_an_interval_is_not_a_finding():
    """The round index, the committed count and the record are the
    round's, not the spec's; nor are the budget, alpha, beta or the rule
    compared with anything: a round does not carry them."""
    state, cache = _round("two cells", 1, "scalar")
    other = _spec(observation=_config(alpha=0.3, element_budget=1, floor_decades=2.0), beta=5, coarsen_rule="unanimous")
    for round_index in (1, 2):
        assert spec_violations(other, cache, start_round(state, round_index, 2)) == []


def test_spec_violations_changes_nothing_and_returns_a_new_list():
    state, cache = _round("two cells", 1, "scalar")
    spec, round_state = _spec(max_level=3), start_round(state, 1, 2)
    before = to_dict(spec)
    first = spec_violations(spec, cache, round_state)
    first.append("changed")
    assert spec_violations(spec, cache, round_state) == ["max_level: the spec says 3, the round has 2"]
    assert to_dict(spec) == before and round_state.committed == state.n_active


def test_the_builders_raise_nothing_for_a_round_the_spec_does_not_describe():
    """Section 2.8.2's sixth point. Position 0 is at level 1: its level
    feature is 1/2 at max_level 2 and 1/3 at max_level 3, and round 1's
    progress is 1/2 of two rounds and 1/3 of three."""
    config = _config()
    level = feature_names(2).index("level")
    progress = feature_names(2).index("round_progress")
    state, cache = _round("two cells", 1, "scalar")
    deeper_mock, deeper_state = _two_cell_mesh(max_level=3)
    deeper = read_round(deeper_mock, deeper_state, INDICATORS, "scalar")
    assert int(state.level[0]) == int(deeper_state.level[0]) == 1
    here = actor_observation(cache, start_round(state, 1, 2), 0, config)
    there = actor_observation(deeper, start_round(deeper_state, 1, 2), 0, config)
    longer = actor_observation(cache, start_round(state, 1, 3), 0, config)
    assert (here[level], there[level]) == (np.float32(0.5), np.float32(1 / 3))
    assert (here[progress], longer[progress]) == (np.float32(0.5), np.float32(1 / 3))
    assert len(spec_violations(_spec(), deeper, start_round(deeper_state, 1, 2))) == 1
    assert len(spec_violations(_spec(), cache, start_round(state, 1, 3))) == 1


# --- The signatures (S10) ----------------------------------------------------------

@pytest.mark.parametrize("function", [to_dict, from_dict, spec_path, save_spec, load_spec, spec_violations])
def test_no_function_takes_a_training_only_type(function):
    assert training_only_params(function) == []


def test_the_module_imports_no_rl_library_and_no_backend():
    roots = set()
    for node in ast.walk(ast.parse(Path(round_spec.__file__).read_text())):
        if isinstance(node, ast.Import):
            roots |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])
    assert roots == {"__future__", "json", "math", "collections", "dataclasses", "numbers", "pathlib", "agent"}
