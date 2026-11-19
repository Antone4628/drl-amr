"""Tests for the training-side surface of agent/round_observation.py: the
feature bounds, the critic builder and the observation dict; and the builder
as a real consumer of the round cache, counted.

AGENT_2D_ROADMAP item 2.2, sitting 2, signals T7-T11 and T13. Five groups:

  - feature_bounds: the range of every feature, by name (T7);
  - critic_observation: the actor's features in this build, and the
    training-only parameter its signature declares (T8);
  - observation and observation_bounds: the dict and its two keys (T9), and
    that nothing in agent/ imports an RL library (T13);
  - on the mock quad backend: one interval of two rounds with every element
    observed through the dict, counted by the counting fixture of
    tests/conftest.py (T10);
  - on Python1DBackend: two intervals of three rounds on real numerics,
    counted, every element compared with the sequential core's
    build_observation on the wrapped contract's own state (T11).

The actor builder's own features are tests/agent/test_round_observation.py.

Run: pytest tests/agent/test_round_dict.py -v
"""
from __future__ import annotations

import ast
import math
from pathlib import Path

import numpy as np
import pytest

import agent
from agent.normalization import normalized_error
from agent.observation import build_observation
from agent.round_cache import RoundCache, read_round
from agent.round_observation import (
    ACTOR_KEY,
    CRITIC_KEY,
    ObservationConfig,
    actor_observation,
    critic_feature_names,
    critic_observation,
    feature_bounds,
    feature_names,
    observation,
    observation_bounds,
)
from agent.round_state import RoundState
from backends.mock_quad.backend import MockQuadBackend
from backends.python_1d.backend import Python1DBackend
from backends.python_1d.solvers.dg_advection_solver_multiround import DGAdvectionSolver
from contract.checks import training_only_params
from contract.reports import AuxFields
from contract.solver_contract import HOLD, REFINE
from contract.state import SolverState

ALPHA = 0.1
BUDGET = 8
FLOOR = 8.0
MOCK_NAMES = ("ind_a", "ind_b")
ONE_D_NAMES = ("zz_style", "raw_jump")
IC = {"x0": 0.5, "y0": 0.5, "width": 0.25}
BASE_XELEM = np.array([-1.0, -0.4, 0.0, 0.4, 1.0])
RL_LIBRARIES = {"gymnasium", "gym", "stable_baselines3", "sb3_contrib"}


def config(normalization="relative", **changes) -> ObservationConfig:
    """The mock instance's configuration, with any field replaced."""
    fields = {
        "indicator": "ind_b",
        "reduction": "scalar",
        "alpha": ALPHA,
        "element_budget": BUDGET,
        "normalization": normalization,
        "floor_decades": FLOOR,
    }
    return ObservationConfig(**{**fields, **changes})


def at(committed=5, round_index=1, n_rounds=2) -> RoundState:
    return RoundState(round_index=round_index, n_rounds=n_rounds, committed=committed)


def make_mock(*, rotate_cells=False, nvar=1) -> MockQuadBackend:
    """Cells A and B on [0,2]x[0,1], periodic in x and walled in y."""
    return MockQuadBackend(
        nx=2,
        ny=1,
        periodic=(True, False),
        lo=(0.0, 0.0),
        hi=(2.0, 1.0),
        max_level=2,
        nvar=nvar,
        dt_base=0.25,
        rotate_cells=rotate_cells,
    )


def refine_cell_a(state: SolverState) -> np.ndarray:
    """Marks that refine whatever lies in cell A, addressed by centroid."""
    return np.where(state.centroid[:, 0] < 1.0, REFINE, HOLD).astype(np.int8)


def five_elements() -> RoundCache:
    """A cache on the mock's five-element mesh: cell A refined, cell B not."""
    mock = make_mock()
    state, _ = mock.apply_marks(refine_cell_a(mock.reset(IC, amr_freq=4, initial_level=0)))
    return read_round(mock, state, MOCK_NAMES, "scalar")


def make_1d(*, icase=1, initial_level=1):
    """A Python1DBackend on the real base mesh, reset; and its first state."""
    solver = DGAdvectionSolver(
        nop=4, xelem=BASE_XELEM.copy(), max_elements=120, max_level=3, icase=icase, balance=False
    )
    backend = Python1DBackend(solver)
    return backend, backend.reset({"icase": icase}, amr_freq=20, initial_level=initial_level)


def within(obs: np.ndarray, bounds: tuple[np.ndarray, np.ndarray]) -> bool:
    """Whether a vector lies within (low, high), inclusive, no tolerance."""
    low, high = bounds
    return bool(((low <= obs) & (obs <= high)).all())


# --- feature_bounds (T7) -----------------------------------------------------------

@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_bounds_are_two_float32_vectors_as_long_as_the_names(ndim):
    low, high = feature_bounds(ndim, config())
    for bound in (low, high):
        assert bound.dtype == np.float32
        assert bound.shape == (len(feature_names(ndim)),) == (6 * ndim + 5,)
    assert (low <= high).all()


@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_relative_bounds_by_name(ndim):
    low, high = feature_bounds(ndim, config())
    for name, lo, hi in zip(feature_names(ndim), low.tolist(), high.tolist(), strict=True):
        if name.startswith("error"):
            assert (lo, hi) == (-FLOOR, 1.0), name
        elif name == "resource":
            assert (lo, hi) == (0.0, math.inf), name
        else:
            assert (lo, hi) == (0.0, 1.0), name


def test_dynamo_error_features_are_unbounded():
    names = feature_names(2)
    low, high = feature_bounds(2, config("dynamo"))
    rel_low, rel_high = feature_bounds(2, config("relative"))
    for i, name in enumerate(names):
        if name.startswith("error"):
            assert (low[i], high[i]) == (-math.inf, math.inf)
        else:
            assert (low[i], high[i]) == (rel_low[i], rel_high[i])
    assert sum(name.startswith("error") for name in names) == 6


@pytest.mark.parametrize("alpha", [0.007, 0.1, 0.3, 0.9])
def test_alpha_sets_the_upper_error_bound(alpha):
    _, high = feature_bounds(2, config(alpha=alpha))
    assert high[0] == np.float32(-math.log10(alpha))
    assert high[0] > 0.0


def test_the_floor_sets_the_lower_error_bound():
    low, _ = feature_bounds(2, config(floor_decades=12.5))
    assert low[0] == -12.5
    # Not read under dynamo.
    assert feature_bounds(2, config("dynamo", floor_decades=12.5))[0][0] == -math.inf


def test_bounds_are_new_arrays_at_every_call():
    first = feature_bounds(2, config())
    second = feature_bounds(2, config())
    assert first[0] is not second[0] and first[1] is not second[1]
    first[0][0] = 99.0
    assert feature_bounds(2, config())[0][0] == -FLOOR


def test_bounds_reject_other_dimensions():
    with pytest.raises(ValueError, match="^ndim: 4, expected 1, 2 or 3"):
        feature_bounds(4, config())


def test_the_budget_and_the_indicator_do_not_move_the_bounds():
    base = feature_bounds(2, config())
    other = feature_bounds(2, config(element_budget=500, indicator="ind_a", reduction="per_variable"))
    assert np.array_equal(base[0], other[0]) and np.array_equal(base[1], other[1])


# --- critic_observation (T8) -------------------------------------------------------

def test_the_critic_sees_the_actors_features():
    cache = five_elements()
    for position in range(cache.n_active):
        actor = actor_observation(cache, at(), position, config())
        critic = critic_observation(cache, at(), position, config())
        assert critic.dtype == np.float32 and critic.shape == (17,)
        assert critic is not actor
        assert np.array_equal(critic, actor)


@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_the_critics_names_are_the_actors(ndim):
    assert critic_feature_names(ndim) == feature_names(ndim)


def test_only_the_training_builders_take_the_auxiliary_type():
    assert training_only_params(actor_observation) == []
    assert training_only_params(critic_observation) == ["aux"]
    assert training_only_params(observation) == ["aux"]


@pytest.mark.parametrize("build", [critic_observation, observation], ids=["the critic builder", "the dict"])
def test_auxiliary_fields_are_rejected_until_a_privileged_feature_exists(build):
    cache = five_elements()
    aux = AuxFields(fields={"true_error": np.zeros((cache.n_active, 1))})
    with pytest.raises(ValueError, match="^aux: no privileged critic feature is defined yet, expected None"):
        build(cache, at(), 0, config(), aux)
    # None, passed or left out, is accepted.
    build(cache, at(), 0, config(), None)
    build(cache, at(), 0, config())


@pytest.mark.parametrize("build", [critic_observation, observation], ids=["the critic builder", "the dict"])
def test_the_training_builders_reject_what_the_actor_builder_rejects(build):
    cache = five_elements()
    with pytest.raises(ValueError, match=r"^position: 5, expected an integer in \[0, 5\)"):
        build(cache, at(), 5, config())
    with pytest.raises(ValueError, match="^indicator: 'zz_style' is not in the cache"):
        build(cache, at(), 0, config(indicator="zz_style"))


# --- observation and observation_bounds (T9, T13) ----------------------------------

def test_the_keys():
    assert (ACTOR_KEY, CRITIC_KEY) == ("actor", "critic")


def test_the_dict_holds_the_two_builders_vectors():
    cache = five_elements()
    for position in range(cache.n_active):
        obs = observation(cache, at(), position, config())
        assert list(obs) == [ACTOR_KEY, CRITIC_KEY]
        assert np.array_equal(obs[ACTOR_KEY], actor_observation(cache, at(), position, config()))
        assert np.array_equal(obs[CRITIC_KEY], critic_observation(cache, at(), position, config()))


def test_the_two_values_are_separate_arrays():
    cache = five_elements()
    obs = observation(cache, at(), 0, config())
    assert obs[ACTOR_KEY] is not obs[CRITIC_KEY]
    kept = obs[CRITIC_KEY].copy()
    obs[ACTOR_KEY][:] = 99.0
    assert np.array_equal(obs[CRITIC_KEY], kept)


@pytest.mark.parametrize("normalization", ["dynamo", "relative"])
def test_the_bounds_dict_has_the_observations_keys(normalization):
    cfg = config(normalization)
    bounds = observation_bounds(2, cfg)
    assert list(bounds) == list(observation(five_elements(), at(), 0, cfg)) == [ACTOR_KEY, CRITIC_KEY]
    low, high = feature_bounds(2, cfg)
    arrays = []
    for key in (ACTOR_KEY, CRITIC_KEY):
        key_low, key_high = bounds[key]
        assert np.array_equal(key_low, low) and np.array_equal(key_high, high)
        arrays += [key_low, key_high]
    # Four separate arrays: a space built from one key cannot alias another.
    assert len({id(a) for a in arrays}) == 4


@pytest.mark.parametrize("normalization", ["dynamo", "relative"])
@pytest.mark.parametrize("indicator", MOCK_NAMES)
def test_every_value_of_the_dict_is_inside_its_bounds(indicator, normalization):
    cache = five_elements()
    cfg = config(normalization, indicator=indicator)
    bounds = observation_bounds(2, cfg)
    for position in range(cache.n_active):
        for committed in (1, 5, 40):
            obs = observation(cache, at(committed), position, cfg)
            for key in (ACTOR_KEY, CRITIC_KEY):
                assert within(obs[key], bounds[key])


def test_nothing_in_agent_imports_an_rl_library():
    """agent/ builds arrays and bounds; the environment builds the space."""
    modules = sorted(Path(agent.__file__).parent.glob("*.py"))
    assert {"round_cache.py", "round_observation.py", "round_state.py"} <= {m.name for m in modules}
    for module in modules:
        roots = set()
        for node in ast.walk(ast.parse(module.read_text())):
            if isinstance(node, ast.Import):
                roots |= {alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module:
                roots.add(node.module.split(".")[0])
        assert not roots & RL_LIBRARIES, module.name


# --- On the mock: the builder as a consumer of the cache (T10) ---------------------

def _observe_a_round(cache: RoundCache, round_index: int, cfg: ObservationConfig) -> None:
    """Observe every element of one round through the dict, as a driver
    would, with the committed count at the round's start."""
    bounds = observation_bounds(cache.state.ndim, cfg)
    round_state = RoundState(round_index=round_index, n_rounds=2, committed=cache.n_active)
    for position in range(cache.n_active):
        obs = observation(cache, round_state, position, cfg)
        for key in (ACTOR_KEY, CRITIC_KEY):
            assert np.isfinite(obs[key]).all()
            assert within(obs[key], bounds[key])


@pytest.mark.parametrize("rotate_cells", [False, True], ids=["pinned", "rotating"])
def test_observing_every_element_of_two_rounds_is_still_five_calls(rotate_cells, counting):
    """2R + 1 for R = 2 with a real consumer: seven observations, two keys
    each, and the backend is asked for errors once per round."""
    mock = make_mock(rotate_cells=rotate_cells)
    state = mock.reset(IC, amr_freq=4, initial_level=0)
    backend = counting(mock)
    cfg = config()

    cache1 = read_round(backend, state, MOCK_NAMES, "scalar")
    _observe_a_round(cache1, 1, cfg)
    state, _ = backend.apply_marks(refine_cell_a(cache1.state))
    cache2 = read_round(backend, state, MOCK_NAMES, "scalar")
    _observe_a_round(cache2, 2, cfg)
    state, _ = backend.apply_marks(np.full(cache2.n_active, HOLD, dtype=np.int8))
    backend.step_interval(["ind_b"], "scalar", 1)

    assert (cache1.n_active, cache2.n_active) == (2, 5)
    assert dict(backend.calls) == {"compute_error": 2, "apply_marks": 2, "step_interval": 1}


def test_the_count_is_the_same_at_three_variables(counting):
    mock = make_mock(nvar=3)
    state = mock.reset(IC, amr_freq=4, initial_level=0)
    backend = counting(mock)
    cfg = config(reduction="per_variable")

    for round_index in (1, 2):
        cache = read_round(backend, state, MOCK_NAMES, "per_variable")
        assert cache.n_comp == 3
        _observe_a_round(cache, round_index, cfg)
        marks = refine_cell_a(state) if round_index == 1 else np.full(state.n_active, HOLD, dtype=np.int8)
        state, _ = backend.apply_marks(marks)
    backend.step_interval(["ind_b"], "per_variable", 1)

    assert dict(backend.calls) == {"compute_error": 2, "apply_marks": 2, "step_interval": 1}


# --- On Python1DBackend: real numerics (T11) ---------------------------------------

SEQUENTIAL_SLOTS = (
    "error", "error_x_min", "error_x_max", "level", "level_x_min", "level_x_max", "resource", "round_progress"
)


def test_1d_first_round_is_eleven_features(counting):
    inner, state = make_1d()
    backend = counting(inner)
    cache = read_round(backend, state, ONE_D_NAMES, "scalar")
    cfg = config(indicator="zz_style", element_budget=30)
    obs = observation(cache, RoundState(round_index=1, n_rounds=3, committed=state.n_active), 0, cfg)

    assert dict(backend.calls) == {"compute_error": 1}
    assert state.ndim == 1 and len(feature_names(1)) == 11
    for key in (ACTOR_KEY, CRITIC_KEY):
        assert obs[key].shape == (11,) and obs[key].dtype == np.float32
        assert np.isfinite(obs[key]).all()
        assert within(obs[key], observation_bounds(1, cfg)[key])


@pytest.mark.parametrize("seed", [0, 1])
def test_1d_two_intervals_of_three_rounds_observed(seed, counting):
    """The builder on real numerics. At every position of every round, under
    dynamo and with the committed count at n_active, the eight slots of the
    sequential core's build_observation appear in the 11-vector bit for bit.
    The sequential state is the wrapped contract's own, read through the
    backend's private handle and so not through the counting wrapper."""
    rng = np.random.default_rng(seed)
    icase = int(rng.choice([1, 12, 16]))
    inner, state = make_1d(icase=icase)
    backend = counting(inner)
    names = feature_names(1)
    picked = [names.index(name) for name in SEQUENTIAL_SLOTS]
    family_slot = names.index("error_family_max")
    wall_slots = [names.index("wall_x_min"), names.index("wall_x_max")]
    budget = 30
    assert state.max_level == 3

    for _ in range(2):
        for round_index in range(1, state.max_level + 1):
            cache = read_round(backend, state, ONE_D_NAMES, "scalar")
            old = inner._contract.get_state()
            assert old.n_active == state.n_active
            round_state = RoundState(round_index=round_index, n_rounds=3, committed=state.n_active)

            for indicator in ONE_D_NAMES:
                errors = cache.errors[indicator][:, 0]
                e_inf = float(errors.max())
                dynamo = config("dynamo", indicator=indicator, element_budget=budget)
                relative = config("relative", indicator=indicator, element_budget=budget)
                bounds = feature_bounds(1, relative)
                for position in range(state.n_active):
                    new = actor_observation(cache, round_state, position, dynamo)
                    sequential = build_observation(
                        old, errors, position, alpha=ALPHA, element_budget=budget, round_number=round_index
                    )
                    assert np.array_equal(new[picked], sequential), f"{indicator}, position {position}"
                    assert new[wall_slots].tolist() == [0.0, 0.0]

                    sibling = int(old.sibling[position])
                    largest = max(errors[position], errors[sibling]) if sibling >= 0 else errors[position]
                    assert new[family_slot] == np.float32(float(normalized_error(float(largest), ALPHA, e_inf)))

                    scale_free = actor_observation(cache, round_state, position, relative)
                    assert np.isfinite(scale_free).all()
                    assert within(scale_free, bounds)

            marks = rng.integers(-1, 2, size=state.n_active).astype(np.int8)
            state, _ = backend.apply_marks(marks)
        backend.step_interval(ONE_D_NAMES[:1], "scalar", 1)

    assert dict(backend.calls) == {"compute_error": 6, "apply_marks": 6, "step_interval": 2}
