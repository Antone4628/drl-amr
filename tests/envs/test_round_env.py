"""Tests for the 2D environment, envs/round_env.py: one element's decision
per step, on the mock quad backend.

AGENT_2D_ROADMAP item 3.1, sitting 1, signals E1-E14. No RL library is
imported: the environment is driven by scripted and by random masked
actions. Ten groups:

  - construction: the spec, the spaces, what is refused (E1);
  - the reset: the first asked element, the seeded initial condition (E2,
    E13);
  - the instance of the explanation file's section 3.1.2, step by step
    (E3-E7);
  - the environment against a reference loop: the interval written as a
    plain loop over the core's functions, as tests/agent/test_round_nvar3.py
    walks it, given the same actions (E14);
  - the reward's delivery: the two lambdas (E5);
  - what the environment hands the backend and the core: the episode's
    values, the spec's values, the round count (E4);
  - the overshoot and the refused action (E8, E9);
  - a backend that fails (E10, E11);
  - the spec check at the first reset (E12);
  - what the module imports.

Run: pytest tests/envs/test_round_env.py -v
"""
from __future__ import annotations

import ast
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces

import envs.round_env as round_env
from agent.round_cache import read_round
from agent.round_commit import commit, overshoot, start_round
from agent.round_mask import ACTION_COARSEN, ACTION_HOLD, ACTION_REFINE, COARSEN_RULES, action_mask
from agent.round_observation import ACTOR_KEY, CRITIC_KEY, ObservationConfig, observation, observation_bounds
from agent.round_queue import next_undecided, round_order, round_thresholds, zone_distance
from agent.round_reward import global_reward, local_reward
from agent.round_spec import ObservationSpec
from backends.mock_quad.backend import MockQuadBackend
from contract.backend import BackendFailure
from contract.reports import IntervalReport
from envs.round_env import RoundEnv

C, H, R = ACTION_COARSEN, ACTION_HOLD, ACTION_REFINE
# The feature: off every centroid, so no two of the instance's errors tie.
IC = {"x0": 0.3, "y0": 0.6, "width": 0.2}
P_UR, P_OR, P_CR = 10.0, 5.0, 2.0
LAMBDA_LOCAL, LAMBDA_GLOBAL = 0.1, 1.0
BETA = 2
ALPHA = 0.1
# Literal values are the explanation file's, printed to six decimals.
TOL = 2e-6


# --- The backends, the configuration and the environment ---------------------------

def _two_cells(max_level: int = 2, nvar: int = 1) -> MockQuadBackend:
    """Cells A and B on [0,2]x[0,1], periodic in x and walled in y, the
    rotation off: position 0 is A and position 1 is B at level 0."""
    return MockQuadBackend(
        nx=2, ny=1, periodic=(True, False), lo=(0.0, 0.0), hi=(2.0, 1.0), max_level=max_level, nvar=nvar,
        dt_base=0.25, rotate_cells=False,
    )


def _grid(periodic: tuple[bool, bool], nvar: int = 1, max_level: int = 2) -> MockQuadBackend:
    """A 4x4 grid on the unit square, the rotation on."""
    return MockQuadBackend(
        nx=4, ny=4, periodic=periodic, max_level=max_level, nvar=nvar, dt_base=0.25, rotate_cells=True
    )


BACKENDS = {
    "two cells": _two_cells,
    "4x4 periodic": lambda nvar=1: _grid((True, True), nvar),
    "4x4 walled": lambda nvar=1: _grid((False, False), nvar),
    "4x4 mixed": lambda nvar=1: _grid((True, False), nvar),
}


def _config(
    indicator: str = "ind_b", reduction: str = "scalar", normalization: str = "relative", budget: int = 8
) -> ObservationConfig:
    return ObservationConfig(
        indicator=indicator, reduction=reduction, alpha=ALPHA, element_budget=budget, normalization=normalization,
        floor_decades=8.0,
    )


def _arguments(**changes) -> dict:
    """The instance's keyword arguments, with any of them changed."""
    arguments = dict(
        observation_config=_config(), beta=BETA, coarsen_rule="first_decider", max_level=2, n_intervals=2,
        amr_freq=2, initial_level=0, p_ur=P_UR, p_or=P_OR, p_cr=P_CR, lambda_local=LAMBDA_LOCAL,
        lambda_global=LAMBDA_GLOBAL,
    )
    arguments.update(changes)
    return arguments


def _env(backend=None, **changes) -> RoundEnv:
    return RoundEnv(_two_cells() if backend is None else backend, **_arguments(**changes))


# --- Two ways to run an episode: through the environment, and as a plain loop ------

def _listed(actions):
    """Actions read off a list, by step."""
    return lambda step, mask: actions[step]


def _drawn(seed: int):
    """Random masked actions fixed by the step's number alone, so the
    environment and the reference loop are given the same ones for as long
    as they agree."""

    def choose(step, mask):
        open_actions = np.flatnonzero(mask)
        return int(open_actions[int(np.random.default_rng([seed, step]).random() * len(open_actions))])

    return choose


def _holds(step, mask):
    return H


def _episode(env: RoundEnv, choose, ic=IC) -> list[dict]:
    """One episode through reset and step. One dict per step: the
    observation and the mask the action was chosen on, the action, and
    what step returned."""
    obs, _ = env.reset(options={"ic": ic})
    steps, terminated = [], False
    while not terminated:
        assert env.observation_space.contains(obs)
        mask = env.action_masks()
        action = choose(len(steps), mask)
        assert mask[action]
        next_obs, reward, terminated, truncated, info = env.step(action)
        assert truncated is False
        steps.append(
            dict(obs=obs, mask=mask, action=action, reward=reward, terminated=terminated, last=next_obs, **info)
        )
        obs = next_obs
        assert len(steps) < 5000
    return steps


def _reference(backend, config, rule, beta, n_intervals, choose, *, ic=IC, amr_freq=2, initial_level=0, cadence=1):
    """The episode as a plain loop over the core's functions, the way
    tests/agent/test_round_nvar3.py walks an interval: the pair from each
    interval's first cache and held; per round one read, the order and the
    distances under that pair, the walk, apply_marks; after the last round
    one step_interval and the global reward. The same dicts as _episode."""
    state = backend.reset(ic, amr_freq, initial_level)
    n_rounds = state.max_level
    steps = []
    for interval in range(1, n_intervals + 1):
        pair = None
        for round_index in range(1, n_rounds + 1):
            cache = read_round(backend, state, [config.indicator], config.reduction)
            if pair is None:
                pair = round_thresholds(cache, config, beta)
            order = round_order(cache, config, *pair)
            distance = zone_distance(cache.errors[config.indicator], config, *pair)
            round_state = start_round(state, round_index, n_rounds)
            position, cursor, skipped = next_undecided(state, round_state, order, 0)
            while position is not None:
                obs = observation(cache, round_state, position, config)
                mask = action_mask(state, round_state, position, rule)
                action = choose(len(steps), mask)
                r_local = local_reward(state, distance, position, action, rule, p_ur=P_UR, p_or=P_OR, p_cr=P_CR)
                round_state = commit(state, round_state, position, action, rule)
                steps.append(
                    dict(
                        obs=obs, mask=mask, action=action, position=position, round_index=round_index,
                        interval=interval, r_local=r_local, r_global=0.0, transition="element",
                    )
                )
                position, cursor, passed = next_undecided(state, round_state, order, cursor)
                skipped += passed
            new_state, _ = backend.apply_marks(round_state.decisions)
            steps[-1].update(
                transition="round", overshoot=overshoot(round_state, new_state), skipped=skipped,
                n_active=new_state.n_active,
            )
            state = new_state
        report = backend.step_interval([config.indicator], config.reduction, cadence)
        r_global = global_reward(
            report.acc[config.indicator], state.level, state.max_level, config, *pair, p_ur=P_UR, p_or=P_OR
        )
        steps[-1].update(r_global=r_global, transition="done" if interval == n_intervals else "interval")
    for step in steps:
        step["reward"] = LAMBDA_LOCAL * step["r_local"] + LAMBDA_GLOBAL * step["r_global"]
    return steps


# --- Construction (E1) -------------------------------------------------------------

def test_it_is_a_gymnasium_environment():
    assert isinstance(_env(), gym.Env)


def test_the_spec_is_built_from_the_arguments_and_the_backend():
    config = _config()
    env = _env(observation_config=config)
    spec = env.observation_spec
    assert isinstance(spec, ObservationSpec)
    assert spec.observation is config
    assert (spec.beta, spec.coarsen_rule, spec.ndim, spec.max_level, spec.n_rounds, spec.n_comp) == (
        BETA, "first_decider", 2, 2, 2, 1,
    )


@pytest.mark.parametrize("max_level", [1, 2, 3])
def test_rounds_per_interval_equal_max_level(max_level):
    spec = _env(_two_cells(max_level), max_level=max_level).observation_spec
    assert spec.n_rounds == spec.max_level == max_level


@pytest.mark.parametrize(
    "nvar, reduction, n_comp", [(1, "scalar", 1), (3, "scalar", 1), (3, "per_variable", 3), (1, "per_variable", 1)]
)
def test_the_column_count_follows_the_reduction(nvar, reduction, n_comp):
    env = _env(_two_cells(nvar=nvar), observation_config=_config(reduction=reduction))
    assert env.observation_spec.n_comp == n_comp


@pytest.mark.parametrize("normalization", ["relative", "dynamo"])
def test_the_space_is_the_bounds_the_core_gives(normalization):
    config = _config(normalization=normalization)
    env = _env(observation_config=config)
    space = env.observation_space
    assert isinstance(space, spaces.Dict)
    assert list(space.spaces) == [ACTOR_KEY, CRITIC_KEY]
    for key, (low, high) in observation_bounds(2, config).items():
        box = space.spaces[key]
        assert isinstance(box, spaces.Box)
        assert box.shape == (17,) and box.dtype == np.float32
        assert np.array_equal(box.low, low) and np.array_equal(box.high, high)


def test_the_instances_bounds():
    box = _env().observation_space.spaces[ACTOR_KEY]
    assert box.low.min() == -8.0 and box.high.max() == np.inf
    assert box.high[0] == pytest.approx(1.0)


def test_three_actions():
    space = _env().action_space
    assert isinstance(space, spaces.Discrete) and space.n == 3


def test_construction_only_describes_the_backend(counting):
    backend = counting(_two_cells())
    _env(backend)
    assert dict(backend.calls) == {"describe": 1}


REFUSED = [
    ("n_intervals", 0, "n_intervals: 0, expected an integer of at least 1"),
    ("n_intervals", 2.0, "n_intervals: 2.0, expected an integer of at least 1"),
    ("amr_freq", 1.5, "amr_freq: 1.5, expected an integer of at least 1"),
    ("amr_freq", 0, "amr_freq: 0, expected an integer of at least 1"),
    ("initial_level", -1, "initial_level: -1, expected an integer of at least 0"),
    ("cadence", True, "cadence: True, expected an integer of at least 1"),
    ("cadence", 0, "cadence: 0, expected an integer of at least 1"),
    ("p_ur", -1.0, "p_ur: -1.0, expected a finite number, not negative"),
    ("p_or", "5", "p_or: '5', expected a finite number, not negative"),
    ("p_cr", float("inf"), "p_cr: inf, expected a finite number, not negative"),
    ("lambda_local", -0.1, "lambda_local: -0.1, expected a finite number, not negative"),
    ("lambda_global", float("nan"), "lambda_global: nan, expected a finite number, not negative"),
    ("lambda_global", False, "lambda_global: False, expected a finite number, not negative"),
    ("max_level", 0, "max_level: 0, expected an integer of at least 1"),
    ("beta", 1, "beta: 1, expected a finite number above 1"),
    ("coarsen_rule", "first", "coarsen_rule: 'first', expected one of ['first_decider', 'any_coarsen', 'unanimous']"),
    ("observation_config", None, "observation_config: None, expected an ObservationConfig"),
    ("observation_config", "ind_b", "observation_config: 'ind_b', expected an ObservationConfig"),
]


@pytest.mark.parametrize("name, value, message", REFUSED, ids=[f"{name}={value!r}" for name, value, _ in REFUSED])
def test_a_bad_argument_is_refused_by_name(name, value, message):
    with pytest.raises(ValueError) as caught:
        _env(**{name: value})
    assert str(caught.value) == message


def test_the_arguments_after_the_backend_are_keyword_only():
    with pytest.raises(TypeError):
        RoundEnv(_two_cells(), _config(), BETA, "first_decider", 2, 2, 2)


@pytest.mark.parametrize("zero", ["p_ur", "p_or", "p_cr", "lambda_local", "lambda_global"])
def test_a_weight_of_zero_is_accepted(zero):
    assert len(_episode(_env(**{zero: 0}), _holds)) == 8


def test_nothing_runs_before_a_reset():
    env = _env()
    with pytest.raises(RuntimeError, match=r"^action_masks: no episode is running, call reset$"):
        env.action_masks()
    with pytest.raises(RuntimeError, match=r"^step: no episode is running, call reset$"):
        env.step(H)


# --- The reset (E2, E13) -----------------------------------------------------------

def test_the_instances_reset():
    env = _env()
    obs, info = env.reset(options={"ic": IC})
    assert set(info) == {"ic", "n_active", "e_max", "e_min"}
    assert info["ic"] == IC and info["n_active"] == 2
    assert info["e_max"] == pytest.approx([1.338154], abs=TOL)
    assert info["e_min"] == pytest.approx([0.133815], abs=TOL)
    assert isinstance(info["e_max"], list) and isinstance(info["e_min"], list)
    assert env.action_masks().tolist() == [False, True, True]
    expected = [-2.2572, 1.0, 1.0, -2.2572, -2.2572, -2.2572, 0, 0, 0, 0, 0, 0, 0, 1.0, 1.0, 0.25, 0.5]
    assert obs[ACTOR_KEY] == pytest.approx(expected, abs=1e-4)


def test_the_reset_returns_the_dict_the_core_builds():
    env = _env()
    obs, _ = env.reset(options={"ic": IC})
    assert list(obs) == [ACTOR_KEY, CRITIC_KEY]
    assert all(obs[key].dtype == np.float32 and obs[key].shape == (17,) for key in obs)
    assert np.array_equal(obs[ACTOR_KEY], obs[CRITIC_KEY]) and obs[ACTOR_KEY] is not obs[CRITIC_KEY]
    assert env.observation_space.contains(obs)
    backend = _two_cells()
    state = backend.reset(IC, 2, 0)
    cache = read_round(backend, state, ["ind_b"], "scalar")
    order = round_order(cache, _config(), *round_thresholds(cache, _config(), BETA))
    theirs = observation(cache, start_round(state, 1, 2), int(order[0]), _config())
    assert np.array_equal(obs[ACTOR_KEY], theirs[ACTOR_KEY])


def test_the_reset_costs_one_reset_and_one_read(counting):
    backend = counting(_two_cells())
    env = _env(backend)
    env.reset(options={"ic": IC})
    assert dict(backend.calls) == {"describe": 1, "reset": 1, "compute_error": 1}


def test_the_mask_is_a_new_boolean_triple():
    env = _env()
    env.reset(options={"ic": IC})
    mask = env.action_masks()
    assert mask.shape == (3,) and mask.dtype == bool and mask[H]
    mask[:] = False
    assert env.action_masks()[H]


def test_the_same_seed_draws_the_same_initial_condition():
    first = _env().reset(seed=7)[1]["ic"]
    assert _env().reset(seed=7)[1]["ic"] == first
    assert _env().reset(seed=8)[1]["ic"] != first


def test_a_second_episode_draws_again_and_the_run_is_reproducible():
    one, other = _env(), _env()
    draws = [one.reset(seed=3)[1]["ic"], one.reset()[1]["ic"], one.reset()[1]["ic"]]
    assert draws[0] != draws[1] != draws[2]
    assert [other.reset(seed=3)[1]["ic"], other.reset()[1]["ic"], other.reset()[1]["ic"]] == draws


@pytest.mark.parametrize("seed", range(20))
def test_a_drawn_initial_condition_is_inside_the_backends_bounds(seed):
    backend = _grid((True, True))
    ic = _env(backend, initial_level=1).reset(seed=seed)[1]["ic"]
    bounds = backend.describe().ic_params
    assert list(ic) == list(bounds)
    assert all(isinstance(ic[name], float) and low <= ic[name] <= high for name, (low, high) in bounds.items())


def test_a_forced_initial_condition_draws_nothing():
    env = _env()
    env.reset(seed=5, options={"ic": IC})
    after_forced = env.reset()[1]["ic"]
    fresh = _env()
    assert fresh.reset(seed=5)[1]["ic"] == after_forced


def test_options_without_an_initial_condition_draw_one():
    assert _env().reset(seed=7, options={})[1]["ic"] == _env().reset(seed=7)[1]["ic"]


def test_a_bad_initial_condition_is_the_backends_to_refuse_and_leaves_no_episode():
    env = _env()
    env.reset(options={"ic": IC})
    with pytest.raises(ValueError, match=r"^ic: unknown names \[\], missing names \['width'\]$"):
        env.reset(options={"ic": {"x0": 0.3, "y0": 0.6}})
    with pytest.raises(RuntimeError):
        env.step(H)


def test_the_episode_values_reach_the_backends_reset():
    seen = {}

    class Recording:
        def __init__(self, inner):
            self.inner = inner

        def __getattr__(self, name):
            return getattr(self.inner, name)

        def reset(self, ic, amr_freq, initial_level=1):
            seen.update(ic=dict(ic), amr_freq=amr_freq, initial_level=initial_level)
            return self.inner.reset(ic, amr_freq, initial_level)

    env = _env(Recording(_grid((True, True))), amr_freq=3, initial_level=1)
    _, info = env.reset(options={"ic": IC})
    assert seen == {"ic": IC, "amr_freq": 3, "initial_level": 1}
    assert info["n_active"] == 64


def test_the_default_initial_level_is_one():
    assert RoundEnv(_grid((True, True)), **{k: v for k, v in _arguments().items() if k != "initial_level"}).reset(
        options={"ic": IC}
    )[1]["n_active"] == 64


def test_a_reset_in_the_middle_of_an_episode_starts_over():
    env = _env()
    first = _episode(env, _listed(SCRIPT))
    env.reset(options={"ic": IC})
    for action in SCRIPT[:5]:
        env.step(action)
    again = _episode(env, _listed(SCRIPT))
    assert [s["reward"] for s in again] == [s["reward"] for s in first]
    assert [s["position"] for s in again] == [s["position"] for s in first]


# --- The instance, step by step (E3-E7) --------------------------------------------

# Section 3.1.2's episode: refine A and hold B; hold everything; coarsen
# A's family from its first-asked child and refine B; hold everything.
SCRIPT = [H, R, H, H, H, H, H, C, R, H, H, H, H, H]
# (interval, round, position, mask, r_local, r_global, reward, transition)
TABLE = [
    (1, 1, 1, [0, 1, 1], 0.0, 0.0, 0.0, "element"),
    (1, 1, 0, [0, 1, 1], 0.0, 0.0, 0.0, "round"),
    (1, 2, 4, [0, 1, 1], 0.0, 0.0, 0.0, "element"),
    (1, 2, 2, [1, 1, 1], 0.0, 0.0, 0.0, "element"),
    (1, 2, 1, [0, 1, 1], 0.0, 0.0, 0.0, "element"),
    (1, 2, 0, [0, 1, 1], 0.0, 0.0, 0.0, "element"),
    (1, 2, 3, [0, 1, 1], 0.0, -2.134628, -2.134628, "interval"),
    (2, 1, 3, [1, 1, 1], -10.0, 0.0, -1.0, "element"),
    (2, 1, 4, [0, 1, 1], 0.0, 0.0, 0.0, "round"),
    (2, 2, 2, [1, 1, 1], 0.0, 0.0, 0.0, "element"),
    (2, 2, 4, [0, 1, 1], 0.0, 0.0, 0.0, "element"),
    (2, 2, 0, [0, 1, 1], 0.0, 0.0, 0.0, "element"),
    (2, 2, 1, [0, 1, 1], 0.0, 0.0, 0.0, "element"),
    (2, 2, 3, [0, 1, 1], 0.0, -5.304082, -5.304082, "done"),
]


@pytest.fixture(scope="module")
def instance():
    return _episode(_env(), _listed(SCRIPT))


def test_the_instance_takes_fourteen_steps(instance):
    assert len(instance) == 14


@pytest.mark.parametrize("number", range(14))
def test_the_instances_step(instance, number):
    interval, round_index, position, mask, r_local, r_global, reward, transition = TABLE[number]
    step = instance[number]
    assert (step["interval"], step["round_index"], step["position"]) == (interval, round_index, position)
    assert step["mask"].astype(int).tolist() == mask
    assert step["r_local"] == pytest.approx(r_local, abs=TOL)
    assert step["r_global"] == pytest.approx(r_global, abs=TOL)
    assert step["reward"] == pytest.approx(reward, abs=TOL)
    assert step["transition"] == transition


def test_the_instances_return(instance):
    assert sum(step["reward"] for step in instance) == pytest.approx(-8.438710, abs=TOL)


def test_the_instances_types(instance):
    for step in instance:
        assert type(step["reward"]) is float and type(step["terminated"]) is bool
        assert type(step["r_local"]) is float and type(step["r_global"]) is float
        assert type(step["position"]) is int


def test_only_the_last_step_terminates(instance):
    assert [step["terminated"] for step in instance] == [False] * 13 + [True]


def test_the_last_observation_is_zeros_in_the_space(instance):
    last = instance[-1]["last"]
    assert list(last) == [ACTOR_KEY, CRITIC_KEY]
    assert all(last[key].dtype == np.float32 and not last[key].any() for key in last)
    assert last[ACTOR_KEY] is not last[CRITIC_KEY]
    assert _env().observation_space.contains(last)


def test_every_other_observation_is_the_next_steps(instance):
    for before, after in zip(instance[:-1], instance[1:], strict=True):
        assert before["last"] is after["obs"]
        assert after["obs"][ACTOR_KEY].any()


def test_the_global_reward_arrives_at_an_intervals_last_step_only(instance):
    paid = [number + 1 for number, step in enumerate(instance) if step["r_global"] != 0.0]
    assert paid == [7, 14]
    assert all(step["r_global"] <= 0.0 for step in instance)


def test_a_rounds_end_reports_the_round(instance):
    ends = {
        number + 1: (step["transition"], step["overshoot"], step["skipped"], step["n_active"])
        for number, step in enumerate(instance)
        if "overshoot" in step
    }
    assert ends == {2: ("round", 0, 0, 5), 7: ("interval", 0, 0, 5), 9: ("round", 0, 3, 5), 14: ("done", 0, 0, 5)}
    assert all(("skipped" in step) == ("overshoot" in step) == ("n_active" in step) for step in instance)
    assert all(step["transition"] == "element" for step in instance if "overshoot" not in step)


def test_a_step_reports_these_keys(instance):
    always = {"position", "round_index", "interval", "r_local", "r_global", "transition"}
    for step in instance:
        info = set(step) - {"obs", "mask", "action", "reward", "terminated", "last"}
        assert info == always | ({"overshoot", "skipped", "n_active"} if "overshoot" in step else set())


def test_the_resource_feature_moves_within_a_round_and_the_progress_between(instance):
    """Features 15 and 16 are the resource and the round progress."""
    seen = [[float(step["obs"][ACTOR_KEY][15]), float(step["obs"][ACTOR_KEY][16])] for step in instance]
    assert seen[0] == [0.25, 0.5] and seen[1] == [0.25, 0.5]
    assert seen[2] == [0.625, 1.0]
    # Step 8 coarsens a family of four: the count falls from 5 to 2.
    assert seen[7] == [0.625, 0.5] and seen[8] == [0.25, 0.5]


def test_the_second_interval_takes_its_own_thresholds(instance):
    """Step 8 coarsens a family whose worst child has an error of 4.572598.
    Under interval 2's own pair that child is the worst element, 1 decade
    above the zone, and the coarsen costs 10. Under interval 1's pair,
    held too long, it would lie 0.533657 above and cost 5.336572."""
    assert instance[7]["r_local"] == pytest.approx(-10.0, abs=TOL)

def test_the_instance_costs_2r_plus_1_calls_per_interval(counting):
    backend = counting(_two_cells())
    _episode(_env(backend), _listed(SCRIPT))
    assert dict(backend.calls) == {"describe": 1, "reset": 1, "compute_error": 4, "apply_marks": 4, "step_interval": 2}


def test_all_holds_on_the_instance():
    steps = _episode(_env(), _holds)
    assert len(steps) == 2 * 2 * 2
    assert [step["r_global"] for step in steps if step["r_global"]] == pytest.approx([-6.017878, -11.357170], abs=TOL)
    assert all(step["r_local"] == 0.0 for step in steps)


def test_a_step_after_the_end_is_refused_and_a_reset_begins_again():
    env = _env()
    first = _episode(env, _listed(SCRIPT))
    with pytest.raises(RuntimeError, match=r"^step: no episode is running, call reset$"):
        env.step(H)
    with pytest.raises(RuntimeError, match=r"^action_masks: no episode is running, call reset$"):
        env.action_masks()
    again = _episode(env, _listed(SCRIPT))
    assert [step["reward"] for step in again] == [step["reward"] for step in first]


# --- The environment against the reference loop (E14) ------------------------------

def _same(ours: list[dict], theirs: list[dict]) -> None:
    """The two runs agree exactly: the same functions on the same arrays."""
    assert len(ours) == len(theirs)
    for one, other in zip(ours, theirs, strict=True):
        for key in other:
            if key == "obs":
                assert all(np.array_equal(one["obs"][k], other["obs"][k]) for k in (ACTOR_KEY, CRITIC_KEY))
            elif key == "mask":
                assert np.array_equal(one["mask"], other["mask"])
            else:
                assert one[key] == other[key], key
        assert set(one) - {"reward", "terminated", "last"} == set(other) - {"reward"}


def test_the_instance_is_the_reference_loop(instance):
    _same(instance, _reference(_two_cells(), _config(), "first_decider", BETA, 2, _listed(SCRIPT)))


CELLS = [
    (1, "scalar", "relative"), (1, "scalar", "dynamo"), (3, "scalar", "relative"), (3, "per_variable", "relative"),
]
CELL_IDS = ["one variable", "one variable, dynamo", "three, scalar", "three, per_variable"]


@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("indicator", ["ind_a", "ind_b"])
@pytest.mark.parametrize("nvar, reduction, normalization", CELLS, ids=CELL_IDS)
@pytest.mark.parametrize("rule", COARSEN_RULES)
@pytest.mark.parametrize("mesh", list(BACKENDS))
def test_random_masked_episodes_are_the_reference_loop(mesh, rule, nvar, reduction, normalization, indicator, seed):
    """Whatever the mesh, the rule, the reduction and the indicator, an
    episode through reset and step is the plain loop, step for step: the
    same positions asked, the same observations and masks, the same
    rewards at the same steps, the same counts at every round's end."""
    config = _config(indicator, reduction, normalization, budget=64)
    arguments = dict(observation_config=config, coarsen_rule=rule, initial_level=1)
    env = _env(BACKENDS[mesh](nvar=nvar), **arguments)
    ours = _episode(env, _drawn(seed))
    theirs = _reference(BACKENDS[mesh](nvar=nvar), config, rule, BETA, 2, _drawn(seed), initial_level=1)
    _same(ours, theirs)
    assert ours[-1]["terminated"] and all(np.isfinite(step["reward"]) for step in ours)


@pytest.mark.parametrize("rule", COARSEN_RULES)
@pytest.mark.parametrize("mesh", list(BACKENDS))
def test_asked_and_skipped_add_up_to_the_elements_of_every_round(mesh, rule):
    env = _env(BACKENDS[mesh](), observation_config=_config(budget=64), coarsen_rule=rule, initial_level=1)
    steps = _episode(env, _drawn(11))
    n_before, asked, rounds = env.reset(options={"ic": IC})[1]["n_active"], 0, 0
    for step in steps:
        asked += 1
        if "overshoot" in step:
            assert asked + step["skipped"] == n_before
            assert step["overshoot"] >= 0 and step["overshoot"] % 3 == 0
            n_before, asked, rounds = step["n_active"], 0, rounds + 1
    assert rounds == 2 * 2


def test_the_random_episodes_are_not_all_holds():
    steps = _episode(_env(_grid((True, True)), observation_config=_config(budget=64), initial_level=1), _drawn(0))
    actions = {step["action"] for step in steps}
    assert actions == {C, H, R}
    assert any(step["r_local"] > 0 for step in steps) and any(step["r_local"] < 0 for step in steps)
    assert any(step.get("skipped", 0) > 0 for step in steps) and any(step.get("overshoot", 0) > 0 for step in steps)


@pytest.mark.parametrize("max_level", [1, 2, 3])
def test_an_interval_costs_2r_plus_1_calls(counting, max_level):
    backend = counting(_two_cells(max_level))
    steps = _episode(_env(backend, max_level=max_level, n_intervals=3), _drawn(2))
    assert dict(backend.calls) == {
        "describe": 1, "reset": 1, "compute_error": 3 * max_level, "apply_marks": 3 * max_level, "step_interval": 3,
    }
    assert max(step["round_index"] for step in steps) == max_level
    assert sorted({step["interval"] for step in steps}) == [1, 2, 3]
    assert [step["transition"] for step in steps if step["transition"] in ("interval", "done")] == [
        "interval", "interval", "done",
    ]


@pytest.mark.parametrize("n_intervals", [1, 2, 4])
def test_an_episode_is_n_intervals_long(n_intervals):
    steps = _episode(_env(n_intervals=n_intervals), _holds)
    assert len(steps) == n_intervals * 2 * 2
    assert sum(step["r_global"] != 0.0 for step in steps) == n_intervals


# --- The reward's delivery (E5) ----------------------------------------------------

@pytest.mark.parametrize("lambda_local, lambda_global", [(0.1, 1.0), (1.0, 1.0), (0.0, 1.0), (0.1, 0.0), (0.5, 2.0)])
def test_every_reward_is_the_two_parts_scaled(instance, lambda_local, lambda_global):
    steps = _episode(_env(lambda_local=lambda_local, lambda_global=lambda_global), _listed(SCRIPT))
    for step, base in zip(steps, instance, strict=True):
        assert step["r_local"] == base["r_local"] and step["r_global"] == base["r_global"]
        assert step["reward"] == lambda_local * step["r_local"] + lambda_global * step["r_global"]
    assert steps[7]["reward"] == pytest.approx(-10.0 * lambda_local, abs=TOL)
    assert steps[13]["reward"] == pytest.approx(-5.304082 * lambda_global, abs=TOL)


def test_the_weights_reach_both_rewards(instance):
    steps = _episode(_env(p_ur=20.0, p_or=0.0, p_cr=7.0), _listed(SCRIPT))
    assert steps[7]["r_local"] == pytest.approx(2 * instance[7]["r_local"])
    assert steps[6]["r_global"] != instance[6]["r_global"]
    doubled = _episode(_env(p_ur=20.0, p_or=10.0), _listed(SCRIPT))
    assert doubled[6]["r_global"] == pytest.approx(2 * instance[6]["r_global"])


@pytest.mark.parametrize("rule, r_local", [("first_decider", 0.0), ("any_coarsen", 0.0), ("unanimous", 6.857362)])
def test_a_coarsen_is_paid_at_once_on_the_distance_the_rule_names(rule, r_local):
    """From level 1 the first element asked is a child of B, 3.428681
    decades below the zone, and one of its siblings is neutral. Its coarsen
    is paid p_cr on its own distance under unanimous, and on the family's
    largest, 0, where it decides for the family."""
    env = _env(coarsen_rule=rule, initial_level=1)
    env.reset(options={"ic": IC})
    assert env.action_masks()[C]
    _, reward, _, _, info = env.step(C)
    assert info["r_local"] == pytest.approx(r_local, abs=TOL)
    assert reward == LAMBDA_LOCAL * info["r_local"]


# --- What the core is handed: the spec's values ------------------------------------

@pytest.mark.parametrize("beta", [1.5, 2, 3])
def test_beta_sets_the_lower_threshold(beta):
    info = _env(beta=beta).reset(options={"ic": IC})[1]
    assert info["e_min"][0] == pytest.approx(ALPHA**beta * info["e_max"][0] / ALPHA)


@pytest.mark.parametrize(
    "rule, expected",
    [("first_decider", [False, True, True]), ("any_coarsen", [True, True, True]), ("unanimous", [True, True, True])],
)
def test_the_coarsen_rule_reaches_the_mask(rule, expected):
    """After one sibling holds, coarsen is closed to the next under
    first_decider and open under the other two."""
    env = _env(coarsen_rule=rule, initial_level=1)
    env.reset(options={"ic": IC})
    assert env.action_masks()[C]
    env.step(H)
    assert env.action_masks().tolist() == expected


@pytest.mark.parametrize(
    "rule, skipped, n_after", [("first_decider", 3, 5), ("any_coarsen", 3, 5), ("unanimous", 0, 8)]
)
def test_the_coarsen_rule_reaches_the_commit(rule, skipped, n_after):
    """One coarsen, then holds: a family commit skips three siblings and
    merges four elements into one; one vote under unanimous does neither."""
    env = _env(coarsen_rule=rule, initial_level=1, n_intervals=1)
    env.reset(options={"ic": IC})
    info = env.step(C)[4]
    while "skipped" not in info:
        info = env.step(H)[4]
    assert (info["skipped"], info["n_active"], info["overshoot"]) == (skipped, n_after, 0)


def test_the_cadence_reaches_step_interval():
    seen = []

    class Recording:
        def __init__(self, inner):
            self.inner = inner

        def __getattr__(self, name):
            return getattr(self.inner, name)

        def step_interval(self, accumulate, reduction, cadence, aux=()):
            seen.append((list(accumulate), reduction, cadence, tuple(aux)))
            return self.inner.step_interval(accumulate, reduction, cadence, aux)

    _episode(_env(Recording(_two_cells()), cadence=3), _holds)
    assert seen == [(["ind_b"], "scalar", 3, ())] * 2


def test_the_indicator_and_the_reduction_reach_compute_error():
    seen = []

    class Recording:
        def __init__(self, inner):
            self.inner = inner

        def __getattr__(self, name):
            return getattr(self.inner, name)

        def compute_error(self, indicators, reduction):
            seen.append((list(indicators), reduction))
            return self.inner.compute_error(indicators, reduction)

    config = _config("ind_a", "per_variable")
    _episode(_env(Recording(_two_cells(nvar=3)), observation_config=config), _holds)
    assert seen == [(["ind_a"], "per_variable")] * 4


def test_the_budget_reaches_the_resource_feature():
    for budget in (8, 64):
        obs, _ = _env(observation_config=_config(budget=budget)).reset(options={"ic": IC})
        assert obs[ACTOR_KEY][15] == np.float32(2 / budget)


# --- The overshoot and the refused action (E8, E9) ---------------------------------

def test_a_refine_that_balance_spreads_is_reported_as_overshoot():
    """Refine A, then its child beside B: B must refine. The count stood at
    8 and the mesh is 11."""
    env = _env(n_intervals=1)
    steps = _episode(env, _listed([H, R, H, H, H, H, R]))
    assert [step["position"] for step in steps][-1] == 3
    assert (steps[1]["overshoot"], steps[1]["n_active"]) == (0, 5)
    assert (steps[-1]["overshoot"], steps[-1]["n_active"], steps[-1]["transition"]) == (3, 11, "done")


def test_a_masked_action_is_refused_and_nothing_moves(counting, instance):
    backend = counting(_two_cells())
    env = _env(backend)
    obs, _ = env.reset(options={"ic": IC})
    calls = dict(backend.calls)
    with pytest.raises(ValueError, match=r"^action: 0 is masked for position 1 under 'first_decider'$"):
        env.step(C)
    assert dict(backend.calls) == calls
    assert env.action_masks().tolist() == [False, True, True]
    steps, terminated = [], False
    while not terminated:
        _, reward, terminated, _, info = env.step(SCRIPT[len(steps)])
        steps.append((info["position"], reward))
    assert steps == [(step["position"], step["reward"]) for step in instance]


@pytest.mark.parametrize("action", [3, -1])
def test_a_number_that_is_no_action_is_refused(action):
    env = _env()
    env.reset(options={"ic": IC})
    with pytest.raises(ValueError, match=r"^action: -?\d, expected 0 \(coarsen\), 1 \(hold\) or 2 \(refine\)$"):
        env.step(action)
    assert env.step(H)[4]["position"] == 1


def test_a_numpy_integer_is_an_action():
    env = _env()
    env.reset(options={"ic": IC})
    assert env.step(np.int64(H))[4]["transition"] == "element"
    assert env.step(np.array(R))[4]["transition"] == "round"


# --- A backend that fails (E10, E11) -----------------------------------------------

# (verb, which call fails, the step the episode ends at): section 3.1.2's table.
FAILURES = [
    ("compute_error", 2, 2), ("compute_error", 3, 7), ("step_interval", 1, 7), ("apply_marks", 3, 9),
    ("step_interval", 2, 14),
]
FAILURE_IDS = [f"{verb} call {at}" for verb, at, _ in FAILURES]


@pytest.mark.parametrize("verb, at, last", FAILURES, ids=FAILURE_IDS)
def test_a_failure_ends_the_episode(failing, instance, verb, at, last):
    env = _env(failing(_two_cells(), verb, at))
    obs, _ = env.reset(options={"ic": IC})
    for number in range(last - 1):
        obs, reward, terminated, truncated, info = env.step(SCRIPT[number])
        assert not terminated and info["transition"] == instance[number]["transition"]
        assert reward == instance[number]["reward"]
    obs, reward, terminated, truncated, info = env.step(SCRIPT[last - 1])
    assert terminated is True and truncated is False
    assert info["transition"] == "failure"
    assert info["backend_failure"] == f"{verb}: failed on demand at call {at}"
    assert info["r_global"] == 0.0 and reward == LAMBDA_LOCAL * info["r_local"]
    assert all(not obs[key].any() for key in (ACTOR_KEY, CRITIC_KEY)) and env.observation_space.contains(obs)


@pytest.mark.parametrize("verb, at, last", FAILURES, ids=FAILURE_IDS)
def test_after_a_failure_only_a_reset_runs(failing, instance, verb, at, last):
    env = _env(failing(_two_cells(), verb, at))
    env.reset(options={"ic": IC})
    for number in range(last):
        env.step(SCRIPT[number])
    with pytest.raises(RuntimeError, match=r"^step: no episode is running, call reset$"):
        env.step(H)
    with pytest.raises(RuntimeError, match=r"^action_masks: no episode is running, call reset$"):
        env.action_masks()
    again = _episode(env, _listed(SCRIPT))
    assert [step["reward"] for step in again] == [step["reward"] for step in instance]


def test_a_failure_keeps_the_steps_own_local_reward(failing):
    """With a neutral zone a hundredth as wide, no element of the level-1
    mesh is neutral, and under unanimous every coarsen vote is paid on its
    own distance: the round's last step carries a local reward. When the
    apply that follows it fails, that step still returns it, and nothing
    else."""
    backend = failing(_two_cells(), "apply_marks", 1)
    env = _env(backend, beta=1.01, coarsen_rule="unanimous", initial_level=1, n_intervals=1)
    env.reset(options={"ic": IC})
    rewards, terminated = [], False
    while not terminated:
        assert env.action_masks()[C]
        _, reward, terminated, _, info = env.step(C)
        rewards.append((reward, info["r_local"]))
    assert len(rewards) == 8 and info["transition"] == "failure"
    assert all(r_local != 0.0 and reward == LAMBDA_LOCAL * r_local for reward, r_local in rewards)


def test_a_failure_that_never_comes_changes_nothing(failing, instance):
    steps = _episode(_env(failing(_two_cells(), "step_interval", 3)), _listed(SCRIPT))
    assert [step["reward"] for step in steps] == [step["reward"] for step in instance]
    assert steps[-1]["transition"] == "done" and "backend_failure" not in steps[-1]


@pytest.mark.parametrize("verb", ["reset", "compute_error"])
def test_a_failure_inside_reset_is_raised(failing, verb):
    env = _env(failing(_two_cells(), verb, 1))
    with pytest.raises(BackendFailure, match=rf"^{verb}: failed on demand at call 1$"):
        env.reset(options={"ic": IC})
    with pytest.raises(RuntimeError):
        env.step(H)
    assert env.reset(options={"ic": IC})[1]["n_active"] == 2


class _NotFinite:
    """Forward every verb, but put a NaN in one array of one call."""

    def __init__(self, inner, verb, at):
        self.inner, self.verb, self.at, self.seen = inner, verb, at, 0

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def compute_error(self, indicators, reduction):
        errors = self.inner.compute_error(indicators, reduction)
        if self.verb == "compute_error":
            self.seen += 1
            if self.seen == self.at:
                errors[indicators[0]][0, 0] = np.nan
        return errors

    def step_interval(self, accumulate, reduction, cadence, aux=()):
        report = self.inner.step_interval(accumulate, reduction, cadence, aux)
        if self.verb == "step_interval":
            self.seen += 1
            if self.seen == self.at:
                acc = {name: values.copy() for name, values in report.acc.items()}
                acc[accumulate[0]][0, 0] = np.inf
                return IntervalReport(time=report.time, n_substeps=report.n_substeps, acc=acc, mass=report.mass)
        return report


@pytest.mark.parametrize("verb, at, last, message", [
    ("compute_error", 2, 2, "errors['ind_b']: not finite"),
    ("step_interval", 1, 7, "errors: not finite"),
])
def test_an_array_that_is_not_finite_ends_the_episode(verb, at, last, message):
    env = _env(_NotFinite(_two_cells(), verb, at))
    env.reset(options={"ic": IC})
    for number in range(last - 1):
        assert not env.step(SCRIPT[number])[2]
    _, _, terminated, truncated, info = env.step(SCRIPT[last - 1])
    assert terminated and not truncated
    assert (info["transition"], info["backend_failure"]) == ("failure", message)


def test_only_a_backend_failure_is_caught():
    class Broken:
        def __init__(self, inner):
            self.inner = inner

        def __getattr__(self, name):
            return getattr(self.inner, name)

        def apply_marks(self, marks):
            raise KeyError("not a backend failure")

    env = _env(Broken(_two_cells()))
    env.reset(options={"ic": IC})
    env.step(H)
    with pytest.raises(KeyError):
        env.step(R)


# --- The spec check (E12) ----------------------------------------------------------

def test_a_round_the_spec_does_not_describe_stops_the_first_reset():
    env = _env(max_level=3)
    for _ in range(2):
        with pytest.raises(ValueError, match=r"^spec: max_level: the spec says 3, the round has 2$"):
            env.reset(options={"ic": IC})
        with pytest.raises(RuntimeError):
            env.action_masks()


def test_every_finding_is_in_the_message(monkeypatch):
    monkeypatch.setattr(round_env, "spec_violations", lambda spec, cache, round_state: ["one: a", "two: b"])
    with pytest.raises(ValueError, match=r"^spec: one: a; two: b$"):
        _env().reset(options={"ic": IC})


def test_the_spec_is_checked_once_per_environment(monkeypatch):
    calls = []
    real = round_env.spec_violations

    def counted(spec, cache, round_state):
        calls.append((spec, cache.state.n_active, round_state.round_index))
        return real(spec, cache, round_state)

    monkeypatch.setattr(round_env, "spec_violations", counted)
    env = _env()
    for _ in range(3):
        _episode(env, _holds)
    assert calls == [(env.observation_spec, 2, 1)]
    other = _env()
    other.reset(options={"ic": IC})
    assert len(calls) == 2


# --- What the module imports -------------------------------------------------------

def _imported(path: Path) -> set[str]:
    names = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def test_the_environment_imports_no_rl_library_and_nothing_sequential():
    names = _imported(Path(round_env.__file__))
    roots = {name.split(".")[0] for name in names}
    assert not roots & {"stable_baselines3", "sb3_contrib", "torch", "drivers", "deployment", "training", "backends"}
    sequential = {"agent.core", "agent.masking", "agent.observation", "agent.queue", "agent.reward", "agent"}
    assert not names & sequential
    assert "gymnasium" in roots
