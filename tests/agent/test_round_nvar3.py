"""Tests for the 2D agent core at three solution variables: the whole round
loop on the mock quad backend at nvar = 3, under both reductions.

AGENT_2D_ROADMAP item 2.7, signals N2-N11. No module is under test alone:
the read (agent/round_cache.py), the observation (round_observation.py),
the queue (round_queue.py), the mask (round_mask.py), the commit
(round_commit.py) and both rewards (round_reward.py) are walked together,
as a driver would walk them. Nine groups:

  - the five-element instance of the explanation file's section 2.7.2:
    three columns, a threshold pair per column, the distances and the order
    (N2);
  - one interval on the instance, in the three cells that run (N3);
  - scalar at three variables is scalar at one variable, exactly (N4);
  - per_variable against scalar in a round: the same classes, distances to
    a tolerance, and an order that agrees wherever priorities differ (N5);
  - whole per_variable intervals of random masked actions (N6), and under
    unanimous, where the order cannot matter, against scalar (N7);
  - the call count (N8);
  - what is refused: the 1D rule at three columns (N9);
  - three columns that disagree, set by hand on the instance (N10);
  - three columns that disagree, stacked from the mock's fields on its
    meshes (N11).

The mock's per_variable columns are proportional (variable v is the field
over v + 1), so the groups on them cannot tell one scalarization from
another. The last two groups can.

The reductions are compared to a tolerance and never on the order of two
elements whose priorities differ in the last place: dividing by 3 is not
exact, and which way the last place falls is the platform's.

Run: pytest tests/agent/test_round_nvar3.py -v
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from agent.round_cache import RoundCache, read_round
from agent.round_commit import commit, overshoot, start_round
from agent.round_mask import ACTION_COARSEN, ACTION_HOLD, ACTION_REFINE, COARSEN_RULES, action_mask
from agent.round_observation import (
    ACTOR_KEY,
    CRITIC_KEY,
    ObservationConfig,
    actor_observation,
    feature_names,
    observation,
    observation_bounds,
)
from agent.round_queue import next_undecided, round_order, round_priority, round_thresholds, zone_distance
from agent.round_reward import global_reward, local_reward
from backends.mock_quad.backend import MockQuadBackend
from contract.solver_contract import COARSEN, HOLD, REFINE
from contract.state import SolverState

NVAR = 3
BETA = 2
ALPHA = 0.1
INDICATORS = ("ind_a", "ind_b")
FAMILY_RULES = ("first_decider", "any_coarsen")
# The feature: off every centroid, so no two of the instance's errors tie.
IC = {"x0": 0.3, "y0": 0.6, "width": 0.2}
# A second feature, for a column that peaks somewhere else.
IC_ELSEWHERE = {"x0": 0.8, "y0": 0.2, "width": 0.2}
P_UR, P_OR, P_CR = 10.0, 5.0, 2.0
W = dict(p_ur=P_UR, p_or=P_OR, p_cr=P_CR)
G = dict(p_ur=P_UR, p_or=P_OR)
# Cell B's position on the instance; positions 0-3 are cell A's children.
B = 4
# Section 2.7.2's first table: ind_b on the instance, variable 0.
E0 = (1.310071, 0.107537, 4.572598, 0.375342, 0.007401)
D0 = (0.457132, 0.0, 1.0, 0.0, -0.790866)
# The three cells that run, as (nvar, reduction): one variable, and three
# variables under each reduction.
CELLS = [(1, "scalar"), (NVAR, "scalar"), (NVAR, "per_variable")]
CELL_IDS = ["one variable", "three, scalar", "three, per_variable"]


# --- The meshes, the configuration and the walk ------------------------------------

def _two_cell_mesh(nvar: int, ic=IC) -> tuple[MockQuadBackend, SolverState]:
    """The mock's five-element mesh: cells A and B on [0,2]x[0,1], periodic
    in x and walled in y, A refined. Pinned: positions 0-3 are A's children."""
    mock = MockQuadBackend(
        nx=2, ny=1, periodic=(True, False), lo=(0.0, 0.0), hi=(2.0, 1.0), max_level=2, nvar=nvar, dt_base=0.25,
        rotate_cells=False,
    )
    state = mock.reset(ic, amr_freq=4, initial_level=0)
    state, _ = mock.apply_marks(np.where(state.centroid[:, 0] < 1.0, REFINE, HOLD).astype(np.int8))
    return mock, state


def _graded_mesh(periodic: tuple[bool, bool], nvar: int, ic=IC) -> tuple[MockQuadBackend, SolverState]:
    """A 4x4 grid at level 1 with the cells near (0.3, 0.6) refined once
    more, the rotation on: levels 1 and 2, with 2:1 faces between them. The
    refined cells are the same whatever feature the mock carries."""
    mock = MockQuadBackend(nx=4, ny=4, periodic=periodic, max_level=2, nvar=nvar, dt_base=0.25, rotate_cells=True)
    state = mock.reset(ic, amr_freq=4, initial_level=1)
    near = np.hypot(state.centroid[:, 0] - IC["x0"], state.centroid[:, 1] - IC["y0"]) < 0.15
    assert near.any()
    state, _ = mock.apply_marks(np.where(near, REFINE, HOLD).astype(np.int8))
    return mock, state


MESHES = {
    "two cells": _two_cell_mesh,
    "4x4 periodic": lambda nvar, ic=IC: _graded_mesh((True, True), nvar, ic),
    "4x4 walled": lambda nvar, ic=IC: _graded_mesh((False, False), nvar, ic),
    "4x4 mixed": lambda nvar, ic=IC: _graded_mesh((True, False), nvar, ic),
}


def _config(indicator: str, reduction: str, normalization: str = "relative") -> ObservationConfig:
    return ObservationConfig(
        indicator=indicator, reduction=reduction, alpha=ALPHA, element_budget=64, normalization=normalization,
        floor_decades=8.0,
    )


def _families(state: SolverState) -> list[tuple[int, ...]]:
    """Each complete family once, as its positions in child order."""
    return sorted({tuple(row) for row in state.family.tolist() if row[0] >= 0})


def _within(obs: np.ndarray, bounds: tuple[np.ndarray, np.ndarray]) -> bool:
    low, high = bounds
    return bool(((low <= obs) & (obs <= high)).all())


def _interval(mock, state, rule, indicator, reduction, choose, normalization="relative", n_rounds=2):
    """One interval as a driver would run it, at any reduction: the
    thresholds from round 1's cache and held; per round one read, the order
    and the distances under that pair, then a walk in which every asked
    position is observed through the dict, masked, paid and committed;
    apply_marks and the overshoot; then one step_interval and the global
    reward on its acc and the last state's levels.

    choose(round_index, position, mask) returns the action.

    Returns a dict: one log per round under "rounds", and the pair, the
    acc array, the report, the last state and the global reward.
    """
    cfg = _config(indicator, reduction, normalization)
    bounds = observation_bounds(state.ndim, cfg)
    pair, rounds = None, []
    for round_index in range(1, n_rounds + 1):
        cache = read_round(mock, state, [indicator], reduction)
        if pair is None:
            pair = round_thresholds(cache, cfg, BETA)
        order = round_order(cache, cfg, *pair)
        distance = zone_distance(cache.errors[indicator], cfg, *pair)
        round_state = start_round(state, round_index, n_rounds)
        asked, actions, rewards, seen, skipped = [], [], [], [], 0
        position, cursor, passed = next_undecided(state, round_state, order, 0)
        while position is not None:
            skipped += passed
            obs = observation(cache, round_state, position, cfg)
            assert list(obs) == [ACTOR_KEY, CRITIC_KEY]
            assert all(np.isfinite(obs[key]).all() and _within(obs[key], bounds[key]) for key in obs)
            mask = action_mask(state, round_state, position, rule)
            action = choose(round_index, position, mask)
            assert mask[action]
            rewards.append(local_reward(state, distance, position, action, rule, **W))
            round_state = commit(state, round_state, position, action, rule)
            asked.append(position)
            actions.append(action)
            seen.append(obs[ACTOR_KEY])
            position, cursor, passed = next_undecided(state, round_state, order, cursor)
        skipped += passed
        marks = round_state.decisions
        new_state, report = mock.apply_marks(marks)
        whole = sum(bool((marks[list(f)] == COARSEN).all()) for f in _families(state))
        kept = sum(bool((report.realized[list(f)] == COARSEN).all()) for f in _families(state))
        rounds.append(
            dict(
                n=state.n_active, errors=cache.errors[indicator], order=order, distance=distance, asked=asked,
                actions=actions, rewards=rewards, seen=np.array(seen), skipped=skipped, marks=marks,
                committed=round_state.committed, n_after=new_state.n_active,
                overshoot=overshoot(round_state, new_state),
                added=int((report.realized == REFINE).sum()) - int((marks == REFINE).sum()), undone=whole - kept,
            )
        )
        state = new_state
    report = mock.step_interval([indicator], reduction, 1)
    acc = report.acc[indicator]
    r_global = global_reward(acc, state.level, state.max_level, cfg, *pair, **G)
    return dict(rounds=rounds, pair=pair, acc=acc, report=report, state=state, r_global=r_global)


def _drawn(seed: int):
    """Random masked actions, one draw per asked position, in the order
    asked: a different order gives different actions."""
    rng = np.random.default_rng(seed)
    return lambda round_index, position, mask: int(rng.choice(np.flatnonzero(mask)))


def _keyed(seed: int):
    """Random masked actions fixed by the round and the position alone,
    whatever order the positions are asked in."""

    def choose(round_index, position, mask):
        open_actions = np.flatnonzero(mask)
        u = np.random.default_rng([seed, round_index, position]).random()
        return int(open_actions[int(u * len(open_actions))])

    return choose


def _holds(round_index, position, mask):
    return ACTION_HOLD


# --- The instance: three columns (N2) ----------------------------------------------

def _instance(reduction: str = "per_variable", nvar: int = NVAR):
    """The instance's first round: the state, the cache, the configuration
    and the interval's pair."""
    mock, state = _two_cell_mesh(nvar)
    cache = read_round(mock, state, ["ind_b"], reduction)
    cfg = _config("ind_b", reduction)
    return state, cache, cfg, round_thresholds(cache, cfg, BETA)


def test_the_instance_carries_three_variables():
    """Section 2.7.2's first table. Variable v is variable 0 over v + 1."""
    state, cache, _, _ = _instance()
    errors = cache.errors["ind_b"]
    assert (state.nvar, state.n_active, cache.n_comp) == (3, 5, 3)
    assert errors.shape == (5, 3) and errors.dtype == np.float64
    assert errors[:, 0].tolist() == pytest.approx(list(E0), abs=1e-6)
    assert errors[:, 1].tolist() == pytest.approx([e / 2 for e in E0], abs=1e-6)
    assert errors[:, 2].tolist() == pytest.approx([e / 3 for e in E0], abs=1e-6)
    assert np.array_equal(errors[:, 1], errors[:, 0] / 2)
    assert errors[:, 2].tolist() == pytest.approx((errors[:, 0] / 3).tolist(), rel=1e-15)


def test_under_scalar_the_instance_is_its_first_column():
    _, per_variable, _, _ = _instance()
    _, scalar, _, _ = _instance("scalar")
    assert scalar.n_comp == 1 and scalar.errors["ind_b"].shape == (5, 1)
    assert np.array_equal(scalar.errors["ind_b"][:, 0], per_variable.errors["ind_b"][:, 0])


def test_the_thresholds_are_one_pair_per_column():
    """Each column is measured against its own largest error: the pair of
    variable v is the pair of variable 0 over v + 1."""
    _, cache, _, (e_max, e_min) = _instance()
    assert e_max.shape == e_min.shape == (3,)
    assert e_max.tolist() == pytest.approx([0.457260, 0.228630, 0.152420], abs=1e-6)
    assert e_min.tolist() == pytest.approx([0.045726, 0.022863, 0.015242], abs=1e-6)
    scale = np.arange(1, NVAR + 1)
    assert e_max.tolist() == pytest.approx((e_max[0] / scale).tolist(), rel=1e-15)
    assert e_min.tolist() == pytest.approx((e_min[0] / scale).tolist(), rel=1e-15)
    assert e_max.tolist() == pytest.approx((ALPHA * cache.errors["ind_b"].max(axis=0)).tolist(), rel=1e-15)


def test_the_scalar_pair_is_the_first_columns():
    _, _, _, per_variable = _instance()
    _, _, _, scalar = _instance("scalar")
    assert scalar[0].shape == (1,)
    assert np.array_equal(scalar[0], per_variable[0][:1]) and np.array_equal(scalar[1], per_variable[1][:1])


@pytest.mark.parametrize("reduction", ["scalar", "per_variable"])
def test_the_instances_distances_and_order(reduction):
    """Position 2 holds the largest error, a decade above e_max; position 0
    is above it too; positions 1 and 3 are in the zone; B is below e_min."""
    _, cache, cfg, pair = _instance(reduction)
    distance = zone_distance(cache.errors["ind_b"], cfg, *pair)
    assert distance.tolist() == pytest.approx(list(D0), abs=1e-6)
    assert distance[1] == 0.0 and distance[3] == 0.0
    assert distance[0] == pytest.approx(math.log10(E0[0] / 0.457260), abs=1e-5)
    assert round_order(cache, cfg, *pair).tolist() == [2, 4, 0, 1, 3]
    assert np.array_equal(round_priority(cache, cfg, *pair), np.abs(distance))


def test_the_instances_observation_at_three_columns():
    """Position 2, the round's worst element: its own error and its family
    read -log10(alpha) = 1; B, across the wrap on its x-min face, reads
    -1.79; its y-max face is a wall and repeats its own."""
    state, cache, cfg, _ = _instance()
    names = feature_names(2)
    obs = actor_observation(cache, start_round(state, 1, 2), 2, cfg)
    assert obs.shape == (17,) and obs.dtype == np.float32
    by_name = dict(zip(names, obs.tolist(), strict=True))
    assert by_name["error"] == pytest.approx(1.0, abs=1e-6)
    assert by_name["error_x_min"] == pytest.approx(-1.790866, abs=1e-5)
    assert by_name["error_x_max"] == pytest.approx(-0.085736, abs=1e-5)
    assert by_name["error_y_min"] == pytest.approx(0.457132, abs=1e-5)
    assert by_name["error_y_max"] == pytest.approx(1.0, abs=1e-6)
    assert by_name["error_family_max"] == pytest.approx(1.0, abs=1e-6)
    assert (by_name["wall_y_max"], by_name["resource"], by_name["round_progress"]) == (1.0, 5 / 64, 0.5)


@pytest.mark.parametrize("position", range(5))
def test_on_the_instance_the_observation_is_the_scalar_one(position):
    state, per_variable, cfg3, _ = _instance()
    _, scalar, cfg1, _ = _instance("scalar")
    round_state = start_round(state, 1, 2)
    three = actor_observation(per_variable, round_state, position, cfg3)
    one = actor_observation(scalar, round_state, position, cfg1)
    assert three.tolist() == pytest.approx(one.tolist(), abs=1e-6)


@pytest.mark.parametrize("rule", FAMILY_RULES)
@pytest.mark.parametrize("decider", range(4))
def test_on_the_instance_a_family_coarsen_costs_p_ur_from_any_member(decider, rule):
    """The family's largest distance is position 2's, one decade."""
    state, cache, cfg, pair = _instance()
    distance = zone_distance(cache.errors["ind_b"], cfg, *pair)
    assert local_reward(state, distance, decider, ACTION_COARSEN, rule, **W) == pytest.approx(-10.0)


def test_on_the_instance_a_unanimous_vote_is_paid_on_its_own_distance():
    state, cache, cfg, pair = _instance()
    distance = zone_distance(cache.errors["ind_b"], cfg, *pair)
    assert local_reward(state, distance, 0, ACTION_COARSEN, "unanimous", **W) == pytest.approx(-4.571319, abs=1e-6)
    assert local_reward(state, distance, 1, ACTION_COARSEN, "unanimous", **W) == 0.0


# --- The instance's interval (N3) --------------------------------------------------

def _script(round_index, position, mask):
    """Round 1: position 2 and cell B refine. Everything else holds."""
    return ACTION_REFINE if round_index == 1 and position in (2, B) else ACTION_HOLD


@pytest.mark.parametrize("nvar, reduction", CELLS, ids=CELL_IDS)
def test_the_instances_interval(nvar, reduction):
    """Section 2.7.2's interval, the same in every cell. Round 1: position
    2, above the zone, refines for nothing; B, 0.79 decades below e_min,
    refines and pays p_or times that; count 11, mesh 11. Round 2: eleven
    holds. Then the advance: five elements are left above the zone below
    max_level, none below it, over 11."""
    mock, state = _two_cell_mesh(nvar)
    out = _interval(mock, state, "first_decider", "ind_b", reduction, _script)
    n_comp = 1 if reduction == "scalar" else nvar
    first, second = out["rounds"]

    assert first["errors"].shape == (5, n_comp)
    assert first["asked"] == [2, 4, 0, 1, 3] and first["skipped"] == 0
    assert first["rewards"] == pytest.approx([0.0, -3.954328, 0.0, 0.0, 0.0], abs=1e-6)
    assert first["rewards"][1] == pytest.approx(P_OR * D0[B], abs=1e-5)
    assert (first["committed"], first["n_after"], first["overshoot"]) == (11, 11, 0)

    assert second["errors"].shape == (11, n_comp) and second["n"] == 11
    assert len(second["asked"]) == 11 and second["rewards"] == [0.0] * 11
    assert (second["committed"], second["n_after"], second["overshoot"]) == (11, 11, 0)

    report, last = out["report"], out["state"]
    assert out["acc"].shape == (11, n_comp) and out["acc"].dtype == np.float64
    assert report.mass.shape == (nvar,) and report.n_substeps == 16
    assert (last.nvar, last.n_active, sorted(last.level.tolist())) == (nvar, 11, [1] * 7 + [2] * 4)
    assert out["r_global"] == pytest.approx(-3.111172, abs=1e-6)

    cfg = _config("ind_b", reduction)
    left = zone_distance(out["acc"], cfg, *out["pair"])
    counted = left[(left > 0.0) & (last.level < last.max_level)]
    assert sorted(counted.tolist()) == pytest.approx([0.457132, 0.469855, 0.469855, 1.012723, 1.012723], abs=1e-6)
    assert not (left < 0.0).any()
    assert out["r_global"] == pytest.approx(-P_UR * float(counted.sum()) / 11)


def test_the_accumulated_first_column_is_the_scalar_accumulator():
    scalar = _interval(*_two_cell_mesh(NVAR), "first_decider", "ind_b", "scalar", _script)
    per_variable = _interval(*_two_cell_mesh(NVAR), "first_decider", "ind_b", "per_variable", _script)
    assert np.array_equal(per_variable["acc"][:, 0], scalar["acc"][:, 0])
    assert np.array_equal(per_variable["acc"][:, 1], per_variable["acc"][:, 0] / 2)


# --- scalar at three variables is scalar at one (N4) -------------------------------

def _same_interval(one: dict, other: dict) -> None:
    """Assert that two intervals are equal in everything, exactly."""
    assert len(one["rounds"]) == len(other["rounds"])
    for a, b in zip(one["rounds"], other["rounds"], strict=True):
        for key in ("errors", "order", "distance", "seen", "marks"):
            assert np.array_equal(a[key], b[key]), key
        for key in ("n", "asked", "actions", "rewards", "skipped", "committed", "n_after", "overshoot"):
            assert a[key] == b[key], key
    assert np.array_equal(one["pair"][0], other["pair"][0]) and np.array_equal(one["pair"][1], other["pair"][1])
    assert np.array_equal(one["acc"], other["acc"])
    assert one["r_global"] == other["r_global"]


@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("normalization", ["relative", "dynamo"])
@pytest.mark.parametrize("indicator", INDICATORS)
@pytest.mark.parametrize("rule", COARSEN_RULES)
@pytest.mark.parametrize("mesh", list(MESHES))
def test_scalar_at_three_variables_is_scalar_at_one(mesh, rule, indicator, normalization, seed):
    """The core reads the cache's columns and never the state's nvar: with
    one column at either nvar, every array, order, observation, record,
    reward and count of a whole interval is the same, bit for bit. Under
    scalar the 1D rule is open too."""
    one = _interval(*MESHES[mesh](1), rule, indicator, "scalar", _drawn(seed), normalization)
    three = _interval(*MESHES[mesh](NVAR), rule, indicator, "scalar", _drawn(seed), normalization)
    _same_interval(one, three)
    assert (one["state"].nvar, three["state"].nvar) == (1, 3)
    assert (one["report"].mass.shape, three["report"].mass.shape) == ((1,), (3,))
    assert three["acc"].shape[1] == 1


# --- per_variable against scalar, in a round (N5) ----------------------------------

def _round_one(mesh: str, indicator: str, reduction: str):
    mock, state = MESHES[mesh](NVAR)
    cache = read_round(mock, state, [indicator], reduction)
    cfg = _config(indicator, reduction)
    pair = round_thresholds(cache, cfg, BETA)
    return state, cache, cfg, pair, zone_distance(cache.errors[indicator], cfg, *pair)


@pytest.mark.parametrize("indicator", INDICATORS)
@pytest.mark.parametrize("mesh", list(MESHES))
def test_on_the_mock_the_thresholds_are_per_column(mesh, indicator):
    _, cache, _, (e_max, e_min), _ = _round_one(mesh, indicator, "per_variable")
    _, _, _, scalar, _ = _round_one(mesh, indicator, "scalar")
    assert cache.errors[indicator].shape == (cache.n_active, 3)
    scale = np.arange(1, NVAR + 1)
    assert np.array_equal(e_max[:1], scalar[0]) and np.array_equal(e_min[:1], scalar[1])
    assert e_max.tolist() == pytest.approx((scalar[0][0] / scale).tolist(), rel=1e-15)
    assert e_min.tolist() == pytest.approx((scalar[1][0] / scale).tolist(), rel=1e-15)


@pytest.mark.parametrize("indicator", INDICATORS)
@pytest.mark.parametrize("mesh", list(MESHES))
def test_on_the_mock_the_distances_are_the_scalar_ones(mesh, indicator):
    """Proportional columns are one column seen three times: the same
    class at every element, and the same distance to a tolerance. Never
    lower: the first column is the scalar array, and an element's distance
    is the largest over its columns."""
    _, _, _, _, three = _round_one(mesh, indicator, "per_variable")
    _, _, _, _, one = _round_one(mesh, indicator, "scalar")
    assert np.array_equal(np.sign(three), np.sign(one))
    assert three.tolist() == pytest.approx(one.tolist(), abs=1e-12)
    assert (three >= one).all()
    assert (one > 0.0).any() and (one == 0.0).any()


@pytest.mark.parametrize("indicator", INDICATORS)
@pytest.mark.parametrize("mesh", list(MESHES))
def test_on_the_mock_the_order_agrees_wherever_priorities_differ(mesh, indicator):
    """The per_variable order is a permutation, and along it the scalar
    priorities never rise by more than a rounding error. Two elements
    whose priorities differ in the last place may swap between the
    reductions, so the two orders are not compared slot by slot."""
    state, cache3, cfg3, pair3, _ = _round_one(mesh, indicator, "per_variable")
    _, cache1, cfg1, pair1, _ = _round_one(mesh, indicator, "scalar")
    order = round_order(cache3, cfg3, *pair3)
    assert sorted(order.tolist()) == list(range(state.n_active))
    along = round_priority(cache1, cfg1, *pair1)[order]
    assert (np.diff(along) <= 1e-12).all()
    own = round_priority(cache3, cfg3, *pair3)[order]
    assert (np.diff(own) <= 0.0).all()


def test_on_the_instance_the_two_orders_are_the_same():
    """Five priorities far apart: here the orders can be compared."""
    _, cache3, cfg3, pair3 = _instance()
    _, cache1, cfg1, pair1 = _instance("scalar")
    assert round_order(cache3, cfg3, *pair3).tolist() == round_order(cache1, cfg1, *pair1).tolist() == [2, 4, 0, 1, 3]


@pytest.mark.parametrize("indicator", INDICATORS)
@pytest.mark.parametrize("mesh", list(MESHES))
def test_on_the_mock_every_observation_is_the_scalar_one(mesh, indicator):
    state, cache3, cfg3, _, _ = _round_one(mesh, indicator, "per_variable")
    _, cache1, cfg1, _, _ = _round_one(mesh, indicator, "scalar")
    round_state = start_round(state, 1, 2)
    bounds = observation_bounds(2, cfg3)
    for position in range(state.n_active):
        three = observation(cache3, round_state, position, cfg3)
        one = actor_observation(cache1, round_state, position, cfg1)
        assert three[ACTOR_KEY].tolist() == pytest.approx(one.tolist(), abs=1e-6)
        assert np.array_equal(three[CRITIC_KEY], three[ACTOR_KEY])
        assert _within(three[ACTOR_KEY], bounds[ACTOR_KEY])


@pytest.mark.parametrize("indicator", INDICATORS)
@pytest.mark.parametrize("mesh", list(MESHES))
def test_on_the_mock_every_local_reward_is_the_scalar_one(mesh, indicator):
    """Every element's hold and refine, every family's coarsen from each
    member, and every sibling's own vote."""
    state, _, _, _, three = _round_one(mesh, indicator, "per_variable")
    _, _, _, _, one = _round_one(mesh, indicator, "scalar")
    for position in range(state.n_active):
        for action in (ACTION_HOLD, ACTION_REFINE):
            paid3 = local_reward(state, three, position, action, "first_decider", **W)
            assert paid3 == pytest.approx(local_reward(state, one, position, action, "first_decider", **W), abs=1e-9)
    families = _families(state)
    assert families
    for family in families:
        for member in family:
            for rule in COARSEN_RULES:
                paid3 = local_reward(state, three, member, ACTION_COARSEN, rule, **W)
                assert paid3 == pytest.approx(local_reward(state, one, member, ACTION_COARSEN, rule, **W), abs=1e-9)


# --- Whole per_variable intervals (N6, N7) -----------------------------------------

@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("indicator", INDICATORS)
@pytest.mark.parametrize("rule", COARSEN_RULES)
@pytest.mark.parametrize("mesh", list(MESHES))
def test_a_per_variable_interval_of_random_masked_actions(mesh, rule, indicator, seed):
    """Two rounds and the advance at three columns, every asked position
    observed, masked, paid and committed. In every round: asked plus
    skipped is n_active; the overshoot is never negative and is 3 for
    every refine balance added and every family coarsen it undid; every
    local reward is finite. The global reward is finite and not positive."""
    mock, state = MESHES[mesh](NVAR)
    out = _interval(mock, state, rule, indicator, "per_variable", _drawn(seed))
    assert len(out["rounds"]) == 2 and out["pair"][0].shape == (3,)
    for log in out["rounds"]:
        assert log["errors"].shape == (log["n"], 3) and log["seen"].shape == (len(log["asked"]), 17)
        assert len(log["asked"]) + log["skipped"] == log["n"] and len(log["rewards"]) == len(log["asked"])
        assert log["overshoot"] == log["n_after"] - log["committed"]
        assert log["overshoot"] >= 0 and log["overshoot"] == 3 * (log["added"] + log["undone"])
        assert all(math.isfinite(r) for r in log["rewards"])
        if rule == "unanimous":
            assert log["skipped"] == 0
    assert out["rounds"][1]["n"] == out["rounds"][0]["n_after"]
    assert out["acc"].shape == (out["state"].n_active, 3) and out["report"].mass.shape == (3,)
    assert math.isfinite(out["r_global"]) and out["r_global"] <= 0.0


def test_the_per_variable_intervals_are_not_checked_on_zeros_alone():
    """Across the random intervals some rounds overshoot and some do not,
    some positions are skipped, some local rewards are penalties and some
    are positive, and some global reward is below zero."""
    over = exact = skipped = penalties = earned = below = 0
    for mesh in MESHES:
        for rule in COARSEN_RULES:
            for seed in (0, 1, 2):
                out = _interval(*MESHES[mesh](NVAR), rule, "ind_b", "per_variable", _drawn(seed))
                for log in out["rounds"]:
                    over += log["overshoot"] > 0
                    exact += log["overshoot"] == 0
                    skipped += log["skipped"]
                    penalties += sum(r < 0.0 for r in log["rewards"])
                    earned += sum(r > 0.0 for r in log["rewards"])
                below += out["r_global"] < 0.0
    assert over > 0 and exact > 0 and skipped > 0 and penalties > 0 and earned > 0 and below > 0


@pytest.mark.parametrize("rule", COARSEN_RULES)
@pytest.mark.parametrize("mesh", list(MESHES))
def test_a_per_variable_interval_of_holds_is_the_scalar_one(mesh, rule):
    """With every position holding, the order decides nothing: the mesh is
    unchanged, no local reward is paid, and the global reward is the
    scalar interval's to a tolerance."""
    three = _interval(*MESHES[mesh](NVAR), rule, "ind_b", "per_variable", _holds)
    one = _interval(*MESHES[mesh](NVAR), rule, "ind_b", "scalar", _holds)
    for log in three["rounds"]:
        assert log["skipped"] == 0 and log["overshoot"] == 0 and log["n_after"] == log["n"]
        assert log["rewards"] == [0.0] * log["n"]
    assert three["r_global"] == pytest.approx(one["r_global"], abs=1e-9)
    assert three["r_global"] < 0.0


@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("indicator", INDICATORS)
@pytest.mark.parametrize("mesh", list(MESHES))
def test_under_unanimous_a_per_variable_interval_is_the_scalar_one(mesh, indicator, seed):
    """Under unanimous the mask does not read the record, so with each
    position's action fixed in advance the order the positions are asked
    in cannot change the outcome. The two reductions then give the same
    records, counts and meshes, and each position the same reward to a
    tolerance."""
    three = _interval(*MESHES[mesh](NVAR), "unanimous", indicator, "per_variable", _keyed(seed))
    one = _interval(*MESHES[mesh](NVAR), "unanimous", indicator, "scalar", _keyed(seed))
    for a, b in zip(three["rounds"], one["rounds"], strict=True):
        assert np.array_equal(a["marks"], b["marks"])
        for key in ("n", "committed", "n_after", "overshoot"):
            assert a[key] == b[key], key
        assert sorted(a["asked"]) == sorted(b["asked"]) == list(range(a["n"]))
        paid_a = dict(zip(a["asked"], a["rewards"], strict=True))
        paid_b = dict(zip(b["asked"], b["rewards"], strict=True))
        for position, reward in paid_a.items():
            assert reward == pytest.approx(paid_b[position], abs=1e-9)
        seen_a = dict(zip(a["asked"], a["seen"], strict=True))
        seen_b = dict(zip(b["asked"], b["seen"], strict=True))
        # The error, level and wall features; the resource feature moves
        # with the order the positions were asked in.
        fixed = [i for i, name in enumerate(feature_names(2)) if name != "resource"]
        for position in seen_a:
            assert seen_a[position][fixed].tolist() == pytest.approx(seen_b[position][fixed].tolist(), abs=1e-6)
    assert np.array_equal(three["acc"][:, 0], one["acc"][:, 0])
    assert three["r_global"] == pytest.approx(one["r_global"], abs=1e-9)
    assert any((log["marks"] != HOLD).any() for log in three["rounds"])


# --- The call count (N8) -----------------------------------------------------------

@pytest.mark.parametrize("rotate", [False, True], ids=["pinned", "rotating"])
def test_a_per_variable_interval_costs_2r_plus_1_calls(counting, rotate):
    """Three columns cost no call: one compute_error and one apply_marks
    per round and one step_interval, after the reset."""
    mock = counting(
        MockQuadBackend(
            nx=2, ny=1, periodic=(True, False), lo=(0.0, 0.0), hi=(2.0, 1.0), max_level=2, nvar=NVAR, dt_base=0.25,
            rotate_cells=rotate,
        )
    )
    state = mock.reset(IC, amr_freq=4, initial_level=1)
    out = _interval(mock, state, "first_decider", "ind_b", "per_variable", _drawn(4))
    assert out["acc"].shape[1] == 3 and out["r_global"] <= 0.0
    assert dict(mock.calls) == {"reset": 1, "compute_error": 2, "apply_marks": 2, "step_interval": 1}


# --- What is refused (N9) ----------------------------------------------------------

CACHE_MESSAGE = r"^normalization: 'dynamo' takes one column per indicator, and the cache has n_comp = 3$"
ARRAY_MESSAGE = r"^normalization: 'dynamo' takes one column, and errors has 3$"


def _refusal_case():
    """The instance at three columns, with a valid pair, its accumulated
    errors, and a configuration that asks for the 1D rule."""
    mock, state = _two_cell_mesh(NVAR)
    cache = read_round(mock, state, ["ind_b"], "per_variable")
    pair = round_thresholds(cache, _config("ind_b", "per_variable"), BETA)
    acc = mock.step_interval(["ind_b"], "per_variable", 1).acc["ind_b"]
    return state, cache, pair, acc, _config("ind_b", "per_variable", "dynamo")


REFUSED = {
    "actor_observation": (lambda s, c, p, a, d: actor_observation(c, start_round(s, 1, 2), 0, d), CACHE_MESSAGE),
    "observation": (lambda s, c, p, a, d: observation(c, start_round(s, 1, 2), 0, d), CACHE_MESSAGE),
    "round_thresholds": (lambda s, c, p, a, d: round_thresholds(c, d, BETA), CACHE_MESSAGE),
    "round_priority": (lambda s, c, p, a, d: round_priority(c, d, *p), CACHE_MESSAGE),
    "round_order": (lambda s, c, p, a, d: round_order(c, d, *p), CACHE_MESSAGE),
    "zone_distance": (lambda s, c, p, a, d: zone_distance(c.errors["ind_b"], d, *p), ARRAY_MESSAGE),
    "global_reward": (lambda s, c, p, a, d: global_reward(a, s.level, s.max_level, d, *p, **G), ARRAY_MESSAGE),
}


@pytest.mark.parametrize("name", list(REFUSED))
def test_the_1d_rule_is_refused_at_three_columns(name):
    """On arrays the mock returned, by every function that reads errors,
    each in its own words."""
    call, message = REFUSED[name]
    state, cache, pair, acc, dynamo = _refusal_case()
    assert cache.n_comp == 3 and acc.shape == (5, 3)
    with pytest.raises(ValueError, match=message):
        call(state, cache, pair, acc, dynamo)


@pytest.mark.parametrize(
    "call",
    [
        lambda s, c, cfg: actor_observation(c, start_round(s, 1, 2), 0, cfg),
        lambda s, c, cfg: round_thresholds(c, cfg, BETA),
    ],
    ids=["actor_observation", "round_thresholds"],
)
def test_a_configuration_that_says_scalar_is_refused_on_a_per_variable_cache(call):
    state, cache, _, _, _ = _refusal_case()
    message = r"^reduction: the cache was read under 'per_variable', the configuration says 'scalar'$"
    with pytest.raises(ValueError, match=message):
        call(state, cache, _config("ind_b", "scalar"))


def test_one_columns_pair_is_refused_at_three_columns():
    _, cache, pair, _, _ = _refusal_case()
    with pytest.raises(ValueError, match=r"^e_max: expected a numpy array of shape \(3,\)$"):
        zone_distance(cache.errors["ind_b"], _config("ind_b", "per_variable"), pair[0][:1], pair[1][:1])


@pytest.mark.parametrize("rule", COARSEN_RULES)
def test_the_1d_rule_is_accepted_at_three_variables_under_scalar(rule):
    """The refusal is of three columns, not of three variables."""
    out = _interval(*_two_cell_mesh(NVAR), rule, "ind_b", "scalar", _drawn(0), "dynamo")
    assert out["state"].nvar == 3 and out["pair"][0].shape == (1,)
    assert math.isfinite(out["r_global"]) and out["r_global"] <= 0.0


# --- Three columns that disagree, on the instance (N10) ----------------------------

# Section 2.7.2's second table: each column's own distance, by position.
COLUMN_DISTANCES = (
    (0.457132, 0.806617, 0.0),
    (0.0, 0.0, 0.457132),
    (1.0, 1.0, 1.0),
    (0.0, 0.391385, 0.0),
    (-0.790866, -0.384409, -0.790866),
)
ELEMENT_DISTANCES = (0.806617, 0.457132, 1.0, 0.391385, -0.384409)


def _stacked_cache(state: SolverState, columns: np.ndarray) -> RoundCache:
    return RoundCache(state=state, errors={"e": columns}, indicators=("e",), reduction="per_variable")


def _hand_instance():
    """The instance with three columns that are not proportional: ind_b;
    1000 times ind_a; and 0.001 times ind_b with its first two rows
    swapped. Returns the state, the cache, the configuration and the pair."""
    mock, state = _two_cell_mesh(1)
    fields = mock.compute_error(["ind_a", "ind_b"], "scalar")
    ring, bump = fields["ind_a"][:, 0], fields["ind_b"][:, 0]
    columns = np.stack([bump, 1e3 * ring, 1e-3 * bump[[1, 0, 2, 3, 4]]], axis=1)
    cache = _stacked_cache(state, columns)
    cfg = _config("e", "per_variable")
    return state, cache, cfg, round_thresholds(cache, cfg, BETA)


def _column_distances(cache: RoundCache, pair) -> np.ndarray:
    """Each column's own signed distance, (n, n_comp): zone_distance of the
    column alone under its own threshold pair."""
    errors = cache.errors["e"]
    cfg = _config("e", "scalar")
    return np.stack(
        [
            zone_distance(errors[:, v : v + 1].copy(), cfg, pair[0][v : v + 1], pair[1][v : v + 1])
            for v in range(errors.shape[1])
        ],
        axis=1,
    )


def test_the_hand_instances_columns_disagree():
    """Three scales six decades apart, and three different patterns."""
    _, cache, _, (e_max, e_min) = _hand_instance()
    assert e_max.tolist() == pytest.approx([0.457260, 144.598235, 0.000457260], rel=1e-5)
    assert e_min.tolist() == pytest.approx((ALPHA * e_max).tolist(), rel=1e-12)
    by_column = _column_distances(cache, (e_max, e_min))
    assert by_column.shape == (5, 3)
    assert by_column.ravel().tolist() == pytest.approx(np.array(COLUMN_DISTANCES).ravel().tolist(), abs=1e-6)


def test_an_elements_distance_is_the_largest_of_its_columns_distances():
    """Above the zone in any column: its worst column (position 0, the
    second column; position 1, the third alone). Below the zone in every
    column: its nearest (B, the second). Exactly."""
    _, cache, cfg, pair = _hand_instance()
    distance = zone_distance(cache.errors["e"], cfg, *pair)
    assert distance.tolist() == pytest.approx(list(ELEMENT_DISTANCES), abs=1e-6)
    assert np.array_equal(distance, _column_distances(cache, pair).max(axis=1))
    assert round_order(cache, cfg, *pair).tolist() == [2, 0, 1, 3, 4]


def test_the_hand_instances_order_is_not_any_one_columns():
    _, cache, cfg, pair = _hand_instance()
    by_column = _column_distances(cache, pair)
    whole = round_order(cache, cfg, *pair).tolist()
    for v in range(3):
        assert np.argsort(-np.abs(by_column[:, v]), kind="stable").tolist() != whole


def test_the_hand_instances_local_rewards():
    """B is below e_min in every column and nearest to it in the second:
    its refine costs p_or * 0.384, where the first column alone would cost
    p_or * 0.791. Position 1 is in the zone in two columns and above it in
    the third: its vote costs p_ur * 0.457. Position 0's refine is free."""
    state, cache, cfg, pair = _hand_instance()
    distance = zone_distance(cache.errors["e"], cfg, *pair)
    first = _column_distances(cache, pair)[:, 0].copy()
    assert local_reward(state, distance, B, ACTION_REFINE, "first_decider", **W) == pytest.approx(-1.922045, abs=1e-6)
    assert local_reward(state, first, B, ACTION_REFINE, "first_decider", **W) == pytest.approx(-3.954328, abs=1e-6)
    assert local_reward(state, distance, 1, ACTION_COARSEN, "unanimous", **W) == pytest.approx(-4.571319, abs=1e-6)
    assert local_reward(state, first, 1, ACTION_COARSEN, "unanimous", **W) == 0.0
    assert local_reward(state, distance, 0, ACTION_REFINE, "unanimous", **W) == 0.0
    assert local_reward(state, distance, 0, ACTION_COARSEN, "unanimous", **W) == pytest.approx(-8.066169, abs=1e-6)
    for member in range(4):
        assert local_reward(state, distance, member, ACTION_COARSEN, "first_decider", **W) == pytest.approx(-10.0)


def test_the_hand_instances_observation_is_decided_column_by_column():
    """B's own error reads -1.384, the second column's value, where the
    first column alone reads -1.791; its x-min face reads 0.457, the third
    column's (position 1), where the first alone reads -0.086."""
    state, cache, cfg, _ = _hand_instance()
    names = feature_names(2)
    obs = dict(zip(names, actor_observation(cache, start_round(state, 1, 2), B, cfg).tolist(), strict=True))
    assert obs["error"] == pytest.approx(-1.384409, abs=1e-5)
    assert obs["error_x_min"] == pytest.approx(0.457132, abs=1e-5)
    assert obs["error_x_max"] == pytest.approx(1.0, abs=1e-6)

    column = cache.errors["e"][:, :1].copy()
    alone = RoundCache(state=state, errors={"e": column}, indicators=("e",), reduction="scalar")
    one = actor_observation(alone, start_round(state, 1, 2), B, _config("e", "scalar"))
    one = dict(zip(names, one.tolist(), strict=True))
    assert one["error"] == pytest.approx(-1.790866, abs=1e-5)
    assert one["error_x_min"] == pytest.approx(-0.085736, abs=1e-5)


def test_the_hand_instances_global_reward():
    """The instance's levels, the three columns as the interval's errors:
    positions 0-3 are above the zone below max_level and B, below it, is
    at level 0 and costs nothing."""
    state, cache, cfg, pair = _hand_instance()
    expected = -P_UR * (0.806617 + 0.457132 + 1.0 + 0.391385) / 5
    paid = global_reward(cache.errors["e"], state.level, state.max_level, cfg, *pair, **G)
    assert paid == pytest.approx(expected, abs=1e-5)
    raised = state.level.copy()
    raised[B] = 1
    more = global_reward(cache.errors["e"], raised, state.max_level, cfg, *pair, **G)
    assert more == pytest.approx(expected - P_OR * 0.384409 / 5, abs=1e-5)


def test_a_dead_column_takes_no_part_where_a_maximum_with_its_zero_would():
    """A third column of zeros has zero thresholds. Its own distance is 0
    everywhere, and B stays below the zone: the rule is the largest over
    the live columns, not over all of them."""
    _, cache, cfg, pair = _hand_instance()
    columns = cache.errors["e"].copy()
    columns[:, 2] = 0.0
    dead = (np.array([pair[0][0], pair[0][1], 0.0]), np.array([pair[1][0], pair[1][1], 0.0]))
    distance = zone_distance(columns, cfg, *dead)
    two = zone_distance(columns[:, :2].copy(), cfg, dead[0][:2], dead[1][:2])
    assert np.array_equal(distance, two)
    assert distance[B] == pytest.approx(-0.384409, abs=1e-6)
    alone = zone_distance(columns[:, 2:].copy(), _config("e", "scalar"), dead[0][2:], dead[1][2:])
    assert (alone == 0.0).all() and max(distance[B], alone[B]) == 0.0


# --- Three columns that disagree, on the mock's meshes (N11) -----------------------

def _stacked(mesh: str):
    """A mesh of the mock with three columns stacked from its fields: the
    bump; 1000 times the ring; and 0.001 times the bump of a second mock
    built alike with its feature elsewhere. Returns the mock, the state,
    the cache, the configuration and the pair."""
    mock, state = MESHES[mesh](1)
    other, other_state = MESHES[mesh](1, IC_ELSEWHERE)
    assert np.array_equal(state.centroid, other_state.centroid) and np.array_equal(state.family, other_state.family)
    fields = mock.compute_error(["ind_a", "ind_b"], "scalar")
    elsewhere = other.compute_error(["ind_b"], "scalar")["ind_b"]
    columns = np.concatenate([fields["ind_b"], 1e3 * fields["ind_a"], 1e-3 * elsewhere], axis=1)
    cache = _stacked_cache(state, columns)
    cfg = _config("e", "per_variable")
    return mock, state, cache, cfg, round_thresholds(cache, cfg, BETA)


@pytest.mark.parametrize("mesh", list(MESHES))
def test_on_the_mocks_meshes_the_distance_is_the_largest_column_distance(mesh):
    """Exactly, at every element, with every column live."""
    _, state, cache, cfg, pair = _stacked(mesh)
    assert cache.errors["e"].shape == (state.n_active, 3) and (pair[1] > 0.0).all()
    distance = zone_distance(cache.errors["e"], cfg, *pair)
    assert np.array_equal(distance, _column_distances(cache, pair).max(axis=1))
    assert np.array_equal(round_priority(cache, cfg, *pair), np.abs(distance))


def test_on_the_mocks_meshes_every_column_decides_somewhere():
    """So that the identity above is not checked on one column's values:
    across the meshes each column alone holds the largest distance of some
    element, and some element is above the zone in one column and below it
    in another."""
    alone, mixed = np.zeros(3, dtype=int), 0
    for mesh in MESHES:
        _, _, cache, cfg, pair = _stacked(mesh)
        by_column = _column_distances(cache, pair)
        distance = zone_distance(cache.errors["e"], cfg, *pair)
        for v in range(3):
            others = np.delete(by_column, v, axis=1).max(axis=1)
            alone[v] += int(((by_column[:, v] == distance) & (others < distance)).sum())
        mixed += int(((by_column.max(axis=1) > 0.0) & (by_column.min(axis=1) < 0.0)).sum())
    assert (alone > 0).all() and mixed > 0


@pytest.mark.parametrize("mesh", list(MESHES))
def test_on_the_mocks_meshes_each_error_feature_is_the_largest_over_the_columns(mesh):
    """The observation's rule at three columns: every error feature is the
    largest of the three one-column features, each column normalized
    against its own largest error. Bit for bit, at every element."""
    _, state, cache, cfg, _ = _stacked(mesh)
    names = feature_names(2)
    error_slots = [i for i, name in enumerate(names) if name.startswith("error")]
    other_slots = [i for i, name in enumerate(names) if not name.startswith("error")]
    round_state = start_round(state, 1, 2)
    singles = [
        RoundCache(state=state, errors={"e": cache.errors["e"][:, v : v + 1].copy()}, indicators=("e",),
                   reduction="scalar")
        for v in range(3)
    ]
    scalar = _config("e", "scalar")
    for position in range(state.n_active):
        whole = actor_observation(cache, round_state, position, cfg)
        parts = np.stack([actor_observation(single, round_state, position, scalar) for single in singles])
        assert np.array_equal(whole[error_slots], parts[:, error_slots].max(axis=0))
        assert np.array_equal(whole[other_slots], parts[0, other_slots])


@pytest.mark.parametrize("rule", COARSEN_RULES)
@pytest.mark.parametrize("mesh", list(MESHES))
def test_on_the_mocks_meshes_a_round_at_columns_that_disagree(mesh, rule):
    """One round of random masked actions in the stacked columns' order,
    every asked position observed and paid: asked plus skipped is
    n_active, the backend takes the record, the overshoot is not negative,
    and under a family rule each family coarsen is paid on the family's
    largest distance."""
    mock, state, cache, cfg, pair = _stacked(mesh)
    order = round_order(cache, cfg, *pair)
    distance = zone_distance(cache.errors["e"], cfg, *pair)
    bounds = observation_bounds(2, cfg)
    rng = np.random.default_rng(7)
    round_state = start_round(state, 1, 2)
    asked, skipped = 0, 0
    position, cursor, passed = next_undecided(state, round_state, order, 0)
    while position is not None:
        skipped += passed
        assert _within(actor_observation(cache, round_state, position, cfg), bounds[ACTOR_KEY])
        action = int(rng.choice(np.flatnonzero(action_mask(state, round_state, position, rule))))
        paid = local_reward(state, distance, position, action, rule, **W)
        assert math.isfinite(paid)
        if action == ACTION_COARSEN and rule != "unanimous":
            worst = float(distance[state.family[position]].max())
            assert paid == pytest.approx(-P_UR * worst if worst > 0.0 else P_CR * -worst)
        round_state = commit(state, round_state, position, action, rule)
        asked += 1
        position, cursor, passed = next_undecided(state, round_state, order, cursor)
    skipped += passed
    assert asked + skipped == state.n_active
    new_state, _ = mock.apply_marks(round_state.decisions)
    assert overshoot(round_state, new_state) >= 0 and overshoot(round_state, new_state) % 3 == 0
