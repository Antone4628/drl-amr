"""Tests for analysis/round_sanity.py, the instrument of the 2D stack's
sanity task (AGENT_2D_ROADMAP item 3.2).

Five groups:

  - the task's arithmetic: why beta is 2 on ind_b (G2), the nine initial
    conditions, the zone's width;
  - level_contrast on hand-built states;
  - the three baselines on the task's configuration: hold and the
    threshold rule to six decimals (G3), the rule's eight rounds on one
    initial condition (G4), the random baseline within its ranges (G5);
  - one episode's record: its identities, the mask as the environment's
    own refusal, the first choice, an episode that does not depend on the
    ones before it;
  - a run measured: the rows of sanity.csv for a run trained on the
    two-cell instance, a checkpoint whose spec differs refused.

The first four groups are numpy alone. The numbers of the third were
computed on copies of the repository's files equal to the originals by
SHA-256, before this file was written. The last group trains, and asserts
names, counts and equalities between two evaluations, never a number that
depends on the initial weights.

Run: pytest tests/analysis/test_round_sanity.py -v
"""
from __future__ import annotations

import ast
import csv
import json
import math
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import analysis.round_sanity as round_sanity
from agent.round_mask import ACTION_REFINE
from agent.round_observation import ObservationConfig
from agent.round_queue import zone_distance
from analysis.round_sanity import (
    BY_WIDTH,
    ICS,
    MEASURES,
    RANDOM_SEEDS,
    baselines,
    by_width,
    checkpoints,
    evaluate,
    hold,
    level_contrast,
    load_checked,
    main,
    model_policy,
    play,
    random_masked,
    sanity,
    summarize,
    table,
    task_env,
    threshold_rule,
    zone_decades,
)
from training.train_round import PROJECT_ROOT, build_backend, load_config, train

# The two-cell instance's initial condition (the explanation file's 3.1.2).
IC = {"x0": 0.3, "y0": 0.6, "width": 0.2}
# The rule's rounds on ICS[3], (0.30, 0.60, 0.10): committed, overshoot,
# realized, skipped (the explanation file's 3.2.2).
RULE_ROUNDS = [
    (1, 1, 67, 6, 73, 27), (1, 2, 88, 9, 97, 0), (2, 1, 91, 0, 91, 15), (2, 2, 91, 0, 91, 0),
    (3, 1, 94, 9, 103, 6), (3, 2, 91, 12, 103, 12), (4, 1, 100, 6, 106, 15), (4, 2, 97, 9, 106, 9),
]


@pytest.fixture(scope="module")
def task() -> dict:
    """The sanity task's configuration, as a run resolves it."""
    return load_config(PROJECT_ROOT / "configs" / "round_sanity.yaml")


@pytest.fixture(scope="module")
def played(task) -> dict:
    """The three baselines on the nine initial conditions, played once."""
    return baselines(task)


def _small() -> dict:
    """The two-cell instance as a run's configuration: 96 steps, a checkpoint every 32."""
    config = load_config(None)
    config["backend"].update(nx=2, ny=1, periodic=[True, False], hi=[2.0, 1.0], dt_base=0.25, rotate_cells=False)
    config["observation"]["element_budget"] = 8
    config["decision"]["beta"] = 2
    config["episode"].update(n_intervals=2, amr_freq=2, initial_level=0)
    config["training"].update(
        total_timesteps=96, n_steps=32, batch_size=16, n_epochs=2, net_arch=[16, 16], seed=0, device="cpu"
    )
    config["checkpointing"]["save_freq"] = 32
    return config


# --- The task's arithmetic -----------------------------------------------------------

@pytest.mark.parametrize("beta, child", [(1.2, -0.197940), (2.0, 0.0)])
def test_a_refine_on_ind_b_is_penalized_at_the_default_beta_and_not_at_the_tasks(beta, child):
    """A level-1 element at 0.16 of the round's largest error, and its
    children at a quarter of that: at beta 1.2 both sides of the refine lie
    outside the neutral zone (G2)."""
    config = ObservationConfig(
        indicator="ind_b", reduction="scalar", alpha=0.1, element_budget=96, normalization="relative",
        floor_decades=8.0,
    )
    errors = np.array([[0.16], [0.04]])
    distance = zone_distance(errors, config, np.array([0.1]), np.array([0.1**beta]))
    assert distance.tolist() == pytest.approx([0.204120, child], abs=1e-6)


def test_the_zone_is_one_decade_at_the_tasks_beta(task):
    assert zone_decades(task) == pytest.approx(1.0)
    assert zone_decades(load_config(None)) == pytest.approx(0.2)
    # One refine moves ind_b by log10(4): past the default zone, inside the task's.
    assert zone_decades(load_config(None)) < math.log10(4) < zone_decades(task)
    finer = load_config(None)
    finer["observation"]["alpha"] = 0.01
    assert zone_decades(finer) == pytest.approx(0.4)


def test_the_initial_conditions_are_nine_inside_the_mocks_bounds(task):
    assert len(ICS) == 9
    assert sorted({ic["width"] for ic in ICS}) == [0.05, 0.10, 0.15]
    assert len({(ic["x0"], ic["y0"]) for ic in ICS}) == 3
    bounds = build_backend(task).describe().ic_params
    for ic in ICS:
        assert set(ic) == set(bounds)
        assert all(bounds[name][0] <= ic[name] <= bounds[name][1] for name in ic), ic


# --- level_contrast ------------------------------------------------------------------

def _state(centroid, level, time=0.0, periodic=(True, True), hi=(1.0, 1.0)) -> SimpleNamespace:
    return SimpleNamespace(
        centroid=np.array(centroid, dtype=np.float64), level=np.array(level, dtype=np.int32), time=time,
        periodic=np.array(periodic), domain_lo=np.zeros(2), domain_hi=np.array(hi, dtype=np.float64),
    )


FEATURE = {"x0": 0.5, "y0": 0.5, "width": 0.1}


def test_the_contrast_is_the_near_mean_level_less_the_far():
    # Two elements within two widths, at levels 2 and 1; two beyond 3.5, at 1 and 0.
    state = _state([(0.5, 0.5), (0.5, 0.65), (0.9, 0.5), (0.5, 0.1)], [2, 1, 1, 0])
    assert level_contrast(state, FEATURE, (0.0, 0.0)) == 1.5 - 0.5


def test_an_element_between_the_two_radii_is_in_neither_set():
    state = _state([(0.5, 0.5), (0.5, 0.8), (0.9, 0.5)], [2, 2, 0])
    assert level_contrast(state, FEATURE, (0.0, 0.0)) == 2.0
    assert level_contrast(state, FEATURE, (0.0, 0.0), near=3.5, far=3.9) == 2.0
    assert level_contrast(state, FEATURE, (0.0, 0.0), near=2.0, far=2.5) == 2.0 - 1.0


def test_a_uniform_mesh_has_no_contrast(task):
    state = build_backend(task).reset(ICS[3], 10, 1)
    assert level_contrast(state, ICS[3], (1.0, 0.0)) == 0.0


@pytest.mark.parametrize("centroid", [[(0.9, 0.5), (0.5, 0.1)], [(0.5, 0.5), (0.5, 0.6)]])
def test_the_contrast_is_nan_when_a_set_is_empty(centroid):
    assert math.isnan(level_contrast(_state(centroid, [1, 0]), FEATURE, (0.0, 0.0)))


def test_the_distance_is_to_the_nearest_image_on_a_periodic_axis_only():
    ic = {"x0": 0.95, "y0": 0.5, "width": 0.1}
    centroid, level = [(0.05, 0.5), (0.5, 0.5)], [2, 0]
    assert level_contrast(_state(centroid, level), ic, (0.0, 0.0)) == 2.0
    # With walls in x the element at 0.05 is nine widths away, and no element is near.
    assert math.isnan(level_contrast(_state(centroid, level, periodic=(False, True)), ic, (0.0, 0.0)))


def test_the_center_moves_with_the_velocity_and_the_states_time():
    state = _state([(0.5, 0.5), (0.9, 0.5)], [2, 0], time=0.4)
    assert level_contrast(state, FEATURE, (0.0, 0.0)) == 2.0
    # At time 0.4 a feature moving at 1 in x has gone from 0.5 to 0.9.
    assert level_contrast(state, FEATURE, (1.0, 0.0)) == -2.0


# --- The three baselines on the task (G3-G5) -----------------------------------------

def test_holding_costs_fifty(played):
    records = played["hold"]
    assert [record["steps"] for record in records] == [512] * 9
    assert [record["ret"] for record in records] == pytest.approx(
        [-105.920, -105.575, -105.563, -33.417, -34.466, -34.402, -11.643, -12.071, -11.926], abs=1e-3
    )
    summary = summarize(records)
    assert summary["ret"] == pytest.approx(-50.553838, abs=1e-6)
    assert (summary["n_end"], summary["n_max"], summary["rounds_over"], summary["overshoot"]) == (64, 64, 0, 0)
    assert (summary["coarsen_closed"], summary["refine_closed"]) == (384, 0)
    assert (summary["local"], summary["contrast"], summary["first_choice_closed"]) == (0, 0, None)


def test_the_threshold_rule_reaches_about_zero(played):
    records = played["rule"]
    assert [record["steps"] for record in records] == [266, 281, 308, 644, 680, 659, 1235, 1301, 1232]
    assert [record["ret"] for record in records] == pytest.approx(
        [-8.192, -2.982, -1.661, 1.064, -0.113, 0.165, 0.045, -0.586, -0.380], abs=1e-3
    )
    assert [record["overshoot"] for record in records] == [114, 57, 93, 51, 102, 105, 18, 12, 12]
    summary = summarize(records)
    assert summary["ret"] == pytest.approx(-1.404341, abs=1e-6)
    assert summary["contrast"] == pytest.approx(0.818524, abs=1e-6)
    assert summary["first_choice_closed"] == pytest.approx(3170 / 9)
    widths = by_width(records)
    assert list(widths) == [0.05, 0.10, 0.15]
    assert [widths[width]["n_end"] for width in widths] == [54, 111, 190]
    assert [widths[width]["rounds_over"] for width in widths] == pytest.approx([0, 19 / 3, 8])


def test_the_rules_eight_rounds_on_one_initial_condition(played):
    record = played["rule"][3]
    assert (record["x0"], record["y0"], record["width"]) == (0.30, 0.60, 0.10)
    assert [tuple(entry.values()) for entry in record["rounds"]] == RULE_ROUNDS
    assert list(record["rounds"][0]) == ["interval", "round", "committed", "overshoot", "n_active", "skipped"]
    assert (record["steps"], record["rounds_over"], record["n_end"]) == (644, 5, 106)
    assert record["ret"] == pytest.approx(1.064150, abs=1e-6)


def test_random_action_is_three_times_worse_than_none(played):
    """Ranges, not values: the generator's stream may differ between numpy
    versions (G5)."""
    records = played["random"]
    assert len(records) == len(RANDOM_SEEDS) * len(ICS)
    for index in range(len(RANDOM_SEEDS)):
        summary = summarize(records[index * len(ICS):(index + 1) * len(ICS)])
        assert -185 < summary["ret"] < -150
        assert 140 < summary["n_end"] < 166
        assert summary["rounds_over"] == 8
        assert abs(summary["contrast"]) < 0.1
        assert summary["local"] < -100
    # Each seed is its own stream.
    assert len({record["ret"] for record in records[::len(ICS)]}) == len(RANDOM_SEEDS)


# --- One episode's record ------------------------------------------------------------

def test_a_record_adds_up(played):
    for name, records in played.items():
        for record in records:
            assert record["coarsens"] + record["holds"] + record["refines"] == record["steps"], name
            assert record["ret"] == pytest.approx(record["local"] + record["global"]), name
            assert len(record["rounds"]) == 8 and not record["failed"], name
            assert all(e["committed"] + e["overshoot"] == e["n_active"] for e in record["rounds"]), name
            assert record["n_end"] == record["rounds"][-1]["n_active"], name
            assert record["n_max"] == max(64, *(e["n_active"] for e in record["rounds"])), name
            assert record["rounds_over"] == sum(e["n_active"] > 96 for e in record["rounds"]), name
            assert record["overshoot"] == sum(e["overshoot"] for e in record["rounds"]), name
            assert record["rounds_overshot"] == sum(e["overshoot"] > 0 for e in record["rounds"]), name
            assert record["global"] <= 0, name


def test_an_episode_does_not_depend_on_the_ones_before_it(task):
    """One environment and one generator per episode: the mock's element
    order moves with every adapt of a backend, and the queue breaks ties by
    position."""
    pair = [ICS[3], ICS[0]]
    forward = evaluate(task, lambda: random_masked(0), pair)
    backward = evaluate(task, lambda: random_masked(0), pair[::-1])
    assert forward == backward[::-1]


def test_a_closed_action_is_refused_by_the_environment(task):
    """The mask is the environment's: a policy that ignores it does not get
    a wrong episode, it gets no episode."""
    env, velocity = task_env(task)
    with pytest.raises(ValueError, match="is masked for position"):
        play(env, lambda obs, mask: ACTION_REFINE, ICS[3], velocity)


def test_the_first_choice_counts_where_the_mask_did_work(task):
    env, velocity = task_env(task)
    zone = zone_decades(task)
    record = play(env, threshold_rule(zone), ICS[3], velocity, threshold_rule(zone, masked=False))
    assert 0 < record["first_choice_closed"] <= record["coarsen_closed"] + record["refine_closed"]
    # A first choice that is always open counts nothing.
    env, velocity = task_env(task)
    assert play(env, hold(), ICS[3], velocity, hold())["first_choice_closed"] == 0


def test_a_models_policy_asks_for_its_surest_action_with_the_mask_or_without():
    """A stand-in model that tells what it was asked: the masked policy
    hands predict the mask, the first choice hands it none, and both ask
    for the action of highest probability."""
    asked = []

    def predict(obs, deterministic=False, action_masks=None):
        asked.append((deterministic, None if action_masks is None else action_masks.tolist()))
        return np.array(0 if action_masks is None else 1), None

    model = SimpleNamespace(predict=predict)
    mask = np.array([False, True, True])
    assert model_policy(model)({}, mask) == 1
    assert model_policy(model, masked=False)({}, mask) == 0
    assert asked == [(True, [False, True, True]), (True, None)]
    assert type(model_policy(model)({}, mask)) is int


def test_saved_models_are_ordered_by_their_step_counts_as_numbers(tmp_path):
    (tmp_path / "checkpoints").mkdir()
    for name in ("checkpoints/round_100_steps.zip", "checkpoints/round_20_steps.zip",
                 "checkpoints/round_20_steps.spec.json", "final_model.zip", "config.yaml"):
        (tmp_path / name).touch()
    assert [name for name, _ in checkpoints(tmp_path)] == ["checkpoint_20", "checkpoint_100", "final"]
    (tmp_path / "final_model.zip").unlink()
    assert [path.name for _, path in checkpoints(str(tmp_path))] == ["round_20_steps.zip", "round_100_steps.zip"]


def test_a_summary_skips_what_an_episode_does_not_have():
    records = [
        dict.fromkeys(MEASURES, 2.0) | {"contrast": math.nan, "first_choice_closed": None},
        dict.fromkeys(MEASURES, 4.0) | {"first_choice_closed": None},
    ]
    summary = summarize(records)
    assert list(summary) == list(MEASURES)
    assert (summary["ret"], summary["contrast"], summary["first_choice_closed"]) == (3.0, 4.0, None)


def test_the_table_and_the_command_line_show_the_baselines(task, capsys):
    main(["--config", str(PROJECT_ROOT / "configs" / "round_sanity.yaml")])
    out = capsys.readouterr().out.splitlines()
    assert out[0].split() == [
        "policy", "timesteps", "return", "contrast", "n_end", "over/8", "overshoot", "c_closed", "r_closed",
        "first_closed",
    ]
    assert [line.split()[0] for line in out[1:4]] == ["hold", "rule", "random"]
    assert out[1].split()[1] == "-50.553838" and out[2].split()[1] == "-1.404341"
    assert "rule on (0.3, 0.6, 0.1): 644 steps, return 1.064150" in out
    assert out[-1] == "  interval 4 round 2 committed 97 overshoot 9 n_active 106 skipped 9"
    with pytest.raises(SystemExit):
        main([])


def test_the_module_imports_no_library_that_trains_at_its_top():
    tree = ast.parse(Path(round_sanity.__file__).read_text(encoding="utf-8"))
    names = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    assert names == {
        "__future__", "argparse", "csv", "math", "collections.abc", "pathlib", "typing", "numpy",
        "agent.round_mask", "agent.round_observation",
    }


# --- A run measured ------------------------------------------------------------------

@pytest.fixture(scope="module")
def run(tmp_path_factory) -> SimpleNamespace:
    """One run trained on the two-cell instance and measured on its one
    initial condition; no test changes what it wrote."""
    folder = tmp_path_factory.mktemp("sanity").resolve()
    results = folder / "results" / "pytest_sanity_run"
    train(_small(), results, runs_dir=folder / "runs")
    return SimpleNamespace(results=results, rows=sanity(results, ics=[IC]))


def test_a_run_is_measured_policy_by_policy(run):
    assert [row["policy"] for row in run.rows] == [
        "hold", "rule", "random", "step_0", "checkpoint_32", "checkpoint_64", "checkpoint_96", "final",
    ]
    assert [row["timesteps"] for row in run.rows] == ["", "", "", 0, 32, 64, 96, 96]
    assert [row["episodes"] for row in run.rows] == [1, 1, len(RANDOM_SEEDS), 1, 1, 1, 1, 1]
    assert list(run.rows[0]) == ["policy", "timesteps", "episodes", *MEASURES, *(f"{key}_w0.2" for key in BY_WIDTH)]
    for row in run.rows:
        assert row["ret"] == pytest.approx(row["local"] + row["global"])
        assert row["ret"] == row["ret_w0.2"]
    # All holds on the instance: 8 steps (the explanation file's 3.1.2).
    assert (run.rows[0]["steps"], run.rows[0]["ret"]) == (8, pytest.approx(-6.017878 - 11.357170, abs=1e-6))


def test_the_rows_are_written_as_they_are_returned(run):
    with (run.results / "sanity.csv").open(newline="", encoding="utf-8") as file:
        written = list(csv.DictReader(file))
    assert [row["policy"] for row in written] == [row["policy"] for row in run.rows]
    assert list(written[0]) == list(run.rows[0])
    for mine, theirs in zip(written, run.rows, strict=True):
        assert mine["ret"] == str(theirs["ret"])
        assert mine["first_choice_closed"] == ("" if theirs["first_choice_closed"] is None else str(
            theirs["first_choice_closed"]))
    assert len(table(run.rows).splitlines()) == 9


def test_a_models_rows_count_its_first_choice_and_a_baselines_do_not(run):
    by_name = {row["policy"]: row for row in run.rows}
    assert by_name["hold"]["first_choice_closed"] is None and by_name["random"]["first_choice_closed"] is None
    for name in ("rule", "step_0", "checkpoint_32", "checkpoint_64", "checkpoint_96", "final"):
        assert 0 <= by_name[name]["first_choice_closed"] <= by_name[name]["steps"], name


def test_a_models_first_choice_is_its_policy_without_the_mask(run, monkeypatch):
    """What sanity hands evaluate for a model: the masked policy to play,
    the unmasked one to count. Each policy is tagged where it is made and
    read where it is handed over; the episodes are played as usual."""
    make, real_evaluate, seen = round_sanity.model_policy, round_sanity.evaluate, []

    def tagged(model, masked=True):
        policy = make(model, masked)
        policy.masked = masked
        return policy

    def watched(config, make_policy, ics=ICS, first_choice=None):
        seen.append((getattr(make_policy(), "masked", None), getattr(first_choice, "masked", None)))
        return real_evaluate(config, make_policy, ics, first_choice)

    monkeypatch.setattr(round_sanity, "model_policy", tagged)
    monkeypatch.setattr(round_sanity, "evaluate", watched)
    rows = sanity(run.results, ics=[IC])
    # The untrained policy, three checkpoints and the final model; the
    # baselines' policies carry no tag.
    assert [pair for pair in seen if pair != (None, None)] == [(True, False)] * 5
    assert len(seen) == 5 + 2 + len(RANDOM_SEEDS)
    assert [row for row in rows if row["policy"] != "random"] == [row for row in run.rows if row["policy"] != "random"]


def test_the_final_model_is_the_last_checkpoint_and_a_second_look_sees_the_same(run):
    by_name = {row["policy"]: row for row in run.rows}
    same = {key: value for key, value in by_name["final"].items() if key != "policy"}
    assert same == {key: value for key, value in by_name["checkpoint_96"].items() if key != "policy"}
    again = sanity(run.results, ics=[IC])
    assert [row for row in again if row["policy"] != "random"] == [
        row for row in run.rows if row["policy"] != "random"
    ]


def test_a_runs_models_are_found_in_step_order(run):
    assert [(name, path.name) for name, path in checkpoints(run.results)] == [
        ("checkpoint_32", "round_32_steps.zip"), ("checkpoint_64", "round_64_steps.zip"),
        ("checkpoint_96", "round_96_steps.zip"), ("final", "final_model.zip"),
    ]


def test_a_model_chooses_an_open_action_and_may_first_choose_a_closed_one(run):
    env, _ = task_env(_small())
    model = load_checked(run.results / "final_model.zip", env)
    obs, _ = env.reset(options={"ic": IC})
    mask = env.action_masks()
    assert mask.tolist() == [False, True, True]
    assert mask[model_policy(model)(obs, mask)]
    assert model_policy(model, masked=False)(obs, mask) in (0, 1, 2)


def test_a_checkpoint_saved_with_another_spec_is_refused(run, tmp_path):
    copy = tmp_path / "copy"
    shutil.copytree(run.results, copy)
    (copy / "sanity.csv").unlink()
    spec_file = copy / "checkpoints" / "round_64_steps.spec.json"
    spec = json.loads(spec_file.read_text(encoding="utf-8"))
    spec["beta"] = 3.0
    spec_file.write_text(json.dumps(spec), encoding="utf-8")
    with pytest.raises(ValueError, match="spec: round_64_steps.zip was saved with another spec"):
        sanity(copy, ics=[IC])
    assert not (copy / "sanity.csv").exists()


def test_a_checkpoint_with_no_spec_is_not_loaded(run, tmp_path):
    copy = tmp_path / "copy"
    shutil.copytree(run.results, copy)
    (copy / "final_model.spec.json").unlink()
    env, _ = task_env(_small())
    with pytest.raises(FileNotFoundError):
        load_checked(copy / "final_model.zip", env)
