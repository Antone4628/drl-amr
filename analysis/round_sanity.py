"""The sanity task of the 2D stack, measured: what a policy does on fixed
initial conditions, beside three policies that need no training.

AGENT_2D_ROADMAP item 3.2 trains MaskablePPO on the mock backend
(configs/round_sanity.yaml) and asks three things of the result: that
returns move, that masks are respected, and that the overshoot is reported.
A run's own monitor file cannot answer the first, since an episode's return
depends mostly on the feature width it happened to draw. This module plays
every policy on the same nine initial conditions, ICS, and reports:

    return              the episode's return, with its local and global parts
    n_end, n_max        the element count after the last round, and the largest
    rounds_over         rounds that ended above the element budget, which
                        nothing enforces
    overshoot           elements realized above the committed count, summed
                        over the episode's rounds
    coarsen_closed, refine_closed
                        steps at which the mask closed that action
    first_choice_closed steps at which the policy's choice without the mask
                        was a closed action: how often the mask did work
    contrast            level_contrast, averaged over the intervals

The policies: a checkpoint's (the action of highest probability among the
open ones); the same network before any training; and three baselines,
hold, random_masked and threshold_rule.

Nothing learned or measured on the mock's synthetic errors is a claim about
mesh adaptation. This is an instrument for the plumbing.

Everything here but the loading of a checkpoint is numpy. The training
entry and the RL libraries are imported inside the functions that need
them.

Usage:
    python -m analysis.round_sanity --config configs/round_sanity.yaml
    python -m analysis.round_sanity results/round_sanity_s0 results/round_sanity_s1
"""
from __future__ import annotations

import argparse
import csv
import math
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from agent.round_mask import ACTION_COARSEN, ACTION_HOLD, ACTION_REFINE
from agent.round_observation import ACTOR_KEY, feature_names

# A policy here: the observation dict and the mask in, an action index out.
Policy = Callable[[dict[str, np.ndarray], np.ndarray], int]

# The fixed initial conditions: three widths by three starts, none of them
# on a centroid of the default mesh. Inside the mock's bounds on the unit
# square.
ICS: tuple[dict[str, float], ...] = tuple(
    {"x0": x0, "y0": y0, "width": width}
    for width in (0.05, 0.10, 0.15)
    for x0, y0 in ((0.30, 0.60), (0.55, 0.20), (0.85, 0.45))
)
# level_contrast's two sets, in feature widths from the feature's center.
NEAR, FAR = 2.0, 3.5
# The random baseline is the mean over these seeds.
RANDOM_SEEDS = (0, 1, 2, 3, 4)

# What summarize averages, in the order sanity.csv holds them.
MEASURES = (
    "ret", "local", "global", "steps", "n_end", "n_max", "rounds_over", "overshoot", "coarsen_closed",
    "refine_closed", "first_choice_closed", "contrast",
)
# What is also reported width by width.
BY_WIDTH = ("ret", "n_end", "rounds_over", "overshoot", "contrast")


# --- The tracking measure ------------------------------------------------------------

def level_contrast(
    state: Any, ic: dict[str, float], velocity: Sequence[float], near: float = NEAR, far: float = FAR
) -> float:
    """How much finer the mesh is near the feature than far from it.

    The mean level of the elements whose centroid lies within near widths
    of the feature's center, minus the mean level of those beyond far
    widths. The center is where the feature is at the state's time,
    ic's start moved by the velocity; on a periodic axis the distance is to
    its nearest image. Geometry alone: no indicator and no reward is read.

    0 on a uniform mesh, and max_level when the near set is at the finest
    level and the far set at the base level. Two dimensions only.

    Args:
        state: A v2 SolverState: centroid, level, time, the domain and
            which axes are periodic.
        ic: The feature's x0, y0 and width.
        velocity: The feature's velocity, (vx, vy).
        near: The near set's radius, in widths.
        far: The far set's inner radius, in widths.

    Returns:
        The difference, or nan when either set is empty.
    """
    delta = np.empty(state.centroid.shape, dtype=np.float64)
    for axis, name in enumerate(("x0", "y0")):
        delta[:, axis] = state.centroid[:, axis] - (ic[name] + velocity[axis] * state.time)
        if state.periodic[axis]:
            length = state.domain_hi[axis] - state.domain_lo[axis]
            delta[:, axis] = (delta[:, axis] + length / 2) % length - length / 2
    distance = np.hypot(delta[:, 0], delta[:, 1]) / ic["width"]
    inside, outside = distance < near, distance > far
    if not inside.any() or not outside.any():
        return math.nan
    return float(state.level[inside].mean() - state.level[outside].mean())


# --- The policies --------------------------------------------------------------------

def hold() -> Policy:
    """The policy that holds every element: what doing nothing returns."""
    return lambda obs, mask: ACTION_HOLD


def random_masked(seed: int) -> Policy:
    """A uniform choice among the open actions, from a generator seeded here."""
    rng = np.random.default_rng(seed)
    return lambda obs, mask: int(rng.choice(np.flatnonzero(mask)))


def zone_decades(config: dict[str, dict[str, Any]]) -> float:
    """The neutral zone's width under relative: (beta - 1) * -log10(alpha)."""
    return (config["decision"]["beta"] - 1.0) * -math.log10(config["observation"]["alpha"])


def threshold_rule(zone: float, ndim: int = 2, masked: bool = True) -> Policy:
    """A rule read from two features of the actor's observation.

    Refine where the element's own error feature is above 0, which under
    relative is above the round's e_max; coarsen where the family's largest
    is below -zone, which is below the round's e_min; hold otherwise. The
    round's thresholds are not the interval's, so this is a reference for
    what a simple reading of the observation earns, not the reward's
    optimum.

    Args:
        zone: The neutral zone's width in decades, zone_decades.
        ndim: The mesh dimension, for the feature names.
        masked: If False the rule ignores the mask: its first choice.
    """
    names = feature_names(ndim)
    own, family = names.index("error"), names.index("error_family_max")

    def policy(obs: dict[str, np.ndarray], mask: np.ndarray) -> int:
        features = obs[ACTOR_KEY]
        if features[own] > 0 and (mask[ACTION_REFINE] or not masked):
            return ACTION_REFINE
        if features[family] < -zone and (mask[ACTION_COARSEN] or not masked):
            return ACTION_COARSEN
        return ACTION_HOLD

    return policy


def model_policy(model: Any, masked: bool = True) -> Policy:
    """A MaskablePPO model's action of highest probability.

    Args:
        model: The model; only its predict is called.
        masked: If True, among the open actions. If False, among all
            three: the model's first choice, which the mask may close.
    """

    def policy(obs: dict[str, np.ndarray], mask: np.ndarray) -> int:
        action, _ = model.predict(obs, deterministic=True, action_masks=mask if masked else None)
        return int(action)

    return policy


# --- One episode ---------------------------------------------------------------------

def play(
    env: Any, policy: Policy, ic: dict[str, float], velocity: Sequence[float], first_choice: Policy | None = None
) -> dict[str, Any]:
    """Play one episode on a forced initial condition and return its record.

    Args:
        env: A RoundEnv, not wrapped.
        policy: Chooses each action; it is given the mask and must choose
            an open action.
        ic: The initial condition, forced through reset's options.
        velocity: The feature's velocity, for level_contrast.
        first_choice: The same policy without the mask, or None. Where its
            choice is a closed action, first_choice_closed counts it.

    Returns:
        The measures of MEASURES for this episode (first_choice_closed is
        None without first_choice); the initial condition; the action
        counts coarsens, holds and refines; rounds_overshot, the rounds
        with an overshoot; failed, whether a backend failure ended it; and
        rounds, one dict per round ended: interval, round, committed,
        overshoot, n_active, skipped.
    """
    budget = env.observation_spec.observation.element_budget
    obs, info = env.reset(options={"ic": ic})
    record: dict[str, Any] = {
        **ic, "ret": 0.0, "local": 0.0, "global": 0.0, "steps": 0, "n_end": info["n_active"],
        "n_max": info["n_active"], "rounds_over": 0, "overshoot": 0, "rounds_overshot": 0, "coarsens": 0,
        "holds": 0, "refines": 0, "coarsen_closed": 0, "refine_closed": 0,
        "first_choice_closed": None if first_choice is None else 0, "failed": False, "rounds": [],
    }
    counted = {ACTION_COARSEN: "coarsens", ACTION_HOLD: "holds", ACTION_REFINE: "refines"}
    contrasts = []
    terminated = False
    while not terminated:
        mask = env.action_masks()
        record["coarsen_closed"] += int(not mask[ACTION_COARSEN])
        record["refine_closed"] += int(not mask[ACTION_REFINE])
        if first_choice is not None and not mask[first_choice(obs, mask)]:
            record["first_choice_closed"] += 1
        action = policy(obs, mask)
        obs, reward, terminated, _, info = env.step(action)
        record[counted[action]] += 1
        record["steps"] += 1
        record["ret"] += reward
        record["local"] += env.lambda_local * info["r_local"]
        record["global"] += env.lambda_global * info["r_global"]
        if "n_active" in info:
            record["rounds"].append({
                "interval": info["interval"], "round": info["round_index"],
                "committed": info["n_active"] - info["overshoot"], "overshoot": info["overshoot"],
                "n_active": info["n_active"], "skipped": info["skipped"],
            })
            record["n_end"] = info["n_active"]
            record["n_max"] = max(record["n_max"], info["n_active"])
            record["rounds_over"] += int(info["n_active"] > budget)
            record["overshoot"] += info["overshoot"]
            record["rounds_overshot"] += int(info["overshoot"] > 0)
        if info["transition"] in ("interval", "done"):
            contrasts.append(level_contrast(env.backend.get_state(), ic, velocity))
        record["failed"] = info["transition"] == "failure"
    known = [value for value in contrasts if not math.isnan(value)]
    record["contrast"] = sum(known) / len(known) if known else math.nan
    return record


def evaluate(
    config: dict[str, dict[str, Any]],
    make_policy: Callable[[], Policy],
    ics: Sequence[dict[str, float]] = ICS,
    first_choice: Policy | None = None,
) -> list[dict[str, Any]]:
    """Play every initial condition once, each on an environment and with a
    policy made anew.

    The mock's element order moves with every adapt and reset of one
    backend, and the queue breaks ties by position. On a fresh environment
    an episode's record does not depend on which episodes were played
    before it.
    """
    records = []
    for ic in ics:
        env, velocity = task_env(config)
        records.append(play(env, make_policy(), ic, velocity, first_choice))
    return records


def summarize(records: Sequence[dict[str, Any]]) -> dict[str, float | None]:
    """The mean of each measure over episodes.

    contrast is averaged over the episodes that have one; a measure no
    episode has (first_choice_closed without a first choice) is None.
    """
    summary: dict[str, float | None] = {}
    for key in MEASURES:
        values = [record[key] for record in records if record[key] is not None and not _is_nan(record[key])]
        summary[key] = sum(values) / len(values) if values else None
    return summary


def _is_nan(value: object) -> bool:
    return isinstance(value, float) and math.isnan(value)


def by_width(records: Sequence[dict[str, Any]]) -> dict[float, dict[str, float | None]]:
    """summarize, for the episodes of each feature width, widths ascending."""
    widths = sorted({record["width"] for record in records})
    return {width: summarize([record for record in records if record["width"] == width]) for width in widths}


# --- The task's environment, the baselines, a run's checkpoints ----------------------

def task_env(config: dict[str, dict[str, Any]]) -> tuple[Any, tuple[float, ...]]:
    """The environment a configuration builds, unwrapped, and the feature's velocity."""
    from training.train_round import build_env

    return build_env(config).unwrapped, tuple(config["backend"]["velocity"])


def baselines(
    config: dict[str, dict[str, Any]], ics: Sequence[dict[str, float]] = ICS
) -> dict[str, list[dict[str, Any]]]:
    """The three baselines' records: hold, rule, and random over RANDOM_SEEDS.

    The rule's first choice is the rule without the mask. The random
    baseline's records are every seed's, one after another.
    """
    zone = zone_decades(config)
    played = {
        "hold": evaluate(config, hold, ics),
        "rule": evaluate(config, lambda: threshold_rule(zone), ics, threshold_rule(zone, masked=False)),
        "random": [],
    }
    for seed in RANDOM_SEEDS:
        # One generator per episode, seeded alike, for the same reason as
        # one environment per episode.
        played["random"] += evaluate(config, lambda seed=seed: random_masked(seed), ics)
    return played


def checkpoints(results_dir: str | Path) -> list[tuple[str, Path]]:
    """A run's saved models: its checkpoints by step count, then the final model."""
    results = Path(results_dir)
    found = sorted(
        (results / "checkpoints").glob("round_*_steps.zip"), key=lambda path: int(path.stem.split("_")[1])
    )
    named = [(f"checkpoint_{path.stem.split('_')[1]}", path) for path in found]
    if (results / "final_model.zip").is_file():
        named.append(("final", results / "final_model.zip"))
    return named


def load_checked(path: str | Path, env: Any) -> Any:
    """Load a checkpoint whose spec is the environment's.

    Raises:
        FileNotFoundError: If the checkpoint has no spec beside it.
        ValueError: Starting with "spec", if the spec beside the checkpoint
            is not the one the run's configuration builds: its policy would
            be fed something else here.
    """
    from sb3_contrib import MaskablePPO

    from agent.round_spec import load_spec

    if load_spec(path) != env.observation_spec:
        raise ValueError(f"spec: {Path(path).name} was saved with another spec than the run's configuration builds")
    return MaskablePPO.load(path, device="cpu")


def _row(policy: str, timesteps: int | str, records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """One row of sanity.csv: the means, then the means width by width."""
    row: dict[str, Any] = {"policy": policy, "timesteps": timesteps, "episodes": len(records), **summarize(records)}
    for width, summary in by_width(records).items():
        for key in BY_WIDTH:
            row[f"{key}_w{width:g}"] = summary[key]
    return row


def sanity(results_dir: str | Path, ics: Sequence[dict[str, float]] = ICS) -> list[dict[str, Any]]:
    """Measure a run: the baselines, its untrained policy and every saved model.

    Reads the run's config.yaml, so the environment is the one the run
    trained on. Writes sanity.csv into the run's folder, one row per
    policy, and returns the rows: hold, rule, random, step_0, each
    checkpoint by its step count, final.

    Raises:
        ValueError: As load_checked raises, before anything is written.
    """
    from training.train_round import build_env, build_model, load_config

    results = Path(results_dir)
    config = load_config(results / "config.yaml")
    env, _ = task_env(config)
    models = [("step_0", build_model(build_env(config), config))]
    models += [(name, load_checked(path, env)) for name, path in checkpoints(results)]
    rows = [_row(name, "", records) for name, records in baselines(config, ics).items()]
    for name, model in models:
        first_choice = model_policy(model, masked=False)
        records = evaluate(config, lambda model=model: model_policy(model), ics, first_choice)
        rows.append(_row(name, model.num_timesteps, records))
    with (results / "sanity.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


# --- The command line ----------------------------------------------------------------

def _number(value: float | None, digits: int = 3) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def table(rows: Sequence[dict[str, Any]]) -> str:
    """The rows as text: one line per policy, the columns a reader compares."""
    lines = [
        f"{'policy':<18}{'timesteps':>10}{'return':>12}{'contrast':>10}{'n_end':>8}{'over/8':>8}{'overshoot':>11}"
        f"{'c_closed':>10}{'r_closed':>10}{'first_closed':>14}"
    ]
    for row in rows:
        lines.append(
            f"{row['policy']:<18}{str(row['timesteps']):>10}{_number(row['ret'], 6):>12}{_number(row['contrast']):>10}"
            f"{_number(row['n_end'], 1):>8}{_number(row['rounds_over'], 2):>8}{_number(row['overshoot'], 1):>11}"
            f"{_number(row['coarsen_closed'], 1):>10}{_number(row['refine_closed'], 1):>10}"
            f"{_number(row['first_choice_closed'], 1):>14}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    """Print the baselines of a configuration, or measure runs and print their tables."""
    parser = argparse.ArgumentParser(description="The sanity task of the 2D stack, on fixed initial conditions.")
    parser.add_argument("runs", nargs="*", help="Run folders, each with config.yaml and its saved models.")
    parser.add_argument("--config", type=str, default=None,
                        help="A YAML file: print its baselines and the rule's rounds on one IC. Loads no model.")
    args = parser.parse_args(argv)
    if args.config is None and not args.runs:
        parser.error("give run folders, or --config")
    if args.config is not None:
        from training.train_round import load_config

        config = load_config(args.config)
        played = baselines(config)
        print(table([_row(name, "", records) for name, records in played.items()]))
        per_seed = len(ICS)
        for index, seed in enumerate(RANDOM_SEEDS):
            summary = summarize(played["random"][index * per_seed:(index + 1) * per_seed])
            print(f"random, seed {seed}: return {summary['ret']:.3f}, contrast {summary['contrast']:.3f}, "
                  f"n_end {summary['n_end']:.1f}")
        for name in ("hold", "rule"):
            print(f"{name}, by IC: " + " ".join(f"{record['ret']:.3f}" for record in played[name]))
            print(f"{name}, steps: " + " ".join(str(record["steps"]) for record in played[name]))
        traced = played["rule"][3]
        print(f"rule on ({traced['x0']}, {traced['y0']}, {traced['width']}): {traced['steps']} steps, "
              f"return {traced['ret']:.6f}")
        for entry in traced["rounds"]:
            print("  " + " ".join(f"{key} {value}" for key, value in entry.items()))
    for run in args.runs:
        print(f"{run}:")
        print(table(sanity(run)))


if __name__ == "__main__":
    main()
