"""Training entry of the 2D stack: MaskablePPO on RoundEnv, from one configuration.

A run is built from seven sections of values, DEFAULTS below. A YAML file
overrides any of them, and the command line a few more:

    backend         which backend, and what it is built with
    observation     the ObservationConfig (agent/round_observation.py)
    decision        beta and the coarsen rule
    episode         intervals, base steps per interval, the starting level
    reward          the three weights and the two lambdas
    training        MaskablePPO's values, the network sizes, the seed
    checkpointing   steps between checkpoints

Every checkpoint is written with its observation spec beside it
(agent/round_spec.py), by one helper, so no weights exist on disk without
what their inputs were built with.

This module belongs to the 2D stack (AGENT_2D_ROADMAP item 3.1). It stands
beside training/train_multiround.py, the sequential entry, and imports
nothing from it.

Usage:
    python -m training.train_round
    python -m training.train_round --config my_run.yaml
    python -m training.train_round --timesteps 4096 --seed 0 --save-freq 1024 \\
        --results-dir results/smoke_round
"""
from __future__ import annotations

import argparse
import copy
import csv
import datetime
from pathlib import Path
from typing import Any

import yaml
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor

from agent.round_observation import ObservationConfig
from agent.round_spec import ObservationSpec, save_spec
from backends.mock_quad.backend import MockQuadBackend
from contract.backend import SolverBackend
from envs.round_env import RoundEnv
from tools.run_manifest import write_manifest
from training.round_policy import RoundPolicy

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# The one source of a run's defaults. A value's kind here (true or false, an
# integer, a number, a name, a list of one of them) is the kind load_config
# accepts for its key. max_level is written once, under backend: the
# environment is given the same value.
DEFAULTS: dict[str, dict[str, Any]] = {
    "backend": {
        "kind": "mock_quad",
        "nx": 4,
        "ny": 4,
        "periodic": [True, True],
        "lo": [0.0, 0.0],
        "hi": [1.0, 1.0],
        "max_level": 2,
        "nvar": 1,
        "dt_base": 0.01,
        "velocity": [1.0, 0.0],
        "rotate_cells": True,
    },
    "observation": {
        "indicator": "ind_b",
        "reduction": "scalar",
        "alpha": 0.1,
        "element_budget": 64,
        # The mock run's plumbing value, not the 2D default: that is undecided.
        "normalization": "relative",
        "floor_decades": 8.0,
    },
    "decision": {
        "beta": 1.2,
        "coarsen_rule": "first_decider",
    },
    "episode": {
        "n_intervals": 4,
        "amr_freq": 5,
        "initial_level": 1,
        "cadence": 1,
    },
    "reward": {
        "p_ur": 10.0,
        "p_or": 5.0,
        "p_cr": 2.0,
        "lambda_local": 0.1,
        "lambda_global": 1.0,
    },
    "training": {
        "total_timesteps": 100_000,
        "learning_rate": 3e-4,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        # One rollout holds a whole episode: about 800 steps on the default mesh.
        "n_steps": 1024,
        "batch_size": 64,
        "n_epochs": 10,
        "ent_coef": 0.01,
        "clip_range": 0.2,
        "net_arch": [256, 256],
        "seed": 42,
        "device": "auto",
    },
    "checkpointing": {
        "save_freq": 10_000,
    },
}


# --- The configuration ---------------------------------------------------------------

def _kind(default: object) -> str:
    """The kind of value a default stands for, in words, for a refusal."""
    if isinstance(default, bool):
        return "true or false"
    if isinstance(default, int):
        return "an integer"
    if isinstance(default, float):
        return "a number"
    if isinstance(default, str):
        return "a name"
    return f"a list, each item {_kind(default[0])}"


def _fits(value: object, default: object) -> bool:
    """Whether a value is of its default's kind.

    An integer fits a number; true and false fit neither. A list fits when
    it is not empty and every item fits the default's first item; its
    length is not compared.
    """
    if isinstance(default, (bool, str)):
        return type(value) is type(default)
    if isinstance(default, int):
        return type(value) is int
    if isinstance(default, float):
        return type(value) in (int, float)
    return isinstance(value, list) and len(value) > 0 and all(_fits(item, default[0]) for item in value)


def load_config(path: str | Path | None = None) -> dict[str, dict[str, Any]]:
    """A run's configuration: the defaults, with a YAML file's values over them.

    The file holds sections of DEFAULTS and, in each, keys of that section.
    A value replaces its default whole, a list included. Anything the
    defaults do not have is refused, and so is a value of another kind than
    its default: YAML reads 3e-4 as a string (a float needs its dot,
    3.0e-4), and that string would otherwise reach the optimizer.

    Args:
        path: The YAML file, or None for the defaults alone. An empty file
            is the defaults too.

    Returns:
        A new dict that shares nothing with DEFAULTS.

    Raises:
        ValueError: Starting with "config" for the file or a section name,
            or "config[<section>]" for a key or its value.
    """
    config = copy.deepcopy(DEFAULTS)
    if path is None:
        return config
    given = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if given is None:
        return config
    if not isinstance(given, dict):
        raise ValueError(f"config: expected sections, got {type(given).__name__}")
    unknown = sorted(str(section) for section in given if section not in DEFAULTS)
    if unknown:
        raise ValueError(f"config: unknown sections {unknown}")
    for section, values in given.items():
        if not isinstance(values, dict):
            raise ValueError(f"config[{section}]: expected keys, got {type(values).__name__}")
        unknown = sorted(str(key) for key in values if key not in DEFAULTS[section])
        if unknown:
            raise ValueError(f"config[{section}]: unknown keys {unknown}")
        for key, value in values.items():
            default = DEFAULTS[section][key]
            if not _fits(value, default):
                raise ValueError(f"config[{section}]: {key}: {value!r}, expected {_kind(default)}")
            config[section][key] = value
    return config


# --- The backend, the environment, the model -------------------------------------------

def build_backend(config: dict[str, dict[str, Any]]) -> MockQuadBackend:
    """The backend a configuration names, built with its backend section.

    The four pairs (periodic, lo, hi, velocity) are lists in YAML and
    tuples in the backend's constructor.

    Raises:
        ValueError: Starting with "config[backend]", for a kind that is not
            mock_quad or a pair that is not two items; or as the backend
            raises.
    """
    values = dict(config["backend"])
    kind = values.pop("kind")
    if kind != "mock_quad":
        raise ValueError(f"config[backend]: kind: {kind!r}, expected 'mock_quad'")
    for key in ("periodic", "lo", "hi", "velocity"):
        if len(values[key]) != 2:
            raise ValueError(f"config[backend]: {key}: {values[key]!r}, expected 2 items")
        values[key] = tuple(values[key])
    return MockQuadBackend(**values)


def build_env(
    config: dict[str, dict[str, Any]], backend: SolverBackend | None = None, monitor_dir: str | Path | None = None
) -> Monitor:
    """The training environment: RoundEnv in a Monitor, and nothing else.

    No wrapper is needed for the masks: MaskablePPO finds RoundEnv's own
    action_masks through the Monitor.

    Args:
        config: A run's configuration.
        backend: The backend to drive; built from the configuration when
            None. A test hands one that fails on demand.
        monitor_dir: A folder for the Monitor's episode log, monitor.csv,
            made if it is missing; None for no log.

    Raises:
        ValueError: As build_backend, ObservationConfig and RoundEnv raise.
    """
    if backend is None:
        backend = build_backend(config)
    # The episode and reward sections hold RoundEnv's argument names, so a
    # key with no taker there is a TypeError, not a value silently dropped.
    env = RoundEnv(
        backend,
        observation_config=ObservationConfig(**config["observation"]),
        beta=config["decision"]["beta"],
        coarsen_rule=config["decision"]["coarsen_rule"],
        max_level=config["backend"]["max_level"],
        **config["episode"],
        **config["reward"],
    )
    if monitor_dir is None:
        return Monitor(env)
    # Monitor writes monitor.csv inside a folder only if the folder exists.
    Path(monitor_dir).mkdir(parents=True, exist_ok=True)
    return Monitor(env, filename=str(monitor_dir))


def build_model(env: Monitor, config: dict[str, dict[str, Any]], verbose: int = 0) -> MaskablePPO:
    """MaskablePPO with the routed policy, from the training section.

    The policy is given net_arch alone, so that is all a saved model holds
    beside the policy's class. The seed reaches the environment here and is
    used by its first reset.
    """
    training = config["training"]
    return MaskablePPO(
        RoundPolicy,
        env,
        learning_rate=training["learning_rate"],
        n_steps=training["n_steps"],
        batch_size=training["batch_size"],
        n_epochs=training["n_epochs"],
        gamma=training["gamma"],
        gae_lambda=training["gae_lambda"],
        ent_coef=training["ent_coef"],
        clip_range=training["clip_range"],
        policy_kwargs={"net_arch": list(training["net_arch"])},
        seed=training["seed"],
        device=training["device"],
        verbose=verbose,
    )


# --- Checkpoints -------------------------------------------------------------------------

def save_checkpoint(model: MaskablePPO, spec: ObservationSpec, path: str | Path) -> Path:
    """Save a model with its observation spec beside it.

    The one way this module writes weights. The spec is written first, so
    at no moment does a zip lie on disk without its spec; the folder is
    made by that write.

    The zip is named here, not by stable-baselines3, which appends .zip
    only to a path with no suffix at all: a stem such as run_0.5 would be
    saved under that name, with no .zip.

    Args:
        model: The model to save.
        spec: What its policy was trained with: the environment's
            observation_spec.
        path: The checkpoint's path, with or without .zip.

    Returns:
        The zip's path. The spec is at agent.round_spec.spec_path of it.
    """
    path = Path(path)
    if path.suffix != ".zip":
        path = path.with_name(path.name + ".zip")
    save_spec(spec, path)
    model.save(path)
    return path


class SpecCheckpointCallback(BaseCallback):
    """Save a checkpoint with its spec every save_freq steps.

    It stands where stable-baselines3's CheckpointCallback stands and names
    its files the same way, <prefix>_<steps>_steps.zip. That callback saves
    the zip inside its own step, so a spec cannot be paired with it from
    outside; this one saves through save_checkpoint.

    Args:
        save_freq: Steps of the environment between checkpoints.
        save_dir: The folder the checkpoints go to.
        spec: The run's observation spec, written beside every zip.
        name_prefix: The first part of every file's name.

    Attributes:
        saved: The zips written so far, in order.

    Raises:
        ValueError: Naming save_freq.
    """

    def __init__(
        self, save_freq: int, save_dir: str | Path, spec: ObservationSpec, name_prefix: str = "round"
    ) -> None:
        super().__init__()
        if isinstance(save_freq, bool) or not isinstance(save_freq, int) or save_freq < 1:
            raise ValueError(f"save_freq: {save_freq!r}, expected an integer of at least 1")
        self.save_freq = save_freq
        self.save_dir = Path(save_dir)
        self.spec = spec
        self.name_prefix = name_prefix
        self.saved: list[Path] = []

    def _on_step(self) -> bool:
        """Save when the step count reaches a multiple of save_freq; never stop training."""
        if self.n_calls % self.save_freq == 0:
            name = f"{self.name_prefix}_{self.num_timesteps}_steps"
            self.saved.append(save_checkpoint(self.model, self.spec, self.save_dir / name))
        return True


class RoundLogCallback(BaseCallback):
    """Write one row per adaptation round to a CSV file.

    The step that ends a round carries the round's counts in its info
    (envs/round_env.py): n_active, the realized count; overshoot, how far
    that is above the committed count; and skipped. This writes them as
    they come, with where in the run they came, so the counts can be set
    against the element budget, which nothing enforces.

    Columns: timesteps, the model's step count at the round's last step;
    episode, counted from 1; interval and round, of the round that ended;
    committed, n_active minus overshoot; overshoot; n_active; skipped.

    A round's row is written once its marks are applied. A backend failure
    in the adapt itself leaves no counts and no row; one after it (the
    interval's advance, or the next round's read) ends the episode at that
    step, and the round's row is still written. One environment is read,
    the first.

    Args:
        path: The file to write, replaced if it exists. Its folder must
            exist.

    Attributes:
        rows: Rows written so far, the header not counted.
    """

    COLUMNS = ("timesteps", "episode", "interval", "round", "committed", "overshoot", "n_active", "skipped")

    def __init__(self, path: str | Path) -> None:
        super().__init__()
        self.path = Path(path)
        self.rows = 0
        self._episode = 1
        self._file = None
        self._writer = None

    def _on_training_start(self) -> None:
        """Open the file and write the header."""
        self._file = self.path.open("w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        self._writer.writerow(self.COLUMNS)

    def _on_step(self) -> bool:
        """Write a row if this step ended a round; never stop training."""
        info = self.locals["infos"][0]
        if "n_active" in info:
            committed = info["n_active"] - info["overshoot"]
            self._writer.writerow(
                [self.num_timesteps, self._episode, info["interval"], info["round_index"], committed,
                 info["overshoot"], info["n_active"], info["skipped"]]
            )
            # A run that stops on an error still leaves its rows.
            self._file.flush()
            self.rows += 1
        if self.locals["dones"][0]:
            self._episode += 1
        return True

    def _on_training_end(self) -> None:
        """Close the file."""
        self._file.close()



# --- The run -----------------------------------------------------------------------------

def train(
    config: dict[str, dict[str, Any]], results_dir: str | Path, runs_dir: str | Path | None = None, verbose: int = 0
) -> MaskablePPO:
    """One training run, from a configuration to its saved final model.

    Written into results_dir: config.yaml, the configuration as resolved;
    monitor.csv, one row per episode; rounds.csv, one row per adaptation
    round; checkpoints/round_<steps>_steps.zip every save_freq steps and
    final_model.zip at the end, each with its spec beside it. Written into
    runs_dir: the run's manifest, <results_dir's name>.yaml
    (tools/run_manifest.py), before anything is built, so its git_dirty is
    the tree's state when the run began.

    Args:
        config: A run's configuration, as load_config returns it.
        results_dir: The run's folder, made if it is missing. Its name is
            the run's id.
        runs_dir: Where the manifest goes; the repository's runs/ when
            None.
        verbose: MaskablePPO's verbosity.

    Returns:
        The trained model.
    """
    results = Path(results_dir).resolve()
    results.mkdir(parents=True, exist_ok=True)
    config_path = results / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    runs = PROJECT_ROOT / "runs" if runs_dir is None else Path(runs_dir)
    training = config["training"]
    write_manifest(
        runs / f"{results.name}.yaml",
        config_path=config_path,
        seed=training["seed"],
        extra={
            "run_id": results.name,
            "results_dir": str(results),
            "total_timesteps": training["total_timesteps"],
            "indicator": config["observation"]["indicator"],
        },
    )

    env = build_env(config, monitor_dir=results)
    model = build_model(env, config, verbose=verbose)
    # The one spec of the run: the environment built it, every save writes it.
    spec = env.unwrapped.observation_spec
    checkpoints = SpecCheckpointCallback(config["checkpointing"]["save_freq"], results / "checkpoints", spec)
    rounds = RoundLogCallback(results / "rounds.csv")
    model.learn(total_timesteps=training["total_timesteps"], callback=CallbackList([checkpoints, rounds]))
    save_checkpoint(model, spec, results / "final_model")
    env.close()
    return model


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """The command line: a YAML file, four values over it, and where to write."""
    parser = argparse.ArgumentParser(description="Train MaskablePPO on RoundEnv, the 2D stack's training entry.")
    parser.add_argument("--config", type=str, default=None, help="A YAML file of overrides; the defaults if omitted.")
    parser.add_argument("--timesteps", type=int, default=None, help="Override training.total_timesteps.")
    parser.add_argument("--seed", type=int, default=None, help="Override training.seed.")
    parser.add_argument("--save-freq", type=int, default=None, help="Override checkpointing.save_freq.")
    parser.add_argument("--results-dir", type=str, default=None, help="Default: results/round_<timestamp>.")
    parser.add_argument("--device", type=str, default=None, choices=["auto", "cpu", "cuda"],
                        help="Override training.device.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Resolve the configuration (defaults, then the YAML, then the command line) and train."""
    args = parse_args(argv)
    config = load_config(args.config)
    overrides = (
        ("training", "total_timesteps", args.timesteps),
        ("training", "seed", args.seed),
        ("training", "device", args.device),
        ("checkpointing", "save_freq", args.save_freq),
    )
    for section, key, value in overrides:
        if value is not None:
            config[section][key] = value
    if args.results_dir is not None:
        results_dir = Path(args.results_dir)
    else:
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = PROJECT_ROOT / "results" / f"round_{stamp}"
    train(config, results_dir, verbose=1)
    print(f"results: {results_dir}")


if __name__ == "__main__":
    main()