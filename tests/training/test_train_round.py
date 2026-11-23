"""Tests for the training entry of the 2D stack, training/train_round.py:
one configuration, a run built from it, and no weights on disk without
their observation spec.

AGENT_2D_ROADMAP item 3.1, sitting 3, signals S7-S14; item 3.2, signals G1
and G6. Seven groups:

  - the configuration: the defaults, a YAML file over them, and what is
    refused by name (S7);
  - the backend, the environment and the model built from it;
  - the checkpoint helper and the callback: the spec first, the zip named
    here, every save_freq steps;
  - a run on the two-cell instance: what it writes (S8), every spec equal
    to the environment's (S9), both networks moved (S10), one seed one run
    (S11), the masks as what keeps it alive (S12), a checkpoint reloaded in
    a fresh interpreter (S13), training past a backend failure (S14);
  - the round log: one row per adaptation round, written once the round's
    marks are applied (G6);
  - the command line;
  - what the module imports.

Nothing here asserts a number that depends on the initial weights: the
checks are names, counts, shapes and exact equalities between two runs of
the same operations.

Run: pytest tests/training/test_train_round.py -v
"""
from __future__ import annotations

import ast
import copy
import csv
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch as th
import yaml
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.utils import is_masking_supported
from stable_baselines3.common.monitor import Monitor

import training.train_round as train_round
from agent.round_observation import ObservationConfig
from agent.round_spec import ObservationSpec, load_spec, spec_path
from backends.mock_quad.backend import MockQuadBackend
from envs.round_env import RoundEnv
from training.round_policy import RoundPolicy
from training.train_round import (
    DEFAULTS,
    PROJECT_ROOT,
    RoundLogCallback,
    SpecCheckpointCallback,
    build_backend,
    build_env,
    build_model,
    load_config,
    main,
    parse_args,
    save_checkpoint,
    train,
)

IC = {"x0": 0.3, "y0": 0.6, "width": 0.2}
# A run id no real run has, so a manifest that strays into the repository's
# runs/ is seen.
RUN = "pytest_round_run"


# --- The instance as a configuration, and what a run leaves ------------------------

def _small(seed: int = 0) -> dict:
    """The two-cell instance of the explanation file's section 3.1.2 as a
    run's configuration: three rollouts of 32 steps, a checkpoint at each."""
    config = load_config(None)
    config["backend"].update(nx=2, ny=1, periodic=[True, False], hi=[2.0, 1.0], dt_base=0.25, rotate_cells=False)
    config["observation"]["element_budget"] = 8
    config["decision"]["beta"] = 2
    config["episode"].update(n_intervals=2, amr_freq=2, initial_level=0)
    config["training"].update(
        total_timesteps=96, n_steps=32, batch_size=16, n_epochs=2, net_arch=[16, 16], seed=seed, device="cpu"
    )
    config["checkpointing"]["save_freq"] = 32
    return config


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "run.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def _files(folder: Path) -> set[str]:
    return {path.relative_to(folder).as_posix() for path in folder.rglob("*") if path.is_file()}


def _episodes(results: Path) -> list[tuple[str, str]]:
    """The return and the length of every episode in a run's monitor file,
    as written. The first line of the file is a comment."""
    lines = (results / "monitor.csv").read_text(encoding="utf-8").splitlines()
    return [(row["r"], row["l"]) for row in csv.DictReader(lines[1:])]


def _parameters(model: MaskablePPO) -> dict[str, th.Tensor]:
    return {name: parameter.detach().clone() for name, parameter in model.policy.named_parameters()}


def _count(model: MaskablePPO) -> int:
    return sum(parameter.numel() for parameter in model.policy.parameters())


def _log_probs(model: MaskablePPO, obs: dict[str, list[float]]) -> list[float]:
    """The log-probability of each action on one observation, unmasked."""
    tensors = {key: th.as_tensor(np.asarray([value], dtype=np.float32)) for key, value in obs.items()}
    model.policy.set_training_mode(False)
    with th.no_grad():
        distribution = model.policy.get_distribution(tensors)
        return [float(distribution.log_prob(th.full((1,), action))[0]) for action in range(3)]


def _env_spec(model: MaskablePPO) -> ObservationSpec:
    return model.get_env().envs[0].unwrapped.observation_spec


@pytest.fixture(scope="module")
def run(tmp_path_factory) -> SimpleNamespace:
    """One run of train on the instance, shared by the tests of its group;
    none of them changes what it wrote."""
    folder = tmp_path_factory.mktemp("round").resolve()
    config = _small()
    model = train(config, folder / "results" / RUN, runs_dir=folder / "runs")
    return SimpleNamespace(model=model, config=config, results=folder / "results" / RUN, runs=folder / "runs")


# --- The configuration (S7) --------------------------------------------------------

def test_the_defaults_are_seven_sections():
    assert list(DEFAULTS) == ["backend", "observation", "decision", "episode", "reward", "training", "checkpointing"]
    assert [len(DEFAULTS[section]) for section in DEFAULTS] == [11, 6, 2, 4, 5, 12, 1]


def test_max_level_is_written_once():
    assert [section for section in DEFAULTS if "max_level" in DEFAULTS[section]] == ["backend"]


def test_the_training_defaults_are_the_sequential_stacks_but_for_the_rollout():
    from training.train_multiround import DEFAULT_CONFIG

    ours, theirs = dict(DEFAULTS["training"]), dict(DEFAULT_CONFIG["training"])
    assert (ours.pop("n_steps"), theirs.pop("n_steps")) == (1024, 256)
    assert ours == theirs


def test_no_file_gives_the_defaults_and_shares_nothing_with_them():
    before = copy.deepcopy(DEFAULTS)
    config = load_config(None)
    assert config == DEFAULTS and config is not DEFAULTS
    for section in config:
        assert config[section] is not DEFAULTS[section]
        for key, value in config[section].items():
            if isinstance(value, list):
                assert value is not DEFAULTS[section][key]
    config["training"]["net_arch"].append(1)
    config["backend"]["periodic"][0] = False
    config["reward"]["p_ur"] = 0
    assert DEFAULTS == before


@pytest.mark.parametrize("text", ["", "{}\n"])
def test_an_empty_file_gives_the_defaults(tmp_path, text):
    assert load_config(_write(tmp_path, text)) == DEFAULTS


def test_one_overridden_leaf_leaves_its_siblings(tmp_path):
    expected = copy.deepcopy(DEFAULTS)
    expected["training"]["n_steps"] = 32
    assert load_config(_write(tmp_path, "training:\n  n_steps: 32\n")) == expected
    assert load_config(str(_write(tmp_path, "training:\n  n_steps: 32\n"))) == expected


def test_a_list_replaces_its_default_whole(tmp_path):
    config = load_config(_write(tmp_path, "training:\n  net_arch: [16]\nbackend:\n  periodic: [true, false]\n"))
    assert config["training"]["net_arch"] == [16] and config["backend"]["periodic"] == [True, False]
    assert DEFAULTS["training"]["net_arch"] == [256, 256]


REFUSED = [
    ("enviroment:\n  alpha: 0.5\n", "config: unknown sections ['enviroment']"),
    ("zeta: {}\nalpha: {}\ntraining: {}\n", "config: unknown sections ['alpha', 'zeta']"),
    ("training:\n  n_step: 32\n", "config[training]: unknown keys ['n_step']"),
    ("- 1\n- 2\n", "config: expected sections, got list"),
    ("training: 5\n", "config[training]: expected keys, got int"),
    ("training:\n", "config[training]: expected keys, got NoneType"),
    ("training:\n  learning_rate: 3e-4\n", "config[training]: learning_rate: '3e-4', expected a number"),
    ("training:\n  n_steps: 32.0\n", "config[training]: n_steps: 32.0, expected an integer"),
    ("training:\n  n_steps: true\n", "config[training]: n_steps: True, expected an integer"),
    ("training:\n  seed: null\n", "config[training]: seed: None, expected an integer"),
    ("backend:\n  rotate_cells: 1\n", "config[backend]: rotate_cells: 1, expected true or false"),
    ("backend:\n  periodic: true\n", "config[backend]: periodic: True, expected a list, each item true or false"),
    ("backend:\n  lo: [0, a]\n", "config[backend]: lo: [0, 'a'], expected a list, each item a number"),
    (
        "training:\n  net_arch: {pi: [8], vf: [8]}\n",
        "config[training]: net_arch: {'pi': [8], 'vf': [8]}, expected a list, each item an integer",
    ),
    ("training:\n  net_arch: []\n", "config[training]: net_arch: [], expected a list, each item an integer"),
    ("observation:\n  indicator: 3\n", "config[observation]: indicator: 3, expected a name"),
    ("reward:\n  p_ur: yes\n", "config[reward]: p_ur: True, expected a number"),
]


@pytest.mark.parametrize("text, message", REFUSED)
def test_what_the_defaults_do_not_have_is_refused_by_name(tmp_path, text, message):
    before = copy.deepcopy(DEFAULTS)
    with pytest.raises(ValueError) as refusal:
        load_config(_write(tmp_path, text))
    assert str(refusal.value) == message
    assert DEFAULTS == before


def test_a_number_takes_an_integer_and_a_float_with_its_dot(tmp_path):
    text = (
        "training:\n  learning_rate: 3.0e-4\n  gamma: 1\n  net_arch: [16, 16, 16]\n"
        "decision:\n  beta: 2\nbackend:\n  hi: [2, 1]\n  periodic: [true, false]\n"
    )
    config = load_config(_write(tmp_path, text))
    assert config["training"]["learning_rate"] == 0.0003 and config["training"]["gamma"] == 1
    assert config["training"]["net_arch"] == [16, 16, 16]
    assert config["decision"]["beta"] == 2 and config["backend"]["hi"] == [2, 1]


def test_the_resolved_configuration_survives_yaml(tmp_path):
    config = _small()
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    assert load_config(path) == config
    assert list(yaml.safe_load(path.read_text(encoding="utf-8"))) == list(DEFAULTS)


def test_the_sanity_tasks_file_overrides_five_leaves():
    """configs/round_sanity.yaml, the task of item 3.2 (G1)."""
    config = load_config(PROJECT_ROOT / "configs" / "round_sanity.yaml")
    changed = {
        (section, key): config[section][key]
        for section in DEFAULTS
        for key in DEFAULTS[section]
        if config[section][key] != DEFAULTS[section][key]
    }
    assert changed == {
        ("observation", "element_budget"): 96,
        ("decision", "beta"): 2.0,
        ("episode", "amr_freq"): 10,
        ("training", "total_timesteps"): 200_000,
        ("checkpointing", "save_freq"): 20_000,
    }


# --- The backend, the environment, the model ---------------------------------------

def test_the_backend_is_the_mock_at_the_configurations_values():
    config = load_config(None)
    backend = build_backend(config)
    assert type(backend) is MockQuadBackend
    assert (backend.describe().ndim, backend.describe().nvar) == (2, 1)
    state = backend.reset({"x0": 0.5, "y0": 0.5, "width": 0.1}, 5, 1)
    assert (state.n_active, state.max_level, state.dt_base) == (64, 2, 0.01)
    assert config == DEFAULTS


def test_the_backend_is_given_its_pairs_as_tuples():
    # The one look at a private attribute in this file: the mock keeps the
    # pairs it was given, and a list there would be a list in a frozen noun.
    grid = build_backend(load_config(None))._grid
    assert all(type(pair) is tuple for pair in (grid.periodic, grid.lo, grid.hi))


@pytest.mark.parametrize(
    "key, value, message",
    [
        ("kind", "jexpresso", "config[backend]: kind: 'jexpresso', expected 'mock_quad'"),
        ("lo", [0.0], "config[backend]: lo: [0.0], expected 2 items"),
        ("periodic", [True, True, True], "config[backend]: periodic: [True, True, True], expected 2 items"),
        ("velocity", [1.0, 0.0, 0.0], "config[backend]: velocity: [1.0, 0.0, 0.0], expected 2 items"),
    ],
)
def test_the_backend_section_is_refused_by_name(key, value, message):
    config = load_config(None)
    config["backend"][key] = value
    with pytest.raises(ValueError) as refusal:
        build_backend(config)
    assert str(refusal.value) == message


def test_the_environment_is_round_env_in_a_monitor_and_nothing_else():
    env = build_env(load_config(None))
    assert type(env) is Monitor and type(env.env) is RoundEnv
    assert env.results_writer is None


def test_the_environment_holds_the_configurations_values():
    config = load_config(None)
    inner = build_env(config).unwrapped
    assert inner.observation_spec == ObservationSpec(
        observation=ObservationConfig(**config["observation"]), beta=1.2, coarsen_rule="first_decider", ndim=2,
        max_level=2, n_rounds=2, n_comp=1,
    )
    assert (inner.n_intervals, inner.amr_freq, inner.initial_level, inner.cadence) == (4, 5, 1, 1)
    assert (inner.lambda_local, inner.lambda_global) == (0.1, 1.0)
    assert type(inner.backend) is MockQuadBackend


def test_every_section_reaches_the_environment():
    config = _small()
    config["reward"].update(lambda_local=0.25, lambda_global=0.5)
    config["episode"]["cadence"] = 2
    config["decision"]["coarsen_rule"] = "unanimous"
    inner = build_env(config).unwrapped
    spec = inner.observation_spec
    assert (spec.beta, spec.coarsen_rule, spec.observation.element_budget) == (2, "unanimous", 8)
    assert (inner.n_intervals, inner.amr_freq, inner.initial_level, inner.cadence) == (2, 2, 0, 2)
    assert (inner.lambda_local, inner.lambda_global) == (0.25, 0.5)


def test_a_backend_handed_in_is_the_one_driven():
    config = _small()
    backend = build_backend(config)
    assert build_env(config, backend=backend).unwrapped.backend is backend


def test_a_key_with_no_taker_is_an_error():
    config = load_config(None)
    config["episode"]["extra"] = 1
    with pytest.raises(TypeError):
        build_env(config)


def test_the_monitor_folder_is_made_and_holds_the_log(tmp_path):
    folder = tmp_path / "not" / "yet"
    env = build_env(_small(), monitor_dir=folder)
    env.close()
    assert _files(tmp_path) == {"not/yet/monitor.csv"}


def test_the_model_is_maskable_ppo_with_the_routed_policy():
    config = load_config(None)
    config["training"]["device"] = "cpu"
    model = build_model(build_env(config), config)
    assert type(model) is MaskablePPO and type(model.policy) is RoundPolicy
    assert _count(model) == 141_828
    assert (model.n_steps, model.batch_size, model.n_epochs) == (1024, 64, 10)
    assert (model.gamma, model.gae_lambda, model.ent_coef, model.learning_rate) == (0.99, 0.95, 0.01, 3e-4)
    assert (model.seed, str(model.device), model.verbose) == (42, "cpu", 0)
    assert model.clip_range(1.0) == 0.2


def test_the_model_takes_every_training_value():
    config = _small(seed=7)
    config["training"].update(gamma=0.9, gae_lambda=0.8, ent_coef=0.0, learning_rate=1e-3, clip_range=0.1)
    model = build_model(build_env(config), config, verbose=1)
    assert _count(model) == 1188
    assert (model.n_steps, model.batch_size, model.n_epochs) == (32, 16, 2)
    assert (model.gamma, model.gae_lambda, model.ent_coef, model.learning_rate) == (0.9, 0.8, 0.0, 1e-3)
    assert (model.seed, model.verbose) == (7, 1)
    assert model.clip_range(1.0) == 0.1


def test_the_policy_is_given_net_arch_alone_and_its_own_copy():
    config = _small()
    model = build_model(build_env(config), config)
    assert model.policy_kwargs == {"net_arch": [16, 16]}
    assert model.policy_kwargs["net_arch"] is not config["training"]["net_arch"]


def test_the_masks_reach_the_model_with_no_wrapper():
    config = _small()
    model = build_model(build_env(config), config)
    assert is_masking_supported(model.get_env())


# --- The checkpoint helper and the callback ----------------------------------------

class _Watcher:
    """A model that only looks: whether the spec is on disk at the moment
    the zip would be written."""

    def save(self, path) -> None:
        self.path = Path(path)
        self.spec_first = spec_path(path).is_file()


@pytest.mark.parametrize(
    "stem, name",
    [("model", "model.zip"), ("model.zip", "model.zip"), ("run_0.5", "run_0.5.zip"), ("deep/er/model", "model.zip")],
)
def test_the_spec_is_written_before_the_zip_and_the_zip_is_named_here(run, tmp_path, stem, name):
    watcher = _Watcher()
    written = save_checkpoint(watcher, _env_spec(run.model), tmp_path / stem)
    assert written == watcher.path == (tmp_path / stem).with_name(name)
    assert watcher.spec_first
    assert _files(tmp_path) == {spec_path(written).relative_to(tmp_path).as_posix()}


@pytest.mark.parametrize("stem", ["model", "model.zip", "run_0.5", "deep/er/model"])
def test_a_checkpoint_is_a_zip_and_its_spec(run, tmp_path, stem):
    spec = _env_spec(run.model)
    written = save_checkpoint(run.model, spec, str(tmp_path / stem))
    assert written.suffix == ".zip" and written.is_file()
    assert _files(tmp_path) == {path.relative_to(tmp_path).as_posix() for path in (written, spec_path(written))}
    assert load_spec(written) == spec
    assert type(MaskablePPO.load(written, device="cpu").policy) is RoundPolicy


def test_the_library_alone_drops_the_zip_from_a_dotted_stem(run, tmp_path):
    """The control for the naming: stable-baselines3 appends .zip only to a
    path with no suffix, so the helper names the zip itself."""
    run.model.save(str(tmp_path / "run_0.5"))
    assert _files(tmp_path) == {"run_0.5"}


@pytest.mark.parametrize("save_freq", [0, -1, True, 2.0, "32", None])
def test_the_callback_refuses_a_save_freq_that_is_not_a_count(tmp_path, save_freq):
    with pytest.raises(ValueError) as refusal:
        SpecCheckpointCallback(save_freq, tmp_path, None)
    assert str(refusal.value) == f"save_freq: {save_freq!r}, expected an integer of at least 1"


def test_the_callback_saves_at_every_multiple_and_names_by_the_step_count(tmp_path):
    config = _small()
    env = build_env(config)
    model = build_model(env, config)
    callback = SpecCheckpointCallback(40, tmp_path, env.unwrapped.observation_spec, name_prefix="mine")
    model.learn(total_timesteps=96, callback=callback)
    assert model.num_timesteps == 96
    assert [path.name for path in callback.saved] == ["mine_40_steps.zip", "mine_80_steps.zip"]
    assert _files(tmp_path) == {
        "mine_40_steps.zip", "mine_40_steps.spec.json", "mine_80_steps.zip", "mine_80_steps.spec.json",
    }
    assert [MaskablePPO.load(path, device="cpu").num_timesteps for path in callback.saved] == [40, 80]


# --- A run on the two cells (S8-S14) -----------------------------------------------

def test_a_run_trains_its_step_count(run):
    assert (run.model.num_timesteps, run.model._n_updates) == (96, 6)
    assert type(run.model.policy) is RoundPolicy


def test_a_run_writes_its_files_and_no_others(run):
    checkpoints = {f"checkpoints/round_{steps}_steps{end}" for steps in (32, 64, 96) for end in (".zip", ".spec.json")}
    assert _files(run.results) == checkpoints | {
        "config.yaml", "monitor.csv", "rounds.csv", "final_model.zip", "final_model.spec.json",
    }


def test_every_zip_has_its_spec_and_every_spec_is_the_environments(run):
    zips = sorted(run.results.rglob("*.zip"))
    assert len(zips) == 4
    for path in zips:
        assert spec_path(path).is_file(), path
        assert load_spec(path) == _env_spec(run.model), path


def test_the_specs_of_a_run_are_the_same_bytes(run):
    texts = {path.read_bytes() for path in run.results.rglob("*.spec.json")}
    assert len(texts) == 1


def test_each_checkpoint_holds_its_own_step_count(run):
    for steps in (32, 64, 96):
        loaded = MaskablePPO.load(run.results / "checkpoints" / f"round_{steps}_steps.zip", device="cpu")
        assert loaded.num_timesteps == steps
    final = MaskablePPO.load(run.results / "final_model", device="cpu")
    assert final.num_timesteps == 96
    for name, parameter in final.policy.named_parameters():
        assert th.equal(parameter, _parameters(run.model)[name]), name


def test_the_resolved_configuration_is_written_and_reads_back(run):
    assert load_config(run.results / "config.yaml") == run.config
    written = yaml.safe_load((run.results / "config.yaml").read_text(encoding="utf-8"))
    assert written == run.config
    assert list(written) == list(DEFAULTS)


def test_the_manifest_goes_where_it_is_told_and_names_the_run(run):
    assert _files(run.runs) == {f"{RUN}.yaml"}
    assert not (PROJECT_ROOT / "runs" / f"{RUN}.yaml").exists()
    manifest = yaml.safe_load((run.runs / f"{RUN}.yaml").read_text(encoding="utf-8"))
    assert list(manifest) == [
        "git_commit", "git_dirty", "config_path", "seed", "created_at", "python", "platform", "run_id",
        "results_dir", "total_timesteps", "indicator",
    ]
    assert (manifest["seed"], manifest["run_id"], manifest["total_timesteps"], manifest["indicator"]) == (
        0, RUN, 96, "ind_b",
    )
    assert manifest["config_path"] == str(run.results / "config.yaml")
    assert manifest["results_dir"] == str(run.results)


def test_the_monitor_file_holds_at_least_one_episode(run):
    episodes = _episodes(run.results)
    assert len(episodes) >= 1
    assert sum(int(length) for _, length in episodes) <= 96
    assert run.model.get_env().envs[0].results_writer.file_handler.closed


def test_training_moves_the_first_layer_of_each_network():
    config = _small()
    model = build_model(build_env(config), config)
    before = _parameters(model)
    model.learn(total_timesteps=96)
    after = _parameters(model)
    for name in ("mlp_extractor.policy_net.0.weight", "mlp_extractor.value_net.0.weight"):
        assert not th.equal(after[name], before[name]), name


def test_one_seed_gives_one_run(run, tmp_path):
    again = train(_small(), tmp_path / "results" / "again", runs_dir=tmp_path / "runs")
    assert _episodes(tmp_path / "results" / "again") == _episodes(run.results)
    mine, theirs = _parameters(run.model), _parameters(again)
    assert all(th.equal(mine[name], theirs[name]) for name in mine)


def test_another_seed_gives_another_run(run, tmp_path):
    other = train(_small(seed=1), tmp_path / "results" / "other", runs_dir=tmp_path / "runs")
    assert _episodes(tmp_path / "results" / "other") != _episodes(run.results)
    mine, theirs = _parameters(run.model), _parameters(other)
    assert not any(th.equal(mine[name], theirs[name]) for name in mine if name.endswith("weight"))


def test_without_the_masks_the_same_run_raises_the_environments_refusal():
    """The control: the masks are what keeps a run alive."""
    config = _small()
    model = build_model(build_env(config), config)
    with pytest.raises(ValueError, match="is masked for position"):
        model.learn(total_timesteps=96, use_masking=False)


_CHILD = """
import json
import sys

import numpy as np
import torch as th
from sb3_contrib import MaskablePPO

model = MaskablePPO.load(sys.argv[1], device="cpu")
obs = json.loads(sys.argv[2])
tensors = {key: th.as_tensor(np.asarray([value], dtype=np.float32)) for key, value in obs.items()}
model.policy.set_training_mode(False)
with th.no_grad():
    distribution = model.policy.get_distribution(tensors)
    log_probs = [float(distribution.log_prob(th.full((1,), action))[0]) for action in range(3)]
cls = type(model.policy)
print(json.dumps({
    "policy": cls.__module__ + "." + cls.__qualname__,
    "policy_kwargs": model.policy_kwargs,
    "num_timesteps": model.num_timesteps,
    "log_probs": log_probs,
    "entry_imported": "training.train_round" in sys.modules,
}))
"""


def test_a_checkpoint_reloads_in_a_fresh_interpreter(run):
    """By the class's import path alone: no custom_objects, and the training
    entry is not needed to load what it saved."""
    checkpoint = run.results / "checkpoints" / "round_32_steps.zip"
    env = build_env(_small()).unwrapped
    first, _ = env.reset(options={"ic": IC})
    obs = {key: [float(x) for x in value] for key, value in first.items()}
    child = subprocess.run(
        [sys.executable, "-c", _CHILD, str(checkpoint), json.dumps(obs)],
        cwd=PROJECT_ROOT, capture_output=True, text=True, check=False,
    )
    assert child.returncode == 0, child.stderr
    told = json.loads(child.stdout.strip().splitlines()[-1])
    assert told["policy"] == "training.round_policy.RoundPolicy"
    assert told["policy_kwargs"] == {"net_arch": [16, 16]}
    assert told["num_timesteps"] == 32
    assert told["entry_imported"] is False
    assert len(told["log_probs"]) == 3
    assert told["log_probs"] == _log_probs(MaskablePPO.load(checkpoint, device="cpu"), obs)


def test_training_continues_past_a_backend_failure(failing):
    config = _small()
    backend = failing(build_backend(config), "step_interval", at=3)
    model = build_model(build_env(config, backend=backend), config)
    model.learn(total_timesteps=96)
    assert model.num_timesteps == 96
    assert backend.seen >= 3


def test_a_spec_finding_stops_the_run_at_its_first_reset():
    config = _small()
    backend = build_backend(config)
    config["backend"]["max_level"] = 3
    model = build_model(build_env(config, backend=backend), config)
    with pytest.raises(ValueError, match="spec: max_level: the spec says 3, the round has 2"):
        model.learn(total_timesteps=32)


# --- The round log (item 3.2, G6) --------------------------------------------------

ROUND = {"interval": 2, "round_index": 1, "n_active": 14, "overshoot": 3, "skipped": 5}


def _logged(results: Path) -> list[dict[str, int]]:
    """A folder's rounds.csv as integers, its header checked on the way."""
    lines = (results / "rounds.csv").read_text(encoding="utf-8").splitlines()
    assert lines[0] == "timesteps,episode,interval,round,committed,overshoot,n_active,skipped"
    return [{key: int(value) for key, value in row.items()} for row in csv.DictReader(lines)]


def _fed(tmp_path: Path) -> tuple[RoundLogCallback, SimpleNamespace]:
    """A round log started by hand, with the stand-in model whose step count
    it reads: what learn does to a callback before the first step."""
    model = SimpleNamespace(num_timesteps=0)
    callback = RoundLogCallback(tmp_path / "rounds.csv")
    callback.init_callback(model)
    callback.on_training_start({}, {})
    return callback, model


def _feed(callback: RoundLogCallback, model: SimpleNamespace, timesteps: int, info: dict, done: bool = False) -> None:
    """One step, as the rollout hands it over: the count, then the locals."""
    model.num_timesteps = timesteps
    callback.update_locals({"infos": [info], "dones": [done]})
    assert callback.on_step() is True


def test_the_round_log_has_one_row_per_round(run):
    rows = _logged(run.results)
    assert rows
    assert all(row["committed"] + row["overshoot"] == row["n_active"] for row in rows)
    steps = [row["timesteps"] for row in rows]
    assert steps == sorted(set(steps)) and steps[-1] <= 96
    # Every complete episode of the instance is four rounds, two intervals
    # of two, and its last row is written at the step that ended it.
    episodes = _episodes(run.results)
    end = 0
    for number, (_, length) in enumerate(episodes, start=1):
        end += int(length)
        mine = [row for row in rows if row["episode"] == number]
        assert [(row["interval"], row["round"]) for row in mine] == [(1, 1), (1, 2), (2, 1), (2, 2)]
        assert mine[-1]["timesteps"] == end
    assert {row["episode"] for row in rows} <= set(range(1, len(episodes) + 2))


def test_the_round_log_is_closed_and_counted_when_a_run_returns(monkeypatch, tmp_path):
    made = []

    class Watched(RoundLogCallback):
        def __init__(self, path) -> None:
            super().__init__(path)
            made.append(self)

    monkeypatch.setattr(train_round, "RoundLogCallback", Watched)
    results = tmp_path / "results" / "logged"
    train(_small(), results, runs_dir=tmp_path / "runs")
    (callback,) = made
    assert callback.path == results.resolve() / "rounds.csv"
    assert callback._file.closed
    assert callback.rows == len(_logged(results)) > 0


def test_a_step_that_ends_no_round_writes_nothing(tmp_path):
    callback, model = _fed(tmp_path)
    _feed(callback, model, 1, {"position": 0, "round_index": 1, "interval": 1, "transition": "element"})
    _feed(callback, model, 2, {"transition": "failure", "backend_failure": "on demand"}, done=True)
    callback.on_training_end()
    assert callback.rows == 0 and _logged(tmp_path) == []


def test_a_row_is_the_rounds_counts_and_where_they_came(tmp_path):
    callback, model = _fed(tmp_path)
    _feed(callback, model, 57, dict(ROUND, position=3, transition="round"))
    callback.on_training_end()
    assert callback.rows == 1
    assert _logged(tmp_path) == [
        {"timesteps": 57, "episode": 1, "interval": 2, "round": 1, "committed": 11, "overshoot": 3, "n_active": 14,
         "skipped": 5},
    ]


def test_the_episode_number_rises_after_the_step_that_ends_one(tmp_path):
    callback, model = _fed(tmp_path)
    _feed(callback, model, 10, ROUND)
    _feed(callback, model, 12, ROUND, done=True)
    _feed(callback, model, 13, {"transition": "element"})
    _feed(callback, model, 15, ROUND)
    callback.on_training_end()
    assert [(row["timesteps"], row["episode"]) for row in _logged(tmp_path)] == [(10, 1), (12, 1), (15, 2)]


def test_a_row_is_on_disk_before_the_file_is_closed(tmp_path):
    callback, model = _fed(tmp_path)
    _feed(callback, model, 4, ROUND)
    assert not callback._file.closed
    assert [row["timesteps"] for row in _logged(tmp_path)] == [4]
    callback.on_training_end()
    assert callback._file.closed


def test_a_file_already_there_is_replaced(tmp_path):
    (tmp_path / "rounds.csv").write_text("left,over\n1,2\n", encoding="utf-8")
    callback, _ = _fed(tmp_path)
    callback.on_training_end()
    assert _logged(tmp_path) == []


@pytest.mark.parametrize("verb, logged", [("apply_marks", False), ("step_interval", True)])
def test_a_round_is_logged_once_its_marks_are_applied(failing, tmp_path, verb, logged):
    """A failed adapt leaves no counts and no row. A failure after the adapt
    ends the episode at that step, and the round's row is still written."""
    config = _small()
    env = build_env(config, backend=failing(build_backend(config), verb, at=1)).unwrapped
    env.reset(options={"ic": IC})
    terminated, steps = False, 0
    while not terminated:
        _, _, terminated, _, info = env.step(1)
        steps += 1
    assert info["transition"] == "failure" and ("n_active" in info) is logged
    callback, model = _fed(tmp_path)
    _feed(callback, model, steps, info, done=True)
    callback.on_training_end()
    assert [row["timesteps"] for row in _logged(tmp_path)] == ([steps] if logged else [])


# --- The command line --------------------------------------------------------------

@pytest.fixture
def called(monkeypatch) -> list:
    """Stand in for train: record what main hands it, and train nothing."""
    calls = []

    def record(config, results_dir, runs_dir=None, verbose=0):
        calls.append(SimpleNamespace(config=config, results_dir=results_dir, runs_dir=runs_dir, verbose=verbose))

    monkeypatch.setattr(train_round, "train", record)
    return calls


def test_no_argument_overrides_nothing():
    assert vars(parse_args([])) == {
        "config": None, "timesteps": None, "seed": None, "save_freq": None, "results_dir": None, "device": None,
    }


def test_the_smoke_runs_command_line(called, capsys):
    main(["--timesteps", "4096", "--seed", "0", "--save-freq", "1024", "--results-dir", "results/smoke_round"])
    (call,) = called
    expected = copy.deepcopy(DEFAULTS)
    expected["training"].update(total_timesteps=4096, seed=0)
    expected["checkpointing"]["save_freq"] = 1024
    assert call.config == expected
    assert (call.results_dir, call.runs_dir, call.verbose) == (Path("results/smoke_round"), None, 1)
    assert capsys.readouterr().out == "results: results/smoke_round\n"


def test_the_command_line_goes_over_the_file_and_the_file_over_the_defaults(called, tmp_path):
    path = _write(tmp_path, "training:\n  total_timesteps: 7\n  seed: 5\n  n_steps: 32\n")
    main(["--config", str(path), "--timesteps", "9", "--device", "cpu", "--results-dir", str(tmp_path / "x")])
    training = called[0].config["training"]
    assert (training["total_timesteps"], training["seed"], training["n_steps"], training["device"]) == (9, 5, 32, "cpu")
    assert called[0].config["checkpointing"] == DEFAULTS["checkpointing"]


def test_with_no_folder_named_a_run_goes_to_a_stamped_one(called, capsys):
    main([])
    folder = called[0].results_dir
    assert folder.parent == PROJECT_ROOT / "results"
    assert folder.name.startswith("round_") and len(folder.name) == 21
    assert called[0].config == DEFAULTS
    assert capsys.readouterr().out == f"results: {folder}\n"


@pytest.mark.parametrize("arguments", [["--device", "gpu"], ["--timesteps", "many"], ["--save_freq", "8"]])
def test_the_command_line_refuses_what_it_does_not_know(called, arguments):
    with pytest.raises(SystemExit):
        main(arguments)
    assert called == []


def test_a_refused_file_trains_nothing(called, tmp_path):
    with pytest.raises(ValueError, match="unknown sections"):
        main(["--config", str(_write(tmp_path, "enviroment: {}\n"))])
    assert called == []


# --- What the module imports -------------------------------------------------------

def test_the_module_stands_beside_the_sequential_stack_and_imports_none_of_it():
    names = set()
    for node in ast.walk(ast.parse(Path(train_round.__file__).read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    ours = {name for name in names if name.split(".")[0] in {
        "agent", "backends", "contract", "deployment", "drivers", "envs", "tools", "training",
    }}
    assert ours == {
        "agent.round_observation", "agent.round_spec", "backends.mock_quad.backend", "contract.backend",
        "envs.round_env", "tools.run_manifest", "training.round_policy",
    }
    assert "sb3_contrib.common.wrappers" not in names
