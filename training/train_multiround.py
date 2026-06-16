#!/usr/bin/env python
"""Multi-round DRL-AMR training entry point (RESTRUCTURE Phase 6).

Ports drl-amr-1d/experiments/train_multiround.py to the contract-layered layout.
The solver -> contract -> core -> driver chain is built by
deployment.build.build_driver_from_config (shared with the deployment runner);
this script wraps that driver in the gym shell + MaskablePPO and owns the
training loop, config resolution, checkpointing, diagnostics, and the run
manifest.

Reconciled to the restructure:
  - error_indicator defaults to zz_style (D-048); pre_advance retired (D-049)
  - initial_refinement_level default = 1 (coarsen live from round 1; D-049)
  - chain built via build_driver_from_config + MultiroundEnv shell (D-047)
  - run manifest wired here (D-037, deferred from Phase 2)

Spec: strategy/proposals/Stage_1_Architecture_Specification.md (§10)
Decisions: D-007 (terminal dual-reward delivery), D-025 (MaskablePPO),
           D-037 (run manifest), D-047 (shared driver), D-048, D-049

Usage:
    python -m training.train_multiround
    python -m training.train_multiround --config configs/multiround_default.yaml
    python -m training.train_multiround --timesteps 10000 --seed 0 \\
        --results-dir results/smoke/
"""
from __future__ import annotations

import argparse
import datetime
from pathlib import Path

import numpy as np
import yaml
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from deployment.build import build_driver_from_config
from envs.multiround_env import MultiroundEnv
from tools.run_manifest import write_manifest

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Default configuration (YAML overrides these; CLI overrides YAML)
# ---------------------------------------------------------------------------
# Mirrors configs/multiround_default.yaml. Any key present in a YAML config
# overrides the default; unspecified keys retain these values.
DEFAULT_CONFIG = {
    "environment": {
        "alpha": 0.1,                    # Error tolerance (Spec §9.2)
        "beta": 1.2,                     # Hysteresis exponent (Spec §9.2)
        "element_budget": 30,            # Soft cap on active elements
        "n_remesh": 4,                   # Remesh intervals per episode (D-027)
        "step_domain_fraction": 0.05,    # Wave travel per interval (Spec §4.1)
        "initial_refinement_level": 1,   # Coarsen live from round 1 (D-049)
        "error_indicator": "zz_style",   # Error indicator (D-048): zz_style, raw_jump
        "ic_pool": [1, 10, 12, 13, 14, 15, 16],  # Multi-IC pool (Spec §11.1)
        "verbosity": 0,                  # 0 = silent for training
    },
    "reward": {
        "p_ur": 10.0,                    # Under-refinement penalty (DynAMO default)
        "p_or": 5.0,                     # Over-refinement penalty (DynAMO default)
        "p_cr": 2.0,                     # Correct coarsening reward (D-020 / D-023)
        "lambda_local": 0.1,             # Local reward scaling (Spec §12.3)
        "lambda_global": 1.0,            # Global reward scaling (D-030)
    },
    "solver": {
        "nop": 4,                        # Polynomial order (D-009)
        "xelem": [-1.0, -0.4, 0.0, 0.4, 1.0],  # Base mesh nodes (4 elements)
        "max_level": 3,                  # Max refinement depth / rounds per interval
        "max_elements": 120,             # Safety cap (~4x budget)
        "courant_max": 0.1,              # CFL number
    },
    "training": {
        "total_timesteps": 100_000,      # Stage 1A default
        "learning_rate": 3e-4,           # PPO default (Spec §10.1)
        "gamma": 0.99,                   # Discount factor
        "gae_lambda": 0.95,              # GAE lambda
        "n_steps": 256,                  # Rollout buffer (> 1 episode ~180 steps)
        "batch_size": 64,                # PPO minibatch size
        "n_epochs": 10,                  # PPO epochs per update
        "ent_coef": 0.01,                # Entropy bonus for exploration
        "clip_range": 0.2,               # PPO clip range
        "net_arch": [256, 256],          # 2x256 FCNN (matching DynAMO)
        "seed": 42,                      # Random seed
        "device": "auto",                # auto, cpu, or cuda
    },
    "checkpointing": {
        "save_freq": 10_000,             # Steps between checkpoint saves
        "keep_best": True,               # Reserved; periodic CheckpointCallback only
    },
}


# ---------------------------------------------------------------------------
# Config resolution
# ---------------------------------------------------------------------------
def load_config(config_path: str | None = None) -> dict:
    """Return DEFAULT_CONFIG deep-merged with the YAML at config_path (if any)."""
    config = _deep_copy_dict(DEFAULT_CONFIG)
    if config_path is not None:
        with open(config_path) as f:
            yaml_config = yaml.safe_load(f) or {}
        _deep_merge(config, yaml_config)
    return config


def _deep_copy_dict(d: dict) -> dict:
    """Deep copy a nested dict (lists copied, not shared)."""
    result = {}
    for k, v in d.items():
        if isinstance(v, dict):
            result[k] = _deep_copy_dict(v)
        elif isinstance(v, list):
            result[k] = list(v)
        else:
            result[k] = v
    return result


def _deep_merge(base: dict, override: dict) -> None:
    """Recursively merge override into base, in place."""
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v


# ---------------------------------------------------------------------------
# Environment + model construction
# ---------------------------------------------------------------------------
def _mask_fn(env: MultiroundEnv) -> np.ndarray:
    """ActionMasker bridge: expose the env's per-element mask to MaskablePPO."""
    return env.action_masks()


def create_env(config: dict, log_dir: str | None = None) -> Monitor:
    """Build the full training env: driver -> gym shell -> ActionMasker -> Monitor.

    The solver -> contract -> core -> driver chain is constructed by
    build_driver_from_config (sequential mode, the D-038 parity target), so this
    function adds only the gym wrapper, the action mask bridge, and episode
    logging. No solver/env parameters are duplicated here — they all live in the
    config sections build_driver_from_config reads.
    """
    driver = build_driver_from_config(config)  # sequential (default)
    env = MultiroundEnv(driver)
    env = ActionMasker(env, _mask_fn)
    monitor_path = str(Path(log_dir) / "monitor") if log_dir else None
    return Monitor(env, filename=monitor_path)


def create_model(env: Monitor, config: dict, log_dir: str) -> MaskablePPO:
    """Create MaskablePPO with the spec §10.1 architecture (2x256 FCNN)."""
    t = config["training"]
    policy_kwargs = {"net_arch": t["net_arch"]}

    model = MaskablePPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=t["learning_rate"],
        n_steps=t["n_steps"],
        batch_size=t["batch_size"],
        n_epochs=t["n_epochs"],
        gamma=t["gamma"],
        gae_lambda=t["gae_lambda"],
        ent_coef=t["ent_coef"],
        clip_range=t["clip_range"],
        policy_kwargs=policy_kwargs,
        tensorboard_log=str(Path(log_dir) / "tensorboard"),
        seed=t["seed"],
        device=t["device"],
        verbose=1,
    )

    total_params = sum(p.numel() for p in model.policy.parameters())
    print("\nModel created:")
    print(f"  Device: {model.device}")
    print(f"  Network: {t['net_arch']}")
    print(f"  Parameters: {total_params:,}")
    print(f"  n_steps={t['n_steps']}, batch_size={t['batch_size']}, n_epochs={t['n_epochs']}")
    return model


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(config: dict, results_dir: str) -> None:
    """Run one training session: set up outputs, build, train, save."""
    t = config["training"]
    ckpt = config["checkpointing"]

    results_path = Path(results_dir).resolve()
    (results_path / "checkpoints").mkdir(parents=True, exist_ok=True)

    # Resolved config — the authoritative per-run record (build chain reads this).
    config_save_path = results_path / "config.yaml"
    with open(config_save_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    # Tracked run manifest (D-037): git commit + seed + resolved config path.
    run_id = results_path.name
    manifest_path = write_manifest(
        PROJECT_ROOT / "runs" / f"{run_id}.yaml",
        config_path=config_save_path,
        seed=t["seed"],
        extra={
            "run_id": run_id,
            "results_dir": str(results_path),
            "total_timesteps": t["total_timesteps"],
            "error_indicator": config["environment"]["error_indicator"],
        },
    )

    print(f"\n{'=' * 60}")
    print("TRAINING SETUP")
    print(f"{'=' * 60}")
    print(f"  Results:  {results_path}")
    print(f"  Manifest: {manifest_path}")
    print(f"  Config:   {config_save_path}")
    print(f"  Timesteps: {t['total_timesteps']:,}   Seed: {t['seed']}")

    env = create_env(config, log_dir=str(results_path))
    model = create_model(env, config, log_dir=str(results_path))

    callbacks = [
        CheckpointCallback(
            save_freq=ckpt["save_freq"],
            save_path=str(results_path / "checkpoints"),
            name_prefix="multiround",
            save_replay_buffer=False,
            save_vecnormalize=False,
        )
    ]

    # Lazy import: keeps this module importable before training/diagnostics.py
    # exists, and only training needs the (matplotlib-heavy) callback.
    from training.diagnostics import MultiroundDiagnosticsCallback

    callbacks.append(
        MultiroundDiagnosticsCallback(
            log_dir=str(results_path),
            log_freq=max(1000, t["total_timesteps"] // 100),
            verbose=1,
        )
    )

    print(f"\n{'=' * 60}")
    print("TRAINING START")
    print(f"{'=' * 60}\n")

    model.learn(total_timesteps=t["total_timesteps"], callback=callbacks, progress_bar=True)

    final_model_path = results_path / "final_model"
    model.save(str(final_model_path))
    env.close()

    print(f"\n{'=' * 60}")
    print("TRAINING COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Final model: {final_model_path}.zip")
    print(f"  TensorBoard: {results_path / 'tensorboard'}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train multi-round DRL-AMR agent with MaskablePPO",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config. If omitted, uses DEFAULT_CONFIG.")
    parser.add_argument("--timesteps", type=int, default=None, help="Override total timesteps.")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed.")
    parser.add_argument("--results-dir", type=str, default=None,
                        help="Output dir. Default: results/multiround_<timestamp>/")
    parser.add_argument("--device", type=str, default=None, choices=["auto", "cpu", "cuda"],
                        help="Override compute device.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    if args.timesteps is not None:
        config["training"]["total_timesteps"] = args.timesteps
    if args.seed is not None:
        config["training"]["seed"] = args.seed
    if args.device is not None:
        config["training"]["device"] = args.device

    if args.results_dir is not None:
        results_dir = args.results_dir
    else:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = str(PROJECT_ROOT / "results" / f"multiround_{timestamp}")

    train(config, results_dir)


if __name__ == "__main__":
    main()