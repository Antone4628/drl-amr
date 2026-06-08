#!/usr/bin/env python3
"""Run-manifest helper — lightweight, tracked provenance for every training run.

Convention (D-037):
  - Heavy artifacts (model, metrics, full config copy, logs) live in the
    gitignored ``results/<run_id>/`` tree (Borah + Mac, not committed).
  - A small, *tracked* manifest per run lives in ``runs/<run_id>.yaml`` so any
    run reproduces from config alone, with the exact commit and seed on record.

The Phase 6 training script calls ``write_manifest(...)`` once per run.
"""
from __future__ import annotations

import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml


def _git_commit() -> str:
    """Return the current git commit hash, or 'unknown' outside a repo."""
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _git_dirty() -> bool:
    """Return True if the working tree has uncommitted changes."""
    try:
        out = (
            subprocess.check_output(["git", "status", "--porcelain"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
        return bool(out)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def write_manifest(
    output_path: str | Path,
    config_path: str | Path,
    seed: int,
    extra: dict | None = None,
) -> Path:
    """Write a tracked run manifest and return its path.

    Args:
        output_path: Destination YAML path (convention: ``runs/<run_id>.yaml``).
        config_path: Path to the YAML config the run used.
        seed: Random seed for the run.
        extra: Optional extra fields to record (e.g. run_id, notes, host).

    Returns:
        The path the manifest was written to.
    """
    manifest = {
        "git_commit": _git_commit(),
        "git_dirty": _git_dirty(),
        "config_path": str(config_path),
        "seed": seed,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    if extra:
        manifest.update(extra)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.safe_dump(manifest, f, sort_keys=False)
    return output_path