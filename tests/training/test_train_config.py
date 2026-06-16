"""Config-resolution tests for the training port (RESTRUCTURE Phase 6).

Pins the merge semantics of training.train_multiround.load_config and the
D-048/D-049 reconciliations baked into DEFAULT_CONFIG (zz_style default, fixed
initial_refinement_level=1, pre_advance retired). Pure functions only — no RL,
no env construction; fast.
"""
import yaml

from training.train_multiround import DEFAULT_CONFIG, _deep_copy_dict, load_config


def test_default_config_sections_present():
    cfg = load_config(None)
    assert set(cfg) >= {"environment", "reward", "solver", "training", "checkpointing"}


def test_default_config_pins_restructure_reconciliations():
    # D-048: zz_style is the default indicator.
    assert cfg_env(load_config(None))["error_indicator"] == "zz_style"
    # D-049: fixed initial_refinement_level = 1 (coarsen live from round 1).
    assert cfg_env(load_config(None))["initial_refinement_level"] == 1
    # D-049: pre_advance retired — the key must not exist (build.py no longer reads it).
    assert "pre_advance_range" not in cfg_env(load_config(None))


def test_load_config_none_equals_default_but_is_independent_copy():
    cfg = load_config(None)
    assert cfg == DEFAULT_CONFIG
    assert cfg is not DEFAULT_CONFIG
    # Mutating a nested list in the copy must not touch DEFAULT_CONFIG.
    cfg["solver"]["xelem"].append(99.0)
    assert DEFAULT_CONFIG["solver"]["xelem"][-1] != 99.0


def test_deep_copy_dict_is_independent():
    src = {"a": {"b": [1, 2]}, "c": 3}
    cp = _deep_copy_dict(src)
    cp["a"]["b"].append(4)
    cp["c"] = 99
    assert src["a"]["b"] == [1, 2]
    assert src["c"] == 3


def test_yaml_override_deep_merges_nested(tmp_path):
    override = {
        "environment": {"alpha": 0.5},
        "training": {"total_timesteps": 1234},
    }
    path = tmp_path / "override.yaml"
    path.write_text(yaml.safe_dump(override))

    cfg = load_config(str(path))

    # Overridden leaves change...
    assert cfg["environment"]["alpha"] == 0.5
    assert cfg["training"]["total_timesteps"] == 1234
    # ...while sibling keys in the same sections retain their defaults...
    assert cfg["environment"]["beta"] == DEFAULT_CONFIG["environment"]["beta"]
    assert cfg["training"]["learning_rate"] == DEFAULT_CONFIG["training"]["learning_rate"]
    # ...and untouched sections are fully retained.
    assert cfg["reward"] == DEFAULT_CONFIG["reward"]
    assert cfg["solver"] == DEFAULT_CONFIG["solver"]


def test_yaml_override_single_env_key(tmp_path):
    path = tmp_path / "ind.yaml"
    path.write_text(yaml.safe_dump({"environment": {"error_indicator": "raw_jump"}}))

    cfg = load_config(str(path))

    assert cfg["environment"]["error_indicator"] == "raw_jump"
    assert cfg["environment"]["ic_pool"] == DEFAULT_CONFIG["environment"]["ic_pool"]


def cfg_env(cfg: dict) -> dict:
    """Small accessor to keep the reconciliation asserts readable."""
    return cfg["environment"]