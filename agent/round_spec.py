"""The observation spec of the 2D agent core: what a policy was trained with.

A trained policy is weights and a way of building its inputs. The weights
are the checkpoint; the rest is here. An ObservationSpec holds every value
the within-round logic is configured with, and the file written from it
names, beside those values, the feature lists and the rules the code holds
fixed. The file is what a deployment loads with the weights
(SOLVER_CONTRACT_V2 section 4, row 19), so it is plain JSON: readable with
no Python and no RL library.

What is saved:

    observation     the ObservationConfig: the indicator and its reduction,
                    alpha, the element budget, the normalization rule and
                    its floor
    beta            the hysteresis exponent of the thresholds
    coarsen_rule    one of agent.round_mask.COARSEN_RULES
    ndim            which sets the feature lists
    max_level, n_rounds, n_comp
                    what the policy was trained at: the finest level, the
                    rounds per remesh interval and the columns per
                    indicator array. Level features divide by the first
                    and round progress by the second.
    fixed_rules     FIXED_RULES: what the code does one way only
    actor_features, critic_features
                    the names, in order, as the builders emit them

The last three are not fields of the noun. They are read from the code
when a spec is written and compared with the code when one is loaded, so a
file written by code whose feature list or fixed rules have since changed
is refused, not reinterpreted.

spec_violations compares a spec with a round, since a policy fed a round
at another max_level, round count or column count gets no error from the
builders.

The reward's weights are not here: the reward is training-only and belongs
to a run's configuration.

This module belongs to the 2D core (AGENT_2D_ROADMAP item 2.8). The
sequential stack saves one config.yaml per run and is not on this path.
"""
from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from numbers import Integral, Real
from pathlib import Path

from agent.round_cache import RoundCache
from agent.round_mask import COARSEN_RULES
from agent.round_observation import ObservationConfig, critic_feature_names, feature_names
from agent.round_state import RoundState

# The layout of the file. Raised when a key is added, removed or renamed.
SPEC_VERSION = 1

# What the 2D core does one way only, by name. None is a configuration
# value; each is written to the file so that a second way, when one is
# built, has a name to differ from.
FIXED_RULES = {
    # round_order: farthest from the neutral zone first, by a stable sort,
    # so equal priorities keep their position order.
    "queue": "priority_magnitude_stable",
    # actor_observation at more than one column: each column normalized
    # against its own largest error, the feature the largest of them.
    "error_feature": "max_over_columns",
    # zone_distance at more than one column: the largest of the columns'
    # signed distances over the live columns.
    "zone_distance": "max_signed_over_columns",
    # The thresholds are computed once per remesh interval, from its first
    # round's cache, and held. Nothing in agent/ enforces this: it is what
    # whoever drives the interval must do.
    "thresholds": "interval_first",
    # The normalization carries nothing from one round to the next.
    "normalization_state": "none",
}

# A checkpoint's spec lies beside it: <checkpoint stem>.spec.json.
SPEC_SUFFIX = ".spec.json"

_KEYS = (
    "spec_version", "ndim", "max_level", "n_rounds", "n_comp", "observation", "beta", "coarsen_rule",
    "fixed_rules", "actor_features", "critic_features",
)
_OBSERVATION_KEYS = ("indicator", "reduction", "alpha", "element_budget", "normalization", "floor_decades")


@dataclass(frozen=True, slots=True, kw_only=True)
class ObservationSpec:
    """What the within-round logic of one trained policy is configured with.

    Fields:
        observation: The observation's configuration.
        beta: The hysteresis exponent, above 1: round_thresholds' argument.
        coarsen_rule: One of COARSEN_RULES.
        ndim: The mesh dimension, 1, 2 or 3.
        max_level: The finest level the policy was trained at, at least 1.
        n_rounds: Rounds per remesh interval it was trained at.
        n_comp: Columns per indicator array it was trained at.
    """

    observation: ObservationConfig
    beta: float
    coarsen_rule: str
    ndim: int
    max_level: int
    n_rounds: int
    n_comp: int

    def __post_init__(self) -> None:
        """Check every field.

        Raises:
            ValueError: Naming the field.
        """
        if not isinstance(self.observation, ObservationConfig):
            raise ValueError(f"observation: {self.observation!r}, expected an ObservationConfig")
        beta = self.beta
        if isinstance(beta, bool) or not isinstance(beta, Real) or not 1 < beta < math.inf:
            raise ValueError(f"beta: {beta!r}, expected a finite number above 1")
        if self.coarsen_rule not in COARSEN_RULES:
            raise ValueError(f"coarsen_rule: {self.coarsen_rule!r}, expected one of {list(COARSEN_RULES)}")
        ndim = self.ndim
        if isinstance(ndim, bool) or not isinstance(ndim, Integral) or ndim not in (1, 2, 3):
            raise ValueError(f"ndim: {ndim!r}, expected 1, 2 or 3")
        for name in ("max_level", "n_rounds", "n_comp"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name}: {value!r}, expected an integer of at least 1")
        if self.n_comp > 1 and self.observation.normalization == "dynamo":
            raise ValueError(
                f"normalization: 'dynamo' takes one column per indicator, and the spec has n_comp = {self.n_comp}"
            )

    @property
    def actor_features(self) -> tuple[str, ...]:
        """The actor's feature names, in order: feature_names(ndim)."""
        return feature_names(self.ndim)

    @property
    def critic_features(self) -> tuple[str, ...]:
        """The critic's feature names, in order."""
        return critic_feature_names(self.ndim)


def _number(value: Real) -> int | float:
    """A Python int for an integer and a Python float for any other number,
    so that json writes it and a numpy scalar does not reach the file."""
    return int(value) if isinstance(value, Integral) else float(value)


def to_dict(spec: ObservationSpec) -> dict[str, object]:
    """The spec as the file holds it: plain values, lists and dicts.

    Adds what the noun does not carry: the layout's version, the fixed
    rules and both feature lists, read from the code as it is now. An
    integer stays an integer and any other number becomes a float, so the
    values read back are the values the builders were given.

    Returns:
        A new dict with the keys of the file, every value one that json
        writes.
    """
    observation = asdict(spec.observation)
    for name in ("alpha", "floor_decades", "element_budget"):
        observation[name] = _number(observation[name])
    return {
        "spec_version": SPEC_VERSION,
        "ndim": int(spec.ndim),
        "max_level": int(spec.max_level),
        "n_rounds": int(spec.n_rounds),
        "n_comp": int(spec.n_comp),
        "observation": observation,
        "beta": _number(spec.beta),
        "coarsen_rule": spec.coarsen_rule,
        "fixed_rules": dict(FIXED_RULES),
        "actor_features": list(spec.actor_features),
        "critic_features": list(spec.critic_features),
    }


def _check_keys(name: str, data: object, keys: tuple[str, ...]) -> None:
    """Reject anything but a mapping with exactly these keys.

    Raises:
        ValueError: Starting with name.
    """
    if not isinstance(data, Mapping):
        raise ValueError(f"{name}: expected a mapping, got {type(data).__name__}")
    missing = [key for key in keys if key not in data]
    unknown = sorted(str(key) for key in data if key not in keys)
    if missing or unknown:
        raise ValueError(f"{name}: missing keys {missing}, unknown keys {unknown}")


def from_dict(data: Mapping[str, object]) -> ObservationSpec:
    """The spec a dict holds, checked against the code as it is now.

    The dict must have to_dict's keys and no others, this layout's version,
    this code's fixed rules, and the feature lists this code builds at the
    dict's ndim. A dict that differs in any of them was written by other
    code, and a policy trained on it would be fed something else here.

    Raises:
        ValueError: Starting with the key at fault: spec, spec_version,
            observation, fixed_rules, actor_features or critic_features;
            or as ObservationConfig and ObservationSpec raise for a value.
    """
    _check_keys("spec", data, _KEYS)
    if type(data["spec_version"]) is not int or data["spec_version"] != SPEC_VERSION:
        raise ValueError(f"spec_version: {data['spec_version']!r}, this code reads version {SPEC_VERSION}")
    _check_keys("observation", data["observation"], _OBSERVATION_KEYS)
    spec = ObservationSpec(
        observation=ObservationConfig(**data["observation"]),
        beta=data["beta"],
        coarsen_rule=data["coarsen_rule"],
        ndim=data["ndim"],
        max_level=data["max_level"],
        n_rounds=data["n_rounds"],
        n_comp=data["n_comp"],
    )
    if data["fixed_rules"] != FIXED_RULES:
        raise ValueError(f"fixed_rules: {data['fixed_rules']!r}, this code's are {FIXED_RULES!r}")
    for key, names in (("actor_features", spec.actor_features), ("critic_features", spec.critic_features)):
        if data[key] != list(names):
            raise ValueError(f"{key}: {data[key]!r}, this code builds {list(names)!r}")
    return spec


def spec_path(checkpoint_path: str | Path) -> Path:
    """Where a checkpoint's spec lies: beside it, <stem>.spec.json.

    A trailing .zip is dropped first, so the path a model was saved to and
    the file that save produced name the same spec.
    """
    path = Path(checkpoint_path)
    if path.suffix == ".zip":
        path = path.with_suffix("")
    return path.with_name(path.name + SPEC_SUFFIX)


def save_spec(spec: ObservationSpec, checkpoint_path: str | Path) -> Path:
    """Write a checkpoint's spec beside it, as JSON.

    The text is to_dict's, keys sorted, two spaces of indent and a final
    newline: the same spec always gives the same bytes. The directory is
    made if it is missing, and a spec already there is replaced.

    Args:
        spec: What the checkpoint's policy was trained with.
        checkpoint_path: The checkpoint's path, with or without .zip.

    Returns:
        The path written, spec_path(checkpoint_path).
    """
    path = spec_path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_dict(spec), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return path


def load_spec(checkpoint_path: str | Path) -> ObservationSpec:
    """Read the spec that lies beside a checkpoint.

    Raises:
        FileNotFoundError: If the checkpoint has no spec beside it.
        ValueError: If the file is not JSON, or as from_dict raises.
    """
    return from_dict(json.loads(spec_path(checkpoint_path).read_text(encoding="utf-8")))


def spec_violations(spec: ObservationSpec, cache: RoundCache, round_state: RoundState) -> list[str]:
    """Where a round differs from what a spec's policy was trained at.

    A policy fed a round its spec does not describe gets no error from the
    builders: a level over another max_level is still a number. This
    compares the two, for whoever drives the round to call once per
    episode or per run.

    Rules, each the stable prefix of its finding:
        ndim, max_level: The spec's against the cache's state.
        n_comp, reduction: The spec's against the cache.
        indicator: The spec's indicator is in the cache.
        n_rounds: The spec's against the round state's.

    Args:
        spec: What the policy was trained with.
        cache: A round's frozen state and indicator arrays.
        round_state: Any RoundState of the interval.

    Returns:
        One finding per broken rule, in the order above; empty when the
        round is one the spec describes.
    """
    state, config = cache.state, spec.observation
    found = []
    for name, here in (("ndim", state.ndim), ("max_level", state.max_level), ("n_comp", cache.n_comp)):
        if getattr(spec, name) != here:
            found.append(f"{name}: the spec says {getattr(spec, name)}, the round has {here}")
    if config.reduction != cache.reduction:
        found.append(f"reduction: the spec says {config.reduction!r}, the cache was read under {cache.reduction!r}")
    if config.indicator not in cache.errors:
        found.append(f"indicator: {config.indicator!r} is not in the cache, which holds {list(cache.indicators)}")
    if spec.n_rounds != round_state.n_rounds:
        found.append(f"n_rounds: the spec says {spec.n_rounds}, the round has {round_state.n_rounds}")
    return found