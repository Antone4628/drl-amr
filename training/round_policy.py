"""The policy of the 2D agent core: each network reads its own key.

An observation of the 2D core is a dict with the keys "actor" and "critic"
(agent/round_observation.py). The actor's features are what deployment
builds; the critic's are training-only and may one day hold what the actor
must not see. sb3-contrib's multi-input policy concatenates every key and
feeds the whole vector to both networks, so its actor reads the critic's
key. RoundPolicy routes them:

    RoundExtractor      the two keys as one feature vector, the actor's
                        columns first, in KEY_ORDER
    SplitMlpExtractor   the policy network on the actor's columns, the
                        value network on the critic's
    RoundPolicy         the multi-input maskable policy with those two
                        installed

The policy keeps one shared features extractor. With it, every path of the
parent class (acting, evaluating a batch, the action distribution alone,
the value alone) reaches the networks through SplitMlpExtractor's
forward_actor and forward_critic on the full feature vector, so the split
is made in one place.

This module belongs to the 2D stack (AGENT_2D_ROADMAP item 3.1). The
sequential stack trains sb3-contrib's MlpPolicy on a flat observation and
is not on this path.
"""
from __future__ import annotations

from typing import Any

import torch as th
from gymnasium import spaces
from sb3_contrib.common.maskable.policies import MaskableMultiInputActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.type_aliases import Schedule
from torch import nn

from agent.round_observation import ACTOR_KEY, CRITIC_KEY

# The order the two keys take in the feature vector. SplitMlpExtractor cuts
# the vector at the first key's width, so the order is named here and not
# left to the order a Dict space happens to iterate in.
KEY_ORDER = (ACTOR_KEY, CRITIC_KEY)


class RoundExtractor(BaseFeaturesExtractor):
    """The observation dict as one feature vector: the actor's key, then the critic's.

    It has no parameters: it only places the two keys side by side.

    Args:
        observation_space: A Dict of exactly the keys of KEY_ORDER, each a
            Box of one dimension.

    Attributes:
        n_actor: The width of the actor's key, the first n_actor columns.
        n_critic: The width of the critic's key, the columns after them.

    Raises:
        ValueError: Starting with observation_space, for any other space.
    """

    def __init__(self, observation_space: spaces.Dict) -> None:
        if not isinstance(observation_space, spaces.Dict) or set(observation_space.spaces) != set(KEY_ORDER):
            raise ValueError(
                f"observation_space: expected a Dict with the keys {list(KEY_ORDER)}, got {observation_space!r}"
            )
        widths = []
        for key in KEY_ORDER:
            box = observation_space.spaces[key]
            if not isinstance(box, spaces.Box) or len(box.shape) != 1:
                raise ValueError(f"observation_space[{key!r}]: expected a Box of one dimension, got {box!r}")
            widths.append(int(box.shape[0]))
        super().__init__(observation_space, features_dim=sum(widths))
        self.n_actor, self.n_critic = widths

    def forward(self, observations: dict[str, th.Tensor]) -> th.Tensor:
        """The (batch, n_actor + n_critic) feature vector of a batch of observations."""
        return th.cat([observations[key] for key in KEY_ORDER], dim=1)


def _mlp(width: int, layers: list[int], activation_fn: type[nn.Module]) -> tuple[nn.Sequential, int]:
    """Linear layers of the given sizes, each followed by the activation.

    Returns:
        The network, and the width of its output: the input's width when
        layers is empty, since the network is then the identity.
    """
    modules: list[nn.Module] = []
    for size in layers:
        modules += [nn.Linear(width, size), activation_fn()]
        width = size
    return nn.Sequential(*modules), width


class SplitMlpExtractor(nn.Module):
    """Two networks on two parts of one feature vector.

    It stands where stable-baselines3's MlpExtractor stands and has its
    face: forward, forward_actor, forward_critic, latent_dim_pi and
    latent_dim_vf, and the submodules policy_net and value_net. The
    difference is what each network is given: policy_net the first n_actor
    columns, value_net the rest.

    Args:
        n_actor: Columns the policy network reads, the first ones.
        n_critic: Columns the value network reads, after them.
        net_arch: Hidden layer sizes: a list for both networks, or a dict
            with the keys "pi" and "vf", a missing key meaning no hidden
            layer.
        activation_fn: The activation's class, built once per layer.

    Raises:
        ValueError: Naming the argument.
    """

    def __init__(
        self, n_actor: int, n_critic: int, net_arch: list[int] | dict[str, list[int]], activation_fn: type[nn.Module]
    ) -> None:
        super().__init__()
        for name, value in (("n_actor", n_actor), ("n_critic", n_critic)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name}: {value!r}, expected an integer of at least 1")
        if isinstance(net_arch, dict):
            unknown = sorted(str(key) for key in net_arch if key not in ("pi", "vf"))
            if unknown:
                raise ValueError(f"net_arch: unknown keys {unknown}, expected 'pi' and 'vf'")
            pi_layers, vf_layers = net_arch.get("pi", []), net_arch.get("vf", [])
        else:
            pi_layers = vf_layers = net_arch
        self.n_actor = n_actor
        self.n_critic = n_critic
        self.policy_net, self.latent_dim_pi = _mlp(n_actor, pi_layers, activation_fn)
        self.value_net, self.latent_dim_vf = _mlp(n_critic, vf_layers, activation_fn)

    def forward(self, features: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        """The policy's and the value's latent vectors of a batch of features."""
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        """The policy network on the actor's columns."""
        return self.policy_net(features[:, : self.n_actor])

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        """The value network on the critic's columns."""
        return self.value_net(features[:, self.n_actor :])


class RoundPolicy(MaskableMultiInputActorCriticPolicy):
    """The maskable multi-input policy, with each network on its own key.

    The arguments are the parent's, in the parent's order, so that
    stable-baselines3 rebuilds the policy from a saved model as it rebuilds
    its own. Two of them have one accepted value each, since any other
    breaks the split: the features extractor is RoundExtractor, and it is
    shared.

    A run passes net_arch alone (policy_kwargs), so that is all a saved
    model holds beside the class.

    Raises:
        ValueError: Naming features_extractor_class or
            share_features_extractor; or as RoundExtractor and
            SplitMlpExtractor raise.
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: list[int] | dict[str, list[int]] | None = None,
        activation_fn: type[nn.Module] = nn.Tanh,
        ortho_init: bool = True,
        features_extractor_class: type[BaseFeaturesExtractor] = RoundExtractor,
        features_extractor_kwargs: dict[str, Any] | None = None,
        share_features_extractor: bool = True,
        normalize_images: bool = True,
        optimizer_class: type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: dict[str, Any] | None = None,
    ) -> None:
        if features_extractor_class is not RoundExtractor:
            raise ValueError(f"features_extractor_class: {features_extractor_class.__name__}, expected RoundExtractor")
        if share_features_extractor is not True:
            raise ValueError(
                f"share_features_extractor: {share_features_extractor!r}, expected True: the split is made on one "
                "feature vector"
            )
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            ortho_init,
            features_extractor_class,
            features_extractor_kwargs,
            share_features_extractor,
            normalize_images,
            optimizer_class,
            optimizer_kwargs,
        )

    def _build_mlp_extractor(self) -> None:
        """Install the split extractor where the parent installs MlpExtractor.

        Called by the parent's _build, after the features extractor exists
        and before the action and value heads are sized from latent_dim_pi
        and latent_dim_vf.
        """
        extractor = self.features_extractor
        self.mlp_extractor = SplitMlpExtractor(extractor.n_actor, extractor.n_critic, self.net_arch, self.activation_fn)