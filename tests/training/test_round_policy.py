"""Tests for the policy of the 2D stack, training/round_policy.py: each
network reads its own key of the observation dict.

AGENT_2D_ROADMAP item 3.1, sitting 2, signals S2-S6. Five groups:

  - the extractor: the two keys side by side in a named order, and what it
    refuses (S2);
  - the split extractor: two networks on two parts of one vector (S3);
  - the policy: its networks and their sizes, and what it refuses (S4, S6);
  - the route: through every path of the parent class, a network does not
    move at all when the other network's key moves, and the stock policy,
    as the control, fails that (S5);
  - the policy under MaskablePPO: it trains, it is still routed, and a
    saved model holds the class and net_arch alone.

Every exact zero here is a comparison of the same numbers through the same
operations, so it holds whatever the initial weights.

Run: pytest tests/training/test_round_policy.py -v
"""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest
import torch as th
from gymnasium import spaces
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableMultiInputActorCriticPolicy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.save_util import load_from_zip_file
from stable_baselines3.common.torch_layers import CombinedExtractor, MlpExtractor
from torch import nn

import training.round_policy as round_policy
from agent.round_observation import ACTOR_KEY, CRITIC_KEY, ObservationConfig
from backends.mock_quad.backend import MockQuadBackend
from envs.round_env import RoundEnv
from training.round_policy import KEY_ORDER, RoundExtractor, RoundPolicy, SplitMlpExtractor

IC = {"x0": 0.3, "y0": 0.6, "width": 0.2}
HOLD = 1


# --- The instance, a policy on it, and a batch of its observations ------------------

def _env() -> RoundEnv:
    """The two-cell instance of the explanation file's section 3.1.2."""
    backend = MockQuadBackend(
        nx=2, ny=1, periodic=(True, False), lo=(0.0, 0.0), hi=(2.0, 1.0), max_level=2, dt_base=0.25,
        rotate_cells=False,
    )
    config = ObservationConfig(
        indicator="ind_b", reduction="scalar", alpha=0.1, element_budget=8, normalization="relative",
        floor_decades=8.0,
    )
    return RoundEnv(
        backend, observation_config=config, beta=2, coarsen_rule="first_decider", max_level=2, n_intervals=2,
        amr_freq=2, initial_level=0, p_ur=10.0, p_or=5.0, p_cr=2.0, lambda_local=0.1, lambda_global=1.0,
    )


def _schedule(_progress: float) -> float:
    return 3e-4


def _space(n_actor: int = 17, n_critic: int = 17) -> spaces.Dict:
    return spaces.Dict(
        {
            ACTOR_KEY: spaces.Box(low=-8.0, high=8.0, shape=(n_actor,), dtype=np.float32),
            CRITIC_KEY: spaces.Box(low=-8.0, high=8.0, shape=(n_critic,), dtype=np.float32),
        }
    )


def _policy(cls=RoundPolicy, space: spaces.Dict | None = None, seed: int = 0, **arguments):
    th.manual_seed(seed)
    return cls(_space() if space is None else space, spaces.Discrete(3), _schedule, **arguments)


def _observations() -> dict[str, th.Tensor]:
    """The eight observations of the instance's all-holds episode, as one
    batch of float32 tensors."""
    env = _env()
    obs, _ = env.reset(options={"ic": IC})
    seen, terminated = [], False
    while not terminated:
        seen.append(obs)
        obs, _, terminated, _, _ = env.step(HOLD)
    assert len(seen) == 8
    return {key: th.as_tensor(np.stack([one[key] for one in seen])) for key in KEY_ORDER}


def _random(n_actor: int, n_critic: int, batch: int = 4) -> dict[str, th.Tensor]:
    generator = th.Generator().manual_seed(1)
    return {
        ACTOR_KEY: th.randn(batch, n_actor, generator=generator),
        CRITIC_KEY: th.randn(batch, n_critic, generator=generator),
    }


def _moved(obs: dict[str, th.Tensor], key: str) -> dict[str, th.Tensor]:
    """The same batch with 1 added to every feature of one key."""
    moved = {name: tensor.clone() for name, tensor in obs.items()}
    moved[key] = moved[key] + 1
    return moved


def _sides(policy, obs: dict[str, th.Tensor]) -> tuple[dict[str, th.Tensor], dict[str, th.Tensor]]:
    """What the actor's network produces and what the critic's produces,
    by each path of the parent class that reaches them."""
    batch = obs[ACTOR_KEY].shape[0]
    policy.set_training_mode(False)
    with th.no_grad():
        _, value_forward, logp_forward = policy(obs, deterministic=True)
        value_evaluate, logp_evaluate, entropy = policy.evaluate_actions(obs, th.full((batch,), HOLD))
        distribution = policy.get_distribution(obs)
        logp_all = th.stack([distribution.log_prob(th.full((batch,), action)) for action in range(3)])
        value_alone = policy.predict_values(obs)
    actor = {
        "forward": logp_forward,
        "evaluate_actions": th.stack([logp_evaluate, entropy]),
        "get_distribution": logp_all,
    }
    critic = {"forward": value_forward, "evaluate_actions": value_evaluate, "predict_values": value_alone}
    return actor, critic


def _count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


# --- The extractor (S2) ------------------------------------------------------------

def test_the_key_order_is_the_actor_then_the_critic():
    assert KEY_ORDER == (ACTOR_KEY, CRITIC_KEY) == ("actor", "critic")


def test_the_extractor_places_the_actor_key_first():
    obs = _moved(_observations(), CRITIC_KEY)
    assert not th.equal(obs[ACTOR_KEY], obs[CRITIC_KEY])
    features = RoundExtractor(_env().observation_space)(obs)
    assert features.shape == (8, 34)
    assert th.equal(features[:, :17], obs[ACTOR_KEY])
    assert th.equal(features[:, 17:], obs[CRITIC_KEY])


@pytest.mark.parametrize("n_actor, n_critic", [(17, 17), (17, 5), (3, 11)])
def test_the_extractor_knows_the_two_widths(n_actor, n_critic):
    extractor = RoundExtractor(_space(n_actor, n_critic))
    assert (extractor.n_actor, extractor.n_critic, extractor.features_dim) == (n_actor, n_critic, n_actor + n_critic)
    obs = _random(n_actor, n_critic)
    features = extractor(obs)
    assert th.equal(features[:, :n_actor], obs[ACTOR_KEY]) and th.equal(features[:, n_actor:], obs[CRITIC_KEY])


def test_the_extractor_has_no_parameters():
    assert _count(RoundExtractor(_space())) == 0


def test_the_extractor_reads_the_environments_space():
    extractor = RoundExtractor(_env().observation_space)
    assert (extractor.n_actor, extractor.n_critic) == (17, 17)


_BOX = spaces.Box(low=-1.0, high=1.0, shape=(17,), dtype=np.float32)
_WHOLE = "observation_space: expected a Dict with the keys ['actor', 'critic'], got "
BAD_SPACES = {
    "not a Dict": (_BOX, _WHOLE),
    "no critic": (spaces.Dict({ACTOR_KEY: _BOX}), _WHOLE),
    "no actor": (spaces.Dict({CRITIC_KEY: _BOX}), _WHOLE),
    "a third key": (spaces.Dict({ACTOR_KEY: _BOX, CRITIC_KEY: _BOX, "aux": _BOX}), _WHOLE),
    "a critic of two dimensions": (
        spaces.Dict({ACTOR_KEY: _BOX, CRITIC_KEY: spaces.Box(low=-1.0, high=1.0, shape=(17, 2), dtype=np.float32)}),
        "observation_space['critic']: expected a Box of one dimension, got ",
    ),
    "an actor that is no Box": (
        spaces.Dict({ACTOR_KEY: spaces.Discrete(3), CRITIC_KEY: _BOX}),
        "observation_space['actor']: expected a Box of one dimension, got ",
    ),
}


@pytest.mark.parametrize("case", list(BAD_SPACES))
def test_the_extractor_refuses_another_space_by_name(case):
    space, start = BAD_SPACES[case]
    with pytest.raises(ValueError) as caught:
        RoundExtractor(space)
    assert str(caught.value).startswith(start)


# --- The split extractor (S3) ------------------------------------------------------

def _first_in(net: nn.Sequential) -> int | None:
    linear = [module for module in net if isinstance(module, nn.Linear)]
    return linear[0].in_features if linear else None


SPLITS = [
    # n_actor, n_critic, net_arch, policy's first input, value's first input, latent pi, latent vf, parameters
    (17, 17, [64, 64], 17, 17, 64, 64, 10624),
    (17, 5, [64, 64], 17, 5, 64, 64, 9856),
    (17, 5, {"pi": [32], "vf": [8, 8]}, 17, 5, 32, 8, 696),
    (17, 5, {"pi": [32]}, 17, None, 32, 5, 576),
    (17, 5, {}, None, None, 17, 5, 0),
    (17, 5, [], None, None, 17, 5, 0),
]


@pytest.mark.parametrize("n_actor, n_critic, net_arch, in_pi, in_vf, latent_pi, latent_vf, parameters", SPLITS)
def test_the_split_builds_each_network_on_its_own_width(
    n_actor, n_critic, net_arch, in_pi, in_vf, latent_pi, latent_vf, parameters
):
    split = SplitMlpExtractor(n_actor, n_critic, net_arch, nn.Tanh)
    assert (_first_in(split.policy_net), _first_in(split.value_net)) == (in_pi, in_vf)
    assert (split.latent_dim_pi, split.latent_dim_vf) == (latent_pi, latent_vf)
    assert _count(split) == parameters
    pi, vf = split(th.zeros(3, n_actor + n_critic))
    assert pi.shape == (3, latent_pi) and vf.shape == (3, latent_vf)


def test_every_linear_layer_is_followed_by_the_activation():
    split = SplitMlpExtractor(17, 5, {"pi": [32, 16], "vf": [8]}, nn.ReLU)
    assert [type(module).__name__ for module in split.policy_net] == ["Linear", "ReLU", "Linear", "ReLU"]
    assert [type(module).__name__ for module in split.value_net] == ["Linear", "ReLU"]
    assert [module.out_features for module in split.policy_net if isinstance(module, nn.Linear)] == [32, 16]


def test_the_split_has_the_stock_extractors_face():
    split = SplitMlpExtractor(17, 17, [64, 64], nn.Tanh)
    stock = MlpExtractor(34, [64, 64], nn.Tanh, device="cpu")
    assert [name for name, _ in split.named_parameters()] == [name for name, _ in stock.named_parameters()]
    for name in ("forward", "forward_actor", "forward_critic", "latent_dim_pi", "latent_dim_vf"):
        assert hasattr(split, name) and hasattr(stock, name)


def test_forward_is_the_two_halves():
    th.manual_seed(0)
    split = SplitMlpExtractor(17, 5, [16], nn.Tanh)
    features = th.randn(4, 22)
    pi, vf = split(features)
    assert th.equal(pi, split.forward_actor(features)) and th.equal(vf, split.forward_critic(features))


def test_each_half_reads_its_own_columns_and_no_others():
    th.manual_seed(0)
    split = SplitMlpExtractor(17, 5, [16], nn.Tanh)
    features = th.randn(4, 22)
    critic_moved, actor_moved = features.clone(), features.clone()
    critic_moved[:, 17:] += 1
    actor_moved[:, :17] += 1
    pi, vf = split(features)
    assert th.equal(split.forward_actor(critic_moved), pi) and not th.equal(split.forward_critic(critic_moved), vf)
    assert th.equal(split.forward_critic(actor_moved), vf) and not th.equal(split.forward_actor(actor_moved), pi)
    for column in range(22):
        one = features.clone()
        one[:, column] += 1
        assert th.equal(split.forward_actor(one), pi) == (column >= 17)
        assert th.equal(split.forward_critic(one), vf) == (column < 17)


BAD_SPLITS = [
    ((0, 5, [8]), "n_actor: 0, expected an integer of at least 1"),
    ((17, 0, [8]), "n_critic: 0, expected an integer of at least 1"),
    ((17, True, [8]), "n_critic: True, expected an integer of at least 1"),
    ((17.0, 5, [8]), "n_actor: 17.0, expected an integer of at least 1"),
    ((17, 5, {"pi": [8], "qf": [8]}), "net_arch: unknown keys ['qf'], expected 'pi' and 'vf'"),
]


@pytest.mark.parametrize("arguments, message", BAD_SPLITS, ids=[message.split(",")[0] for _, message in BAD_SPLITS])
def test_the_split_refuses_a_bad_argument_by_name(arguments, message):
    with pytest.raises(ValueError) as caught:
        SplitMlpExtractor(*arguments, nn.Tanh)
    assert str(caught.value) == message


# --- The policy: what it is made of (S4, S6) ---------------------------------------

def test_the_policy_is_the_maskable_multi_input_policy_with_the_two_pieces():
    policy = _policy()
    assert isinstance(policy, MaskableMultiInputActorCriticPolicy)
    assert type(policy.features_extractor) is RoundExtractor
    assert type(policy.mlp_extractor) is SplitMlpExtractor
    assert policy.share_features_extractor is True
    assert policy.pi_features_extractor is policy.features_extractor is policy.vf_features_extractor


def test_the_default_networks():
    policy = _policy()
    assert policy.net_arch == {"pi": [64, 64], "vf": [64, 64]}
    assert [(name, tuple(parameter.shape)) for name, parameter in policy.named_parameters()] == [
        ("mlp_extractor.policy_net.0.weight", (64, 17)),
        ("mlp_extractor.policy_net.0.bias", (64,)),
        ("mlp_extractor.policy_net.2.weight", (64, 64)),
        ("mlp_extractor.policy_net.2.bias", (64,)),
        ("mlp_extractor.value_net.0.weight", (64, 17)),
        ("mlp_extractor.value_net.0.bias", (64,)),
        ("mlp_extractor.value_net.2.weight", (64, 64)),
        ("mlp_extractor.value_net.2.bias", (64,)),
        ("action_net.weight", (3, 64)),
        ("action_net.bias", (3,)),
        ("value_net.weight", (1, 64)),
        ("value_net.bias", (1,)),
    ]


COUNTS = [
    (None, 10884),
    ([64, 64], 10884),
    ([256, 256], 141828),
    ({"pi": [32], "vf": [8, 8]}, 900),
    ([], 72),
]


@pytest.mark.parametrize("net_arch, parameters", COUNTS, ids=[str(net_arch) for net_arch, _ in COUNTS])
def test_the_parameter_count(net_arch, parameters):
    assert _count(_policy(net_arch=net_arch)) == parameters


def test_the_stock_policy_is_larger_by_the_columns_each_network_no_longer_reads():
    stock = _policy(MaskableMultiInputActorCriticPolicy)
    assert _count(stock) == 13060
    assert _count(stock) - _count(_policy()) == 2 * 17 * 64


def test_the_critics_width_may_differ_from_the_actors():
    policy = _policy(space=_space(17, 5))
    assert _first_in(policy.mlp_extractor.policy_net) == 17 and _first_in(policy.mlp_extractor.value_net) == 5
    assert policy.features_dim == 22


def test_the_policy_is_rebuilt_from_its_own_constructor_parameters():
    policy = _policy(net_arch=[16, 16])
    again = RoundPolicy(**policy._get_constructor_parameters())
    assert type(again.mlp_extractor) is SplitMlpExtractor
    assert [tuple(p.shape) for p in again.parameters()] == [tuple(p.shape) for p in policy.parameters()]


BAD_POLICIES = [
    (
        {"share_features_extractor": False},
        "share_features_extractor: False, expected True: the split is made on one feature vector",
    ),
    (
        {"features_extractor_class": CombinedExtractor},
        "features_extractor_class: CombinedExtractor, expected RoundExtractor",
    ),
]


@pytest.mark.parametrize("arguments, message", BAD_POLICIES, ids=[next(iter(a)) for a, _ in BAD_POLICIES])
def test_the_policy_refuses_what_would_break_the_split(arguments, message):
    with pytest.raises(ValueError) as caught:
        _policy(**arguments)
    assert str(caught.value) == message


def test_the_policy_refuses_a_space_the_extractor_refuses():
    with pytest.raises(ValueError, match=r"^observation_space: expected a Dict"):
        _policy(space=spaces.Dict({ACTOR_KEY: _BOX}))


# --- The route (S5) ----------------------------------------------------------------

@pytest.mark.parametrize("net_arch", [None, [16], {"pi": [32], "vf": [8, 8]}, []], ids=str)
def test_the_actor_does_not_read_the_critic_key(net_arch):
    policy, obs = _policy(net_arch=net_arch), _observations()
    actor, critic = _sides(policy, obs)
    moved_actor, moved_critic = _sides(policy, _moved(obs, CRITIC_KEY))
    for path in actor:
        assert th.equal(moved_actor[path], actor[path]), path
    for path in critic:
        assert not th.equal(moved_critic[path], critic[path]), path


@pytest.mark.parametrize("net_arch", [None, [16], {"pi": [32], "vf": [8, 8]}, []], ids=str)
def test_the_critic_does_not_read_the_actor_key(net_arch):
    policy, obs = _policy(net_arch=net_arch), _observations()
    actor, critic = _sides(policy, obs)
    moved_actor, moved_critic = _sides(policy, _moved(obs, ACTOR_KEY))
    for path in critic:
        assert th.equal(moved_critic[path], critic[path]), path
    for path in actor:
        assert not th.equal(moved_actor[path], actor[path]), path


def test_the_route_holds_at_unequal_widths():
    policy, obs = _policy(space=_space(17, 5)), _random(17, 5)
    actor, critic = _sides(policy, obs)
    after_critic = _sides(policy, _moved(obs, CRITIC_KEY))
    after_actor = _sides(policy, _moved(obs, ACTOR_KEY))
    assert all(th.equal(after_critic[0][path], actor[path]) for path in actor)
    assert all(th.equal(after_actor[1][path], critic[path]) for path in critic)
    assert not any(th.equal(after_critic[1][path], critic[path]) for path in critic)
    assert not any(th.equal(after_actor[0][path], actor[path]) for path in actor)


@pytest.mark.parametrize("key", [ACTOR_KEY, CRITIC_KEY])
def test_the_stock_policy_fails_the_route(key):
    """The control: on sb3-contrib's own multi-input policy both networks
    move whichever key moves, so the comparison above can see a leak."""
    policy, obs = _policy(MaskableMultiInputActorCriticPolicy), _observations()
    actor, critic = _sides(policy, obs)
    moved_actor, moved_critic = _sides(policy, _moved(obs, key))
    assert not any(th.equal(moved_actor[path], actor[path]) for path in actor)
    assert not any(th.equal(moved_critic[path], critic[path]) for path in critic)


def _gradients(policy, loss: th.Tensor) -> dict[str, th.Tensor | None]:
    policy.zero_grad()
    loss.backward()
    return {name: parameter.grad for name, parameter in policy.named_parameters()}


def test_the_actors_loss_reaches_no_parameter_of_the_critic():
    policy, obs = _policy(), _observations()
    _, log_prob, _ = policy.evaluate_actions(obs, th.full((8,), HOLD))
    gradients = _gradients(policy, log_prob.sum())
    for name, gradient in gradients.items():
        if name.startswith(("mlp_extractor.value_net", "value_net")):
            assert gradient is None, name
        else:
            assert gradient is not None and bool(gradient.abs().sum() > 0), name


def test_the_critics_loss_reaches_no_parameter_of_the_actor():
    policy, obs = _policy(), _observations()
    values, _, _ = policy.evaluate_actions(obs, th.full((8,), HOLD))
    gradients = _gradients(policy, values.sum())
    for name, gradient in gradients.items():
        if name.startswith(("mlp_extractor.policy_net", "action_net")):
            assert gradient is None, name
        else:
            assert gradient is not None and bool(gradient.abs().sum() > 0), name


@pytest.mark.parametrize("action", [0, 1, 2])
def test_a_mask_that_opens_one_action_gets_that_action(action):
    policy = _policy()
    obs, _ = _env().reset(options={"ic": IC})
    mask = np.zeros(3, dtype=bool)
    mask[action] = True
    for deterministic in (True, False):
        chosen, _ = policy.predict(obs, action_masks=mask, deterministic=deterministic)
        assert int(chosen) == action


def test_a_closed_action_is_never_sampled():
    policy = _policy()
    obs, _ = _env().reset(options={"ic": IC})
    mask = np.array([False, True, True])
    th.manual_seed(0)
    chosen = {int(policy.predict(obs, action_masks=mask, deterministic=False)[0]) for _ in range(200)}
    assert chosen == {1, 2}


# --- Under MaskablePPO -------------------------------------------------------------

def _model(seed: int = 0) -> MaskablePPO:
    return MaskablePPO(
        RoundPolicy, Monitor(_env()), n_steps=32, batch_size=16, n_epochs=2, seed=seed, device="cpu",
        policy_kwargs={"net_arch": [16, 16]}, verbose=0,
    )


@pytest.fixture(scope="module")
def trained() -> MaskablePPO:
    """One model trained three rollouts on the instance, shared by the
    tests of this group; none of them changes it."""
    model = _model()
    model.learn(total_timesteps=96)
    return model


def test_it_trains_under_maskable_ppo_with_no_wrapper_for_the_masks(trained):
    assert type(trained.policy) is RoundPolicy
    assert (trained.num_timesteps, trained._n_updates) == (96, 6)
    assert type(trained.get_env().envs[0]) is Monitor and type(trained.get_env().envs[0].env) is RoundEnv


def test_training_moves_both_networks():
    model = _model()
    before = {name: parameter.detach().clone() for name, parameter in model.policy.named_parameters()}
    model.learn(total_timesteps=96)
    for name, parameter in model.policy.named_parameters():
        assert not th.equal(parameter, before[name]), name


def test_the_trained_policy_is_still_routed(trained):
    obs = _observations()
    actor, critic = _sides(trained.policy, obs)
    after_critic = _sides(trained.policy, _moved(obs, CRITIC_KEY))
    after_actor = _sides(trained.policy, _moved(obs, ACTOR_KEY))
    assert all(th.equal(after_critic[0][path], actor[path]) for path in actor)
    assert all(th.equal(after_actor[1][path], critic[path]) for path in critic)
    assert not any(th.equal(after_critic[1][path], critic[path]) for path in critic)
    assert not any(th.equal(after_actor[0][path], actor[path]) for path in actor)


def test_a_saved_model_holds_the_class_and_net_arch_alone(trained, tmp_path):
    trained.save(str(tmp_path / "model"))
    data, _, _ = load_from_zip_file(str(tmp_path / "model"))
    assert data["policy_class"] is RoundPolicy
    assert data["policy_kwargs"] == {"net_arch": [16, 16]}


def test_a_saved_model_loads_as_the_same_policy(trained, tmp_path):
    trained.save(str(tmp_path / "model"))
    loaded = MaskablePPO.load(str(tmp_path / "model"), device="cpu")
    assert type(loaded.policy) is RoundPolicy and type(loaded.policy.mlp_extractor) is SplitMlpExtractor
    obs = _observations()
    for mine, theirs in zip(_sides(trained.policy, obs), _sides(loaded.policy, obs), strict=True):
        for path in mine:
            assert th.equal(mine[path], theirs[path]), path


# --- What the module imports -------------------------------------------------------

def test_the_module_imports_one_name_of_the_core_and_nothing_of_the_environment():
    names = set()
    for node in ast.walk(ast.parse(Path(round_policy.__file__).read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    roots = {name.split(".")[0] for name in names}
    assert not roots & {"envs", "backends", "drivers", "deployment", "contract", "training"}
    assert {name for name in names if name.split(".")[0] == "agent"} == {"agent.round_observation"}
