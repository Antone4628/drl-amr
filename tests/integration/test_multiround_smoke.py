"""End-to-end integration smoke for the multiround stack (RESTRUCTURE Phase 7).

Ports the Stage 1A Phase-2.5 environment smoke (drl-amr-1d
tests/multiround_buildout/smoke_test_multiround.py) onto the contract-layered
stack: it drives a real Python1DSolverContract -> AgentCore -> MultiroundDriver
-> MultiroundEnv chain through full episodes. This is the integration-level
counterpart to tests/drivers/test_driver.py (which drives the driver against a
*fake* contract): here the real python_1d backend is in the loop, so the test
catches wiring/NaN bugs the fakes cannot.

Five checks (escalating scope):
    - episode structure + info-dict contract (deterministic action cycle)
    - per-IC completion (all seven Stage 1A ICs run to termination)
    - multi-episode random-action stress (no crash, no NaN)
    - deterministic transition counts (all-hold, fixed 4-element mesh)
    - reward invariants (finiteness, global <= 0, global only at boundaries,
      reward == lambda_local*r_local + lambda_global*r_global)

The env uses error_indicator="zz_style" (D-048) so errors are nonzero at t=0
without pre-advance (retired, D-049).
"""

import numpy as np

from agent.core import AgentCore
from agent.masking import ACTION_COARSEN, ACTION_HOLD, ACTION_REFINE
from backends.python_1d.contract_impl import Python1DSolverContract
from backends.python_1d.solvers.dg_advection_solver_multiround import DGAdvectionSolver
from drivers.multiround_driver import MultiroundDriver
from envs.multiround_env import MultiroundEnv

# Base mesh + IC pool match the new-repo convention (test_zz_indicator_sanity,
# deployment build): 4 non-uniform elements symmetric about x=0.
_BASE_XELEM = np.array([-1.0, -0.4, 0.0, 0.4, 1.0])
_IC_POOL = [1, 10, 12, 13, 14, 15, 16]

_RESET_KEYS = {"icase", "n_active", "e_max", "e_min", "resource_usage"}
_STEP_KEYS = {
    "element_id", "action", "pre_action_error", "n_active_pre", "n_active_post",
    "n_cascade", "resource_usage", "r_local", "r_global", "reward",
    "transition", "queue_skipped", "remesh_step", "round_number", "episode_steps",
}

_STEP_CAP = 2000  # generous upper bound; tripping it means a stuck episode
_N_STRESS_EPISODES = 50


def make_env(*, n_remesh=4, max_level=3, element_budget=30, initial_refinement_level=0):
    """Build the real contract -> core -> driver -> env chain for the smoke."""
    solver = DGAdvectionSolver(
        nop=4,
        xelem=_BASE_XELEM.copy(),
        max_elements=120,
        max_level=max_level,
        icase=1,
        balance=False,
    )
    contract = Python1DSolverContract(solver)
    core = AgentCore(contract, element_budget=element_budget, error_indicator="zz_style")
    driver = MultiroundDriver(
        core,
        n_remesh=n_remesh,
        step_domain_fraction=0.05,
        initial_refinement_level=initial_refinement_level,
        ic_pool=list(_IC_POOL),
    )
    return MultiroundEnv(driver)


def _random_valid_action(env, mask):
    """Pick a uniformly-random valid action from the mask (env RNG)."""
    valid = np.where(mask)[0]
    return int(env.np_random.choice(valid))


def test_episode_structure_and_info_contract():
    """One episode with a refine/hold/coarsen action cycle: shapes, info keys,
    transition bookkeeping, and the n_remesh interval structure."""
    env = make_env(n_remesh=4, max_level=3)
    obs, info = env.reset(seed=0, options={"icase": 1})

    assert obs.shape == (8,), f"reset obs shape {obs.shape}"
    assert np.all(np.isfinite(obs)), f"non-finite reset obs: {obs}"
    assert not (_RESET_KEYS - set(info)), f"missing reset info keys: {_RESET_KEYS - set(info)}"

    cycle = [ACTION_REFINE, ACTION_HOLD, ACTION_COARSEN]
    interval_count = 0
    done_count = 0
    steps = 0
    terminated = False
    while not terminated:
        mask = env.action_masks()
        assert mask.shape == (3,), f"mask shape {mask.shape}"
        assert mask.dtype == bool, f"mask dtype {mask.dtype}"
        assert mask[ACTION_HOLD], "hold must always be valid"

        preferred = cycle[steps % len(cycle)]
        action = preferred if mask[preferred] else ACTION_HOLD

        obs, reward, terminated, truncated, info = env.step(action)
        steps += 1

        assert obs.shape == (8,), f"step {steps}: obs shape {obs.shape}"
        assert np.all(np.isfinite(obs)), f"step {steps}: non-finite obs"
        assert np.isfinite(reward), f"step {steps}: non-finite reward"
        assert truncated is False, f"step {steps}: truncated should be False"
        assert not (_STEP_KEYS - set(info)), f"step {steps}: missing info keys {_STEP_KEYS - set(info)}"

        transition = info["transition"]
        assert transition in ("element", "interval", "done"), f"step {steps}: bad transition {transition!r}"
        if transition in ("interval", "done"):
            assert "solver_T" in info and "solver_n_steps" in info, f"step {steps}: missing solver advance info"
        if transition == "interval":
            interval_count += 1
        elif transition == "done":
            done_count += 1

        assert steps <= _STEP_CAP, f"episode exceeded {_STEP_CAP} steps — likely stuck"

    assert interval_count == 3, f"expected 3 interval transitions, got {interval_count}"
    assert done_count == 1, f"expected 1 done transition, got {done_count}"
    assert terminated is True


def test_per_ic_completion():
    """Each of the seven Stage 1A ICs completes a full random-action episode."""
    env = make_env()
    env.reset(seed=1)  # initialize env.np_random
    for icase in _IC_POOL:
        obs, info = env.reset(options={"icase": icase})
        assert obs.shape == (8,), f"icase={icase}: obs shape {obs.shape}"
        assert info["icase"] == icase

        steps = 0
        terminated = False
        while not terminated:
            mask = env.action_masks()
            obs, reward, terminated, truncated, info = env.step(_random_valid_action(env, mask))
            steps += 1
            assert np.isfinite(reward), f"icase={icase} step {steps}: non-finite reward"
            assert np.all(np.isfinite(obs)), f"icase={icase} step {steps}: non-finite obs"
            assert steps <= _STEP_CAP, f"icase={icase}: exceeded {_STEP_CAP} steps"


def test_multi_episode_random_stress():
    """Many episodes with the env's own IC sampling + random valid actions:
    no crashes, no NaN. (IC coverage is checked per-IC above, so this drops the
    old 'all ICs sampled' RNG assertion to stay deterministic.)"""
    env = make_env()
    env.reset(seed=2)
    for ep in range(_N_STRESS_EPISODES):
        env.reset()
        steps = 0
        terminated = False
        while not terminated:
            mask = env.action_masks()
            obs, reward, terminated, truncated, info = env.step(_random_valid_action(env, mask))
            steps += 1
            assert np.isfinite(reward), f"ep {ep + 1} step {steps}: non-finite reward"
            assert np.all(np.isfinite(obs)), f"ep {ep + 1} step {steps}: non-finite obs"
            assert steps <= _STEP_CAP, f"ep {ep + 1}: exceeded {_STEP_CAP} steps"


def test_transition_counts_all_hold():
    """All-hold actions hold the mesh at the 4-element base, making every count
    deterministic: with n_remesh=4, max_level=3, n_active=4 -> 48 total steps,
    44 'element' / 3 'interval' / 1 'done', and 8 round advances."""
    n_remesh, max_level = 4, 3
    env = make_env(n_remesh=n_remesh, max_level=max_level, initial_refinement_level=0)
    _, info = env.reset(seed=0, options={"icase": 1})
    n_active = info["n_active"]
    assert n_active == 4, f"expected 4 base elements, got {n_active}"

    counts = {"element": 0, "interval": 0, "done": 0}
    round_advances = 0
    prev_round = 1
    steps = 0
    terminated = False
    while not terminated:
        obs, reward, terminated, truncated, info = env.step(ACTION_HOLD)
        steps += 1
        counts[info["transition"]] += 1
        if info["round_number"] == prev_round + 1:
            round_advances += 1
        prev_round = info["round_number"]
        assert steps <= _STEP_CAP

    steps_per_interval = n_active * max_level
    expected_total = steps_per_interval * n_remesh
    assert steps == expected_total, f"total steps {steps} != {expected_total}"
    assert counts["interval"] == n_remesh - 1, f"interval {counts['interval']} != {n_remesh - 1}"
    assert counts["done"] == 1, f"done {counts['done']} != 1"
    assert counts["element"] == expected_total - (n_remesh - 1) - 1, "element count mismatch"
    assert round_advances == (max_level - 1) * n_remesh, (
        f"round advances {round_advances} != {(max_level - 1) * n_remesh}"
    )


def test_reward_invariants():
    """Reward structure holds every step: r_local finite, r_global <= 0,
    r_global nonzero only on interval/done, and the dual-reward combination."""
    env = make_env(n_remesh=4, max_level=3)
    env.reset(seed=3, options={"icase": 1})
    core = env.driver.core
    lam_l, lam_g = core.lambda_local, core.lambda_global

    steps = 0
    terminated = False
    while not terminated:
        mask = env.action_masks()
        obs, reward, terminated, truncated, info = env.step(_random_valid_action(env, mask))
        steps += 1

        r_local = info["r_local"]
        r_global = info["r_global"]
        transition = info["transition"]

        assert np.isfinite(r_local), f"step {steps}: non-finite r_local"
        assert r_global <= 0.0, f"step {steps}: positive r_global {r_global}"
        if r_global != 0.0:
            assert transition in ("interval", "done"), f"step {steps}: nonzero r_global on '{transition}'"

        expected = lam_l * r_local + lam_g * r_global
        assert abs(reward - expected) < 1e-10, f"step {steps}: reward {reward} != combine {expected}"
        assert steps <= _STEP_CAP