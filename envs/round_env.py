"""The environment of the 2D agent core: one element's decision per step.

RoundEnv puts the round loop of the 2D core (agent/round_*.py) behind the
two calls an RL library makes. The library owns the loop, so the
environment stops after every decision and holds what the loop would have
held: the interval's thresholds, the round's cache, order and distances,
the cursor into the order, and the RoundState.

One remesh interval is max_level rounds (D-018). A round is one read of
the backend, then one step() per position the queue asks; when the queue
is exhausted the round's marks are applied. After the interval's last
round the backend advances one interval. An episode is n_intervals of
them.

The reward of a step is lambda_local times the decision's local reward;
the step that ends an interval adds lambda_global times the interval's
global reward (D-007).

The environment builds one ObservationSpec and hands its values to every
function of the core, so what a policy trains with is what is saved beside
its checkpoints (agent/round_spec.py).

This module belongs to the 2D stack (AGENT_2D_ROADMAP item 3.1) and stands
beside envs/multiround_env.py, the sequential environment, which is not on
this path.
"""
from __future__ import annotations

import math
from numbers import Integral, Real
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from agent.round_cache import read_round
from agent.round_commit import commit, overshoot, start_round
from agent.round_mask import action_mask
from agent.round_observation import ObservationConfig, observation, observation_bounds
from agent.round_queue import next_undecided, round_order, round_thresholds, zone_distance
from agent.round_reward import global_reward, local_reward
from agent.round_spec import ObservationSpec, spec_violations
from contract.backend import BackendFailure, SolverBackend


class RoundEnv(gym.Env):
    """A backend's remesh intervals as a Gymnasium environment.

    An action is 0 (coarsen), 1 (hold) or 2 (refine) for the element the
    queue is asking; action_masks says which are open. An observation is a
    dict with the keys "actor" and "critic" (agent/round_observation.py).

    Args:
        backend: The backend to drive. Only its verbs are called.
        observation_config: What the observation is built with.
        beta: The hysteresis exponent of the thresholds, above 1.
        coarsen_rule: One of agent.round_mask.COARSEN_RULES.
        max_level: The backend's finest level. It is not in describe(), so
            it is given here, and the first reset checks it against the
            first state. Rounds per interval equal it.
        n_intervals: Remesh intervals per episode.
        amr_freq: Base steps per remesh interval, as the backend's reset
            takes it.
        initial_level: The uniform level an episode starts at.
        cadence: Substeps between the samples of the interval's
            accumulator, as step_interval takes it.
        p_ur: Reward weight of an element left, or coarsened, above the
            neutral zone.
        p_or: Reward weight of an element left, or refined, below it.
        p_cr: Reward weight of a coarsen below it.
        lambda_local: Scale of the local reward, paid at every step.
        lambda_global: Scale of the global reward, paid at the step that
            ends an interval.

    Raises:
        ValueError: Naming the argument; or as ObservationSpec raises.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        backend: SolverBackend,
        *,
        observation_config: ObservationConfig,
        beta: float,
        coarsen_rule: str,
        max_level: int,
        n_intervals: int,
        amr_freq: int,
        initial_level: int = 1,
        cadence: int = 1,
        p_ur: float,
        p_or: float,
        p_cr: float,
        lambda_local: float,
        lambda_global: float,
    ) -> None:
        super().__init__()
        counts = {"n_intervals": (n_intervals, 1), "amr_freq": (amr_freq, 1), "initial_level": (initial_level, 0),
                  "cadence": (cadence, 1)}
        for name, (value, least) in counts.items():
            if isinstance(value, bool) or not isinstance(value, Integral) or value < least:
                raise ValueError(f"{name}: {value!r}, expected an integer of at least {least}")
        scales = {"p_ur": p_ur, "p_or": p_or, "p_cr": p_cr, "lambda_local": lambda_local,
                  "lambda_global": lambda_global}
        for name, value in scales.items():
            if isinstance(value, bool) or not isinstance(value, Real) or not 0 <= value < math.inf:
                raise ValueError(f"{name}: {value!r}, expected a finite number, not negative")

        described = backend.describe()
        self.backend = backend
        # The one spec of the run. gym.Env already has an attribute named
        # spec, so this one carries the noun's full name.
        self.observation_spec = ObservationSpec(
            observation=observation_config,
            beta=beta,
            coarsen_rule=coarsen_rule,
            ndim=described.ndim,
            max_level=max_level,
            n_rounds=max_level,
            n_comp=1 if observation_config.reduction == "scalar" else described.nvar,
        )
        self.n_intervals = int(n_intervals)
        self.amr_freq = int(amr_freq)
        self.initial_level = int(initial_level)
        self.cadence = int(cadence)
        self.lambda_local = float(lambda_local)
        self.lambda_global = float(lambda_global)
        self._local_weights = {"p_ur": float(p_ur), "p_or": float(p_or), "p_cr": float(p_cr)}
        self._global_weights = {"p_ur": float(p_ur), "p_or": float(p_or)}
        self._ic_params = dict(described.ic_params)

        bounds = observation_bounds(described.ndim, observation_config)
        self.observation_space = spaces.Dict(
            {key: spaces.Box(low=low, high=high, dtype=np.float32) for key, (low, high) in bounds.items()}
        )
        self.action_space = spaces.Discrete(3)

        # What moves between calls. Set by reset; _position is None when no
        # episode is running.
        self._spec_checked = False
        self._interval = 0
        self._state = None
        self._cache = None
        self._pair = None
        self._order = None
        self._distance = None
        self._round_state = None
        self._position = None
        self._cursor = 0
        self._skipped = 0

    # --- The round: what item 4.1's driver will hold --------------------------------

    def _begin_round(self, round_index: int) -> None:
        """Read the backend once and set up the round's walk.

        In round 1 the thresholds are taken from this read and become the
        interval's: every later round of the interval orders its queue and
        measures its distances under the same pair (D-021).

        Raises:
            BackendFailure: If the backend's arrays are not finite.
        """
        spec = self.observation_spec
        config = spec.observation
        self._cache = read_round(self.backend, self._state, [config.indicator], config.reduction)
        if round_index == 1:
            self._pair = round_thresholds(self._cache, config, spec.beta)
        self._order = round_order(self._cache, config, *self._pair)
        self._distance = zone_distance(self._cache.errors[config.indicator], config, *self._pair)
        self._round_state = start_round(self._state, round_index, spec.n_rounds)
        self._position, self._cursor, self._skipped = next_undecided(self._state, self._round_state, self._order, 0)

    def _advance(self, info: dict[str, Any]) -> tuple[str, float]:
        """Move to the next position to ask, across whatever ends on the way.

        Called after a commit. While the round's queue still holds an
        undecided position, that is all that happens. When it is exhausted
        the round's marks are applied; then either the next round begins,
        or, after the interval's last round, the backend advances one
        interval, the interval's global reward is computed, and the next
        interval begins unless the episode is over.

        Args:
            info: The step's info dict. At a round's end it gains overshoot
                (the realized count above the committed count), skipped
                (positions the round passed over as already decided) and
                n_active (the realized count).

        Returns:
            What ended: "element" (nothing), "round", "interval" or "done";
            and the interval's global reward, 0.0 unless an interval ended.
            After "done" no position is asked.

        Raises:
            BackendFailure: If a verb fails or an array is not finite.
        """
        spec = self.observation_spec
        config = spec.observation
        self._position, self._cursor, passed = next_undecided(
            self._state, self._round_state, self._order, self._cursor
        )
        self._skipped += passed
        if self._position is not None:
            return "element", 0.0

        # The queue is exhausted: the record is the round's marks.
        round_index = self._round_state.round_index
        new_state, _ = self.backend.apply_marks(self._round_state.decisions)
        info["overshoot"] = overshoot(self._round_state, new_state)
        info["skipped"] = self._skipped
        info["n_active"] = new_state.n_active
        self._state = new_state
        if round_index < spec.n_rounds:
            self._begin_round(round_index + 1)
            return "round", 0.0

        # The interval's last round: advance, and judge the mesh it left.
        report = self.backend.step_interval([config.indicator], config.reduction, self.cadence)
        r_global = global_reward(
            report.acc[config.indicator],
            self._state.level,
            self._state.max_level,
            config,
            *self._pair,
            **self._global_weights,
        )
        self._interval += 1
        if self._interval == self.n_intervals:
            return "done", r_global
        self._begin_round(1)
        return "interval", r_global

    def _observation(self) -> dict[str, np.ndarray]:
        """The observation dict of the position the queue is asking."""
        return observation(self._cache, self._round_state, self._position, self.observation_spec.observation)

    def _zeros(self) -> dict[str, np.ndarray]:
        """The observation returned when an episode ends: zeros, in the space."""
        return {key: np.zeros(box.shape, dtype=np.float32) for key, box in self.observation_space.spaces.items()}

    # --- Gymnasium ------------------------------------------------------------------

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Begin an episode and return the first asked element's observation.

        The initial condition is options["ic"] when given, a mapping the
        backend's reset takes; otherwise each parameter is drawn uniformly
        within its bounds from describe(), by the generator that seed
        seeds.

        The first reset of the environment's life compares the spec with
        the round it has just read (spec_violations).

        Returns:
            The observation dict, and an info dict: ic, n_active, and the
            interval's thresholds as lists, e_max and e_min.

        Raises:
            ValueError: Starting with "spec", if the first round is not one
                the spec describes; or as the backend's reset raises.
            BackendFailure: If the backend fails. Nothing is caught here:
                there is no episode to end.
        """
        super().reset(seed=seed)
        self._position = None
        if options is not None and "ic" in options:
            ic = {name: float(value) for name, value in options["ic"].items()}
        else:
            ic = {name: float(self.np_random.uniform(low, high)) for name, (low, high) in self._ic_params.items()}
        self._state = self.backend.reset(ic, self.amr_freq, self.initial_level)
        self._interval = 0
        self._begin_round(1)
        if not self._spec_checked:
            found = spec_violations(self.observation_spec, self._cache, self._round_state)
            if found:
                self._position = None
                raise ValueError("spec: " + "; ".join(found))
            self._spec_checked = True
        info = {
            "ic": ic,
            "n_active": self._state.n_active,
            "e_max": self._pair[0].tolist(),
            "e_min": self._pair[1].tolist(),
        }
        return self._observation(), info

    def action_masks(self) -> np.ndarray:
        """The actions open to the asked element: (3,) bool, coarsen, hold, refine.

        Raises:
            RuntimeError: If no episode is running.
        """
        if self._position is None:
            raise RuntimeError("action_masks: no episode is running, call reset")
        return action_mask(self._state, self._round_state, self._position, self.observation_spec.coarsen_rule)

    def step(self, action: int) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        """Commit one decision for the asked element and move to the next.

        The commit comes first, so an action the mask closes raises before
        anything is paid or moved. The reward is lambda_local times the
        decision's local reward, plus lambda_global times the interval's
        global reward at the step that ends an interval (D-007).

        A backend that fails ends the episode (SOLVER_CONTRACT_V2 section
        6.5, V7): the step returns terminated, with the decision's local
        reward alone and the failure's message in info. No penalty is
        added for the failure.

        Args:
            action: 0 (coarsen), 1 (hold) or 2 (refine), open in
                action_masks.

        Returns:
            The next asked element's observation (zeros when the episode
            ends), the reward, terminated, truncated (always False) and an
            info dict: position, round_index and interval of the decision,
            r_local, r_global, transition, and at a round's end overshoot,
            skipped and n_active. After a failure transition is "failure"
            and backend_failure holds the message.

        Raises:
            RuntimeError: If no episode is running.
            ValueError: As commit raises, for an action that is not an
                action index or is masked. Nothing has changed.
        """
        if self._position is None:
            raise RuntimeError("step: no episode is running, call reset")
        rule = self.observation_spec.coarsen_rule
        action = int(action)
        position = self._position
        self._round_state = commit(self._state, self._round_state, position, action, rule)
        r_local = local_reward(self._state, self._distance, position, action, rule, **self._local_weights)
        info: dict[str, Any] = {
            "position": position,
            "round_index": self._round_state.round_index,
            "interval": self._interval + 1,
            "r_local": r_local,
            "r_global": 0.0,
        }
        try:
            transition, r_global = self._advance(info)
        except BackendFailure as failure:
            # The one place a failure is caught. The episode is over; what
            # the backend holds is unknown until the next reset.
            self._position = None
            info["transition"] = "failure"
            info["backend_failure"] = str(failure)
            return self._zeros(), self.lambda_local * r_local, True, False, info
        info["transition"] = transition
        info["r_global"] = r_global
        reward = self.lambda_local * r_local + self.lambda_global * r_global
        if transition == "done":
            return self._zeros(), reward, True, False, info
        return self._observation(), reward, False, False, info