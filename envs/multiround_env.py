"""Gymnasium front-end shell over the shared MultiroundDriver (RESTRUCTURE Phase 6).

A thin gym.Env wrapper (D-047): it declares the gym observation/action spaces and
delegates reset / step / action_masks to a pre-built MultiroundDriver. SB3 owns
the loop — MaskablePPO calls env.step(action) and queries env.action_masks() per
element. The deployment adapter (Phase 6, next) drives the *same* driver.step in
its own internal loop, so behavioral train/deploy consistency is by construction
(the old "must mirror the env exactly" instruction is retired).

The env holds no episode state and no episode parameters: IC pool, n_remesh,
step_domain_fraction, element_budget, pre_advance_range all live on the
driver/core. The only things here are the gym spaces and the seeded-RNG handoff.

RNG handoff (D-038 parity). reset() calls super().reset(seed=seed) to seed
self.np_random, then passes that same Gymnasium generator into driver.reset(
rng=...). Because the driver draws the IC (choice) then the D-029 pre-advance
multiplier (uniform) in that order, the stream is bit-identical to the old
dg_amr_env_multiround.py — the parity target.

Spaces (lifted verbatim from the old env; Architecture Spec Section 6.2):
    action_space      = Discrete(3)   [0 coarsen, 1 hold, 2 refine]
    observation_space = Box(8,)       per-component bounds below

NOTE (loose obs lower bound — carried over verbatim, flagged for reconciliation).
Components [0:3] are alpha-normalized log-errors whose boundary is o = -1 and
which are NEGATIVE across the healthy operating regime (alpha * e_inf < 1). The
declared low = 0.0 on those three components is therefore too tight: real
observations routinely fall below it. This matches the old env exactly (the
bound was wrong there too); it never surfaced because SB3 does not validate obs
against the Box at train time. Kept verbatim here to hold the D-038 parity line;
reconcile to low = -inf as a deliberate, logged change AFTER the parity gate, not
silently now. (gymnasium.utils.env_checker would flag this — see the test note.)
"""
from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from agent.observation import OBS_DIM
from drivers.multiround_driver import MultiroundDriver


class MultiroundEnv(gym.Env):
    """Thin Gymnasium wrapper over a MultiroundDriver (MaskablePPO front-end).

    Construct the backend chain first, then pass the driver in:

        contract = Python1DSolverContract(...)
        core = AgentCore(contract, ...)
        driver = MultiroundDriver(core, ...)
        env = MultiroundEnv(driver)

    All hyperparameters and episode structure live on the core/driver; this
    wrapper adds only the gym spaces and the seeded-RNG handoff.
    """

    metadata = {"render_modes": []}

    def __init__(self, driver: MultiroundDriver):
        """Bind the driver and declare the gym spaces.

        Args:
            driver: A fully-constructed MultiroundDriver (already bound to an
                AgentCore + SolverContract). The env owns none of the episode
                parameters — they are read from / acted on through the driver.
        """
        super().__init__()
        self.driver = driver

        # Action space (D-025): 0 = coarsen, 1 = hold, 2 = refine. Budget is
        # NOT masked — the agent learns conservation via resource_usage + reward.
        self.action_space = spaces.Discrete(3)

        # Observation space — 8 components, lifted verbatim from the old env.
        # The per-component bounds are not recoverable from OBS_DIM alone, so
        # they are declared here. See the module-level NOTE on the [0:3] lows.
        #   Index | Component              | Declared range
        #   ------|------------------------|----------------
        #     0   | alpha-normalized error | [0, inf)   (actually can be < 0)
        #     1   | left neighbor error    | [0, inf)   (actually can be < 0)
        #     2   | right neighbor error   | [0, inf)   (actually can be < 0)
        #     3   | refinement level       | [0, 1]
        #     4   | left neighbor level    | [0, 1]
        #     5   | right neighbor level   | [0, 1]
        #     6   | resource_usage         | [0, 2]     (can exceed 1.0)
        #     7   | round_progress         | [0, 1]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([np.inf, np.inf, np.inf, 1.0, 1.0, 1.0, 2.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # Guard against env/core drift: the declared space must match the
        # length of the vector the core actually emits.
        assert self.observation_space.shape == (OBS_DIM,), (
            f"observation_space {self.observation_space.shape} != core OBS_DIM {OBS_DIM}"
        )

    # --- Gymnasium API (pure delegation) -----------------------------------

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Seed self.np_random, then hand it to the driver for a new episode.

        Passing the Gymnasium-seeded generator into driver.reset(rng=...) makes
        the IC -> pre-advance draw order bit-identical to the old env (D-038).
        options keys are forwarded unchanged ('icase', 'refinement_level').
        """
        super().reset(seed=seed)
        return self.driver.reset(options=options, rng=self.np_random)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Delegate one element decision to the driver (SB3-style 5-tuple)."""
        return self.driver.step(int(action))

    def action_masks(self) -> np.ndarray:
        """Valid-action mask for the current element (MaskablePPO interface)."""
        return self.driver.action_masks()