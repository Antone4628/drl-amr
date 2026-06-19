"""Gymnasium front-end shell over the shared MultiroundDriver (RESTRUCTURE Phase 6).

A thin gym.Env wrapper (D-047): it declares the gym observation/action spaces and
delegates reset / step / action_masks to a pre-built MultiroundDriver. SB3 owns
the loop — MaskablePPO calls env.step(action) and queries env.action_masks() per
element. The deployment adapter (Phase 6, next) drives the *same* driver.step in
its own internal loop, so behavioral train/deploy consistency is by construction
(the old "must mirror the env exactly" instruction is retired).

The env holds no episode state and no episode parameters: IC pool, n_remesh,
step_domain_fraction, element_budget all live on the driver/core. The only
things here are the gym spaces and the seeded-RNG handoff.

RNG handoff (D-038 parity). reset() calls super().reset(seed=seed) to seed
self.np_random, then passes that same Gymnasium generator into driver.reset(
rng=...). The driver draws the IC (choice); the D-029 pre-advance was retired
(D-049), so the IC-choice stream alone is bit-identical to the old
dg_amr_env_multiround.py run (pre-advance was off there) — the parity target.

Spaces (lifted verbatim from the old env; Architecture Spec Section 6.2):
    action_space      = Discrete(3)   [0 coarsen, 1 hold, 2 refine]
    observation_space = Box(8,)       per-component bounds below

NOTE (obs bounds — reconciled post-parity; see DECISION_LOG D-053).
Components [0:3] are alpha-normalized log-errors whose boundary is o = -1 and
which are NEGATIVE across the healthy operating regime (alpha * e_inf < 1), so
their low is -inf (o -> -inf as e_k -> 0). resource_usage [6] has high = inf:
n_active / budget can exceed 2.0 once the unmasked mesh grows past 2x the soft
budget. Both were carried verbatim from the old dg_amr_env_multiround.py
(low = 0.0, high = 2.0 — too tight there too) and held through the D-038/D-052
parity gate so the declared interface matched the target exactly; widened only
AFTER v0.1-parity was tagged (D-053). The change is metadata-only — SB3 feeds
the raw obs vector to MlpPolicy, so the Box bounds are read only by env-checkers
and by normalization/clipping wrappers (none in use); widening removes a latent
VecNormalize/obs-clip trap without altering any emitted observation.
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

        # Observation space — 8 components, lifted from the old env, bounds
        # reconciled post-parity (D-053). The per-component bounds are not
        # recoverable from OBS_DIM alone, so they are declared here. See the
        # module-level NOTE on the [0:3] and [6] bounds.
        #   Index | Component              | Declared range
        #   ------|------------------------|----------------
        #     0   | alpha-normalized error | (-inf, inf)
        #     1   | left neighbor error    | (-inf, inf)
        #     2   | right neighbor error   | (-inf, inf)
        #     3   | refinement level       | [0, 1]
        #     4   | left neighbor level    | [0, 1]
        #     5   | right neighbor level   | [0, 1]
        #     6   | resource_usage         | [0, inf)
        #     7   | round_progress         | [0, 1]
        self.observation_space = spaces.Box(
            low=np.array([-np.inf, -np.inf, -np.inf, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([np.inf, np.inf, np.inf, 1.0, 1.0, 1.0, np.inf, 1.0], dtype=np.float32),
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
        the IC draw bit-identical to the old env (D-038; pre-advance retired per
        D-049). options keys are forwarded unchanged ('icase', 'refinement_level').
        """
        super().reset(seed=seed)
        return self.driver.reset(options=options, rng=self.np_random)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Delegate one element decision to the driver (SB3-style 5-tuple)."""
        return self.driver.step(int(action))

    def action_masks(self) -> np.ndarray:
        """Valid-action mask for the current element (MaskablePPO interface)."""
        return self.driver.action_masks()