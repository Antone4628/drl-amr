"""Build the solver -> contract -> core -> driver chain from a run config, plus
config/model loading for the deployment runner (RESTRUCTURE Phase 6, deployment
adapter build-order step 5).

The chain mirrors the env shell's construction (envs/multiround_env.py) minus the
Gym wrapper: a Python1DSolverContract over a balance=False solver, an AgentCore
bound to the run's hyperparameters, and a MultiroundDriver. Sequential mode is the
deployment default (D-038 parity target). This builder is a candidate to share
with the training/config port (the 4th Phase 6 item).
"""
from __future__ import annotations

import numpy as np
import yaml

from agent.core import MODE_SEQUENTIAL, AgentCore
from backends.python_1d.contract_impl import Python1DSolverContract
from backends.python_1d.solvers.dg_advection_solver_multiround import DGAdvectionSolver
from drivers.multiround_driver import MultiroundDriver


def load_config(path: str) -> dict:
    """Load a training run's resolved config.yaml."""
    with open(path) as f:
        return yaml.safe_load(f)


def build_driver_from_config(config: dict, *, mode: str = MODE_SEQUENTIAL) -> MultiroundDriver:
    """Construct the solver -> contract -> core -> driver chain from a run config.

    Reads the same sections train_multiround writes (solver / environment /
    reward). The solver is built balance=False (the contract enforces 2:1 balance
    explicitly so cascades are reported). icase here is a placeholder — reset()
    sets the actual IC. Reward weights are threaded through for fidelity even
    though the deployment rollout never calls the reward (env-free).
    """
    s = config["solver"]
    e = config["environment"]
    r = config["reward"]

    solver = DGAdvectionSolver(
        nop=s["nop"],
        xelem=np.array(s["xelem"], dtype=float),
        max_elements=s["max_elements"],
        max_level=s["max_level"],
        courant_max=s["courant_max"],
        icase=e["ic_pool"][0],  # placeholder; reset() sets the actual icase
        balance=False,
    )
    contract = Python1DSolverContract(solver)
    core = AgentCore(
        contract,
        alpha=e["alpha"],
        beta=e["beta"],
        p_ur=r["p_ur"],
        p_or=r["p_or"],
        p_cr=r["p_cr"],
        lambda_local=r["lambda_local"],
        lambda_global=r["lambda_global"],
        element_budget=e["element_budget"],
        error_indicator=e["error_indicator"],
        mode=mode,
    )
    driver = MultiroundDriver(
        core,
        n_remesh=e["n_remesh"],
        step_domain_fraction=e["step_domain_fraction"],
        initial_refinement_level=e["initial_refinement_level"],
        pre_advance_range=tuple(e["pre_advance_range"]),
        ic_pool=e["ic_pool"],
        verbosity=e.get("verbosity", 0),
    )
    return driver


def load_model(path: str):
    """Load a trained MaskablePPO model (lazy import — sb3_contrib is only needed
    for the model decide_fn path, not for random rollouts)."""
    from sb3_contrib import MaskablePPO

    return MaskablePPO.load(path)