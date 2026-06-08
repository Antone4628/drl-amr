# drl-amr

Deep reinforcement learning for adaptive mesh refinement (AMR) in discontinuous
Galerkin (DG) solvers. A single RL **agent core** decides *what* action to take
on *which* element; numerical backends, behind an explicit **solver contract**,
know *how* their mesh is structured. Backends are swappable (Python 1D now;
Python 2D and a Julia/Jexpresso backend later).

## Status

Pre-parity rebuild of the multiround architecture (RESTRUCTURE Phase 2). Behavioral
parity with the `drl-amr-1d` archive is the v0.1 gate. Not yet runnable end to end.

## Layout

| Path | Role |
|------|------|
| `agent/` | Agent core: observation assembly, action masking, queue ordering, reward, budget counter |
| `contract/` | Solver-contract Protocol + per-element state schema |
| `backends/python_1d/` | First backend — DG primitives, grid, AMR, multiround solver, error indicators |
| `envs/` | Gym environment shell (training) |
| `deployment/` | Env-free deployment adapter (location provisional) |
| `training/` | MaskablePPO training + diagnostics |
| `analysis/` | Post-hoc analysis |
| `configs/` | YAML run configs (tracked) |
| `runs/` | Lightweight run manifests (tracked) |
| `tests/` | Unit + smoke tests |
| `tools/` | `tree_gen.py`, `run_manifest.py` |

## Setup

```bash
conda activate rl-amr
pip install -e .
```

## Documentation

- Architecture, decisions, and roadmaps live in the companion notes repo
  `drl-amr-notes` (Mac-only): see `INDEX.md`, `strategy/decisions/DECISION_LOG.md`,
  and `strategy/proposals/Stage_1_Architecture_Specification.md`.
- Provenance: the prior single-source codebase is archived at `drl-amr-1d`
  (frozen once parity is confirmed).
