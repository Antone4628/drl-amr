# Jexpresso case bank — the Phase-6 reproduction surface

The Jexpresso cases that produced the Phase-6 (2D DG-AMR) evidence are untracked in
the Jexpresso repo by decision (they point at locally generated 4×4/8×8 gate meshes
that are not in `JexpressoMeshes`; the committed DG cases are `AdvDiff/advection1d_dg`
and `AdvDiff/advection2d_dg`). This directory banks them, so the evidence can be
regenerated from a fresh Jexpresso checkout at the recorded commit.

- `cases/<case>/` — the case directory as it sat under
  `Jexpresso/problems/AdvDiff/<case>/` on 2026-10-18 (`output/` and
  `.jexpresso_cache/` excluded). Restore with
  `cp -R ~/projects/drl-amr/tools/jexpresso/cases/<case> ~/projects/Jexpresso/problems/AdvDiff/<case>`.
- `meshes/generated/*.msh` — the gate-family meshes as generated; each is also
  regenerable from the `.geo` files one directory up (`gate_family_*.geo`) with the
  gmsh invocations recorded in `drl-amr-notes/knowledge/JEXPRESSO_ENVIRONMENT.md` §8.
  They belong under `JexpressoMeshes/meshes/gmsh_grids/`.

On the walled cases (`advection2d_dg_walled`, `advection2d_dg_insolve_walled`;
meshes from `gate_family_walls*.geo`): "walled" names the *mesh* — the
non-periodic topology, gmsh physical tag `wall` on the closed sides (both axes
on the walled mesh, the y-axis sides only on the channel, which stays periodic
in x). The condition the cases impose is q = 0 on every `wall`-tagged face,
weakly through the mirror ghost (`user_bc.jl`). For scalar advection with
c = (0.5, 1.0) that is an outflow boundary at the top and right with zero
inflow at the bottom and left: the pulse exits through the outflow boundary and
no reflection is expected (one characteristic family, nothing to carry one);
refinement follows the pulse to the boundary and the mesh coarsens back. Read
the walled runs against the exact remaining-mass fraction (`bracket_2d --walls`),
not against conserved mass.

Which case belongs to which item, with the recipe and reference values, is in
`JEXPRESSO_ENVIRONMENT.md` §8.5–§8.13. The probes the runs used are never committed
anywhere; the environment doc banks them as pastes and as patches under
`~/projects/jx/probes/`.

The bank is a snapshot, not a source of truth: the live case on disk may move on.
Re-copy at each item close that changes a case.
