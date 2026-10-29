# Jexpresso case bank — the Phase-6 reproduction surface

The Jexpresso cases that produced the Phase-6 (2D DG-AMR) evidence are untracked in
the Jexpresso repo by decision (they point at locally generated 4×4/8×8 gate meshes
that are not in `JexpressoMeshes`; the committed DG cases are `AdvDiff/advection1d_dg`
and `AdvDiff/advection2d_dg`). This directory banks them, so the evidence can be
regenerated from a fresh Jexpresso checkout at the recorded commit.

- `cases/<case>/` — the case directory as it sits under
  `Jexpresso/problems/AdvDiff/<case>/` (`output/` and `.jexpresso_cache/` excluded;
  re-copied 2026-10-29). Restore into a checkout with
  `cp -R <drl-amr>/tools/jexpresso/cases/<case> <Jexpresso>/problems/AdvDiff/<case>`.
- `meshes/generated/*.msh` — the gate-family meshes as generated. They belong under
  `JexpressoMeshes/meshes/gmsh_grids/`; every deck reads them as
  `./meshes/gmsh_grids/<name>.msh` from the Jexpresso root. Each is regenerable from
  the `.geo` files one directory up:

| Mesh | Generator | `gmsh <geo> -2 -setnumber …` |
|---|---|---|
| `gate2d_4x4_periodic`, `gate2d_8x8_periodic` | `conv_family_periodic.geo` | `nelemx N -setnumber nelemy N` |
| `gate2d_4x4_periodic_native`, `gate2d_8x8_periodic_native`, `gate2d_16x16_periodic_native` | `gate_family_periodic_native.geo` | `nelemx N -setnumber nelemy N` |
| `gate2d_4x4_walled`, `gate2d_8x8_walled` | `gate_family_walls.geo` | `nelemx N -setnumber nelemy N -setnumber walls 2` |
| `gate2d_4x4_channel` (periodic by tags only) | `gate_family_walls.geo` | `nelemx 4 -setnumber nelemy 4 -setnumber walls 1` |
| `gate2d_4x4_channel_native`, `gate2d_8x8_channel_native` | `gate_family_walls_native.geo` | `nelemx N -setnumber nelemy N -setnumber walls 1` |

  For example, from `tools/jexpresso/meshes/`:
  `gmsh gate_family_periodic_native.geo -2 -setnumber nelemx 8 -setnumber nelemy 8 -o <JexpressoMeshes>/meshes/gmsh_grids/gate2d_8x8_periodic_native.msh`.
  A natively periodic mesh carries one `$Periodic` section; the tag-only ones carry none.

## Running a case

From the Jexpresso root, one case per Julia process (case-level definitions leak
between `run_case` calls in one session):

```
cd <Jexpresso>
<SELECTORS> julia --project=. -e 'using Jexpresso; Jexpresso.run_case("AdvDiff", "<case>")'
```

Selectors are environment variables read by the case's `user_inputs.jl` and
`initialize.jl`; an unset one takes the default shown in parentheses (or the first
value listed). Static cases cache their mesh in `problems/AdvDiff/<case>/.jexpresso_cache/`:
remove it after any change under `src/` and between `advection2d_dg_conv` runs on
different meshes. Adaptive runs keep no cache. Output lands in
`output/AdvDiff/<case>/output-<date>-<time>/`.

The instruments run from this repository's root, `python -m tools.<instrument> …`
(each has `--help` and a test file under `tests/dg_2d/`). They read only what
Jexpresso writes — VTU frames, run logs, probe dumps — and never Jexpresso's
element ids or face lists: geometry and topology are rebuilt from node coordinates.

**Second velocity leg.** Operator checks run twice: at the cases' velocity
c = (0.5, 1.0), where cx/Δx = cy/Δy on the 4×4 (0.2 each) and a class of x/y swap
errors is invisible, and at c = (0.7, 0.3). Leg 2 is one line in the case's
`user_flux.jl`, restored afterwards:

```
sed -i '' 's|advection_velocity() = (0.5, 1.0)|advection_velocity() = (0.7, 0.3)|' problems/AdvDiff/<case>/user_flux.jl
```

(the reverse substitution restores it); the instrument then takes `--cx 0.7 --cy 0.3`.

**Probes.** Two kinds of number come from code that is not committed anywhere: the
operator dumps (the extracted matrix, node coordinates, the solver's diagonal mass
and the element levels), written by an extraction probe pasted into
`problems/drivers.jl` and switched on with `JE_GATE2D_PROBE=1` (the run stops after
writing them); and per-adapt counts (mass before and after each transfer, child
order on refine and coarsen, wrap-mortar parent faces), printed by probes in `src/`.
Rows marked † below need the first and rows marked ‡ the second; both are available
on request. Everything else is read from committed output: the VTU frames, and the
build log's census lines (`… conforming faces; … mortar parent faces …; census N/N`
and `one-cell facets: …`), which every DG build prints.

## Cases

| Case | Runs | Selectors | Leg 2 | Scored by |
|---|---|---|---|---|
| `advection2d_dg_gate` | conforming operator, 4×4 periodic by tags, t = 4 | — | yes | `gate_2d_compare --operator <dir>/GATE2D_operator_*.txt --nodes <dir>/GATE2D_nodes_*.txt` † |
| `advection2d_dg_gate_native` | the same on the natively periodic 4×4 | — | yes | `gate_2d_compare` as above †; `gate_2d_permute_compare --ref <gate dump dir> --new <native dump dir>` † |
| `advection2d_dg_amr_native` | static 2:1 — one element preadapted to level 1, 19 elements — on the natively periodic 4×4, t = 4 | `AMR_NATIVE_PREADAPT` = `interior` \| `xwrap` \| `ywrap` \| `corner` | yes | `gate_2d_mortar --dir <dump dir> --expect-nelem 19 --expect-mortar-halves 8 --expect-wrap-mortar-halves H`, H = 0 / 2 / 2 / 4 by preadapt †; without the probe the run integrates to t = 4 for `bracket_2d` |
| `advection2d_dg_conv` | uniform runs, the brackets' coarse and fine members | `CONV_MESH` (a file name in `gmsh_grids/`), `CONV_DT` (0.0125), `CONV_NOP` (4), `CONV_TEND` (4.0) | — | as a `--run` of `bracket_2d` |
| `advection2d_dg_insolve_native` | in-solve adaptation to level 1 across both periodic wraps, natively periodic 4×4, t = 20 | `INSOLVE_TEND` (20) | — | `bracket_2d --adaptive --mass-tol 3e-12`; `amr_overlay_2d --nex 4 --ney 4`; `tools.jexpresso.wrap_identity_check --nex 4 --ney 4 <log>` ‡ |
| `advection2d_dg_insolve_l2_native` | in-solve adaptation to level 2 across both wraps, natively periodic 8×8, t = 20 | `INSOLVE_TEND` (20); `INSOLVE_SCRIPT=1` (a fixed five-adapt script in place of the indicator — run it with `INSOLVE_TEND=3`); `INSOLVE_DT_OUT` (0.5; 0.125 for the animation) | — | `bracket_2d --adaptive --no-l2-gate --nex 8 --ney 8 --mass-tol 3e-12`; `amr_overlay_2d --nex 8 --ney 8`; `tools.jexpresso.wrap_identity_check --nex 8 --ney 8 --n1-from centroid <log>` ‡ |
| `advection2d_dg_insolve_walled` | walls: the walled 4×4 and the channel (walls in y, natively periodic in x) — census, operator, in-solve adaptation to level 1 to t = 14, uniform runs | `WALL_MESH` (`./meshes/gmsh_grids/gate2d_4x4_walled.msh`); `WALL_PREADAPT` = `start` \| `corner_ll` \| `corner_ur` \| `interior`; `WALL_GRID_ONLY=1` (build, print the census, stop); `WALL_LAMR=0` (hold the preadapted mesh — the operator runs); `WALL_STATIC=1` (no preadapt; with `WALL_LAMR=0`, the uniform run on `WALL_MESH`); `INSOLVE_TEND` (14); `INSOLVE_DT_OUT` (0.5) | operator runs | operator: `gate_2d_mortar --dir <dump dir> --walls xy --expect-wall-sides 18 --expect-nelem 19 --expect-mortar-halves 4 --expect-wrap-mortar-halves 0` (walled, `corner_ll` and `corner_ur`), `--walls y --expect-wall-sides 9 --expect-nelem 19 --expect-mortar-halves 6 --expect-wrap-mortar-halves 2` (channel, `corner_ll`) †; in-solve: `bracket_2d --adaptive --no-l2-gate --walls xy` (walled) / `--walls y` (channel); `amr_overlay_2d --walls xy` / `--walls y` |
| `advection2d_dg_walled` | build-only census (`:lgrid_only`) on the walled 4×4 | `WALL_MESH` (`./meshes/gmsh_grids/gate2d_4x4_walled.msh`), `WALL_PREADAPT` = `0` \| `1` | — | the build log's census lines |

**What to expect** (polynomial order 4; ρ is the spectral radius of the extracted
operator, leg 1 / leg 2; L² is the error against the exact solution, coarse >
adaptive > fine):

- `advection2d_dg_gate`, `_gate_native`: 32 conforming faces, census 64/64; ρ
  5.892 / 5.008; entrywise against the analytic operator ≤ 3.2e-11 (gmsh node
  round-off). The native dumps match the tag-only ones under the identity
  permutation to 3.4e-11.
- `advection2d_dg_amr_native`: census 76/76, `one-cell facets: … 12 mortar`; ρ 7.3047
  / 5.8777 for every preadapt (the refined element's position is a translation);
  `wᵀL` with the solver's own mass 3.4e-15 / 4.4e-15; polynomial exactness ~1e-15.
- `advection2d_dg_insolve_native`: census N/N on all 41 builds; at t = 20, L²
  4.967e-01 (4×4) > 1.821e-01 (adaptive, ≤ 34 elements) > 1.789e-01 (8×8).
- `advection2d_dg_insolve_l2_native`: at t = 20, L² 1.789e-01 (8×8) > 1.759e-02
  (adaptive, 70–142 elements) > 1.472e-02 (16×16). The scripted leg's element counts
  run 64 → 70 → 73 → 85 → 85 → 85 → 82.
- `advection2d_dg_insolve_walled`: operator ρ 22.8907 / 19.4571 (walled `corner_ll`),
  11.4454 / 9.7286 (walled `corner_ur`), 13.5769 / 8.0166 (channel `corner_ll`), no
  kernel modes; in-solve, `untagged walls` 0 on every census line; at t = 14, walled
  L² 5.29e-03 > 5.26e-04 > 3.23e-04 and channel 1.94e-01 > 5.35e-02 ≈ 5.37e-02.
- `advection2d_dg_walled`: `census 64/64` with 16 tagged walls; with
  `WALL_PREADAPT=1`, `census 76/76` with 12 mortar facets.

## The brackets and the animations

Each bracket is three runs scored together, coarse to fine. Run each Jexpresso line
as its own process from the Jexpresso root and each `python` line from this
repository's root, with the output directories the runs printed:

```
rm -rf problems/AdvDiff/advection2d_dg_conv/.jexpresso_cache
CONV_MESH=gate2d_4x4_periodic_native.msh CONV_DT=0.025 CONV_NOP=4 CONV_TEND=20.0 julia --project=. -e 'using Jexpresso; Jexpresso.run_case("AdvDiff", "advection2d_dg_conv")'
rm -rf problems/AdvDiff/advection2d_dg_conv/.jexpresso_cache
CONV_MESH=gate2d_8x8_periodic_native.msh CONV_DT=0.025 CONV_NOP=4 CONV_TEND=20.0 julia --project=. -e 'using Jexpresso; Jexpresso.run_case("AdvDiff", "advection2d_dg_conv")'
julia --project=. -e 'using Jexpresso; Jexpresso.run_case("AdvDiff", "advection2d_dg_insolve_native")'
python -m tools.bracket_2d --adaptive --mass-tol 3e-12 --run 4x4=<dir> --run insolve=<dir> --run 8x8=<dir>
```

Level 2 is the same with the 8×8 native base, the 16×16 native base at
`CONV_DT=0.0125`, the case `advection2d_dg_insolve_l2_native`, and
`bracket_2d --adaptive --no-l2-gate --nex 8 --ney 8 --mass-tol 3e-12`. The walled and
channel bases are uniform runs of the walled case itself —
`WALL_STATIC=1 WALL_LAMR=0 WALL_MESH=./meshes/gmsh_grids/<mesh>` on
`gate2d_4x4_walled.msh` and `gate2d_8x8_walled.msh`, or on the two `channel_native`
meshes — and the in-solve run is `WALL_PREADAPT=start` on the 4×4; score the 4×4
base with the in-solve run (`--walls xy` or `--walls y`) and the 8×8 base in its own
call with `--nex 8 --ney 8`.

The three animations are the level-2 run and the walled and channel in-solve runs,
re-run with `INSOLVE_DT_OUT=0.125` and drawn by
`amr_overlay_2d --run <dir> --out <dir> --no-vtp --mp4 <file>.mp4` with the base grid
and walls of the run (`--nex 8 --ney 8` for level 2; `--walls xy` or `--walls y` for
the walled case). The overlay classifies every element side of every frame from the
node coordinates — conforming, mortar parent, mortar child, across the wrap, wall —
and fails the frame if any side is unclaimed or the wrap count does not close.

## Cases kept as history

`advection2d_dg_amr` (static 2:1), `advection2d_dg_insolve` (level 1) and
`advection2d_dg_insolve_l2` (level 2) ran on the periodic-by-tags meshes before the
natively periodic ones existed, with refinement kept off the periodic boundary by a
mask. On this branch each stops at its first mesh build with the refusal for DG
adaptivity on a periodic mesh whose periodicity is carried by boundary tags only —
the expected result; the `_native` cases above replace them. `advection2d_dg_walled`
with `WALL_MESH=./meshes/gmsh_grids/gate2d_4x4_channel.msh WALL_PREADAPT=1`, and
`advection2d_dg_insolve_walled` on the same mesh with a preadapt or in-solve
adaptation, stop the same way.

## Walls

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

The bank is a snapshot, not a source of truth: the live case may move on, and the
bank is re-copied whenever a case changes.
