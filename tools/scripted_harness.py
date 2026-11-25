"""The scripted-action harness: fixed mark sequences, run on any backend.

A script is a base grid and a list of steps. A step is one round of marks,
addressed by where each element is and never by its position, with what the
round must realize pinned beside it. run_script walks one backend through a
script and checks, after the reset and after every adapt, everything that
can be checked from the contract's nouns alone. It returns findings, empty
when the backend did what the script says, and the realized leaf set after
every adapt as text of integers, which two backends' runs can be compared
by diff (JEXPRESSO_BRIDGE_ROADMAP B9.1).

The identical script runs on the mock and on Jexpresso, so the harness asks
only for what the bridge has when its scripts first run (B3.3):

    describe, reset (at level 0), apply_marks, get_state, and dry_run if
    describe() lists it

and never compute_error or step_interval. A level-1 start is a first step
that refines everything.

Addresses are fractions of the domain: (u, v) with u = (x - lo) / (hi - lo)
along each axis, so a script names no domain and its leaf files are the
same on any domain. An address matches the one element whose centroid is
within rtol of it on every axis; no match, or two, stops the script.

What is checked (each finding starts with its step and its rule):

    topology.*    contract.checks.topology_violations, on every state
    script.*      the grid the script names; an address that matches no
                  element; a mark the agent's mask would refuse; the pinned
                  element count; the pinned departures of realized from the
                  marks
    state.*       get_state returns the state apply_marks returned
    report.*      the AdaptReport against the two states: epochs, counts,
                  time, every row of new_to_old by its kind, realized as
                  the levels define it, every old element accounted for
    counter.*     the committed count of the 2D core (agent.round_commit)
                  against the realized count: never above it, and apart by
                  the pinned overshoot
    dry_run.*     dry_run against apply_marks, and that it changed nothing
    backend.*     a verb that raised: the run ends there, as a finding

The committed count is the core's own: the step's marks are replayed
through start_round and commit under the unanimous rule, where a coarsen
marks one element and the count moves when a family's last member votes,
so any marks array has a count and the order of the replay does not
matter. The record that replay leaves is the array handed to apply_marks.

Run on the mock:  python -m tools.scripted_harness --out DIR

AGENT_2D_ROADMAP item 4.2.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from agent.round_commit import commit, overshoot, start_round
from agent.round_mask import ACTION_COARSEN, ACTION_HOLD, ACTION_REFINE
from contract.checks import topology_violations
from contract.reports import KIND_CHILD, KIND_MERGED, KIND_UNCHANGED, AdaptReport
from contract.solver_contract import COARSEN, HOLD, REFINE
from contract.state import SolverState

C, H, R = COARSEN, HOLD, REFINE

# The rule the committed count is replayed under: one vote per element.
REPLAY_RULE = "unanimous"

_MARK_TO_ACTION = {COARSEN: ACTION_COARSEN, HOLD: ACTION_HOLD, REFINE: ACTION_REFINE}
_MARK_NAMES = {COARSEN: "COARSEN", HOLD: "HOLD", REFINE: "REFINE"}

Address = tuple[float, float]


class _Ended(Exception):
    """A verb raised; the finding is recorded and the run is over."""


@dataclass(frozen=True, kw_only=True)
class Step:
    """One round of marks, and what it must realize.

    Fields:
        where: Address -> mark, for the elements named one by one.
        everywhere: The mark of every element not in where.
        n_active: The element count after the adapt.
        overshoot: The realized count above the committed count.
        departures: Address (on the mesh before the adapt) -> realized
            flag, for exactly the elements whose realized flag differs
            from their mark.
    """

    where: Mapping[Address, int] = field(default_factory=dict)
    everywhere: int = HOLD
    n_active: int
    overshoot: int = 0
    departures: Mapping[Address, int] = field(default_factory=dict)


@dataclass(frozen=True, kw_only=True)
class Script:
    """A base grid and the rounds of marks to run on it.

    Fields:
        name: The script's name, and its leaf file's.
        nx: Base cells along x.
        ny: Base cells along y.
        periodic: Whether each axis wraps, (x, y).
        max_level: The finest level.
        steps: The rounds, in order, from the base grid.
    """

    name: str
    nx: int
    ny: int
    periodic: tuple[bool, bool]
    max_level: int
    steps: tuple[Step, ...]

    def grid(self) -> dict:
        """What a backend is built from: plain values, no domain."""
        return {"nx": self.nx, "ny": self.ny, "periodic": list(self.periodic), "max_level": self.max_level}

    def to_dict(self) -> dict:
        """The script as JSON-ready values."""

        def pairs(mapping: Mapping[Address, int]) -> list:
            return [[u, v, _MARK_NAMES[m]] for (u, v), m in mapping.items()]

        return {
            "name": self.name,
            **self.grid(),
            "steps": [
                {
                    "where": pairs(s.where),
                    "everywhere": _MARK_NAMES[s.everywhere],
                    "n_active": s.n_active,
                    "overshoot": s.overshoot,
                    "departures": pairs(s.departures),
                }
                for s in self.steps
            ],
        }


@dataclass(frozen=True, kw_only=True)
class ScriptResult:
    """What one run of one script found.

    Fields:
        name: The script's name.
        findings: Every broken rule, in order; empty when the run is clean.
        skipped: Checks not made, each with its reason.
        leaves: The realized leaf set after the reset and after every
            adapt that returned, as the leaf file's text.
    """

    name: str
    findings: tuple[str, ...]
    skipped: tuple[str, ...]
    leaves: str


# --- Addresses and cells -------------------------------------------------------------


def fractions(state: SolverState) -> np.ndarray:
    """Every centroid as fractions of the domain. (n_active, ndim) float64."""
    return (state.centroid - state.domain_lo) / (state.domain_hi - state.domain_lo)


def locate(state: SolverState, address: Sequence[float], rtol: float = 1e-9) -> int | None:
    """The position of the one element whose centroid is at an address.

    Args:
        state: The state to look in.
        address: Fractions of the domain, one per axis.
        rtol: Tolerance on each fraction.

    Returns:
        The position, or None when no element or more than one matches.
    """
    close = (np.abs(fractions(state) - np.asarray(address, dtype=np.float64)) <= rtol).all(axis=1)
    matches = np.flatnonzero(close)
    return int(matches[0]) if len(matches) == 1 else None


def cells(state: SolverState, base: Sequence[int], rtol: float = 1e-9) -> np.ndarray | None:
    """Each element's cell on the uniform grid of its own level.

    Args:
        state: The state to read.
        base: Base cells along each axis.
        rtol: Tolerance on a centroid, as a fraction of the domain.

    Returns:
        (n_active, 1 + ndim) int64, rows (level, ix, iy), by position; or
        None when some centroid is not a cell center of its level's grid.
    """
    level = state.level.astype(np.int64)
    count = np.asarray(base, dtype=np.int64)[None, :] << level[:, None]
    u = fractions(state)
    index = np.round(u * count - 0.5)
    if (np.abs((index + 0.5) / count - u) > rtol).any() or (index < 0).any() or (index >= count).any():
        return None
    return np.column_stack([level, index.astype(np.int64)])


def _leaf_block(adapt: int, keys: np.ndarray) -> str:
    """One block of the leaf file: a header line, then the sorted leaves."""
    rows = sorted(tuple(int(x) for x in row) for row in keys)
    lines = [f"adapt {adapt} n_active {len(rows)}"]
    lines += [" ".join(str(x) for x in row) for row in rows]
    return "\n".join(lines) + "\n"


# --- One step's marks and their committed count --------------------------------------


def step_marks(state: SolverState, step: Step, rtol: float = 1e-9) -> tuple[np.ndarray | None, list[str]]:
    """A step's marks on a state's ordering.

    Returns:
        The marks, (n_active,) int8, or None with one finding per address
        that matches no single element.
    """
    found: list[str] = []
    marks = np.full(state.n_active, step.everywhere, dtype=np.int8)
    for address, mark in step.where.items():
        position = locate(state, address, rtol)
        if position is None:
            found.append(f"script.address: no single element at {tuple(address)}")
        else:
            marks[position] = mark
    return (None, found) if found else (marks, found)


def replay(state: SolverState, marks: np.ndarray):
    """The core's RoundState after committing a marks array, mark by mark.

    Every position is committed, in position order, under REPLAY_RULE.

    Raises:
        ValueError: As agent.round_commit.commit raises, for a mark the
            mask refuses: a refine at max_level, a coarsen with no
            complete family.
    """
    round_state = start_round(state, 1, 1)
    for position in range(state.n_active):
        round_state = commit(state, round_state, position, _MARK_TO_ACTION[int(marks[position])], REPLAY_RULE)
    return round_state


# --- The report against the two states -----------------------------------------------


def _children(key: tuple[int, ...]) -> list[tuple[int, ...]]:
    """A cell's children, in child order: c has one bit per axis, x first."""
    level, *index = key
    d = len(index)
    return [(level + 1, *(2 * index[a] + ((c >> a) & 1) for a in range(d))) for c in range(2**d)]


def _parent(key: tuple[int, ...]) -> tuple[int, ...]:
    """The cell one level up that contains a cell."""
    level, *index = key
    return (level - 1, *(i >> 1 for i in index))


def realized_by_levels(old_keys: Sequence[tuple[int, ...]], new_keys: Sequence[tuple[int, ...]]) -> np.ndarray:
    """What happened to each old element, read from the two leaf sets.

    HOLD if the element is still a leaf, COARSEN if its parent is one,
    REFINE if all its children are (SOLVER_CONTRACT_V2 section 5.4).
    Anything else is recorded as -128, which no realized flag equals.
    """
    new = set(new_keys)
    out = np.full(len(old_keys), -128, dtype=np.int8)
    for p, key in enumerate(old_keys):
        if key in new:
            out[p] = HOLD
        elif key[0] > 0 and _parent(key) in new:
            out[p] = COARSEN
        elif all(child in new for child in _children(key)):
            out[p] = REFINE
    return out


def report_violations(
    old: SolverState,
    new: SolverState,
    report: AdaptReport,
    old_keys: Sequence[tuple[int, ...]],
    new_keys: Sequence[tuple[int, ...]],
) -> list[str]:
    """Check an AdaptReport against the states on either side of its adapt.

    Rules:
        report.epochs: epoch_old and epoch_new are the two states'.
        report.counts: n_old and n_new are the two states', and new_to_old
            is 2**ndim wide.
        report.time: The adapt did not advance time.
        report.rows: Every row of new_to_old is right for its kind: an
            unchanged element names the old element in the same cell; a
            child names the old element in its parent's cell; a merged
            parent names the old elements in its children's cells, in
            child order. Unused slots are -1.
        report.realized: realized is what the two leaf sets say.
        report.cover: Every old element is named by the rows as often as
            its realized flag says: once if held or merged, 2**ndim times
            if refined.

    Args:
        old: The state before the adapt.
        new: The state apply_marks returned.
        report: The report it returned.
        old_keys: Each old element's cell, (level, ix, iy), by position.
        new_keys: Each new element's cell, by position.

    Returns:
        One finding per broken rule; empty when clean.
    """
    found: list[str] = []
    if report.epoch_old != old.mesh_epoch or report.epoch_new != new.mesh_epoch:
        found.append(
            f"report.epochs: {report.epoch_old} -> {report.epoch_new}, "
            f"the states say {old.mesh_epoch} -> {new.mesh_epoch}"
        )
    width = 2**old.ndim
    if report.n_old != old.n_active or report.n_new != new.n_active or report.new_to_old.shape[1] != width:
        found.append(
            f"report.counts: n_old {report.n_old}, n_new {report.n_new}, width {report.new_to_old.shape[1]}; "
            f"the states say {old.n_active}, {new.n_active}, {width}"
        )
        return found
    if new.time != old.time:
        found.append(f"report.time: {old.time} before the adapt, {new.time} after")

    n_old = old.n_active
    bad_rows = 0
    first = None
    named = np.zeros(n_old, dtype=np.int64)
    for q in range(new.n_active):
        row = [int(x) for x in report.new_to_old[q]]
        kind = int(report.new_kind[q])
        used = row if kind == KIND_MERGED else row[:1]
        ok = all(0 <= o < n_old for o in used) and all(o == -1 for o in row[len(used):])
        if ok:
            for o in used:
                named[o] += 1
            if kind == KIND_UNCHANGED:
                ok = old_keys[row[0]] == new_keys[q]
            elif kind == KIND_CHILD:
                ok = new_keys[q][0] > 0 and old_keys[row[0]] == _parent(new_keys[q])
            else:
                ok = [old_keys[o] for o in row] == _children(new_keys[q])
        if not ok:
            bad_rows += 1
            first = q if first is None else first
    if bad_rows:
        found.append(f"report.rows: {bad_rows}, first at new position {first}")

    expected = realized_by_levels(old_keys, new_keys)
    bad = report.realized != expected
    if bad.any():
        found.append(f"report.realized: {int(bad.sum())}, first at old position {int(np.flatnonzero(bad)[0])}")

    times = np.where(report.realized == REFINE, width, 1)
    bad = named != times
    if bad.any():
        found.append(f"report.cover: {int(bad.sum())}, first at old position {int(np.flatnonzero(bad)[0])}")
    return found


# --- One script on one backend -------------------------------------------------------


def _middle_ic(info) -> dict[str, float]:
    """The middle of every published IC bound."""
    return {name: (low + high) / 2 for name, (low, high) in info.ic_params.items()}


def run_script(backend, script: Script, *, rtol: float = 1e-9, amr_freq: int = 1) -> ScriptResult:
    """Walk a backend through a script and check every state on the way.

    Calls describe once, reset once at level 0, then for each step dry_run
    (if describe() lists it), apply_marks and get_state. A finding that
    leaves nothing to stand on (the grid is not the script's, an address
    matches no element, a mark is ineligible, a centroid is not on its
    level's grid, a verb raised) ends the run; any other is recorded and
    the run goes on. No exception of the backend's comes out.

    Args:
        backend: The backend, built on the script's grid.
        script: The script.
        rtol: Tolerance on a coordinate, as a fraction of the domain's
            extent: for the addresses, the cells and the topology checks.
        amr_freq: Handed to reset; no step advances time.

    Returns:
        The findings, the skipped checks and the leaf file's text.
    """
    found: list[str] = []
    skipped: list[str] = []
    leaves = [f"script {script.name} nx {script.nx} ny {script.ny} max_level {script.max_level}\n"]
    base = (script.nx, script.ny)

    def result() -> ScriptResult:
        return ScriptResult(name=script.name, findings=tuple(found), skipped=tuple(skipped), leaves="".join(leaves))

    def add_leaves(block: str) -> None:
        leaves.append(block)

    def note(step: int, findings: Sequence[str]) -> None:
        found.extend(f"step {step}: {finding}" for finding in findings)

    def call(step: int, verb: str, *args):
        try:
            return getattr(backend, verb)(*args)
        except Exception as error:
            note(step, [f"backend.error: {verb} raised {type(error).__name__}: {error}"])
            raise _Ended from None

    try:
        _walk(script, rtol, amr_freq, base, call, note, skipped, add_leaves)
    except _Ended:
        pass
    return result()


def _walk(script: Script, rtol: float, amr_freq: int, base, call, note, skipped: list[str], add_leaves) -> None:
    """run_script's walk: every backend call goes through call, which ends
    the walk by raising when a verb does."""
    info = call(0, "describe")
    has_dry_run = "dry_run" in info.optional_verbs
    if not has_dry_run:
        skipped.append("dry_run: not offered by this backend")

    state = call(0, "reset", _middle_ic(info), amr_freq, 0)
    grid_is = (state.ndim, state.n_active, int(state.level.max()), state.max_level, tuple(state.periodic.tolist()))
    grid_wanted = (2, script.nx * script.ny, 0, script.max_level, tuple(script.periodic))
    if grid_is != grid_wanted:
        note(0, [f"script.grid: (ndim, n_active, finest, max_level, periodic) is {grid_is}, expected {grid_wanted}"])
        return
    note(0, [f"topology.{finding}" for finding in topology_violations(state, rtol=rtol)])
    keys = cells(state, base, rtol)
    if keys is None:
        note(0, ["script.cells: a centroid is not a cell center of its level's grid"])
        return
    add_leaves(_leaf_block(0, keys))

    for k, step in enumerate(script.steps, start=1):
        marks, missing = step_marks(state, step, rtol)
        if marks is None:
            note(k, missing)
            return
        try:
            round_state = replay(state, marks)
        except ValueError as error:
            note(k, [f"script.ineligible: {error}"])
            return

        if has_dry_run:
            predicted_n, predicted = call(k, "dry_run", round_state.decisions)
            after = call(k, "get_state")
            if after.mesh_epoch != state.mesh_epoch or after.n_active != state.n_active:
                note(k, [f"dry_run.changed: epoch {state.mesh_epoch} -> {after.mesh_epoch}"])

        new, report = call(k, "apply_marks", round_state.decisions)
        current = call(k, "get_state")
        if current.mesh_epoch != new.mesh_epoch or current.n_active != new.n_active:
            note(k, [f"state.current: get_state says epoch {current.mesh_epoch}, apply_marks said {new.mesh_epoch}"])
        note(k, [f"topology.{finding}" for finding in topology_violations(new, rtol=rtol)])
        new_keys = cells(new, base, rtol)
        if new_keys is None:
            note(k, ["script.cells: a centroid is not a cell center of its level's grid"])
            return
        add_leaves(_leaf_block(k, new_keys))

        old_list = [tuple(int(x) for x in row) for row in keys]
        new_list = [tuple(int(x) for x in row) for row in new_keys]
        note(k, report_violations(state, new, report, old_list, new_list))

        if has_dry_run:
            if predicted_n != new.n_active:
                note(k, [f"dry_run.count: predicted {predicted_n}, realized {new.n_active}"])
            if not np.array_equal(predicted, report.realized):
                note(k, ["dry_run.realized: the predicted flags are not the realized flags"])

        over = overshoot(round_state, new)
        if over < 0:
            note(k, [f"counter.negative: realized {new.n_active}, committed {round_state.committed}"])
        if over != step.overshoot:
            note(k, [f"counter.overshoot: {over}, expected {step.overshoot}"])
        if new.n_active != step.n_active:
            note(k, [f"script.n_active: {new.n_active}, expected {step.n_active}"])

        if len(report.realized) == state.n_active:
            wanted = marks.copy()
            unplaced = []
            for address, flag in step.departures.items():
                position = locate(state, address, rtol)
                if position is None:
                    unplaced.append(tuple(address))
                else:
                    wanted[position] = flag
            bad = report.realized != wanted
            if unplaced or bad.any():
                at = [tuple(round(float(x), 12) for x in fractions(state)[p]) for p in np.flatnonzero(bad)[:3]]
                note(k, [f"script.departures: {int(bad.sum())} elements differ, first at {at}; unplaced {unplaced}"])

        state, keys = new, new_keys


# --- The scripts ---------------------------------------------------------------------

# Every script is on a 4x4 base grid at max_level 3, so two meshes serve all
# of them on any backend: one periodic on both axes, one walled on both.
_PERIODIC = {"nx": 4, "ny": 4, "periodic": (True, True), "max_level": 3}
_WALLED = {"nx": 4, "ny": 4, "periodic": (False, False), "max_level": 3}

# The family of the interior base cell [1/4, 1/2]^2, in child order, and
# the family of the corner base cell [0, 1/4]^2.
_INNER = ((0.3125, 0.3125), (0.4375, 0.3125), (0.3125, 0.4375), (0.4375, 0.4375))
_CORNER = ((0.0625, 0.0625), (0.1875, 0.0625), (0.0625, 0.1875), (0.1875, 0.1875))


def _partial(family: Sequence[Address]) -> tuple[Step, ...]:
    """Three siblings marked COARSEN beside a hold, then beside a refine."""
    three = {address: C for address in family[:3]}
    held = {address: H for address in family[:3]}
    return (
        Step(everywhere=R, n_active=64),
        Step(where=three, n_active=64, departures=held),
        Step(where={**three, family[3]: R}, n_active=67, departures=held),
    )


SCRIPTS: tuple[Script, ...] = (
    # A partly marked family is dropped at coarsen: nothing merges, and the
    # fourth sibling's refine in the same round happens.
    Script(name="partial", **_PERIODIC, steps=_partial(_INNER)),
    # A refine that forces balance refines: the level-2 children touch two
    # base cells across faces and one across a corner alone.
    Script(
        name="corner",
        **_PERIODIC,
        steps=(
            Step(where={(0.375, 0.375): R}, n_active=19),
            Step(
                where={(0.4375, 0.4375): R},
                n_active=31,
                overshoot=9,
                departures={(0.625, 0.375): R, (0.375, 0.625): R, (0.625, 0.625): R},
            ),
        ),
    ),
    # A complete family's coarsen undone by balance: an element touching the
    # family at one corner refines in the same round.
    Script(
        name="undone",
        **_PERIODIC,
        steps=(
            Step(everywhere=R, n_active=64),
            Step(
                where={**{address: C for address in _INNER}, (0.5625, 0.5625): R},
                n_active=67,
                overshoot=3,
                departures={address: H for address in _INNER},
            ),
        ),
    ),
    # A family coarsen that succeeds, after a round of all HOLD.
    Script(
        name="merge",
        **_PERIODIC,
        steps=(
            Step(where={(0.375, 0.375): R}, n_active=19),
            Step(n_active=19),
            Step(where={address: C for address in _INNER}, n_active=16),
        ),
    ),
    # A COARSEN mark that comes back REFINE: the merge is undone by balance,
    # and balance then splits one member.
    Script(
        name="coarsen_to_refine",
        **_PERIODIC,
        steps=(
            Step(everywhere=R, n_active=64),
            Step(where={(0.5625, 0.3125): R}, n_active=67),
            Step(
                where={**{address: C for address in _INNER}, (0.53125, 0.28125): R},
                n_active=79,
                overshoot=12,
                departures={
                    (0.3125, 0.3125): H,
                    (0.4375, 0.3125): R,
                    (0.3125, 0.4375): H,
                    (0.4375, 0.4375): H,
                    (0.4375, 0.1875): R,
                    (0.5625, 0.1875): R,
                },
            ),
        ),
    ),
    # The same three with the family in the corner base cell, so that every
    # neighbor that matters is across a periodic wrap.
    Script(name="wrap_partial", **_PERIODIC, steps=_partial(_CORNER)),
    Script(
        name="wrap_corner",
        **_PERIODIC,
        steps=(
            Step(where={(0.125, 0.125): R}, n_active=19),
            Step(
                where={(0.0625, 0.0625): R},
                n_active=31,
                overshoot=9,
                departures={(0.875, 0.125): R, (0.125, 0.875): R, (0.875, 0.875): R},
            ),
        ),
    ),
    Script(
        name="wrap_undone",
        **_PERIODIC,
        steps=(
            Step(everywhere=R, n_active=64),
            Step(
                where={**{address: C for address in _CORNER}, (0.9375, 0.9375): R},
                n_active=67,
                overshoot=3,
                departures={address: H for address in _CORNER},
            ),
        ),
    ),
    # The two wrap scripts' marks on walls: nothing is across the boundary,
    # so the refine forces nothing and the coarsen stands.
    Script(
        name="wall_corner",
        **_WALLED,
        steps=(
            Step(where={(0.125, 0.125): R}, n_active=19),
            Step(where={(0.0625, 0.0625): R}, n_active=22),
        ),
    ),
    Script(
        name="wall_coarsen",
        **_WALLED,
        steps=(
            Step(everywhere=R, n_active=64),
            Step(where={**{address: C for address in _CORNER}, (0.9375, 0.9375): R}, n_active=64),
        ),
    ),
)


# --- Every script, and the files -----------------------------------------------------


def run_scripts(
    make_backend: Callable[[dict], object],
    out_dir: str | Path | None = None,
    *,
    scripts: Sequence[Script] = SCRIPTS,
    rtol: float = 1e-9,
) -> list[ScriptResult]:
    """Run scripts, each on a backend built for its grid, and write the files.

    Args:
        make_backend: Given a script's grid (nx, ny, periodic, max_level),
            returns a backend on it. One backend is built per script.
        out_dir: Where to write <name>.leaves.txt per script and
            harness.json; None writes nothing.
        scripts: The scripts to run.
        rtol: As run_script takes it.

    Returns:
        One result per script, in order.
    """
    results = [run_script(make_backend(script.grid()), script, rtol=rtol) for script in scripts]
    if out_dir is not None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        for item in results:
            (out / f"{item.name}.leaves.txt").write_text(item.leaves)
        summary = {
            "rtol": rtol,
            "replay_rule": REPLAY_RULE,
            "scripts": [
                {**script.to_dict(), "findings": list(item.findings), "skipped": list(item.skipped)}
                for script, item in zip(scripts, results, strict=True)
            ],
        }
        (out / "harness.json").write_text(json.dumps(summary, indent=2) + "\n")
    return results


def make_mock(grid: dict):
    """The mock quad backend on a script's grid, on the unit square."""
    from backends.mock_quad.backend import MockQuadBackend

    return MockQuadBackend(
        nx=grid["nx"], ny=grid["ny"], periodic=tuple(grid["periodic"]), max_level=grid["max_level"]
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run every script on the mock; exit 1 if any has a finding."""
    parser = argparse.ArgumentParser(description="The scripted-action harness, on the mock quad backend.")
    parser.add_argument("--out", required=True, help="directory for the leaf files and harness.json")
    parser.add_argument("--rtol", type=float, default=1e-9, help="tolerance on a coordinate, as a fraction")
    args = parser.parse_args(argv)

    results = run_scripts(make_mock, args.out, rtol=args.rtol)
    for item in results:
        status = "clean" if not item.findings else f"{len(item.findings)} findings"
        print(f"{item.name}: {status}" + (f"; skipped {list(item.skipped)}" if item.skipped else ""))
        for finding in item.findings:
            print(f"  {finding}")
    return 1 if any(item.findings for item in results) else 0


if __name__ == "__main__":
    sys.exit(main())
