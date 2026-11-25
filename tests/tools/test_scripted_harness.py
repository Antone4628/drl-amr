"""The scripted-action harness (tools/scripted_harness.py) on the mock quad
backend, and on mock backends tampered with so that each check has
something to find.

AGENT_2D_ROADMAP item 4.2, signals H1 to H9. The ten scripts' pins were
derived by hand from p4est's rules and are asserted here on the mock; that
Jexpresso realizes the same is the bridge roadmap's B3.3 and B9.1. The leaf
files are text of integers, so their checksums below do not depend on the
numpy version.

Run: pytest tests/tools/test_scripted_harness.py -v
"""
from __future__ import annotations

import ast
import dataclasses
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from agent.round_commit import commit, start_round
from backends.mock_quad.backend import MockQuadBackend
from contract.reports import KIND_CHILD, KIND_MERGED, KIND_UNCHANGED, AdaptReport
from contract.solver_contract import COARSEN, HOLD, REFINE
from tools import scripted_harness
from tools.scripted_harness import (
    SCRIPTS,
    Script,
    Step,
    cells,
    fractions,
    locate,
    make_mock,
    realized_by_levels,
    replay,
    run_script,
    run_scripts,
    step_marks,
)

C, H, R = COARSEN, HOLD, REFINE

BY_NAME = {script.name: script for script in SCRIPTS}

# A second domain, not square and not at the origin: nothing in a script
# or a leaf file may depend on it.
DOMAINS = {"unit": {}, "offset": {"lo": (-5.0, -10.0), "hi": (5.0, 10.0)}}

# SHA-256 of each script's leaf file.
LEAF_SHA = {
    "partial": "67895cd5d984aa947c1dfebeefa5a4a5d7d6c1592c02eceb5f41631e75e70244",
    "corner": "0a503b9edde1157325adedd49b1915460c15ee8f9e0a5300a1d93a3bc89c7677",
    "undone": "e36e953f5bfa69c4c50327bb015891d7958f5f67bb45f132b8d4af9b146e31f1",
    "merge": "74a67969fdf0f4b7f2a73551ce9b12b188d010c45be8bfebf9a9a746d13a38f3",
    "coarsen_to_refine": "cf6b89f59d4558e77bf01d06cd8036a9383504a8972606ca95bd0379c381925d",
    "wrap_partial": "3545aceda61d5ae6bfbc283b46143b4c4b0a0c01ebaa980987d1e8c2dd3c6fa3",
    "wrap_corner": "fb75a4b980c48d771dae3cbc26da85e53323dced5925375a203aed213c9d37aa",
    "wrap_undone": "7ffbced6bf57d3af5d455d31b2f3b2402c352c8f8ae07c791b6059aed86200e5",
    "wall_corner": "79183dcddc5c520769deb97d2792198450e05363c660d943ee4332388fd21118",
    "wall_coarsen": "943f350f4705134cb186a3c6b39622d478dc247b93ee8ec7ee6abf8eef783668",
}

# (n_active, overshoot, number of departures) for each step of each script.
PINS = {
    "partial": [(64, 0, 0), (64, 0, 3), (67, 0, 3)],
    "corner": [(19, 0, 0), (31, 9, 3)],
    "undone": [(64, 0, 0), (67, 3, 4)],
    "merge": [(19, 0, 0), (19, 0, 0), (16, 0, 0)],
    "coarsen_to_refine": [(64, 0, 0), (67, 0, 0), (79, 12, 6)],
    "wrap_partial": [(64, 0, 0), (64, 0, 3), (67, 0, 3)],
    "wrap_corner": [(19, 0, 0), (31, 9, 3)],
    "wrap_undone": [(64, 0, 0), (67, 3, 4)],
    "wall_corner": [(19, 0, 0), (22, 0, 0)],
    "wall_coarsen": [(64, 0, 0), (64, 0, 0)],
}


def mock(script, rotate=True, **overrides):
    """The mock on a script's grid."""
    grid = {**script.grid(), **overrides}
    return MockQuadBackend(
        nx=grid["nx"],
        ny=grid["ny"],
        periodic=tuple(grid["periodic"]),
        max_level=grid["max_level"],
        rotate_cells=rotate,
        **{key: grid[key] for key in ("lo", "hi") if key in grid},
    )


def rules(result):
    """The rule of each finding, in order: 'step 2: report.rows: ...' -> 'report.rows'."""
    return [finding.split(": ")[1] for finding in result.findings]


def steps_of(result):
    """The step of each finding, in order."""
    return [int(finding.split(":")[0].removeprefix("step ")) for finding in result.findings]


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def blocks(leaves):
    """A leaf file's blocks: the header line, then (adapt, rows) pairs."""
    lines = leaves.splitlines()
    out = []
    for line in lines[1:]:
        if line.startswith("adapt"):
            _, adapt, _, count = line.split()
            out.append((int(adapt), int(count), []))
        else:
            out[-1][2].append(tuple(int(x) for x in line.split()))
    return lines[0], out


class Tampered:
    """Forward every verb to a real backend, and pass what some return
    through a function first.

    Keyword arguments, each a function or absent:
        reset(state) -> state
        marks(marks) -> marks            what apply_marks and dry_run are handed
        apply(state, report) -> (state, report)
        dry_run(n, realized) -> (n, realized)
        get_state(state) -> state
    at: the apply_marks call whose return is changed; None means every one.
    """

    def __init__(self, inner, at=None, **changes):
        self.inner = inner
        self.at = at
        self.changes = changes
        self.applies = 0
        self.calls = []

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def reset(self, *args, **kwargs):
        self.calls.append(("reset", args, kwargs))
        state = self.inner.reset(*args, **kwargs)
        change = self.changes.get("reset")
        return change(state) if change else state

    def get_state(self):
        state = self.inner.get_state()
        change = self.changes.get("get_state")
        return change(state) if change else state

    def _marks(self, marks):
        change = self.changes.get("marks")
        return change(marks) if change else marks

    def dry_run(self, marks):
        self.calls.append(("dry_run",))
        n, realized = self.inner.dry_run(self._marks(marks))
        change = self.changes.get("dry_run")
        return change(n, realized) if change else (n, realized)

    def apply_marks(self, marks):
        self.applies += 1
        state, report = self.inner.apply_marks(self._marks(marks))
        change = self.changes.get("apply")
        if change and self.at in (None, self.applies):
            return change(state, report)
        return state, report


def with_report(**fields):
    """An apply change that replaces fields of the report; a value may be a
    function of the report."""

    def change(state, report):
        values = {name: (value(report) if callable(value) else value) for name, value in fields.items()}
        return state, dataclasses.replace(report, **values)

    return change


# --- H1, H3: the ten scripts on the mock ---------------------------------------------


def test_there_are_ten_scripts_on_two_grids():
    assert list(BY_NAME) == list(PINS) == list(LEAF_SHA)
    grids = {(s.nx, s.ny, s.periodic, s.max_level) for s in SCRIPTS}
    assert grids == {(4, 4, (True, True), 3), (4, 4, (False, False), 3)}


@pytest.mark.parametrize("name", PINS)
def test_the_pins_are_the_documented_ones(name):
    steps = BY_NAME[name].steps
    assert [(s.n_active, s.overshoot, len(s.departures)) for s in steps] == PINS[name]


@pytest.mark.parametrize("domain", DOMAINS)
@pytest.mark.parametrize("rotate", [False, True], ids=["pinned", "rotating"])
@pytest.mark.parametrize("name", PINS)
def test_script_is_clean_and_its_leaf_file_is_the_pinned_one(name, rotate, domain):
    script = BY_NAME[name]
    result = run_script(mock(script, rotate, **DOMAINS[domain]), script)
    assert result.findings == ()
    assert result.skipped == ()
    assert result.name == name
    assert sha(result.leaves) == LEAF_SHA[name]


@pytest.mark.parametrize("name", PINS)
def test_leaf_file_is_sorted_integers_one_block_per_adapt(name):
    script = BY_NAME[name]
    result = run_script(mock(script), script)
    header, found = blocks(result.leaves)
    assert header == f"script {name} nx 4 ny 4 max_level 3"
    assert result.leaves.endswith("\n")
    assert [adapt for adapt, _, _ in found] == list(range(len(script.steps) + 1))
    assert [count for _, count, _ in found] == [16] + [step.n_active for step in script.steps]
    for _, count, rows in found:
        assert len(rows) == count == len(set(rows))
        assert rows == sorted(rows)
        assert all(len(row) == 3 for row in rows)
        # The leaves tile the grid: areas in units of the finest cell.
        assert sum(4 ** (3 - level) for level, _, _ in rows) == 16 * 4**3


def test_the_instance_of_beat_1():
    # explanation file, section 4.2.2: the corner script's last block.
    script = BY_NAME["corner"]
    result = run_script(mock(script), script)
    assert len(result.leaves.splitlines()) == 70
    _, found = blocks(result.leaves)
    levels = [level for level, _, _ in found[-1][2]]
    assert (levels.count(0), levels.count(1), levels.count(2)) == (12, 15, 4)
    departures = script.steps[1].departures
    assert departures == {(0.625, 0.375): R, (0.375, 0.625): R, (0.625, 0.625): R}


# --- H4: the wrap and the walls ------------------------------------------------------


@pytest.mark.parametrize("name", ["partial", "corner", "undone"])
def test_wrap_script_has_the_interior_scripts_pins(name):
    assert PINS[f"wrap_{name}"] == PINS[name]
    interior, wrap = BY_NAME[name], BY_NAME[f"wrap_{name}"]
    assert [len(s.where) for s in interior.steps] == [len(s.where) for s in wrap.steps]
    assert LEAF_SHA[f"wrap_{name}"] != LEAF_SHA[name]


@pytest.mark.parametrize("name", ["wrap_partial", "wrap_corner", "wrap_undone"])
def test_wrap_script_departs_only_through_a_wrap(name):
    # On walls the same marks realize as asked wherever the wrap script's
    # departures came from a neighbor across the boundary.
    script = BY_NAME[name]
    walled = dataclasses.replace(script, periodic=(False, False))
    result = run_script(mock(walled), walled)
    expected = [] if name == "wrap_partial" else ["counter.overshoot", "script.n_active", "script.departures"]
    assert rules(result) == expected


def test_the_walled_scripts_are_the_wrap_scripts_marks():
    for wall, wrap in (("wall_corner", "wrap_corner"), ("wall_coarsen", "wrap_undone")):
        assert [dict(s.where) for s in BY_NAME[wall].steps] == [dict(s.where) for s in BY_NAME[wrap].steps]
        assert [s.everywhere for s in BY_NAME[wall].steps] == [s.everywhere for s in BY_NAME[wrap].steps]
    assert (PINS["wall_corner"][-1], PINS["wrap_corner"][-1]) == ((22, 0, 0), (31, 9, 3))
    assert (PINS["wall_coarsen"][-1], PINS["wrap_undone"][-1]) == ((64, 0, 0), (67, 3, 4))


def test_a_coarsen_mark_comes_back_refine():
    step = BY_NAME["coarsen_to_refine"].steps[-1]
    back = [address for address, flag in step.departures.items() if step.where.get(address) == C and flag == R]
    assert back == [(0.4375, 0.3125)]


# --- H2: the calls -------------------------------------------------------------------


@pytest.mark.parametrize("name", PINS)
def test_the_calls_are_the_five_verbs_and_no_other(name, counting):
    script = BY_NAME[name]
    backend = counting(mock(script))
    run_script(backend, script)
    n = len(script.steps)
    assert dict(backend.calls) == {"describe": 1, "reset": 1, "dry_run": n, "apply_marks": n, "get_state": 2 * n}


def test_reset_is_at_level_zero_with_the_middle_of_the_bounds():
    script = BY_NAME["corner"]
    backend = Tampered(mock(script, lo=(-5.0, -10.0), hi=(5.0, 10.0)))
    run_script(backend, script, amr_freq=7)
    (_, args, kwargs), = [call for call in backend.calls if call[0] == "reset"]
    assert not kwargs
    ic, amr_freq, initial_level = args
    assert ic == {"x0": 0.0, "y0": 0.0, "width": 1.5}
    assert (amr_freq, initial_level) == (7, 0)


# --- H5: a backend without dry_run ---------------------------------------------------


class NoDryRun:
    """The mock with dry_run not offered: absent from describe(), and raising."""

    def __init__(self, inner):
        self.inner = inner

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def describe(self):
        return dataclasses.replace(self.inner.describe(), optional_verbs=frozenset())

    def dry_run(self, marks):
        raise NotImplementedError("dry_run: not offered")


@pytest.mark.parametrize("name", PINS)
def test_without_dry_run_the_check_is_skipped_by_name(name):
    script = BY_NAME[name]
    result = run_script(NoDryRun(mock(script)), script)
    assert result.findings == ()
    assert result.skipped == ("dry_run: not offered by this backend",)
    assert sha(result.leaves) == LEAF_SHA[name]


def test_without_dry_run_it_is_never_called(counting):
    script = BY_NAME["corner"]
    backend = counting(NoDryRun(mock(script)))
    run_script(backend, script)
    assert dict(backend.calls) == {"describe": 1, "reset": 1, "apply_marks": 2, "get_state": 2}


# --- H6: each rule, on a backend that breaks it --------------------------------------


def flip_first_hold(report):
    realized = report.realized.copy()
    realized[np.flatnonzero(realized == HOLD)[0]] = REFINE
    return realized


def reverse_merged_row(report):
    rows = report.new_to_old.copy()
    q = np.flatnonzero(report.new_kind == KIND_MERGED)[0]
    rows[q] = rows[q][::-1]
    return rows


def swap_two_unchanged_rows(report):
    rows = report.new_to_old.copy()
    a, b = np.flatnonzero(report.new_kind == KIND_UNCHANGED)[:2]
    rows[[a, b]] = rows[[b, a]]
    return rows


def name_one_old_element_twice(report):
    rows = report.new_to_old.copy()
    a, b = np.flatnonzero(report.new_kind == KIND_UNCHANGED)[:2]
    rows[a] = rows[b]
    return rows


def fill_an_unused_slot(report):
    rows = report.new_to_old.copy()
    q = np.flatnonzero(report.new_kind == KIND_UNCHANGED)[0]
    rows[q, 1] = rows[q, 0]
    return rows


def swap_two_child_rows(report):
    rows = report.new_to_old.copy()
    children = np.flatnonzero(report.new_kind == KIND_CHILD)
    a = children[0]
    b = children[rows[children, 0] != rows[a, 0]][0]
    rows[[a, b]] = rows[[b, a]]
    return rows


def call_a_child_unchanged(report):
    kind = report.new_kind.copy()
    kind[np.flatnonzero(kind == KIND_CHILD)[0]] = KIND_UNCHANGED
    return kind


def drop_the_first_refine(marks):
    marks = np.array(marks)
    marks[np.flatnonzero(marks == REFINE)[0]] = HOLD
    return marks


def flip_the_wrap(state, report):
    return dataclasses.replace(state, neighbor_wrap=~state.neighbor_wrap), report


def later(state, report):
    return dataclasses.replace(state, time=state.time + 1.0), report


def shifted_epochs(state, report):
    return state, dataclasses.replace(report, epoch_old=report.epoch_old + 1, epoch_new=report.epoch_new + 1)


def a_later_epoch(state, report):
    return dataclasses.replace(state, mesh_epoch=state.mesh_epoch + 1), report


def one_row_short(state, report):
    short = AdaptReport(
        realized=report.realized.copy(),
        new_to_old=report.new_to_old[:-1].copy(),
        new_kind=report.new_kind[:-1].copy(),
        epoch_old=report.epoch_old,
        epoch_new=report.epoch_new,
    )
    return state, short


# (script, the apply call tampered with, the changes, the rules found, in order)
TAMPERS = {
    "realized_flipped": (
        "corner", 2, {"apply": with_report(realized=flip_first_hold)},
        ["report.realized", "report.cover", "dry_run.realized", "script.departures"],
    ),
    "merged_row_reversed": (
        "merge", 3, {"apply": with_report(new_to_old=reverse_merged_row)}, ["report.rows"],
    ),
    "unchanged_rows_swapped": (
        "corner", 2, {"apply": with_report(new_to_old=swap_two_unchanged_rows)}, ["report.rows"],
    ),
    "old_element_named_twice": (
        "corner", 2, {"apply": with_report(new_to_old=name_one_old_element_twice)}, ["report.rows", "report.cover"],
    ),
    "unused_slot_filled": (
        "corner", 2, {"apply": with_report(new_to_old=fill_an_unused_slot)}, ["report.rows", "report.cover"],
    ),
    "child_rows_swapped": (
        "corner", 2, {"apply": with_report(new_to_old=swap_two_child_rows)}, ["report.rows"],
    ),
    "child_called_unchanged": (
        "corner", 2, {"apply": with_report(new_kind=call_a_child_unchanged)}, ["report.rows"],
    ),
    "epochs_shifted": ("corner", 2, {"apply": shifted_epochs}, ["report.epochs"]),
    "state_at_a_later_epoch": ("corner", 2, {"apply": a_later_epoch}, ["state.current", "report.epochs"]),
    "report_one_row_short": ("corner", 2, {"apply": one_row_short}, ["report.counts"]),
    "time_advanced": ("corner", 2, {"apply": later}, ["report.time"]),
    "wrap_flipped": ("corner", 2, {"apply": flip_the_wrap}, ["topology.faces.wrap"]),
    "dry_run_count_off": ("corner", None, {"dry_run": lambda n, realized: (n + 3, realized)}, ["dry_run.count"] * 2),
    "dry_run_flags_off": (
        "corner", None, {"dry_run": lambda n, realized: (n, np.zeros_like(realized))}, ["dry_run.realized"] * 2,
    ),
    "a_refine_dropped": (
        "corner", None, {"marks": drop_the_first_refine},
        ["counter.negative", "counter.overshoot", "script.n_active", "script.departures", "script.address"],
    ),
}


@pytest.mark.parametrize("name", TAMPERS)
def test_a_tampered_backend_is_found_by_the_rule(name):
    script_name, at, changes, expected = TAMPERS[name]
    script = BY_NAME[script_name]
    result = run_script(Tampered(mock(script), at=at, **changes), script)
    assert rules(result) == expected
    if at is not None:
        assert set(steps_of(result)) == {at}


def test_a_stale_get_state_is_found():
    script = BY_NAME["corner"]
    inner = mock(script)
    first = []

    def stale(state):
        first.append(state)
        return first[0]

    result = run_script(Tampered(inner, get_state=stale), script)
    assert rules(result) == ["state.current", "dry_run.changed", "state.current"]
    assert steps_of(result) == [1, 2, 2]


def test_a_dry_run_that_adapts_is_found():
    script = BY_NAME["corner"]
    inner = mock(script, rotate=False)

    def adapting(n, realized):
        inner.apply_marks(np.zeros(len(realized), dtype=np.int8))
        return n, realized

    result = run_script(Tampered(inner, dry_run=adapting), script)
    assert rules(result) == ["dry_run.changed", "report.epochs"] * 2


def test_a_wrong_pin_is_found_by_its_rule():
    script = BY_NAME["undone"]
    last = script.steps[-1]
    for change, expected in (
        ({"n_active": 64}, ["script.n_active"]),
        ({"overshoot": 0}, ["counter.overshoot"]),
        ({"departures": {}}, ["script.departures"]),
        ({"departures": {**last.departures, (0.5625, 0.5625): H}}, ["script.departures"]),
        ({"departures": {**last.departures, (0.3, 0.3): H}}, ["script.departures"]),
    ):
        wrong = dataclasses.replace(script, steps=(script.steps[0], dataclasses.replace(last, **change)))
        result = run_script(mock(wrong), wrong)
        assert rules(result) == expected
        assert steps_of(result) == [2]


def test_a_finding_names_its_step_and_what_was_found():
    script = BY_NAME["corner"]
    wrong = dataclasses.replace(script, steps=(script.steps[0], dataclasses.replace(script.steps[1], overshoot=0)))
    assert run_script(mock(wrong), wrong).findings == ("step 2: counter.overshoot: 9, expected 0",)


# --- H7: the findings that end a run -------------------------------------------------


@pytest.mark.parametrize(
    "overrides",
    [{"nx": 3}, {"periodic": (True, False)}, {"max_level": 2}],
    ids=["cells", "periodic", "max_level"],
)
def test_another_grid_than_the_scripts_ends_the_run(overrides):
    script = BY_NAME["corner"]
    result = run_script(mock(script, **overrides), script)
    assert rules(result) == ["script.grid"]
    assert result.leaves == "script corner nx 4 ny 4 max_level 3\n"


def test_an_address_with_no_element_ends_the_run():
    script = BY_NAME["corner"]
    lost = dataclasses.replace(script.steps[1], where={(0.4375, 0.4375): R, (0.3, 0.3): R, (0.375, 0.375): R})
    wrong = dataclasses.replace(script, steps=(script.steps[0], lost))
    backend = Tampered(mock(wrong))
    result = run_script(backend, wrong)
    assert result.findings == (
        "step 2: script.address: no single element at (0.3, 0.3)",
        "step 2: script.address: no single element at (0.375, 0.375)",
    )
    assert backend.applies == 1
    assert [adapt for adapt, _, _ in blocks(result.leaves)[1]] == [0, 1]


def test_a_mark_the_mask_refuses_ends_the_run():
    coarsen_at_level_0 = Script(
        name="x", nx=4, ny=4, periodic=(True, True), max_level=3, steps=(Step(where={(0.125, 0.125): C}, n_active=16),)
    )
    refine_at_max_level = Script(
        name="x",
        nx=4,
        ny=4,
        periodic=(True, True),
        max_level=1,
        steps=(Step(everywhere=R, n_active=64), Step(everywhere=R, n_active=64)),
    )
    for script, step in ((coarsen_at_level_0, 1), (refine_at_max_level, 2)):
        backend = Tampered(mock(script))
        result = run_script(backend, script)
        assert rules(result) == ["script.ineligible"]
        assert steps_of(result) == [step]
        assert backend.applies == step - 1


def test_a_centroid_off_its_grid_ends_the_run():
    script = BY_NAME["corner"]
    backend = Tampered(mock(script), reset=nudged)
    result = run_script(backend, script)
    topology = ["topology.faces.normal", "topology.faces.tangential", "topology.uniform_base.centroid"]
    assert rules(result) == [*topology, "script.cells"]
    assert backend.applies == 0


def nudged(state):
    centroid = state.centroid.copy()
    centroid[0, 0] += 1e-6
    return dataclasses.replace(state, centroid=centroid)


def test_a_looser_tolerance_accepts_nudged_centroids():
    script = BY_NAME["corner"]

    def all_nudged(state):
        centroid = state.centroid.copy()
        centroid[:, 0] += 1e-6
        return dataclasses.replace(state, centroid=centroid)

    def every_state(inner):
        return Tampered(inner, reset=all_nudged, apply=lambda state, report: (all_nudged(state), report))

    assert run_script(every_state(mock(script)), script).findings != ()
    assert run_script(every_state(mock(script)), script, rtol=1e-5).findings == ()
    after_the_adapts = Tampered(mock(script), apply=lambda state, report: (nudged(state), report))
    assert run_script(after_the_adapts, script, rtol=1e-5).findings == ()


@pytest.mark.parametrize(
    ("verb", "at", "step", "applies", "adapts"),
    [
        ("describe", 1, 0, 0, []),
        ("reset", 1, 0, 0, []),
        ("dry_run", 2, 2, 1, [0, 1]),
        ("apply_marks", 2, 2, 1, [0, 1]),
        ("get_state", 4, 2, 2, [0, 1]),
    ],
)
def test_a_verb_that_raises_ends_the_run_as_a_finding(verb, at, step, applies, adapts, failing, counting):
    script = BY_NAME["corner"]
    backend = counting(failing(mock(script), verb, at))
    result = run_script(backend, script)
    assert result.findings == (
        f"step {step}: backend.error: {verb} raised BackendFailure: {verb}: failed on demand at call {at}",
    )
    assert backend.calls["apply_marks"] == applies + (verb == "apply_marks")
    assert [adapt for adapt, _, _ in blocks(result.leaves)[1]] == adapts


def test_any_exception_of_a_verb_is_a_finding_and_the_other_scripts_still_run(tmp_path):
    def make(grid):
        inner = make_mock(grid)
        if not grid["periodic"][0]:
            return Tampered(inner, apply=lambda state, report: 1 / 0)
        return inner

    results = run_scripts(make, tmp_path)
    assert [bool(r.findings) for r in results] == [False] * 8 + [True] * 2
    assert results[8].findings == ("step 1: backend.error: apply_marks raised ZeroDivisionError: division by zero",)
    assert len(list(tmp_path.iterdir())) == 11


# --- The parts -----------------------------------------------------------------------


def state_of(name, n_steps, **overrides):
    """The state a script's first n_steps steps leave on the mock."""
    script = BY_NAME[name]
    backend = mock(script, **overrides)
    info = backend.describe()
    state = backend.reset({k: (lo + hi) / 2 for k, (lo, hi) in info.ic_params.items()}, 1, 0)
    for step in script.steps[:n_steps]:
        marks, _ = step_marks(state, step)
        state, _ = backend.apply_marks(marks)
    return state


def test_fractions_do_not_depend_on_the_domain():
    a = state_of("corner", 2, rotate=False)
    b = state_of("corner", 2, rotate=False, lo=(-5.0, -10.0), hi=(5.0, 10.0))
    assert not np.array_equal(a.centroid, b.centroid)
    assert np.abs(fractions(a) - fractions(b)).max() < 1e-12


def test_locate_finds_one_element_or_none():
    state = state_of("corner", 2)
    p = locate(state, (0.4375, 0.4375))
    assert p is None  # refined away by step 2
    p = locate(state, (0.46875, 0.46875))
    assert state.level[p] == 2
    assert locate(state, (0.46875, 0.46875 + 2e-9)) is None
    assert locate(state, (0.46875, 0.46875 + 2e-9), rtol=1e-8) == p
    assert locate(state, (0.5, 0.5)) is None
    # A tolerance wide enough for two elements matches neither.
    assert locate(state, (0.46875, 0.46875), rtol=0.07) is None


def test_cells_are_the_level_and_the_integer_cell():
    state = state_of("corner", 2, lo=(-5.0, -10.0), hi=(5.0, 10.0))
    keys = cells(state, (4, 4))
    assert keys.dtype == np.int64 and keys.shape == (31, 3)
    assert np.array_equal(keys[:, 0], state.level)
    p = locate(state, (0.46875, 0.46875))
    assert keys[p].tolist() == [2, 7, 7]
    assert cells(state, (3, 4)) is None
    assert cells(state, (2, 2)) is None


def test_step_marks_names_every_address_it_cannot_place():
    state = state_of("corner", 1)
    marks, found = step_marks(state, Step(where={(0.4375, 0.4375): R}, everywhere=H, n_active=0))
    assert found == [] and marks.dtype == np.int8
    assert (marks == R).sum() == 1 and (marks == H).sum() == 18
    marks, found = step_marks(state, Step(everywhere=R, n_active=0))
    assert (marks == R).all()
    marks, found = step_marks(state, Step(where={(0.1, 0.1): R, (0.2, 0.2): C}, n_active=0))
    assert marks is None and len(found) == 2


@pytest.mark.parametrize("seed", range(6))
def test_the_replayed_count_does_not_depend_on_the_order(seed):
    # Marks the mask accepts, on a mesh with complete and incomplete families.
    state = state_of("corner", 2)
    rng = np.random.default_rng(seed)
    marks = rng.choice([C, H, R], size=state.n_active).astype(np.int8)
    marks[(marks == C) & ~state.can_coarsen] = H
    marks[(marks == R) & ~state.can_refine] = H
    forward = replay(state, marks)
    assert np.array_equal(forward.decisions, marks)

    action = {C: 0, H: 1, R: 2}
    backward = start_round(state, 1, 1)
    for position in rng.permutation(state.n_active):
        backward = commit(state, backward, int(position), action[int(marks[position])], "unanimous")
    assert backward.committed == forward.committed

    # The count by its definition: 3 per refine, 3 per family marked whole.
    whole = {tuple(row) for row in state.family[state.can_coarsen].tolist() if (marks[list(row)] == C).all()}
    assert forward.committed == state.n_active + 3 * int((marks == R).sum()) - 3 * len(whole)


def test_replay_refuses_what_the_mask_refuses():
    state = state_of("corner", 1)
    marks = np.zeros(state.n_active, dtype=np.int8)
    marks[np.flatnonzero(~state.can_coarsen)[0]] = C
    with pytest.raises(ValueError, match="masked"):
        replay(state, marks)


def test_realized_by_levels():
    old = [(0, 0, 0), (1, 2, 0), (1, 3, 0), (1, 2, 1), (1, 3, 1), (0, 0, 1), (0, 1, 1)]
    new = [(0, 0, 0), (0, 1, 0), (1, 0, 2), (1, 1, 2), (1, 0, 3), (1, 1, 3), (1, 2, 2)]
    assert realized_by_levels(old, new).tolist() == [H, C, C, C, C, R, -128]


# --- H9: the files and the command ---------------------------------------------------


def test_run_scripts_writes_a_leaf_file_per_script_and_the_summary(tmp_path):
    results = run_scripts(make_mock, tmp_path / "out")
    assert [r.name for r in results] == list(PINS)
    written = sorted(p.name for p in (tmp_path / "out").iterdir())
    assert written == sorted([f"{name}.leaves.txt" for name in PINS] + ["harness.json"])
    for name in PINS:
        assert sha((tmp_path / "out" / f"{name}.leaves.txt").read_text()) == LEAF_SHA[name]

    summary = json.loads((tmp_path / "out" / "harness.json").read_text())
    assert summary["rtol"] == 1e-9 and summary["replay_rule"] == "unanimous"
    assert [s["name"] for s in summary["scripts"]] == list(PINS)
    corner = summary["scripts"][1]
    assert corner["findings"] == [] and corner["skipped"] == []
    assert (corner["nx"], corner["ny"], corner["periodic"], corner["max_level"]) == (4, 4, [True, True], 3)
    assert corner["steps"][1] == {
        "where": [[0.4375, 0.4375, "REFINE"]],
        "everywhere": "HOLD",
        "n_active": 31,
        "overshoot": 9,
        "departures": [[0.625, 0.375, "REFINE"], [0.375, 0.625, "REFINE"], [0.625, 0.625, "REFINE"]],
    }


def test_run_scripts_writes_nothing_without_a_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run_scripts(make_mock, scripts=[BY_NAME["merge"]])
    assert list(tmp_path.iterdir()) == []


def test_the_summary_carries_findings_and_skips(tmp_path):
    script = BY_NAME["corner"]
    wrong = dataclasses.replace(script, steps=(script.steps[0], dataclasses.replace(script.steps[1], overshoot=0)))
    run_scripts(lambda grid: NoDryRun(make_mock(grid)), tmp_path, scripts=[wrong])
    entry = json.loads((tmp_path / "harness.json").read_text())["scripts"][0]
    assert entry["findings"] == ["step 2: counter.overshoot: 9, expected 0"]
    assert entry["skipped"] == ["dry_run: not offered by this backend"]


def test_make_mock_builds_the_scripts_grid_on_the_unit_square():
    state = make_mock(BY_NAME["wall_corner"].grid()).reset({"x0": 0.5, "y0": 0.5, "width": 0.1}, 1, 0)
    assert (state.n_active, state.max_level, state.periodic.tolist()) == (16, 3, [False, False])
    assert (state.domain_lo.tolist(), state.domain_hi.tolist()) == ([0.0, 0.0], [1.0, 1.0])


def test_the_command_exits_0_on_the_mock(tmp_path, capsys):
    assert scripted_harness.main(["--out", str(tmp_path)]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out == [f"{name}: clean" for name in PINS]
    assert len(list(tmp_path.iterdir())) == 11


def test_the_command_exits_1_on_a_finding(tmp_path, capsys, monkeypatch):
    def broken(grid):
        return Tampered(make_mock(grid), dry_run=lambda n, realized: (n + 3, realized))

    monkeypatch.setattr(scripted_harness, "make_mock", broken)
    assert scripted_harness.main(["--out", str(tmp_path)]) == 1
    out = capsys.readouterr().out.splitlines()
    assert out[0] == "partial: 3 findings"
    assert out[1] == "  step 1: dry_run.count: predicted 67, realized 64"


def test_the_command_passes_its_tolerance_on(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(scripted_harness, "make_mock", lambda grid: Tampered(make_mock(grid), reset=nudged))
    assert scripted_harness.main(["--out", str(tmp_path)]) == 1
    assert scripted_harness.main(["--out", str(tmp_path), "--rtol", "1e-5"]) == 0
    assert json.loads((tmp_path / "harness.json").read_text())["rtol"] == 1e-5


def test_the_command_names_a_skip(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(scripted_harness, "make_mock", lambda grid: NoDryRun(make_mock(grid)))
    assert scripted_harness.main(["--out", str(tmp_path)]) == 0
    assert capsys.readouterr().out.splitlines()[0] == "partial: clean; skipped ['dry_run: not offered by this backend']"


# --- H8: what the module imports -----------------------------------------------------


def test_the_modules_project_imports():
    tree = ast.parse(Path(scripted_harness.__file__).read_text())
    top, nested = set(), set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            (top if node in tree.body else nested).add(node.module)
        elif isinstance(node, ast.Import):
            top.update(alias.name for alias in node.names)
    project = {m for m in top if m.split(".")[0] in ("agent", "backends", "contract", "drivers", "envs", "training")}
    assert project == {
        "agent.round_commit",
        "agent.round_mask",
        "contract.checks",
        "contract.reports",
        "contract.solver_contract",
        "contract.state",
    }
    assert nested == {"backends.mock_quad.backend"}
