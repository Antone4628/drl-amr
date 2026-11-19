"""Tests for agent/round_mask.py and the decision record of
agent/round_state.py: the action mask of the 2D agent core and the coarsen
rule a decision follows.

AGENT_2D_ROADMAP item 2.3, signals M2-M11. Seven groups:

  - the record on RoundState: what the field accepts and rejects (M2);
  - the vocabulary: the action indices, the marks they stand for and the
    rule names, and that no function here takes a training-only type (M11);
  - the mask at a round's start, on the five-element instance of the
    explanation file's section 2.3.2 (M3);
  - the three rules on that instance: the two tables of section 2.3.2, cell
    by cell (M4, M5), unanimity (M6), and what the writer returns (M7);
  - what the two functions reject (M8);
  - on a literal 1D state: the same rules with a family of two (M10);
  - on the mock quad backend: whole rounds under each rule, scripted and
    random, and the record handed to apply_marks as its marks (M9).

Records are written as text, one letter per position: C, H and R for the
marks and a dot for an undecided position. Masks are written the same way,
"CHR" with a dash where an action is closed.

Run: pytest tests/agent/test_round_mask.py -v
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from agent import masking as sequential
from agent.round_mask import (
    ACTION_COARSEN,
    ACTION_HOLD,
    ACTION_REFINE,
    ACTION_TO_MARK,
    COARSEN_RULES,
    action_mask,
    decisions_after,
    is_decided,
)
from agent.round_state import UNDECIDED, RoundState
from backends.mock_quad.backend import MockQuadBackend
from contract.checks import training_only_params
from contract.solver_contract import COARSEN, HOLD, REFINE
from contract.state import SolverState

FAMILY_RULES = ("first_decider", "any_coarsen")
LETTER = {".": UNDECIDED, "C": COARSEN, "H": HOLD, "R": REFINE}
ACTION = {"coarsen": ACTION_COARSEN, "hold": ACTION_HOLD, "refine": ACTION_REFINE}
EXPECTED = "expected UNDECIDED, COARSEN, HOLD or REFINE"
# Cell B's position on the instance; positions 0-3 are cell A's children.
B = 4


def record(text: str) -> np.ndarray:
    """A decision record from its letters."""
    return np.array([LETTER[c] for c in text], dtype=np.int8)


def letters(decisions: np.ndarray) -> str:
    """A decision record as letters."""
    by_value = {value: letter for letter, value in LETTER.items()}
    return "".join(by_value[v] for v in decisions.tolist())


def at(text: str | None = None, *, committed: int = 5, round_index: int = 1, n_rounds: int = 2) -> RoundState:
    """Round 1 of 2 holding the record the letters spell; None at the start."""
    decisions = None if text is None else record(text)
    return RoundState(round_index=round_index, n_rounds=n_rounds, committed=committed, decisions=decisions)


def after(state: SolverState, round_state: RoundState, position: int, action: int, rule: str) -> RoundState:
    """The round state one decision later. The committed count is left as it
    was: how it advances is item 2.4's."""
    return dataclasses.replace(round_state, decisions=decisions_after(state, round_state, position, action, rule))


def walk(state: SolverState, rule: str, *steps: tuple[int, str], start: RoundState | None = None) -> RoundState:
    """Take decisions in order, each (position, "coarsen" | "hold" | "refine")."""
    round_state = at() if start is None else start
    for position, name in steps:
        round_state = after(state, round_state, position, ACTION[name], rule)
    return round_state


def opened(state: SolverState, round_state: RoundState, position: int, rule: str) -> str:
    """A mask as letters: "CHR" with a dash where the action is closed."""
    mask = action_mask(state, round_state, position, rule)
    return "".join(letter if is_open else "-" for letter, is_open in zip("CHR", mask.tolist(), strict=True))


def instance_state(*, max_level: int = 2, can_coarsen=None, family=None) -> SolverState:
    """The five-element mesh, built by hand: cell A's children at level 1
    (positions 0-3 in child order) and cell B at level 0 (position 4), on
    [0,2]x[0,1], periodic in x and walled in y."""
    level = np.array([1, 1, 1, 1, 0], dtype=np.int32)
    if family is None:
        family = [[0, 1, 2, 3]] * 4 + [[-1, -1, -1, -1]]
    if can_coarsen is None:
        can_coarsen = [True, True, True, True, False]
    return SolverState(
        level=level,
        neighbors=np.array(
            [
                [[4, -1], [1, -1], [-1, -1], [2, -1]],
                [[0, -1], [4, -1], [-1, -1], [3, -1]],
                [[4, -1], [3, -1], [0, -1], [-1, -1]],
                [[2, -1], [4, -1], [1, -1], [-1, -1]],
                [[1, 3], [0, 2], [-1, -1], [-1, -1]],
            ],
            dtype=np.int32,
        ),
        neighbor_wrap=np.zeros((5, 4), dtype=bool),
        family=np.array(family, dtype=np.int32),
        can_refine=level < max_level,
        can_coarsen=np.array(can_coarsen, dtype=bool),
        centroid=np.array([[0.25, 0.25], [0.75, 0.25], [0.25, 0.75], [0.75, 0.75], [1.5, 0.5]]),
        h=np.array([[0.5, 0.5]] * 4 + [[1.0, 1.0]]),
        attributes=np.zeros((5, 0)),
        n_active=5,
        ndim=2,
        nvar=1,
        max_level=max_level,
        mesh_epoch=2,
        time=0.0,
        interval_T=1.0,
        dt_base=0.25,
        amr_freq=4,
        dt_stable=0.125,
        domain_lo=np.array([0.0, 0.0]),
        domain_hi=np.array([2.0, 1.0]),
        periodic=np.array([True, False]),
    )


# --- The record on RoundState (M2) -------------------------------------------------

def test_a_round_state_holds_no_record_by_default():
    assert RoundState(round_index=1, n_rounds=2, committed=5).decisions is None


def test_undecided_is_an_int8_value_that_is_not_a_mark():
    assert UNDECIDED == -128 == np.iinfo(np.int8).min
    assert UNDECIDED not in (COARSEN, HOLD, REFINE)
    assert UNDECIDED not in ACTION_TO_MARK and UNDECIDED not in ACTION_TO_MARK.values()


def test_the_record_is_the_array_passed_in_and_is_made_read_only():
    decisions = record("CHR..")
    assert decisions.flags.writeable
    state = RoundState(round_index=1, n_rounds=2, committed=5, decisions=decisions)
    assert state.decisions is decisions
    assert not decisions.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        state.decisions[0] = HOLD
    assert state.decisions.tolist() == [COARSEN, HOLD, REFINE, UNDECIDED, UNDECIDED]


def test_a_record_that_is_already_read_only_is_accepted():
    decisions = record("....H")
    decisions.flags.writeable = False
    assert at().decisions is None
    assert RoundState(round_index=1, n_rounds=2, committed=5, decisions=decisions).decisions is decisions


@pytest.mark.parametrize(
    "bad, pattern",
    [
        ([0, 0, 0], r"^decisions: expected a 1-D numpy array or None"),
        ((0, 0, 0), r"^decisions: expected a 1-D numpy array or None"),
        ("CHR", r"^decisions: expected a 1-D numpy array or None"),
        (np.zeros((2, 2), dtype=np.int8), r"^decisions: expected a 1-D numpy array or None"),
        (np.int8(0), r"^decisions: expected a 1-D numpy array or None"),
        (np.zeros(3, dtype=np.int32), r"^decisions: dtype int32, expected int8"),
        (np.zeros(3, dtype=np.int64), r"^decisions: dtype int64, expected int8"),
        (np.zeros(3, dtype=bool), r"^decisions: dtype bool, expected int8"),
        (np.zeros(3), r"^decisions: dtype float64, expected int8"),
        (np.array([0, 2, 5], dtype=np.int8), rf"^decisions: values \[2, 5\], {EXPECTED}$"),
        (np.array([-2, 0], dtype=np.int8), r"^decisions: values \[-2\], expected UNDECIDED, COARSEN"),
        (np.array([127, -127], dtype=np.int8), r"^decisions: values \[-127, 127\], expected"),
    ],
    ids=[
        "a list",
        "a tuple",
        "a string",
        "two dimensions",
        "a numpy scalar",
        "int32",
        "int64",
        "bool",
        "float64",
        "an action index and a 5",
        "minus two",
        "the neighbors of the sentinel",
    ],
)
def test_record_rejections(bad, pattern):
    with pytest.raises(ValueError, match=pattern):
        RoundState(round_index=1, n_rounds=2, committed=5, decisions=bad)


def test_a_rejected_record_stays_writable():
    """The array is frozen last, so a caller whose record was refused still
    owns an ordinary array."""
    bad = np.array([0, 7], dtype=np.int8)
    with pytest.raises(ValueError, match="^decisions: values"):
        RoundState(round_index=1, n_rounds=2, committed=5, decisions=bad)
    assert bad.flags.writeable


def test_the_record_field_is_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        at("....H").decisions = None


def test_the_counts_are_still_checked_beside_a_record():
    with pytest.raises(ValueError, match="^committed: 0, expected at least 1"):
        at(".....", committed=0)


def test_a_record_of_any_length_passes_the_noun():
    """RoundState does not know n_active; the length is checked where the
    record is read (test_a_record_of_the_wrong_length)."""
    for n in (0, 1, 5, 400):
        assert at("." * n).decisions.shape == (n,)


# --- The vocabulary (M11) ----------------------------------------------------------

def test_the_action_indices():
    assert (ACTION_COARSEN, ACTION_HOLD, ACTION_REFINE) == (0, 1, 2)


def test_the_actions_are_the_sequential_cores():
    """Defined twice until the sequential stack retires; this is what keeps
    the two from drifting."""
    assert ACTION_COARSEN == sequential.ACTION_COARSEN
    assert ACTION_HOLD == sequential.ACTION_HOLD
    assert ACTION_REFINE == sequential.ACTION_REFINE
    assert ACTION_TO_MARK == sequential.ACTION_TO_MARK


def test_each_action_stands_for_its_mark():
    assert ACTION_TO_MARK == {0: COARSEN, 1: HOLD, 2: REFINE}
    assert (COARSEN, HOLD, REFINE) == (-1, 0, 1)


def test_the_three_rules():
    assert COARSEN_RULES == ("first_decider", "any_coarsen", "unanimous")
    assert set(FAMILY_RULES) < set(COARSEN_RULES)


@pytest.mark.parametrize("fn", [action_mask, decisions_after, is_decided], ids=lambda fn: fn.__name__)
def test_no_function_takes_a_training_only_type(fn):
    """All three belong to the within-round logic that is implemented twice,
    so none may take a type that exists in training only."""
    assert training_only_params(fn) == []


# --- The mask at a round's start (M3) ----------------------------------------------

@pytest.mark.parametrize("rule", COARSEN_RULES)
def test_at_the_start_the_mask_is_the_structural_one(rule):
    state = instance_state()
    for position in range(4):
        assert opened(state, at(), position, rule) == "CHR"
    assert opened(state, at(), B, rule) == "-HR"


@pytest.mark.parametrize("rule", COARSEN_RULES)
def test_an_explicit_empty_record_is_the_start(rule):
    state = instance_state()
    for position in range(5):
        assert opened(state, at("....."), position, rule) == opened(state, at(), position, rule)
        assert not is_decided(state, at("....."), position)
        assert not is_decided(state, at(), position)


@pytest.mark.parametrize("rule", COARSEN_RULES)
def test_refine_follows_can_refine(rule):
    """At max_level = 1 cell A's children are at the finest level."""
    state = instance_state(max_level=1)
    assert state.can_refine.tolist() == [False, False, False, False, True]
    for position in range(4):
        assert opened(state, at(), position, rule) == "CH-"
    assert opened(state, at(), B, rule) == "-HR"


@pytest.mark.parametrize("rule", COARSEN_RULES)
def test_coarsen_follows_can_coarsen(rule):
    """A backend that reports no element coarsenable closes coarsen
    everywhere, whatever the family rows say."""
    state = instance_state(can_coarsen=[False] * 5)
    for position in range(5):
        assert opened(state, at(), position, rule) == "-HR"


def test_the_mask_is_a_new_bool_array_of_three():
    state = instance_state()
    first = action_mask(state, at(), 0, "first_decider")
    second = action_mask(state, at(), 0, "first_decider")
    assert first.dtype == np.bool_ and first.shape == (3,)
    assert first is not second
    first[:] = False
    assert action_mask(state, at(), 0, "first_decider").tolist() == [True, True, True]


@pytest.mark.parametrize("rule", COARSEN_RULES)
@pytest.mark.parametrize("text", [None, "...H.", "...R.", ".H.R.", "H.RH."])
def test_hold_is_always_open(text, rule):
    state = instance_state()
    for position in range(5):
        if text is None or text[position] == ".":
            assert action_mask(state, at(text), position, rule)[ACTION_HOLD]


@pytest.mark.parametrize("rule", COARSEN_RULES)
def test_the_budget_is_not_masked(rule):
    """D-025: the committed count does not reach the mask, however far past
    any budget it is. Nor does the round."""
    state = instance_state()
    for position in (0, 1, 2, B):
        base = opened(state, at("...H."), position, rule)
        for committed, round_index in ((1, 1), (5, 2), (10**6, 1)):
            moved = at("...H.", committed=committed, round_index=round_index)
            assert opened(state, moved, position, rule) == base
    assert opened(state, at(committed=10**6), 0, rule) == "CHR"
    assert opened(state, at(committed=10**6), B, rule) == "-HR"


def test_a_numpy_integer_position_is_accepted():
    state = instance_state()
    assert opened(state, at(), np.int32(0), "first_decider") == "CHR"
    assert is_decided(state, at("C...."), np.int64(0))


# --- First-decider family commit: the first table (M4) -----------------------------

@pytest.mark.parametrize(
    "choice, expected, decided, others",
    [
        ("coarsen", "CCCC.", [0, 1, 2, 3], None),
        ("hold", "...H.", [3], "-HR"),
        ("refine", "...R.", [3], "-HR"),
    ],
)
def test_first_decider_position_3_decides_for_its_family(choice, expected, decided, others):
    state = instance_state()
    round_state = walk(state, "first_decider", (3, choice))
    assert letters(round_state.decisions) == expected
    assert [p for p in range(5) if is_decided(state, round_state, p)] == decided
    for position in (0, 1, 2):
        if others is None:
            assert is_decided(state, round_state, position)
        else:
            assert opened(state, round_state, position, "first_decider") == others
    assert opened(state, round_state, B, "first_decider") == "-HR"


@pytest.mark.parametrize("decider", range(4))
def test_first_decider_any_sibling_can_be_the_first(decider):
    state = instance_state()
    assert letters(walk(state, "first_decider", (decider, "coarsen")).decisions) == "CCCC."
    held = walk(state, "first_decider", (decider, "hold"))
    for position in set(range(4)) - {decider}:
        assert opened(state, held, position, "first_decider") == "-HR"


def test_first_decider_the_veto_lasts_the_round():
    """After the first sibling holds, each later sibling is still asked, and
    coarsen stays closed to every one of them."""
    state = instance_state()
    round_state = walk(state, "first_decider", (3, "hold"))
    for position, choice in ((0, "refine"), (1, "hold")):
        assert opened(state, round_state, position, "first_decider") == "-HR"
        round_state = walk(state, "first_decider", (position, choice), start=round_state)
    assert opened(state, round_state, 2, "first_decider") == "-HR"
    assert letters(round_state.decisions) == "RH.H."


def test_first_decider_cell_b_decides_nothing_for_the_family():
    """Cell B has no family: its decision is its own entry, and cell A's
    children are as open afterward as before."""
    state = instance_state()
    for choice, expected in (("hold", "....H"), ("refine", "....R")):
        round_state = walk(state, "first_decider", (B, choice))
        assert letters(round_state.decisions) == expected
        for position in range(4):
            assert opened(state, round_state, position, "first_decider") == "CHR"


def test_first_decider_the_examples_round():
    """The order of section 2.3.2: 3, 0, 4, 1, 2. Position 3 coarsens, so
    0, 1 and 2 are skipped and two decisions make the round."""
    state = instance_state()
    round_state, asked = at(), []
    for position in (3, 0, B, 1, 2):
        if is_decided(state, round_state, position):
            continue
        asked.append(position)
        choice = "coarsen" if position == 3 else "refine"
        round_state = walk(state, "first_decider", (position, choice), start=round_state)
    assert asked == [3, B]
    assert letters(round_state.decisions) == "CCCCR"


# --- The three rules side by side: the second table (M5) ---------------------------

@pytest.mark.parametrize(
    "first, rule, expected",
    [
        ("hold", "first_decider", "-HR"),
        ("hold", "any_coarsen", "CHR"),
        ("hold", "unanimous", "CHR"),
        ("refine", "first_decider", "-HR"),
        ("refine", "any_coarsen", "-HR"),
        ("refine", "unanimous", "CHR"),
        ("coarsen", "first_decider", "skipped"),
        ("coarsen", "any_coarsen", "skipped"),
        ("coarsen", "unanimous", "CHR"),
    ],
)
def test_what_position_0_may_do_after_position_3(first, rule, expected):
    state = instance_state()
    round_state = walk(state, rule, (3, first))
    if expected == "skipped":
        assert is_decided(state, round_state, 0)
    else:
        assert not is_decided(state, round_state, 0)
        assert opened(state, round_state, 0, rule) == expected


@pytest.mark.parametrize(
    "rule, expected",
    [("any_coarsen", "CCCC."), ("unanimous", "C..H.")],
)
def test_a_coarsen_by_position_0_after_a_hold_by_position_3(rule, expected):
    """Under the 1D rule the hold does not bind and is overwritten; under
    unanimity the coarsen is one vote and the family is partly marked."""
    state = instance_state()
    assert letters(walk(state, rule, (3, "hold"), (0, "coarsen")).decisions) == expected


def test_any_coarsen_overwrites_every_earlier_hold():
    state = instance_state()
    round_state = walk(state, "any_coarsen", (3, "hold"), (0, "hold"), (1, "coarsen"))
    assert letters(round_state.decisions) == "CCCC."
    assert is_decided(state, round_state, 2)


def test_any_coarsen_a_refine_closes_coarsen_for_the_round():
    state = instance_state()
    round_state = walk(state, "any_coarsen", (3, "refine"), (0, "hold"))
    for position in (1, 2):
        assert opened(state, round_state, position, "any_coarsen") == "-HR"
    assert letters(round_state.decisions) == "H..R."


def test_the_hold_is_the_whole_difference_between_the_two_family_rules():
    """At the start, after a refine and after a coarsen the two rules agree
    on every position: the same record, the same skips, the same masks.
    After a hold they part."""
    state = instance_state()
    for steps in ((), ((3, "refine"),), ((3, "coarsen"),)):
        first = walk(state, "first_decider", *steps)
        other = walk(state, "any_coarsen", *steps)
        if steps:
            assert np.array_equal(first.decisions, other.decisions)
        for position in range(5):
            skipped = is_decided(state, first, position)
            assert skipped == is_decided(state, other, position)
            if not skipped:
                assert opened(state, first, position, "first_decider") == opened(state, other, position, "any_coarsen")
    held = {rule: walk(state, rule, (3, "hold")) for rule in FAMILY_RULES}
    assert np.array_equal(held["first_decider"].decisions, held["any_coarsen"].decisions)
    assert opened(state, held["first_decider"], 0, "first_decider") == "-HR"
    assert opened(state, held["any_coarsen"], 0, "any_coarsen") == "CHR"


# --- Unanimity (M6) ----------------------------------------------------------------

def test_unanimous_four_coarsens_one_at_a_time():
    state = instance_state()
    round_state, seen = at(), []
    for position in (3, 0, 1, 2):
        assert opened(state, round_state, position, "unanimous") == "CHR"
        round_state = walk(state, "unanimous", (position, "coarsen"), start=round_state)
        seen.append(letters(round_state.decisions))
    assert seen == ["...C.", "C..C.", "CC.C.", "CCCC."]


@pytest.mark.parametrize("choice", ["coarsen", "hold", "refine"])
def test_unanimous_a_decision_changes_one_entry_and_no_mask(choice):
    state = instance_state()
    before = at()
    round_state = walk(state, "unanimous", (3, choice))
    changed = np.flatnonzero(round_state.decisions != record("....."))
    assert changed.tolist() == [3]
    for position in (0, 1, 2, B):
        assert not is_decided(state, round_state, position)
        assert opened(state, round_state, position, "unanimous") == opened(state, before, position, "unanimous")


def test_unanimous_three_coarsens_and_a_hold_leave_a_partly_marked_family():
    state = instance_state()
    round_state = walk(state, "unanimous", (0, "coarsen"), (1, "coarsen"), (2, "coarsen"), (3, "hold"))
    assert letters(round_state.decisions) == "CCCH."


def test_unanimous_a_coarsen_stays_open_after_a_sibling_refines():
    """It cannot take effect, and the rule does not say so: one vote each."""
    state = instance_state()
    round_state = walk(state, "unanimous", (3, "refine"), (0, "coarsen"))
    assert letters(round_state.decisions) == "C..R."
    assert opened(state, round_state, 1, "unanimous") == "CHR"


# --- What the writer returns (M7) --------------------------------------------------

@pytest.mark.parametrize("rule", COARSEN_RULES)
@pytest.mark.parametrize("choice", ["coarsen", "hold", "refine"])
def test_the_writer_returns_a_new_read_only_int8_record(choice, rule):
    state = instance_state()
    start = at()
    result = decisions_after(state, start, 3, ACTION[choice], rule)
    assert result.dtype == np.int8 and result.shape == (5,)
    assert not result.flags.writeable
    assert start.decisions is None
    again = decisions_after(state, start, 3, ACTION[choice], rule)
    assert again is not result and np.array_equal(again, result)


@pytest.mark.parametrize("rule", COARSEN_RULES)
def test_the_writer_changes_nothing_it_was_handed(rule):
    state = instance_state()
    held = at("...H.")
    kept = held.decisions.copy()
    result = decisions_after(state, held, 0, ACTION_REFINE, rule)
    assert result is not held.decisions
    assert np.array_equal(held.decisions, kept)
    assert letters(result) == "R..H."
    assert (held.round_index, held.n_rounds, held.committed) == (1, 2, 5)


def test_the_writer_does_not_touch_the_committed_count():
    """It returns a record, not a round state; the count is item 2.4's."""
    state = instance_state()
    start = at(committed=5)
    result = decisions_after(state, start, 3, ACTION_REFINE, "first_decider")
    assert isinstance(result, np.ndarray)
    assert start.committed == 5


def test_a_returned_record_goes_back_into_a_round_state():
    state = instance_state()
    result = decisions_after(state, at(), 3, ACTION_COARSEN, "first_decider")
    assert RoundState(round_index=1, n_rounds=2, committed=5, decisions=result).decisions is result


def test_numpy_integers_are_accepted_as_position_and_action():
    state = instance_state()
    result = decisions_after(state, at(), np.int32(B), np.int64(ACTION_REFINE), "first_decider")
    assert letters(result) == "....R"
    assert letters(decisions_after(state, at(), B, ACTION_HOLD, "first_decider")) == "....H"


@pytest.mark.parametrize("rule", COARSEN_RULES)
def test_hold_and_refine_write_one_entry_under_every_rule(rule):
    state = instance_state()
    for position in range(5):
        for choice, letter in (("hold", "H"), ("refine", "R")):
            expected = "." * position + letter + "." * (4 - position)
            assert letters(decisions_after(state, at(), position, ACTION[choice], rule)) == expected


@pytest.mark.parametrize("rule", FAMILY_RULES)
@pytest.mark.parametrize("decider", range(4))
def test_a_family_coarsen_writes_every_sibling(decider, rule):
    state = instance_state()
    assert letters(decisions_after(state, at(), decider, ACTION_COARSEN, rule)) == "CCCC."


# --- What the two functions reject (M8) --------------------------------------------

@pytest.mark.parametrize("rule", ["majority", "", None, "FIRST_DECIDER", 0])
def test_an_unknown_rule(rule):
    state = instance_state()
    pattern = r"^rule: .*expected one of \['first_decider', 'any_coarsen', 'unanimous'\]"
    with pytest.raises(ValueError, match=pattern):
        action_mask(state, at(), 0, rule)
    with pytest.raises(ValueError, match=pattern):
        decisions_after(state, at(), 0, ACTION_HOLD, rule)


@pytest.mark.parametrize("position", [-1, 5, 1.0, True, None], ids=["negative", "past the end", "a float", "a bool",
                                                                    "None"])
def test_a_position_that_names_no_element(position):
    state = instance_state()
    pattern = r"^position: .*expected an integer in \[0, 5\)"
    with pytest.raises(ValueError, match=pattern):
        is_decided(state, at(), position)
    with pytest.raises(ValueError, match=pattern):
        action_mask(state, at(), position, "first_decider")
    with pytest.raises(ValueError, match=pattern):
        decisions_after(state, at(), position, ACTION_HOLD, "first_decider")


@pytest.mark.parametrize("action", [3, -1, 1.0, True, None, "hold"], ids=["3", "-1", "a float", "a bool", "None",
                                                                          "a name"])
def test_an_action_that_is_not_an_action_index(action):
    with pytest.raises(ValueError, match=r"^action: .*expected 0 \(coarsen\), 1 \(hold\) or 2 \(refine\)"):
        decisions_after(instance_state(), at(), 0, action, "first_decider")


@pytest.mark.parametrize("rule", COARSEN_RULES)
def test_a_masked_action_is_not_written(rule):
    state = instance_state(max_level=1)
    with pytest.raises(ValueError, match=rf"^action: 0 is masked for position 4 under '{rule}'"):
        decisions_after(state, at(), B, ACTION_COARSEN, rule)
    with pytest.raises(ValueError, match=rf"^action: 2 is masked for position 0 under '{rule}'"):
        decisions_after(state, at(), 0, ACTION_REFINE, rule)


@pytest.mark.parametrize(
    "rule, text", [("first_decider", "...H."), ("first_decider", "...R."), ("any_coarsen", "...R.")]
)
def test_a_coarsen_the_rule_has_closed_is_not_written(rule, text):
    with pytest.raises(ValueError, match=rf"^action: 0 is masked for position 0 under '{rule}'"):
        decisions_after(instance_state(), at(text), 0, ACTION_COARSEN, rule)


@pytest.mark.parametrize("rule", COARSEN_RULES)
@pytest.mark.parametrize("text, position", [("CCCC.", 0), ("...H.", 3), ("....R", B), ("C....", 0)])
def test_a_decided_position_is_not_asked_again(text, position, rule):
    state = instance_state()
    assert is_decided(state, at(text), position)
    with pytest.raises(ValueError, match=rf"^position: {position} is already decided this round"):
        action_mask(state, at(text), position, rule)
    with pytest.raises(ValueError, match=rf"^position: {position} is already decided this round"):
        decisions_after(state, at(text), position, ACTION_HOLD, rule)


@pytest.mark.parametrize("text", ["", "....", "......"], ids=["empty", "one short", "one long"])
def test_a_record_of_the_wrong_length(text):
    state = instance_state()
    pattern = rf"^decisions: length {len(text)}, expected n_active = 5"
    with pytest.raises(ValueError, match=pattern):
        is_decided(state, at(text), 0)
    with pytest.raises(ValueError, match=pattern):
        action_mask(state, at(text), 0, "first_decider")
    with pytest.raises(ValueError, match=pattern):
        decisions_after(state, at(text), 0, ACTION_HOLD, "first_decider")


@pytest.mark.parametrize("rule", COARSEN_RULES)
def test_can_coarsen_on_an_incomplete_family_is_rejected(rule):
    """Indexing the record by a row of -1 would read the last element. A
    state that says an element can coarsen and gives it no family is a
    backend's error, and it is named."""
    state = instance_state(can_coarsen=[True] * 5)
    pattern = r"^family: position 4 can coarsen, but its family row is \[-1, -1, -1, -1\]"
    with pytest.raises(ValueError, match=pattern):
        action_mask(state, at(), B, rule)
    with pytest.raises(ValueError, match=pattern):
        decisions_after(state, at(), B, ACTION_HOLD, rule)
    partly = instance_state(family=[[0, 1, 2, -1]] * 4 + [[-1, -1, -1, -1]])
    with pytest.raises(ValueError, match=r"^family: position 0 can coarsen, but its family row is \[0, 1, 2, -1\]"):
        action_mask(partly, at(), 0, rule)


def test_a_rejected_decision_changes_nothing():
    state = instance_state()
    held = at("...H.")
    kept = held.decisions.copy()
    with pytest.raises(ValueError):
        decisions_after(state, held, 0, ACTION_COARSEN, "first_decider")
    with pytest.raises(ValueError):
        decisions_after(state, held, 3, ACTION_HOLD, "first_decider")
    assert np.array_equal(held.decisions, kept)


# --- At ndim = 1: a family of two (M10) --------------------------------------------

def _state_1d() -> SolverState:
    """Four elements on a periodic line. Positions 1 and 2 are a sibling
    pair at the finest level; positions 0 and 3 have no sibling."""
    level = np.array([1, 2, 2, 1], dtype=np.int32)
    return SolverState(
        level=level,
        neighbors=np.array([[[3], [1]], [[0], [2]], [[1], [3]], [[2], [0]]], dtype=np.int32),
        neighbor_wrap=np.zeros((4, 2), dtype=bool),
        family=np.array([[-1, -1], [1, 2], [1, 2], [-1, -1]], dtype=np.int32),
        can_refine=level < 2,
        can_coarsen=np.array([False, True, True, False]),
        centroid=np.array([[0.125], [0.3125], [0.4375], [0.75]]),
        h=np.array([[0.25], [0.125], [0.125], [0.5]]),
        attributes=np.zeros((4, 0)),
        n_active=4,
        ndim=1,
        nvar=1,
        max_level=2,
        mesh_epoch=1,
        time=0.0,
        interval_T=1.0,
        dt_base=0.05,
        amr_freq=20,
        dt_stable=0.05,
        domain_lo=np.array([0.0]),
        domain_hi=np.array([1.0]),
        periodic=np.array([True]),
    )


def at_1d(text: str | None = None) -> RoundState:
    return at(text, committed=4, n_rounds=2)


@pytest.mark.parametrize("rule", COARSEN_RULES)
def test_1d_the_mask_at_the_start(rule):
    state = _state_1d()
    assert [opened(state, at_1d(), p, rule) for p in range(4)] == ["-HR", "CH-", "CH-", "-HR"]


@pytest.mark.parametrize("rule, expected", [("first_decider", ".CC."), ("any_coarsen", ".CC."), ("unanimous", ".C..")])
def test_1d_a_coarsen_by_one_of_the_pair(rule, expected):
    state = _state_1d()
    round_state = walk(state, rule, (1, "coarsen"), start=at_1d())
    assert letters(round_state.decisions) == expected
    assert is_decided(state, round_state, 2) is (rule != "unanimous")


@pytest.mark.parametrize("rule, expected", [("first_decider", "-H-"), ("any_coarsen", "CH-"), ("unanimous", "CH-")])
def test_1d_what_the_sibling_may_do_after_a_hold(rule, expected):
    """The validated 1D behavior is any_coarsen's: an earlier hold by the
    sibling does not bind. first_decider is where the hold vetoes."""
    state = _state_1d()
    round_state = walk(state, rule, (1, "hold"), start=at_1d())
    assert opened(state, round_state, 2, rule) == expected


@pytest.mark.parametrize("rule, expected", [("any_coarsen", ".CC."), ("unanimous", ".HC.")])
def test_1d_the_siblings_coarsen_after_a_hold(rule, expected):
    state = _state_1d()
    assert letters(walk(state, rule, (1, "hold"), (2, "coarsen"), start=at_1d()).decisions) == expected


# --- On the mock quad backend (M9) -------------------------------------------------

def _two_cell_mesh() -> tuple[MockQuadBackend, SolverState]:
    """The mock's own five-element mesh: cells A and B, A refined."""
    mock = MockQuadBackend(
        nx=2, ny=1, periodic=(True, False), lo=(0.0, 0.0), hi=(2.0, 1.0), max_level=2, dt_base=0.25,
        rotate_cells=False,
    )
    state = mock.reset({"x0": 0.5, "y0": 0.5, "width": 0.25}, amr_freq=4, initial_level=0)
    state, _ = mock.apply_marks(np.where(state.centroid[:, 0] < 1.0, REFINE, HOLD).astype(np.int8))
    return mock, state


def _graded_mesh(periodic: tuple[bool, bool]) -> tuple[MockQuadBackend, SolverState]:
    """A 4x4 grid at level 1 with the cells near the feature refined once
    more, the rotation on: levels 1 and 2, with 2:1 faces between them."""
    mock = MockQuadBackend(nx=4, ny=4, periodic=periodic, max_level=2, dt_base=0.25, rotate_cells=True)
    ic = {"x0": 0.3, "y0": 0.6, "width": 0.2}
    state = mock.reset(ic, amr_freq=4, initial_level=1)
    near = np.hypot(state.centroid[:, 0] - ic["x0"], state.centroid[:, 1] - ic["y0"]) < 0.15
    assert near.any()
    state, _ = mock.apply_marks(np.where(near, REFINE, HOLD).astype(np.int8))
    return mock, state


MESHES = {
    "two cells": _two_cell_mesh,
    "4x4 periodic": lambda: _graded_mesh((True, True)),
    "4x4 walled": lambda: _graded_mesh((False, False)),
    "4x4 mixed": lambda: _graded_mesh((True, False)),
}


def _families(state: SolverState) -> list[tuple[int, ...]]:
    """Each complete family once, as its positions in child order."""
    return sorted({tuple(row) for row in state.family.tolist() if row[0] >= 0})


def _run_round(state: SolverState, rule: str, order, choose) -> tuple[RoundState, list[int], list[int]]:
    """One round as a driver would run it: visit the positions in order,
    skip a decided one, ask the mask, take choose(mask)'s action, record it.

    Returns the final round state, the positions asked and those skipped.
    """
    round_state = RoundState(round_index=1, n_rounds=2, committed=state.n_active)
    asked, skipped = [], []
    for position in order:
        position = int(position)
        if is_decided(state, round_state, position):
            skipped.append(position)
            continue
        mask = action_mask(state, round_state, position, rule)
        asked.append(position)
        round_state = after(state, round_state, position, choose(mask), rule)
    return round_state, asked, skipped


def _prefer(action: int):
    """A policy: the given action wherever it is open, else hold."""
    return lambda mask: action if mask[action] else ACTION_HOLD


@pytest.mark.parametrize("mesh", list(MESHES))
def test_the_meshes_have_families_to_decide(mesh):
    """Every mesh has complete families and elements with none; the family
    rows of the mock's five elements are the instance's."""
    _, state = MESHES[mesh]()
    families = _families(state)
    assert families and (state.family[:, 0] < 0).any()
    assert all(state.can_coarsen[list(family)].all() for family in families)
    assert state.can_coarsen.sum() == 4 * len(families)
    if mesh == "two cells":
        hand = instance_state()
        assert np.array_equal(state.family, hand.family)
        assert np.array_equal(state.can_coarsen, hand.can_coarsen)
        assert np.array_equal(state.can_refine, hand.can_refine)


@pytest.mark.parametrize("rule", COARSEN_RULES)
@pytest.mark.parametrize("mesh", list(MESHES))
def test_a_round_of_coarsen_wherever_open(mesh, rule):
    """Every complete family ends all COARSEN under every rule. Under the
    two family rules one decision does it and three siblings are skipped;
    under unanimity all four are asked."""
    mock, state = MESHES[mesh]()
    families = _families(state)
    round_state, asked, skipped = _run_round(state, rule, range(state.n_active), _prefer(ACTION_COARSEN))
    expected = np.where(state.can_coarsen, COARSEN, HOLD).astype(np.int8)
    assert np.array_equal(round_state.decisions, expected)
    assert sorted(asked + skipped) == list(range(state.n_active))
    if rule == "unanimous":
        assert skipped == []
    else:
        assert len(skipped) == 3 * len(families)
        assert all(state.can_coarsen[p] for p in skipped)
        # The decider of each family is its first member in the order.
        assert sorted(set(asked) & set(np.flatnonzero(state.can_coarsen).tolist())) == [min(f) for f in families]


@pytest.mark.parametrize("rule", COARSEN_RULES)
@pytest.mark.parametrize("choice", ["hold", "refine"])
@pytest.mark.parametrize("mesh", list(MESHES))
def test_a_round_with_no_coarsen_skips_nothing(mesh, choice, rule):
    _, state = MESHES[mesh]()
    round_state, asked, skipped = _run_round(state, rule, range(state.n_active), _prefer(ACTION[choice]))
    if choice == "hold":
        expected = np.full(state.n_active, HOLD, dtype=np.int8)
    else:
        expected = np.where(state.can_refine, REFINE, HOLD).astype(np.int8)
    assert np.array_equal(round_state.decisions, expected)
    assert asked == list(range(state.n_active)) and skipped == []


@pytest.mark.parametrize("mesh", list(MESHES))
def test_first_decider_a_hold_first_closes_every_family(mesh):
    """Each family's first member holds, and every other element takes
    coarsen if it is open. Under first_decider none is; under any_coarsen
    the second member coarsens the family."""
    _, state = MESHES[mesh]()
    families = _families(state)
    first = {min(family) for family in families}

    def run(rule):
        round_state = RoundState(round_index=1, n_rounds=2, committed=state.n_active)
        for position in range(state.n_active):
            if is_decided(state, round_state, position):
                continue
            mask = action_mask(state, round_state, position, rule)
            action = ACTION_HOLD if position in first or not mask[ACTION_COARSEN] else ACTION_COARSEN
            round_state = after(state, round_state, position, action, rule)
        return round_state.decisions

    assert (run("first_decider") == HOLD).all()
    assert np.array_equal(run("any_coarsen"), np.where(state.can_coarsen, COARSEN, HOLD))
    partly = run("unanimous")
    for family in families:
        assert partly[list(family)].tolist().count(COARSEN) == 3
        assert partly[min(family)] == HOLD


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("rule", COARSEN_RULES)
@pytest.mark.parametrize("mesh", list(MESHES))
def test_a_round_of_random_masked_actions(mesh, rule, seed):
    """Actions drawn uniformly from the mask, in a random order. Whatever
    is drawn: every position ends decided, each is asked or skipped exactly
    once, no mark is one the structural mask closes, and under the two
    family rules every complete family is all COARSEN or holds none."""
    mock, state = MESHES[mesh]()
    rng = np.random.default_rng(seed)
    order = rng.permutation(state.n_active)
    round_state, asked, skipped = _run_round(
        state, rule, order, lambda mask: int(rng.choice(np.flatnonzero(mask)))
    )
    decisions = round_state.decisions

    assert (decisions != UNDECIDED).all()
    assert sorted(asked + skipped) == list(range(state.n_active))
    assert state.can_coarsen[decisions == COARSEN].all()
    assert state.can_refine[decisions == REFINE].all()
    assert all(decisions[p] == COARSEN for p in skipped)

    coarsened = 0
    for family in _families(state):
        marks = decisions[list(family)].tolist()
        if rule in FAMILY_RULES:
            assert marks.count(COARSEN) in (0, 4), marks
        coarsened += marks.count(COARSEN) == 4
    if rule == "first_decider":
        assert len(skipped) == 3 * coarsened
    elif rule == "any_coarsen":
        assert len(skipped) <= 3 * coarsened
    else:
        assert skipped == []

    new_state, report = mock.apply_marks(decisions)
    assert report.realized.shape == (state.n_active,)
    assert new_state.mesh_epoch == state.mesh_epoch + 1


@pytest.mark.parametrize("rule", COARSEN_RULES)
@pytest.mark.parametrize("mesh", list(MESHES))
def test_apply_marks_takes_a_finished_record_as_its_marks(mesh, rule):
    """The record of a whole round goes to the backend as it stands: int8,
    read-only, every entry a mark. The backend does not write into it."""
    mock, state = MESHES[mesh]()
    round_state, _, _ = _run_round(state, rule, range(state.n_active), _prefer(ACTION_REFINE))
    marks = round_state.decisions
    assert marks.dtype == np.int8 and not marks.flags.writeable
    predicted, _ = mock.dry_run(marks)
    new_state, report = mock.apply_marks(marks)
    assert new_state.n_active == predicted
    assert not marks.flags.writeable


@pytest.mark.parametrize("rule", COARSEN_RULES)
def test_an_unfinished_record_is_not_a_set_of_marks(rule):
    """Why the sentinel is not a mark: a record with a position still
    undecided is refused by the backend, by name, and nothing changes."""
    mock, state = _two_cell_mesh()
    round_state = walk(state, rule, (3, "hold"), start=RoundState(round_index=1, n_rounds=2, committed=5))
    assert (round_state.decisions == UNDECIDED).any()
    with pytest.raises(ValueError, match="^marks: values must be COARSEN, HOLD or REFINE"):
        mock.apply_marks(round_state.decisions)
    assert mock.get_state().mesh_epoch == state.mesh_epoch


def test_on_the_mock_a_family_commit_merges_and_a_partial_vote_does_not():
    """What the rule is for (spec section 2 item 7). One coarsen under
    first_decider marks the family and the backend merges it: five elements
    become two. Three coarsen votes and a hold under unanimity are a partly
    marked family, which the backend drops: five stay five."""
    mock, state = _two_cell_mesh()
    start = RoundState(round_index=1, n_rounds=2, committed=5)
    committed = walk(state, "first_decider", (3, "coarsen"), (B, "hold"), start=start)
    assert letters(committed.decisions) == "CCCCH"
    merged, report = mock.apply_marks(committed.decisions)
    assert merged.n_active == 2
    assert report.realized.tolist() == [COARSEN, COARSEN, COARSEN, COARSEN, HOLD]

    mock, state = _two_cell_mesh()
    voted = walk(
        state, "unanimous", (0, "coarsen"), (1, "coarsen"), (2, "coarsen"), (3, "hold"), (B, "hold"), start=start
    )
    assert letters(voted.decisions) == "CCCHH"
    unchanged, report = mock.apply_marks(voted.decisions)
    assert unchanged.n_active == 5
    assert report.realized.tolist() == [HOLD] * 5
