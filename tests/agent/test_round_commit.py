"""Tests for agent/round_commit.py: the commit of the 2D agent core, which
records one decision and moves the committed element count, and the
RoundState a round begins with.

AGENT_2D_ROADMAP item 2.4, signals K2-K12. Eight groups:

  - start_round: the count, the record and what it rejects (K2);
  - the commit's count on the five-element instance of the explanation
    file's section 2.4.2: the example round (K3), every action under every
    rule (K4), unanimity (K5) and the hold that binds or does not (K6);
  - on a literal 1D state: the two batch cases the sequential core's
    counter gets wrong, a realized pair and a lone mark (K7);
  - what the commit returns and what it leaves alone (K8);
  - what it rejects: everything decisions_after rejects, word for word (K9);
  - the vocabulary: no function here takes a training-only type (K12);
  - on the mock quad backend, scripted: where the count and the realized
    mesh agree and where balance parts them (K11);
  - on the mock quad backend, whole rounds: the count against a recount
    from the record after every decision, and the realized count against
    the counter at the round's end (K10).

Records are written as text, one letter per position: C, H and R for the
marks and a dot for an undecided position.

Run: pytest tests/agent/test_round_commit.py -v
"""
from __future__ import annotations

import itertools

import numpy as np
import pytest

from agent.round_commit import commit, start_round
from agent.round_mask import (
    ACTION_COARSEN,
    ACTION_HOLD,
    ACTION_REFINE,
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
# Cell B's position on the instance; positions 0-3 are cell A's children.
B = 4


def record(text: str) -> np.ndarray:
    """A decision record from its letters."""
    return np.array([LETTER[c] for c in text], dtype=np.int8)


def letters(decisions: np.ndarray) -> str:
    """A decision record as letters."""
    by_value = {value: letter for letter, value in LETTER.items()}
    return "".join(by_value[v] for v in decisions.tolist())


def walk(
    state: SolverState, rule: str, *steps: tuple[int, str], start: RoundState | None = None
) -> tuple[RoundState, list[int]]:
    """Commit decisions in order, each (position, "coarsen" | "hold" |
    "refine"), from start_round or from a given round state.

    Returns the last round state and the count after each decision.
    """
    round_state = start_round(state, 1, 2) if start is None else start
    counts = []
    for position, name in steps:
        round_state = commit(state, round_state, position, ACTION[name], rule)
        counts.append(round_state.committed)
    return round_state, counts


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


# --- start_round (K2) --------------------------------------------------------------

def test_start_round_on_the_instance():
    state = instance_state()
    first = start_round(state, 1, 2)
    assert isinstance(first, RoundState)
    assert (first.round_index, first.n_rounds, first.committed) == (1, 2, 5)
    assert letters(first.decisions) == "....."
    assert first.decisions.dtype == np.int8 and first.decisions.shape == (5,)
    assert not first.decisions.flags.writeable


@pytest.mark.parametrize("rule", COARSEN_RULES)
def test_start_round_masks_as_a_round_with_no_record(rule):
    """The real all-undecided record and None are the same round's start."""
    state = instance_state()
    first = start_round(state, 1, 2)
    bare = RoundState(round_index=1, n_rounds=2, committed=5)
    for position in range(5):
        assert not is_decided(state, first, position)
        assert np.array_equal(action_mask(state, first, position, rule), action_mask(state, bare, position, rule))


def test_start_round_makes_a_new_record_every_time():
    state = instance_state()
    first, second = start_round(state, 1, 2), start_round(state, 1, 2)
    assert first.decisions is not second.decisions
    assert np.array_equal(first.decisions, second.decisions)


def test_start_round_takes_the_count_and_the_length_from_the_state():
    line = start_round(_state_1d(), 2, 2)
    assert line.committed == 4 and letters(line.decisions) == "...."
    assert line.round_index == 2


@pytest.mark.parametrize("round_index, n_rounds", [(1, 1), (1, 3), (3, 3), (np.int64(2), np.int32(4))])
def test_start_round_keeps_the_round_it_is_given(round_index, n_rounds):
    first = start_round(instance_state(), round_index, n_rounds)
    assert (first.round_index, first.n_rounds) == (round_index, n_rounds)


@pytest.mark.parametrize(
    "round_index, n_rounds, pattern",
    [
        (0, 2, r"^round_index: 0, expected 1 to n_rounds = 2"),
        (3, 2, r"^round_index: 3, expected 1 to n_rounds = 2"),
        (1, 0, r"^n_rounds: 0, expected at least 1"),
        (1.0, 2, r"^round_index: 1.0, expected an integer"),
        (True, 2, r"^round_index: True, expected an integer"),
        (1, None, r"^n_rounds: None, expected an integer"),
    ],
    ids=["round zero", "past the last round", "no rounds", "a float", "a bool", "None"],
)
def test_start_round_rejections_are_the_nouns(round_index, n_rounds, pattern):
    with pytest.raises(ValueError, match=pattern):
        start_round(instance_state(), round_index, n_rounds)


# --- The example round (K3) --------------------------------------------------------

def test_first_decider_the_examples_round_counted():
    """Section 2.4.2's first table. The order is 3, 0, 4, 1, 2: position 3
    coarsens for its family (minus 3, once), positions 0, 1 and 2 are
    skipped, and cell B refines (plus 3)."""
    state = instance_state()
    round_state = start_round(state, 1, 2)
    seen = [("start", letters(round_state.decisions), round_state.committed)]
    for position in (3, 0, B, 1, 2):
        if is_decided(state, round_state, position):
            seen.append((position, "skipped", round_state.committed))
            continue
        action = ACTION_COARSEN if position == 3 else ACTION_REFINE
        round_state = commit(state, round_state, position, action, "first_decider")
        seen.append((position, letters(round_state.decisions), round_state.committed))
    assert seen == [
        ("start", ".....", 5),
        (3, "CCCC.", 2),
        (0, "skipped", 2),
        (B, "CCCCR", 5),
        (1, "skipped", 5),
        (2, "skipped", 5),
    ]


# --- Every action under every rule (K4) --------------------------------------------

@pytest.mark.parametrize(
    "rule, position, choice, expected",
    [
        ("first_decider", 3, "coarsen", 2),
        ("first_decider", 3, "hold", 5),
        ("first_decider", 3, "refine", 8),
        ("first_decider", B, "hold", 5),
        ("first_decider", B, "refine", 8),
        ("any_coarsen", 3, "coarsen", 2),
        ("any_coarsen", 3, "hold", 5),
        ("any_coarsen", 3, "refine", 8),
        ("any_coarsen", B, "hold", 5),
        ("any_coarsen", B, "refine", 8),
        ("unanimous", 3, "coarsen", 5),
        ("unanimous", 3, "hold", 5),
        ("unanimous", 3, "refine", 8),
        ("unanimous", B, "hold", 5),
        ("unanimous", B, "refine", 8),
    ],
)
def test_one_decision_from_the_start(rule, position, choice, expected):
    """Refine is plus 3 and hold is nothing under every rule. A coarsen is
    minus 3 where it marks the family and nothing where it is one vote."""
    _, counts = walk(instance_state(), rule, (position, choice))
    assert counts == [expected]


@pytest.mark.parametrize("rule", FAMILY_RULES)
@pytest.mark.parametrize("decider", range(4))
def test_a_family_coarsen_costs_three_once_whoever_decides(decider, rule):
    round_state, counts = walk(instance_state(), rule, (decider, "coarsen"))
    assert counts == [2]
    assert letters(round_state.decisions) == "CCCC."


@pytest.mark.parametrize("rule", COARSEN_RULES)
def test_every_refine_adds_three(rule):
    state = instance_state()
    round_state, counts = walk(state, rule, *((p, "refine") for p in range(5)))
    assert counts == [8, 11, 14, 17, 20]
    assert letters(round_state.decisions) == "RRRRR"


@pytest.mark.parametrize("rule", COARSEN_RULES)
def test_a_round_of_holds_leaves_the_count(rule):
    round_state, counts = walk(instance_state(), rule, *((p, "hold") for p in range(5)))
    assert counts == [5] * 5
    assert letters(round_state.decisions) == "HHHHH"


@pytest.mark.parametrize("rule", FAMILY_RULES)
def test_a_family_coarsen_and_a_refine_cancel(rule):
    """The count is the sum of the decisions, in either order."""
    state = instance_state()
    assert walk(state, rule, (0, "coarsen"), (B, "refine"))[1] == [2, 5]
    assert walk(state, rule, (B, "refine"), (0, "coarsen"))[1] == [8, 5]


def test_the_count_is_never_moved_per_mark():
    """What the sequential core's counter does in batch mode: one step per
    coarsen mark. Four marks would be minus 12 at 2D. Here a family is one
    decision under the family rules and one completed row under unanimity:
    minus 3 either way."""
    state = instance_state()
    for rule in FAMILY_RULES:
        round_state, _ = walk(state, rule, (3, "coarsen"))
        assert (round_state.decisions == COARSEN).sum() == 4 and round_state.committed == 5 - 3
    round_state, _ = walk(state, "unanimous", *((p, "coarsen") for p in range(4)))
    assert (round_state.decisions == COARSEN).sum() == 4 and round_state.committed == 5 - 3


# --- Unanimity (K5) ----------------------------------------------------------------

def test_unanimous_the_count_moves_at_the_fourth_vote():
    round_state, counts = walk(instance_state(), "unanimous", (3, "coarsen"), (0, "coarsen"), (1, "coarsen"),
                               (2, "coarsen"))
    assert counts == [5, 5, 5, 2]
    assert letters(round_state.decisions) == "CCCC."


@pytest.mark.parametrize("order", list(itertools.permutations(range(4))), ids=lambda order: "".join(map(str, order)))
def test_unanimous_the_last_vote_is_the_one_that_counts_in_any_order(order):
    _, counts = walk(instance_state(), "unanimous", *((p, "coarsen") for p in order))
    assert counts == [5, 5, 5, 2]


@pytest.mark.parametrize("holder", range(4))
def test_unanimous_three_votes_and_a_hold_cost_nothing(holder):
    """Wherever the hold falls, first or last: the family is partly marked,
    the backend will drop it, and the count says so."""
    state = instance_state()
    voters = [p for p in range(4) if p != holder]
    last, _ = walk(state, "unanimous", *((p, "coarsen") for p in voters), (holder, "hold"))
    first, _ = walk(state, "unanimous", (holder, "hold"), *((p, "coarsen") for p in voters))
    assert last.committed == first.committed == 5
    assert np.array_equal(last.decisions, first.decisions)
    assert letters(last.decisions).count("C") == 3


def test_unanimous_votes_beside_a_refine_cost_nothing():
    """The sibling's refine counts; the three votes cannot take effect."""
    round_state, counts = walk(instance_state(), "unanimous", (3, "refine"), (0, "coarsen"), (1, "coarsen"),
                               (2, "coarsen"))
    assert counts == [8, 8, 8, 8]
    assert letters(round_state.decisions) == "CCCR."


def test_unanimous_a_refine_after_three_votes():
    _, counts = walk(instance_state(), "unanimous", (0, "coarsen"), (1, "coarsen"), (2, "coarsen"), (3, "refine"))
    assert counts == [5, 5, 5, 8]


# --- The hold that binds, and the hold that does not (K6) --------------------------

def test_any_coarsen_an_overwritten_hold_reverses_nothing():
    """The hold cost nothing when it was made, so the sibling's coarsen is
    the family's one step."""
    round_state, counts = walk(instance_state(), "any_coarsen", (3, "hold"), (0, "coarsen"))
    assert counts == [5, 2]
    assert letters(round_state.decisions) == "CCCC."


def test_any_coarsen_three_holds_and_a_coarsen():
    round_state, counts = walk(instance_state(), "any_coarsen", (3, "hold"), (0, "hold"), (1, "hold"), (2, "coarsen"))
    assert counts == [5, 5, 5, 2]
    assert letters(round_state.decisions) == "CCCC."


def test_first_decider_after_a_hold_the_family_can_only_grow():
    state = instance_state()
    round_state, counts = walk(state, "first_decider", (3, "hold"), (0, "refine"), (1, "hold"), (2, "refine"))
    assert counts == [5, 8, 8, 11]
    assert letters(round_state.decisions) == "RHRH."


# --- At ndim = 1: the cases the sequential counter gets wrong (K7) -----------------

@pytest.mark.parametrize("rule", FAMILY_RULES)
@pytest.mark.parametrize("decider", [1, 2])
def test_1d_a_pair_that_merges_is_minus_one(decider, rule):
    """The roadmap's section 1: a realized pair is one element fewer. The
    sequential core's batch counter says two."""
    state = _state_1d()
    round_state, counts = walk(state, rule, (decider, "coarsen"), start=start_round(state, 1, 2))
    assert counts == [3]
    assert letters(round_state.decisions) == ".CC."


def test_1d_unanimous_the_pair_is_minus_one_at_the_second_vote():
    state = _state_1d()
    _, counts = walk(state, "unanimous", (1, "coarsen"), (2, "coarsen"), start=start_round(state, 1, 2))
    assert counts == [4, 3]


@pytest.mark.parametrize("voter, other, expected", [(1, 2, ".CH."), (2, 1, ".HC.")])
def test_1d_a_lone_mark_is_nothing(voter, other, expected):
    """The roadmap's section 1: a lone coarsen mark changes no element. The
    sequential core's batch counter says one fewer."""
    state = _state_1d()
    round_state, counts = walk(state, "unanimous", (voter, "coarsen"), (other, "hold"), start=start_round(state, 1, 2))
    assert counts == [4, 4]
    assert letters(round_state.decisions) == expected


@pytest.mark.parametrize("rule", COARSEN_RULES)
def test_1d_a_refine_is_plus_one(rule):
    state = _state_1d()
    _, counts = walk(state, rule, (0, "refine"), (3, "refine"), start=start_round(state, 1, 2))
    assert counts == [5, 6]


# --- What the commit returns (K8) --------------------------------------------------

@pytest.mark.parametrize("rule", COARSEN_RULES)
@pytest.mark.parametrize("choice", ["coarsen", "hold", "refine"])
def test_the_commit_returns_a_new_round_state_and_changes_nothing(choice, rule):
    state = instance_state()
    before = start_round(state, 1, 2)
    kept = before.decisions.copy()
    result = commit(state, before, 3, ACTION[choice], rule)
    assert isinstance(result, RoundState) and result is not before
    assert result.decisions is not before.decisions
    assert before.committed == 5 and np.array_equal(before.decisions, kept)
    assert not result.decisions.flags.writeable and result.decisions.dtype == np.int8


@pytest.mark.parametrize("rule", COARSEN_RULES)
@pytest.mark.parametrize("choice", ["coarsen", "hold", "refine"])
@pytest.mark.parametrize("text", [".....", "..H..", ".R..."])
def test_the_record_is_decisions_afters(text, choice, rule):
    """The commit adds a count to item 2.3's writer and changes no entry.
    Where the rule has closed the action, the commit refuses it too."""
    state = instance_state()
    before = RoundState(round_index=1, n_rounds=2, committed=5, decisions=record(text))
    if not action_mask(state, before, 3, rule)[ACTION[choice]]:
        with pytest.raises(ValueError, match="^action: 0 is masked for position 3"):
            commit(state, before, 3, ACTION[choice], rule)
        return
    expected = decisions_after(state, before, 3, ACTION[choice], rule)
    assert np.array_equal(commit(state, before, 3, ACTION[choice], rule).decisions, expected)


@pytest.mark.parametrize("rule", COARSEN_RULES)
def test_a_round_state_with_no_record_comes_back_with_one(rule):
    state = instance_state()
    bare = RoundState(round_index=1, n_rounds=2, committed=5)
    result = commit(state, bare, B, ACTION_HOLD, rule)
    assert bare.decisions is None
    assert letters(result.decisions) == "....H" and result.committed == 5
    assert letters(commit(state, bare, 3, ACTION_REFINE, rule).decisions) == "...R."


@pytest.mark.parametrize("round_index, n_rounds", [(1, 1), (2, 3), (3, 3)])
def test_the_round_is_carried_through(round_index, n_rounds):
    state = instance_state()
    result = commit(state, start_round(state, round_index, n_rounds), 0, ACTION_REFINE, "first_decider")
    assert (result.round_index, result.n_rounds) == (round_index, n_rounds)


def test_the_count_comes_back_a_python_int():
    state = instance_state()
    numpy_count = RoundState(round_index=1, n_rounds=2, committed=np.int64(5))
    for choice, expected in (("coarsen", 2), ("hold", 5), ("refine", 8)):
        result = commit(state, numpy_count, 3, ACTION[choice], "first_decider")
        assert type(result.committed) is int and result.committed == expected


def test_numpy_integers_are_accepted_as_position_and_action():
    state = instance_state()
    result = commit(state, start_round(state, 1, 2), np.int32(B), np.int64(ACTION_REFINE), "first_decider")
    assert letters(result.decisions) == "....R" and result.committed == 8


def test_the_count_handed_in_is_not_checked():
    """The commit moves the count it is given. A round state built by hand
    with a wrong count stays wrong by the same amount; start_round is what
    makes it right."""
    state = instance_state()
    wrong = RoundState(round_index=1, n_rounds=2, committed=100)
    assert commit(state, wrong, 0, ACTION_REFINE, "first_decider").committed == 103
    assert commit(state, wrong, 0, ACTION_COARSEN, "first_decider").committed == 97


def test_a_count_cannot_fall_below_one():
    """The noun's check runs again on the new round state."""
    state = instance_state()
    with pytest.raises(ValueError, match=r"^committed: -2, expected at least 1"):
        commit(state, RoundState(round_index=1, n_rounds=2, committed=1), 0, ACTION_COARSEN, "first_decider")


# --- What the commit rejects (K9) --------------------------------------------------

@pytest.mark.parametrize(
    "text, position, action, rule, pattern",
    [
        (".....", 0, ACTION_HOLD, "majority", r"^rule: 'majority', expected one of"),
        (".....", 5, ACTION_HOLD, "first_decider", r"^position: 5, expected an integer in \[0, 5\)"),
        (".....", -1, ACTION_HOLD, "first_decider", r"^position: -1, expected an integer in \[0, 5\)"),
        (".....", True, ACTION_HOLD, "first_decider", r"^position: True, expected an integer"),
        (".....", 0, 3, "first_decider", r"^action: 3, expected 0 \(coarsen\), 1 \(hold\) or 2 \(refine\)"),
        (".....", 0, "hold", "first_decider", r"^action: 'hold', expected 0"),
        (".....", B, ACTION_COARSEN, "first_decider", r"^action: 0 is masked for position 4 under 'first_decider'"),
        (".....", B, ACTION_COARSEN, "unanimous", r"^action: 0 is masked for position 4 under 'unanimous'"),
        ("...H.", 0, ACTION_COARSEN, "first_decider", r"^action: 0 is masked for position 0 under 'first_decider'"),
        ("...R.", 0, ACTION_COARSEN, "any_coarsen", r"^action: 0 is masked for position 0 under 'any_coarsen'"),
        ("CCCC.", 0, ACTION_HOLD, "first_decider", r"^position: 0 is already decided this round"),
        ("...H.", 3, ACTION_REFINE, "unanimous", r"^position: 3 is already decided this round"),
        ("....", 0, ACTION_HOLD, "first_decider", r"^decisions: length 4, expected n_active = 5"),
        ("......", 0, ACTION_HOLD, "first_decider", r"^decisions: length 6, expected n_active = 5"),
    ],
    ids=[
        "an unknown rule",
        "a position past the end",
        "a negative position",
        "a bool position",
        "an action of 3",
        "an action's name",
        "a coarsen with no family",
        "a vote with no family",
        "a coarsen after the first decider held",
        "a coarsen after a sibling refined",
        "a skipped sibling",
        "a position asked twice",
        "a record one short",
        "a record one long",
    ],
)
def test_the_commit_rejects_what_the_writer_rejects_in_its_words(text, position, action, rule, pattern):
    state = instance_state()
    before = RoundState(round_index=1, n_rounds=2, committed=5, decisions=record(text))
    kept = before.decisions.copy()
    with pytest.raises(ValueError, match=pattern) as from_commit:
        commit(state, before, position, action, rule)
    with pytest.raises(ValueError) as from_writer:
        decisions_after(state, before, position, action, rule)
    assert str(from_commit.value) == str(from_writer.value)
    assert before.committed == 5 and np.array_equal(before.decisions, kept)


@pytest.mark.parametrize("rule", COARSEN_RULES)
def test_a_refine_at_the_finest_level_is_refused_not_counted(rule):
    """The mask closes refine at max_level, so the commit has no case for
    a refine that would not happen."""
    state = instance_state(max_level=1)
    with pytest.raises(ValueError, match=rf"^action: 2 is masked for position 0 under '{rule}'"):
        commit(state, start_round(state, 1, 1), 0, ACTION_REFINE, rule)


@pytest.mark.parametrize("rule", COARSEN_RULES)
def test_can_coarsen_on_an_incomplete_family_is_refused(rule):
    """The commit reads the family row only after the mask has passed it."""
    state = instance_state(can_coarsen=[True] * 5)
    with pytest.raises(ValueError, match=r"^family: position 4 can coarsen, but its family row is \[-1, -1, -1, -1\]"):
        commit(state, start_round(state, 1, 2), B, ACTION_COARSEN, rule)


# --- The vocabulary (K12) ----------------------------------------------------------

@pytest.mark.parametrize("fn", [start_round, commit], ids=lambda fn: fn.__name__)
def test_no_function_takes_a_training_only_type(fn):
    """Both belong to the within-round logic that is implemented twice."""
    assert training_only_params(fn) == []


# --- On the mock quad backend ------------------------------------------------------

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


def _whole_families(state: SolverState, marks: np.ndarray) -> int:
    """How many complete families carry COARSEN on every member."""
    return sum(bool((marks[list(family)] == COARSEN).all()) for family in _families(state))


def _recount(state: SolverState, decisions: np.ndarray) -> int:
    """The committed count from scratch: the slow reference. n_active, plus
    3 for every REFINE entry, minus 3 for every family whose entries are all
    COARSEN. It reads the whole record and knows no rule."""
    arity = 2**state.ndim - 1
    refines = int((decisions == REFINE).sum())
    return state.n_active + arity * refines - arity * _whole_families(state, decisions)


def _run_round(state: SolverState, rule: str, order, choose, round_index: int = 1) -> tuple[RoundState, list[int]]:
    """One round as a driver would run it: start_round, then for each
    position in order skip a decided one, ask the mask, commit choose(mask).
    After every commit the count is checked against the recount.

    Returns the final round state and the positions skipped.
    """
    round_state = start_round(state, round_index, 2)
    skipped = []
    for position in order:
        position = int(position)
        if is_decided(state, round_state, position):
            skipped.append(position)
            continue
        mask = action_mask(state, round_state, position, rule)
        round_state = commit(state, round_state, position, choose(mask), rule)
        assert round_state.committed == _recount(state, round_state.decisions)
    return round_state, skipped


def _overshoot(state: SolverState, round_state: RoundState, realized: np.ndarray) -> tuple[int, int]:
    """What balance did to a round's request, read from the realized flags
    against the record: refines it added, and family coarsens it undid."""
    added = int((realized == REFINE).sum()) - int((round_state.decisions == REFINE).sum())
    merged, remainder = divmod(int((realized == COARSEN).sum()), 2**state.ndim)
    assert remainder == 0
    undone = _whole_families(state, round_state.decisions) - merged
    return added, undone


def _prefer(action: int):
    """A policy: the given action wherever it is open, else hold."""
    return lambda mask: action if mask[action] else ACTION_HOLD


# --- Scripted: where the count and the mesh agree, and where they part (K11) -------

def test_on_the_mock_the_examples_round_is_exact():
    """A family coarsen and a refine: the count says 5 and the mesh is 5."""
    mock, state = _two_cell_mesh()
    round_state, counts = walk(state, "first_decider", (3, "coarsen"), (B, "refine"))
    assert counts == [2, 5]
    predicted, _ = mock.dry_run(round_state.decisions)
    new_state, report = mock.apply_marks(round_state.decisions)
    assert predicted == new_state.n_active == round_state.committed == 5
    assert letters(report.realized) == "CCCCR"


def test_on_the_mock_a_family_commit_is_exact():
    mock, state = _two_cell_mesh()
    round_state, _ = walk(state, "first_decider", (3, "coarsen"), (B, "hold"))
    new_state, _ = mock.apply_marks(round_state.decisions)
    assert new_state.n_active == round_state.committed == 2


def test_on_the_mock_three_votes_and_a_hold_are_exact():
    """Unanimity's partly marked family: the count did not move and neither
    does the mesh. A counter that stepped per mark would read 5 - 9."""
    mock, state = _two_cell_mesh()
    round_state, _ = walk(state, "unanimous", (0, "coarsen"), (1, "coarsen"), (2, "coarsen"), (3, "hold"), (B, "hold"))
    assert letters(round_state.decisions) == "CCCHH"
    new_state, report = mock.apply_marks(round_state.decisions)
    assert new_state.n_active == round_state.committed == 5
    assert letters(report.realized) == "HHHHH"


@pytest.mark.parametrize("refiner", range(4))
def test_on_the_mock_balance_adds_a_refine_the_count_does_not_see(refiner):
    """Section 2.4.2's third table, round 1. Any child of cell A refined to
    level 2 touches cell B, at level 0, across a face or across the wrap, so
    balance splits B. The count is 8; the mesh is 11."""
    mock, state = _two_cell_mesh()
    steps = [(refiner, "refine")] + [(p, "hold") for p in range(5) if p != refiner]
    round_state, _ = walk(state, "first_decider", *steps)
    assert round_state.committed == 8
    predicted, _ = mock.dry_run(round_state.decisions)
    new_state, report = mock.apply_marks(round_state.decisions)
    assert predicted == new_state.n_active == 11
    assert report.realized[refiner] == REFINE and report.realized[B] == REFINE
    assert (report.realized == HOLD).sum() == 3
    assert _overshoot(state, round_state, report.realized) == (1, 0)
    assert new_state.n_active - round_state.committed == 3


def test_on_the_mock_balance_undoes_a_coarsen_the_count_has_taken():
    """Section 2.4.2's third table, round 2. On the 11 elements cell B's
    four children coarsen. The count goes to 8. B merges, touches level 2
    across the wrap, and balance splits it again: 11 elements, and the four
    come back HOLD."""
    mock, state = _two_cell_mesh()
    round_state, _ = walk(state, "first_decider", (0, "refine"), (1, "hold"), (2, "hold"), (3, "hold"), (B, "hold"))
    state, report = mock.apply_marks(round_state.decisions)
    assert letters(report.realized) == "RHHHR" and state.n_active == 11

    in_b = state.centroid[:, 0] > 1.0
    assert in_b.sum() == 4 and state.can_coarsen[in_b].all()
    round_state = start_round(state, 2, 2)
    assert round_state.committed == 11
    for position in range(state.n_active):
        if is_decided(state, round_state, position):
            continue
        action = ACTION_COARSEN if in_b[position] else ACTION_HOLD
        round_state = commit(state, round_state, position, action, "first_decider")
    assert (round_state.decisions[in_b] == COARSEN).all() and (round_state.decisions[~in_b] == HOLD).all()
    assert round_state.committed == 8

    predicted, _ = mock.dry_run(round_state.decisions)
    new_state, report = mock.apply_marks(round_state.decisions)
    assert predicted == new_state.n_active == 11
    assert (report.realized == HOLD).all()
    assert _overshoot(state, round_state, report.realized) == (0, 1)


def test_the_next_round_starts_from_the_realized_mesh():
    """The overshoot does not carry: start_round reads n_active, so round 2
    begins at 11 where round 1's count ended at 8."""
    mock, state = _two_cell_mesh()
    round_state, _ = walk(state, "first_decider", (0, "refine"), (1, "hold"), (2, "hold"), (3, "hold"), (B, "hold"))
    new_state, _ = mock.apply_marks(round_state.decisions)
    second = start_round(new_state, 2, 2)
    assert round_state.committed == 8
    assert second.committed == new_state.n_active == 11
    assert second.decisions.shape == (11,) and (second.decisions == UNDECIDED).all()


# --- Whole rounds: the recount, and the realized count against the counter (K10) ---

@pytest.mark.parametrize("mesh", list(MESHES))
def test_the_recount_on_a_fresh_round(mesh):
    """The reference itself, on a record with nothing decided."""
    _, state = MESHES[mesh]()
    first = start_round(state, 1, 2)
    assert _recount(state, first.decisions) == state.n_active == first.committed


@pytest.mark.parametrize("rule", COARSEN_RULES)
@pytest.mark.parametrize("mesh", list(MESHES))
def test_a_round_of_holds_is_exact(mesh, rule):
    mock, state = MESHES[mesh]()
    round_state, skipped = _run_round(state, rule, range(state.n_active), _prefer(ACTION_HOLD))
    new_state, report = mock.apply_marks(round_state.decisions)
    assert skipped == []
    assert round_state.committed == state.n_active == new_state.n_active
    assert _overshoot(state, round_state, report.realized) == (0, 0)


@pytest.mark.parametrize("rule", COARSEN_RULES)
@pytest.mark.parametrize("mesh", list(MESHES))
def test_a_round_of_coarsen_wherever_open(mesh, rule):
    """Every complete family is taken, one decision or four votes each: the
    count is n_active less 3 per family under every rule."""
    mock, state = MESHES[mesh]()
    families = _families(state)
    round_state, skipped = _run_round(state, rule, range(state.n_active), _prefer(ACTION_COARSEN))
    assert round_state.committed == state.n_active - 3 * len(families)
    assert len(skipped) == (0 if rule == "unanimous" else 3 * len(families))
    new_state, report = mock.apply_marks(round_state.decisions)
    added, undone = _overshoot(state, round_state, report.realized)
    assert added >= 0 and undone >= 0
    assert new_state.n_active - round_state.committed == 3 * (added + undone)


@pytest.mark.parametrize("rule", COARSEN_RULES)
@pytest.mark.parametrize("mesh", list(MESHES))
def test_a_round_of_refine_wherever_open(mesh, rule):
    mock, state = MESHES[mesh]()
    round_state, skipped = _run_round(state, rule, range(state.n_active), _prefer(ACTION_REFINE))
    assert skipped == []
    assert round_state.committed == state.n_active + 3 * int(state.can_refine.sum())
    new_state, report = mock.apply_marks(round_state.decisions)
    added, undone = _overshoot(state, round_state, report.realized)
    assert undone == 0 and added >= 0
    assert new_state.n_active - round_state.committed == 3 * added


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("rule", COARSEN_RULES)
@pytest.mark.parametrize("mesh", list(MESHES))
def test_a_round_of_random_masked_actions(mesh, rule, seed):
    """Actions drawn uniformly from the mask, in a random order. After
    every commit the count equals the recount (checked in _run_round). At
    the round's end the realized mesh is the count plus 3 for every refine
    balance added and every family coarsen balance undid, and never below
    the count. The dry run predicts the same mesh."""
    mock, state = MESHES[mesh]()
    rng = np.random.default_rng(seed)
    order = rng.permutation(state.n_active)
    round_state, _ = _run_round(state, rule, order, lambda mask: int(rng.choice(np.flatnonzero(mask))))
    assert (round_state.decisions != UNDECIDED).all()
    assert round_state.committed == _recount(state, round_state.decisions)

    predicted, predicted_flags = mock.dry_run(round_state.decisions)
    new_state, report = mock.apply_marks(round_state.decisions)
    assert predicted == new_state.n_active
    assert np.array_equal(predicted_flags, report.realized)

    added, undone = _overshoot(state, round_state, report.realized)
    assert added >= 0 and undone >= 0
    assert new_state.n_active - round_state.committed == 3 * (added + undone)
    assert new_state.n_active >= round_state.committed


def test_the_random_rounds_show_both_an_exact_count_and_an_overshoot():
    """So that the identity above is not checked on zeros alone: over the
    same meshes, rules and seeds, some rounds realize exactly the count and
    some realize more, by a balance-added refine and by an undone coarsen."""
    exact = overshot = added_somewhere = undone_somewhere = 0
    for mesh, rule, seed in itertools.product(MESHES, COARSEN_RULES, (0, 1, 2)):
        mock, state = MESHES[mesh]()
        rng = np.random.default_rng(seed)
        order = rng.permutation(state.n_active)
        round_state, _ = _run_round(state, rule, order, lambda mask, rng=rng: int(rng.choice(np.flatnonzero(mask))))
        new_state, report = mock.apply_marks(round_state.decisions)
        added, undone = _overshoot(state, round_state, report.realized)
        exact += new_state.n_active == round_state.committed
        overshot += new_state.n_active > round_state.committed
        added_somewhere += added > 0
        undone_somewhere += undone > 0
    assert exact > 0 and overshot > 0
    assert added_somewhere > 0 and undone_somewhere > 0


@pytest.mark.parametrize("rule", COARSEN_RULES)
@pytest.mark.parametrize("mesh", list(MESHES))
def test_two_rounds_each_started_from_its_own_mesh(mesh, rule):
    """Two rounds as a driver would run them. Each begins at its mesh's
    n_active, whatever the round before overshot, and the recount holds
    after every commit of both."""
    mock, state = MESHES[mesh]()
    rng = np.random.default_rng(7)
    for round_index in (1, 2):
        order = rng.permutation(state.n_active)
        round_state, _ = _run_round(
            state, rule, order, lambda mask: int(rng.choice(np.flatnonzero(mask))), round_index=round_index
        )
        assert round_state.round_index == round_index
        new_state, report = mock.apply_marks(round_state.decisions)
        added, undone = _overshoot(state, round_state, report.realized)
        assert new_state.n_active - round_state.committed == 3 * (added + undone)
        state = new_state
