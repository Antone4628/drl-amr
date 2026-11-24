"""The interval driver of the 2D agent core: one remesh interval, held in one place.

RoundDriver walks a backend through remesh intervals the way the contract
lays one out (SOLVER_CONTRACT_V2 section 6.4): for each round one read of
the backend, one decision per position the queue asks, then the round's
marks applied; after the interval's last round, one advance. That is
2R + 1 backend calls for R rounds.

It holds what moves between decisions: the current state, the round's
cache, order and RoundState, the cursor into the order, and the interval's
thresholds. The thresholds are taken once per interval, from its first
read, and held for every round of it (D-021): begin_interval is the one
place they are assigned.

Whoever calls decides what a decision is. A training environment asks its
policy, pays rewards around commit and advance, and ends the episode
(envs/round_env.py). The same calls with a policy and no rewards are
bridge deployment: run_interval is that loop. Nothing here computes a
reward, draws an initial condition or counts an episode's intervals.

This module belongs to the 2D stack (AGENT_2D_ROADMAP item 4.1) and stands
beside drivers/multiround_driver.py, the sequential driver, which is not
on this path.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral

import numpy as np

from agent.round_cache import RoundCache, read_round
from agent.round_commit import commit, overshoot, start_round
from agent.round_mask import action_mask
from agent.round_observation import actor_observation
from agent.round_queue import next_undecided, round_order, round_thresholds
from agent.round_spec import FIXED_RULES, ObservationSpec
from agent.round_state import RoundState
from contract.backend import SolverBackend
from contract.reports import IntervalReport
from contract.state import SolverState

# The one threshold timing this driver implements, by its name in
# agent.round_spec.FIXED_RULES.
THRESHOLD_TIMING = "interval_first"

# Where the driver stands between calls.
_IDLE = "idle"  # no episode, or a backend call did not return: only reset runs
_ASKING = "asking"  # a position is asked: action_mask, actor_observation, commit
_COMMITTED = "committed"  # the asked position is decided: advance
_ENDED = "ended"  # an interval has ended: begin_interval, or reset


@dataclass(frozen=True, slots=True, kw_only=True)
class RoundRecord:
    """One adaptation round's counts, made once its marks are applied.

    Fields:
        interval: The remesh interval the round belongs to, counted from 1
            at the last reset.
        round_index: The round within the interval, 1 to n_rounds.
        committed: The committed element count at the round's end: what
            the requested marks alone would give.
        overshoot: The realized count above the committed count.
        skipped: Positions the round passed over as already decided.
        n_active: The realized element count, after apply_marks.
    """

    interval: int
    round_index: int
    committed: int
    overshoot: int
    skipped: int
    n_active: int


class RoundDriver:
    """A backend's remesh intervals, one decision at a time.

    One interval is

        driver.reset(ic, amr_freq, initial_level)     # or begin_interval()
        while True:
            mask = driver.action_mask()
            driver.commit(action)                     # chosen from the mask
            ended = driver.advance()
            if ended == "interval":
                break

    commit calls no verb of the backend; advance makes every call. After
    "interval" nothing is asked until begin_interval, so whether another
    interval follows is the caller's to say.

    Args:
        backend: The backend to drive. Only reset, compute_error,
            apply_marks and step_interval are called; never describe or
            get_state.
        spec: What the within-round logic is configured with: the
            indicator and its reduction, beta, the coarsen rule and the
            rounds per interval.
        cadence: Substeps between the samples of the interval's
            accumulator, as step_interval takes it.
        accumulate: Indicator names step_interval accumulates over the
            interval. None by default: a deployment needs none. A training
            environment names its indicator, for the global reward.

    Raises:
        ValueError: Naming spec, cadence or accumulate; or starting with
            "thresholds", if agent.round_spec.FIXED_RULES names another
            threshold timing than the one this driver implements.
    """

    def __init__(
        self,
        backend: SolverBackend,
        spec: ObservationSpec,
        *,
        cadence: int = 1,
        accumulate: Sequence[str] = (),
    ) -> None:
        if not isinstance(spec, ObservationSpec):
            raise ValueError(f"spec: {spec!r}, expected an ObservationSpec")
        if isinstance(cadence, bool) or not isinstance(cadence, Integral) or cadence < 1:
            raise ValueError(f"cadence: {cadence!r}, expected an integer of at least 1")
        if isinstance(accumulate, str):
            raise ValueError(f"accumulate: expected a sequence of names, got the string {accumulate!r}")
        if FIXED_RULES["thresholds"] != THRESHOLD_TIMING:
            raise ValueError(
                f"thresholds: the fixed rule is {FIXED_RULES['thresholds']!r}, this driver holds {THRESHOLD_TIMING!r}"
            )
        self.backend = backend
        self.spec = spec
        self.cadence = int(cadence)
        self.accumulate = tuple(accumulate)

        # What moves between calls. Set by reset.
        self._phase = _IDLE
        self._interval = 0
        self._state: SolverState | None = None
        self._cache: RoundCache | None = None
        self._thresholds: tuple[np.ndarray, np.ndarray] | None = None
        self._order: np.ndarray | None = None
        self._round_state: RoundState | None = None
        self._position: int | None = None
        self._cursor = 0
        self._skipped = 0
        self._round_record: RoundRecord | None = None
        self._interval_report: IntervalReport | None = None

    # --- What the driver holds ------------------------------------------------------

    @property
    def state(self) -> SolverState | None:
        """The current state: the one reset or the last apply_marks returned."""
        return self._state

    @property
    def cache(self) -> RoundCache | None:
        """The round's read: its state and indicator arrays."""
        return self._cache

    @property
    def thresholds(self) -> tuple[np.ndarray, np.ndarray] | None:
        """The interval's thresholds, e_max and e_min. After an interval has
        ended they are still that interval's, until begin_interval."""
        return self._thresholds

    @property
    def round_state(self) -> RoundState | None:
        """The round's index, committed count and decision record."""
        return self._round_state

    @property
    def position(self) -> int | None:
        """The position the queue is asking, or None when nothing is asked."""
        return self._position if self._phase == _ASKING else None

    @property
    def interval(self) -> int:
        """Remesh intervals completed since the last reset."""
        return self._interval

    @property
    def round_record(self) -> RoundRecord | None:
        """The record of the round the last advance ended, or None if it
        ended none. Set once the round's marks are applied, so it is there
        when a backend call after the adapt fails."""
        return self._round_record

    @property
    def interval_report(self) -> IntervalReport | None:
        """What step_interval returned, if the last advance ended an
        interval; else None."""
        return self._interval_report

    # --- The interval and its rounds ------------------------------------------------

    def reset(self, ic: Mapping[str, float], amr_freq: int, initial_level: int = 1) -> SolverState:
        """Begin an episode on the backend and its first interval.

        Args:
            ic: The initial condition, as the backend's reset takes it.
            amr_freq: Base steps per remesh interval.
            initial_level: The uniform level the mesh starts at.

        Returns:
            The first state.

        Raises:
            ValueError: As the backend's reset raises.
            BackendFailure: If the backend fails. Nothing is asked after
                that until a reset that returns.
        """
        self._phase = _IDLE
        self._round_record = None
        self._interval_report = None
        self._state = self.backend.reset(ic, amr_freq, initial_level)
        self._interval = 0
        self._phase = _ENDED
        self.begin_interval()
        return self._state

    def begin_interval(self) -> None:
        """Begin the next interval: one read, its thresholds, round 1.

        The one place the thresholds are assigned. They are computed from
        this read and held until the next call, so every round of the
        interval orders its queue under the same pair (D-021).

        Raises:
            RuntimeError: Unless an interval has just ended (or reset is
                beginning the first).
            BackendFailure: If the read fails or is not finite.
        """
        if self._phase != _ENDED:
            raise RuntimeError("begin_interval: no interval has just ended")
        self._phase = _IDLE
        self._read()
        self._thresholds = round_thresholds(self._cache, self.spec.observation, self.spec.beta)
        self._start_round(1)

    def _read(self) -> None:
        """The round's one read of the backend, on the state the driver holds."""
        config = self.spec.observation
        self._cache = read_round(self.backend, self._state, [config.indicator], config.reduction)

    def _start_round(self, round_index: int) -> None:
        """Order the round's queue under the interval's thresholds and ask its first position."""
        self._order = round_order(self._cache, self.spec.observation, *self._thresholds)
        self._round_state = start_round(self._state, round_index, self.spec.n_rounds)
        self._position, self._cursor, self._skipped = next_undecided(self._state, self._round_state, self._order, 0)
        self._phase = _ASKING

    # --- One decision ---------------------------------------------------------------

    def action_mask(self) -> np.ndarray:
        """The actions open to the asked position: (3,) bool, coarsen, hold, refine.

        Raises:
            RuntimeError: If nothing is asked.
        """
        if self._phase != _ASKING:
            raise RuntimeError("action_mask: nothing is asked")
        return action_mask(self._state, self._round_state, self._position, self.spec.coarsen_rule)

    def actor_observation(self) -> np.ndarray:
        """The actor's feature vector for the asked position.

        The one builder shared with deployment (agent/round_observation.py).

        Raises:
            RuntimeError: If nothing is asked.
        """
        if self._phase != _ASKING:
            raise RuntimeError("actor_observation: nothing is asked")
        return actor_observation(self._cache, self._round_state, self._position, self.spec.observation)

    def commit(self, action: int) -> None:
        """Record the asked position's decision. No backend call.

        Args:
            action: 0 (coarsen), 1 (hold) or 2 (refine), open in
                action_mask.

        Raises:
            RuntimeError: If nothing is asked.
            ValueError: As agent.round_commit.commit raises, for an action
                that is not an action index or is masked. Nothing has
                changed: the same position is still asked.
        """
        if self._phase != _ASKING:
            raise RuntimeError("commit: nothing is asked")
        self._round_state = commit(self._state, self._round_state, self._position, action, self.spec.coarsen_rule)
        self._phase = _COMMITTED

    def advance(self) -> str:
        """Move to the next position to ask, across whatever ends on the way.

        While the round's queue still holds an undecided position, that is
        all that happens. When it is exhausted the round's marks are
        applied and the round's record is made; then either the next round
        is read and begun, or, after the interval's last round, the
        backend advances one interval and the driver stops.

        Returns:
            What ended: "element" (nothing), "round" or "interval". After
            "interval" nothing is asked until begin_interval.

        Raises:
            RuntimeError: Unless a decision has just been committed.
            BackendFailure: If a verb fails or an array is not finite.
                Nothing is asked after that until a reset.
        """
        if self._phase != _COMMITTED:
            raise RuntimeError("advance: no decision has been committed")
        self._round_record = None
        self._interval_report = None
        self._position, self._cursor, passed = next_undecided(
            self._state, self._round_state, self._order, self._cursor
        )
        self._skipped += passed
        if self._position is not None:
            self._phase = _ASKING
            return "element"

        # The queue is exhausted: the record is the round's marks. Until
        # the backend's calls have returned, the driver stands idle, so a
        # call that fails leaves nothing asked.
        self._phase = _IDLE
        round_index = self._round_state.round_index
        new_state, _ = self.backend.apply_marks(self._round_state.decisions)
        self._round_record = RoundRecord(
            interval=self._interval + 1,
            round_index=round_index,
            committed=int(self._round_state.committed),
            overshoot=overshoot(self._round_state, new_state),
            skipped=self._skipped,
            n_active=int(new_state.n_active),
        )
        self._state = new_state
        if round_index < self.spec.n_rounds:
            self._read()
            self._start_round(round_index + 1)
            return "round"

        # The interval's last round: advance, and stop.
        self._interval_report = self.backend.step_interval(
            self.accumulate, self.spec.observation.reduction, self.cadence
        )
        self._interval += 1
        self._phase = _ENDED
        return "interval"

    # --- The loop with a policy and no rewards --------------------------------------

    def run_interval(
        self, decide: Callable[[np.ndarray, np.ndarray], int]
    ) -> tuple[list[RoundRecord], IntervalReport]:
        """Decide every asked position until the interval ends.

        The loop of bridge deployment: the same commit and advance a
        training environment calls, with nothing paid.

        Args:
            decide: Given the actor's feature vector and the mask of the
                asked position, returns an action open in the mask.

        Returns:
            The records of the rounds that ended, in order, and what
            step_interval returned.

        Raises:
            RuntimeError: If nothing is asked.
            ValueError: As commit raises, for an action decide returned.
            BackendFailure: As advance raises.
        """
        if self._phase != _ASKING:
            raise RuntimeError("run_interval: nothing is asked")
        records = []
        while True:
            action = decide(self.actor_observation(), self.action_mask())
            self.commit(int(action))
            ended = self.advance()
            if ended != "element":
                records.append(self._round_record)
            if ended == "interval":
                return records, self._interval_report