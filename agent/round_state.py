"""The round state: what moves within one adaptation round.

Within a batch round the mesh and the indicator arrays are frozen in a
RoundCache (agent/round_cache.py). What still moves from one decision to the
next is held here: which round this is, and how many elements the marks
committed so far would give. The actor observation builder takes a
RoundState, a RoundCache and a configuration, and nothing else.

AGENT_2D_ROADMAP item 2.2 defines the noun. Item 2.3 adds the round's
family-decision record as a field, and item 2.4 adds how committed advances.
"""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral


@dataclass(frozen=True, slots=True, kw_only=True)
class RoundState:
    """Where one round stands at one decision.

    Fields:
        round_index: The round within the remesh interval, 1 to n_rounds.
        n_rounds: Rounds per remesh interval.
        committed: The committed element count: n_active at the round's
            start, moved by every mark committed since. It is exact for the
            requested marks and not for the realized mesh.
    """

    round_index: int
    n_rounds: int
    committed: int

    def __post_init__(self) -> None:
        """Check the three counts.

        Raises:
            ValueError: Naming the field, for a value that is not an integer
                or is out of range.
        """
        for name in ("round_index", "n_rounds", "committed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise ValueError(f"{name}: {value!r}, expected an integer")
        if self.n_rounds < 1:
            raise ValueError(f"n_rounds: {self.n_rounds}, expected at least 1")
        if not 1 <= self.round_index <= self.n_rounds:
            raise ValueError(f"round_index: {self.round_index}, expected 1 to n_rounds = {self.n_rounds}")
        if self.committed < 1:
            raise ValueError(f"committed: {self.committed}, expected at least 1")