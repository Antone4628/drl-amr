"""The round state: what moves within one adaptation round.

Within a batch round the mesh and the indicator arrays are frozen in a
RoundCache (agent/round_cache.py). What still moves from one decision to the
next is held here: which round this is, how many elements the marks
committed so far would give, and what has been decided so far. The actor
observation builder takes a RoundState, a RoundCache and a configuration,
and nothing else.

AGENT_2D_ROADMAP item 2.2 defines the noun. Item 2.3 adds the round's
decision record, which agent/round_mask.py reads and writes; item 2.4 adds
how committed advances.
"""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np

from contract.solver_contract import COARSEN, HOLD, REFINE

# A record entry for a position no decision has reached yet. It is not a
# mark: apply_marks takes COARSEN, HOLD and REFINE only.
UNDECIDED = -128


@dataclass(frozen=True, slots=True, kw_only=True)
class RoundState:
    """Where one round stands at one decision.

    Fields:
        round_index: The round within the remesh interval, 1 to n_rounds.
        n_rounds: Rounds per remesh interval.
        committed: The committed element count: n_active at the round's
            start, moved by every mark committed since. It is exact for the
            requested marks and not for the realized mesh.
        decisions: The round's decision record, or None at the round's
            start, when nothing is decided. (n_active,) int8, by position:
            UNDECIDED, or the mark committed for that element, by itself or
            by a sibling deciding for the family. Made read-only. Its length
            is checked where it is read, against the state.
    """

    round_index: int
    n_rounds: int
    committed: int
    decisions: np.ndarray | None = None

    def __post_init__(self) -> None:
        """Check the three counts and the record.

        Raises:
            ValueError: Naming the field, for a count that is not an integer
                or is out of range, or a record of the wrong form or values.
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
        record = self.decisions
        if record is not None:
            if not isinstance(record, np.ndarray) or record.ndim != 1:
                raise ValueError("decisions: expected a 1-D numpy array or None")
            if record.dtype != np.int8:
                raise ValueError(f"decisions: dtype {record.dtype}, expected int8")
            unknown = sorted(set(record.tolist()) - {UNDECIDED, COARSEN, HOLD, REFINE})
            if unknown:
                raise ValueError(f"decisions: values {unknown}, expected UNDECIDED, COARSEN, HOLD or REFINE")
            record.flags.writeable = False