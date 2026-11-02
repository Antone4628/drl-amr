"""Fixture tests for tools/notes/split_records.py (system hygiene roadmap, Phase 0.2).

The fixture reproduces every header and tail form found in the real INDEX.md
chain and Current-work section, plus a three-decision log with one closed
pending row. Expected outputs are written by hand below, so a parser that
conserves characters while cutting in the wrong place still fails.
"""
import importlib.util
import sys
from pathlib import Path

import pytest

TOOL = Path(__file__).resolve().parents[2] / "tools" / "notes" / "split_records.py"
spec = importlib.util.spec_from_file_location("split_records", TOOL)
sr = importlib.util.module_from_spec(spec)
sys.modules["split_records"] = sr
spec.loader.exec_module(sr)

# ---------------------------------------------------------------------------
# Synthetic INDEX
# ---------------------------------------------------------------------------

HEAD = "2026-10-30 — the site rolled back (D-092); the state is in the hq report above."
E_1029 = "the state is in the hq report."
E_1028 = "**THE PR TARGET SETTLED.** Three sittings. **Next: open the PR.**"
E_1026B = "**SESSION 3 CONTINUED.** Run 5 PASS."
E_1026A = "**THE SITE (D-086).** Studied together."
E_0831B = "**THE REBASE IS DONE.** 18 commits."
E_0831A = "**STEP 7 DONE.** Gate GREEN."
E_0814 = "Session began as demo prep."
E_0811 = "**STEP 6 DONE.** First complete run."
E_0810U = "**D-070 + PHASE-5 STEPS 1–2 BUILT**: a correction-policy change and the first two edits."

CHAIN = (
    f"**Last updated:** {HEAD} "
    f"*(Prior 2026-10-29: {E_1029})* "
    f"*(Prior 2026-10-28:* {E_1028} Prior entry follows.) "
    f"*(Prior 2026-10-26, second entry:* {E_1026B} Prior entry follows.) "
    f"*(Prior 2026-10-26, first entry:* {E_1026A} Prior entry follows.) "
    f"*(Prior 2026-08-31, second session:* {E_0831B} Prior same-day summary follows.) "
    f"*(Prior 2026-08-31, first session:* {E_0831A} Prior 2026-08-14 summary follows, corrected in place per D-070.) "
    f"*(Prior 2026-08-14, corrected 08-15:* {E_0814} Prior 2026-08-11 summary follows.) "
    f"*(Prior 2026-08-11:* {E_0811} Prior same-day summary follows.) "
    f"*(Prior:* {E_0810U}"
)

B_0604 = "**Facet 2 documentation cleanup** (RESTRUCTURE Phase 1): complete (2026-06-04) — repo renamed."
B_0701 = "**Jexpresso DG — source-verification pass (2026-07-01).** Read the source."
B_0811B = "**Phase-5 step 6 done — face infrastructure (2026-08-11, second session).** Full arc."
B_0811A = "**Phase-5 step 5 done — the four leaves (2026-08-11, first session).** Settle-before-code."
B_1006 = "**Phase-6 item 8a-ii done — 8a closed (2026-10-06).** Level 2."

INDEX_SRC = f"""# INDEX — drl-amr-notes

**Purpose:** The first document to open.

---

## hq report

- **State:** fine.
- **Updated:** 2026-10-30

---

{CHAIN}
**Convention:** see `DOC_MANAGEMENT.md`

---

## Active roadmaps (one per track)

| Track | Roadmap | Current state |
|---|---|---|
| Jexpresso DG | `roadmaps/JEXPRESSO_DG_ROADMAP.md` | Phase 6 complete. |

## Ground-truth documents

- `decisions/DECISION_LOG.md`

## Current work (update at session start/end)

- {B_0604}
- {B_0701}

- {B_0811B}
- {B_0811A}
- {B_1006}

## Archived (historical — do not use for current work)

- `STAGE_1_IMPLEMENTATION_ROADMAP.md`

## Handoffs

`handoffs/` — an inbox.
"""


@pytest.fixture
def index_run():
    return sr.process_index(INDEX_SRC, window_n=5, today="2026-11-01")


def test_index_conservation(index_run):
    acc, outputs, entries, notes = index_run
    assert acc.ok(), (acc.source_len, acc.total())
    assert sr.verify_bodies(acc, outputs) == []


def test_index_counts(index_run):
    acc, outputs, entries, notes = index_run
    chain = [e for e in entries if e.source == "chain"]
    bullets = [e for e in entries if e.source == "bullet"]
    assert len(chain) == 10
    assert len(bullets) == 5
    # the undated *(Prior:* entry is dated by the override table
    assert all(e.date for e in chain)
    assert [e for e in chain if e.body == E_0810U][0].date == "2026-08-10"
    assert any("undated after overrides: 0" in n for n in notes)


def test_index_window_and_proposal(index_run):
    acc, outputs, entries, notes = index_run
    prop = outputs["INDEX.proposed.md"]
    # the five newest stay in the INDEX, newest first, under their dates
    for body in (HEAD, E_1029, E_1028, E_1026B, E_1026A):
        assert body in prop
    assert prop.index(HEAD) < prop.index(E_1029) < prop.index(E_1028) < prop.index(E_1026B) < prop.index(E_1026A)
    assert "### 2026-10-26 — second entry" in prop and "### 2026-10-26 — first entry" in prop
    # the sixth and older are not in the INDEX
    for body in (E_0831B, E_0831A, E_0814, E_0811, E_0810U):
        assert body not in prop
    # the Current work section is gone, its neighbors intact
    assert "## Current work" not in prop
    assert "## Ground-truth documents" in prop and "## Archived" in prop
    assert "## Recent sessions" in prop
    assert prop.index("## hq report") < prop.index("## Recent sessions") < prop.index("**Convention:**")
    # the chain syntax is gone
    assert "*(Prior" not in prop and "follows.)" not in prop


def test_index_chronicles(index_run):
    acc, outputs, entries, notes = index_run
    assert "journal/chronicles/RESTRUCTURE_CHRONICLE.md" in outputs
    assert "journal/chronicles/JEXPRESSO_DG_1D_CHRONICLE.md" in outputs
    assert "journal/chronicles/JEXPRESSO_DG_PHASE5_CHRONICLE.md" in outputs
    assert "journal/chronicles/JEXPRESSO_DG_PHASE6_CHRONICLE.md" in outputs
    r = outputs["journal/chronicles/RESTRUCTURE_CHRONICLE.md"]
    assert "### 2026-06-04\n\n" + B_0604[0:20] in r
    d1 = outputs["journal/chronicles/JEXPRESSO_DG_1D_CHRONICLE.md"]
    assert B_0701 in d1
    p5 = outputs["journal/chronicles/JEXPRESSO_DG_PHASE5_CHRONICLE.md"]
    # chronological, same-day order by reversal (first before second), bullets after chain entries on a date
    order = [E_0810U, E_0811, B_0811A, B_0811B, E_0814, E_0831A, E_0831B]
    pos = [p5.index(x) for x in order]
    assert pos == sorted(pos), pos
    # the 08-11 bullets are duplicate candidates (a chain entry shares the date); 08-10 bullet does not exist
    assert p5.count(sr.DUP_MARK.strip()) == 2
    assert "### 2026-08-10 — (date by override)" in p5
    assert "### 2026-08-14 — corrected 08-15" in p5
    assert "### 2026-08-31 — first session" in p5 and "### 2026-08-31 — second session" in p5
    p6 = outputs["journal/chronicles/JEXPRESSO_DG_PHASE6_CHRONICLE.md"]
    assert B_1006 in p6 and E_1028 not in p6  # 10-28 is in the window


def test_index_discarded_is_only_syntax(index_run):
    acc, outputs, entries, notes = index_run
    joined = "".join(acc.discarded)
    # every discarded piece is header/tail syntax or whitespace — none of the bodies' words leak into it
    for word in ("Three sittings", "Run 5 PASS", "Gate GREEN", "repo renamed", "Level 2"):
        assert word not in joined
    assert "## Current work (update at session start/end)" in joined
    assert "Prior 2026-08-14 summary follows, corrected in place per D-070.)" in joined


def test_index_marker_errors():
    with pytest.raises(sr.SplitError):
        sr.process_index(INDEX_SRC.replace("**Convention:**", "**Conv:**"), 5, "2026-11-01")
    with pytest.raises(sr.SplitError):
        sr.process_index(INDEX_SRC.replace("## Archived", "## Archive"), 5, "2026-11-01")


# ---------------------------------------------------------------------------
# Synthetic decision log
# ---------------------------------------------------------------------------

LOG_SRC = """# Decision Log

**Purpose:** Track decisions.

## Decision Register

| ID | Gist | Status |
|----|------|--------|
| D-001 | First | Final |
| D-002 | Second | Final |
| D-003 | Third | Final |

---

## Decisions

### D-001: First
**Date:** 2026-03-09
**Decision:** one.

---

### D-002: Second
**Date:** 2026-03-10
**Decision:** two.

---

### D-003: Third
**Date:** 2026-03-11
**Decision:** three.

---

## Pending Decisions (Updated 2026-10-18)

| ID | Topic | Blocking | Target Resolution |
|----|-------|----------|-------------------|
| P-001 | Open thing | nothing | later |
| P-002 | **CLOSED 2026-10-04.** Was a thing. | nothing | nothing |
| P-003 | Another open | nothing | later |
**Retired pending decisions:**
- ~~P-004~~ retired.
"""


def test_log_split():
    acc, outputs, notes = sr.process_log(LOG_SRC, split_at=[3], today="2026-11-01")
    assert acc.ok(), (acc.source_len, acc.total())
    assert sr.verify_bodies(acc, outputs) == []
    vol = outputs["decisions/volumes/DECISION_LOG_D001-D002.md"]
    prop = outputs["DECISION_LOG.proposed.md"]
    assert "### D-001: First" in vol and "### D-002: Second" in vol and "### D-003" not in vol
    assert "### D-003: Third" in prop and "### D-001" not in prop and "### D-002" not in prop
    # register stays whole in the current log
    assert "| D-001 | First | Final |" in prop
    # closed row relocated, open rows stay in the table
    assert "| P-002 | **CLOSED 2026-10-04.**" in prop
    assert prop.index("## Closed pending items") < prop.index("| P-002 |")
    assert prop.index("| P-001 |") < prop.index("| P-003 |") < prop.index("## Closed pending items")
    assert "**Retired pending decisions:**" in prop
    assert any("relocated as closed: 1 → P-002" in n for n in notes)
    assert any("left open: 2" in n for n in notes)


def test_log_bad_order():
    with pytest.raises(sr.SplitError):
        sr.process_log(LOG_SRC.replace("### D-002", "### D-005"), [3], "2026-11-01")


# ---------------------------------------------------------------------------
# End to end: dry run writes nothing, --write writes only new files, never twice
# ---------------------------------------------------------------------------

def test_cli_dry_run_and_write(tmp_path, capsys):
    notes = tmp_path / "notes"
    (notes / "decisions").mkdir(parents=True)
    (notes / "INDEX.md").write_text(INDEX_SRC, encoding="utf-8")
    (notes / "decisions" / "DECISION_LOG.md").write_text(LOG_SRC, encoding="utf-8")
    before = sorted(p.relative_to(notes).as_posix() for p in notes.rglob("*") if p.is_file())

    rc = sr.main(["--notes", str(notes), "--split-at", "3", "--today", "2026-11-01"])
    assert rc == 0
    after = sorted(p.relative_to(notes).as_posix() for p in notes.rglob("*") if p.is_file())
    assert after == before, "dry run must write nothing"
    out = capsys.readouterr().out
    assert "Dry run — nothing written" in out and "discrepancy 0  OK" in out

    rc = sr.main(["--notes", str(notes), "--split-at", "3", "--today", "2026-11-01", "--write"])
    assert rc == 0
    assert (notes / "INDEX.proposed.md").exists()
    assert (notes / "DECISION_LOG.proposed.md").exists()
    assert (notes / "decisions" / "volumes" / "DECISION_LOG_D001-D002.md").exists()
    assert (notes / "journal" / "chronicles" / "JEXPRESSO_DG_PHASE6_CHRONICLE.md").exists()
    # sources untouched
    assert (notes / "INDEX.md").read_text(encoding="utf-8") == INDEX_SRC
    assert (notes / "decisions" / "DECISION_LOG.md").read_text(encoding="utf-8") == LOG_SRC
    # a second --write refuses because the targets exist
    rc = sr.main(["--notes", str(notes), "--split-at", "3", "--today", "2026-11-01", "--write"])
    assert rc == 1
    assert "target exists" in capsys.readouterr().out


def test_log_two_volumes():
    acc, outputs, notes = sr.process_log(LOG_SRC, split_at=[2, 3], today="2026-11-01")
    assert acc.ok() and sr.verify_bodies(acc, outputs) == []
    v1 = outputs["decisions/volumes/DECISION_LOG_D001-D001.md"]
    v2 = outputs["decisions/volumes/DECISION_LOG_D002-D002.md"]
    prop = outputs["DECISION_LOG.proposed.md"]
    assert "### D-001" in v1 and "### D-002" not in v1
    assert "### D-002" in v2 and "### D-001" not in v2 and "### D-003" not in v2
    assert "### D-003" in prop and "### D-002" not in prop
    assert "Volume 2: D-002 … D-002" in v2
    with pytest.raises(sr.SplitError):
        sr.process_log(LOG_SRC, split_at=[3, 2], today="2026-11-01")
