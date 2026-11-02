#!/usr/bin/env python3
"""Split the drl-amr-notes session records out of INDEX.md and DECISION_LOG.md.

System hygiene roadmap, Phase 0.2 (roadmaps/SYSTEM_HYGIENE_ROADMAP.md);
explained in knowledge/explanations/SYSTEM_HYGIENE_EXPLAINED.md §1.

What it does
------------
INDEX.md
  * The "**Last updated:**" paragraph is a nested chain of session entries,
    newest first, each wrapped as ``*(Prior <date>[, qualifier]:* ... follows.)``.
    The chain is unwrapped into dated entries.
  * The "## Current work" section is a list of dated bullets, one per session.
  * The newest WINDOW chain entries stay in the INDEX as a "## Recent sessions"
    section; every other entry (chain and bullet) goes to the chronicle of the
    phase its date falls in (journal/chronicles/<PHASE>_CHRONICLE.md),
    oldest first.
DECISION_LOG.md
  * Entries D-001 .. D-(SPLIT-1) move verbatim to a volume file; the current
    log keeps the header, the register and D-SPLIT onward.
  * Pending-table rows whose topic begins "**CLOSED" move verbatim to a
    "## Closed pending items" section at the foot.

The one invariant: conservation. Every character of a source file lands in
exactly one of: a kept fragment of the proposed file, a moved body, or a
discarded piece of syntax that the report prints verbatim. The run asserts
    len(source) == sum(kept fragments) + sum(moved bodies) + sum(discarded)
for each source, and additionally checks that every moved body appears
verbatim in exactly one output. A failed check stops the run before any file
is written.

Safety
------
* Dry run by default: parses, accounts, prints the report, writes nothing.
* ``--write`` writes ONLY new files: the chronicles, the volume, and
  ``INDEX.proposed.md`` / ``DECISION_LOG.proposed.md`` beside their sources.
  It refuses to overwrite an existing path, refuses while any entry is
  undated or unassigned, and refuses on any conservation failure.
* The sources are opened read-only and never written.

Usage
-----
    python3 tools/notes/split_records.py --notes ~/projects/drl-amr-notes
    python3 tools/notes/split_records.py --notes ~/projects/drl-amr-notes --only index --write
"""
from __future__ import annotations

import argparse
import datetime as dt
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

# ----------------------------------------------------------------------------
# Tables the researcher rules on
# ----------------------------------------------------------------------------

# Chronicle files by date range (inclusive). A date outside every range is
# reported as UNASSIGNED and blocks --write.
PHASES: list[tuple[str, str, str, str]] = [
    # (file stem, title, first date, last date)
    ("RESTRUCTURE_CHRONICLE", "RESTRUCTURE (notes and code restructure)", "2026-06-04", "2026-06-21"),
    ("JEXPRESSO_DG_1D_CHRONICLE", "Jexpresso DG, the 1D arc (Phases 0–2) and the 2D pivot", "2026-06-22", "2026-08-06"),
    ("JEXPRESSO_DG_PHASE5_CHRONICLE", "Jexpresso DG Phase 5 (2D conforming DG core)", "2026-08-07", "2026-09-06"),
    ("JEXPRESSO_DG_PHASE6_CHRONICLE", "Jexpresso DG Phase 6 (2D DG-AMR)", "2026-09-07", "2026-10-31"),
]

# Dates ruled by the researcher for entries the text does not date.
# Key: the entry body's first 40 characters, exactly. Value: ISO date.
OVERRIDES: dict[str, str] = {
    # The fourth session of 2026-08-10, headed "*(Prior:*" with no date.
    "**D-070 + PHASE-5 STEPS 1–2 BUILT**: a c": "2026-08-10",
}

DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
ORDINALS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6}


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------

@dataclass
class Entry:
    source: str            # "chain" | "bullet"
    seq: int               # chronological rank within its source (0 = oldest)
    body: str              # verbatim text, exactly as it will be written
    header_syntax: str     # discarded text before the body
    tail_syntax: str       # discarded text after the body
    date: str | None       # ISO date or None
    qualifier: str = ""    # e.g. "second entry", "corrected 08-15"
    ordinal: int = 0       # for bullets: first/second/... session in the title
    duplicate: bool = False

    @property
    def first_line(self) -> str:
        return self.body.replace("\n", " ")[:80]


@dataclass
class Accounting:
    name: str
    source_len: int
    kept: list[str] = field(default_factory=list)       # verbatim fragments of the source kept in the proposal
    moved: list[str] = field(default_factory=list)      # verbatim bodies written elsewhere
    discarded: list[str] = field(default_factory=list)  # syntax printed in the report

    def total(self) -> int:
        return sum(map(len, self.kept)) + sum(map(len, self.moved)) + sum(map(len, self.discarded))

    def ok(self) -> bool:
        return self.total() == self.source_len


class SplitError(Exception):
    pass


# ----------------------------------------------------------------------------
# INDEX.md
# ----------------------------------------------------------------------------

CHAIN_START = "**Last updated:**"
CHAIN_END_MARK = "\n**Convention:**"
CW_START = "\n## Current work"
CW_END = "\n## Archived"
PRIOR = "*(Prior"

HEAD_HDR_RE = re.compile(r"^\*\*Last updated:\*\*\s*")
PRIOR_HDR_RE = re.compile(
    r"^\*\(Prior"                 # literal
    r"(?:\s+(\d{4}-\d{2}-\d{2}))?"  # optional date
    r"([^:*]*)"                   # optional qualifier (", second entry")
    r":\*?\s*"                    # ":*" or ": "
)
# Tail: a final sentence beginning "Prior ... follows" through the closing paren,
# or a bare closing ")" / ")*" at the very end.
TAIL_RE = re.compile(r"(\s*Prior\b[^()]*?\bfollows\b[^()]*?\.\)\*?)\s*$")
BARE_CLOSE_RE = re.compile(r"(\)\*?)\s*$")


def _locate(text: str, marker: str, what: str) -> int:
    i = text.find(marker)
    if i < 0:
        raise SplitError(f"INDEX: marker not found for {what}: {marker!r}")
    if text.find(marker, i + 1) >= 0:
        raise SplitError(f"INDEX: marker for {what} occurs more than once: {marker!r}")
    return i


def parse_chain(chain: str) -> list[Entry]:
    """Split the Last-updated paragraph into entries, newest first as found."""
    pieces: list[str] = []
    idx = [m.start() for m in re.finditer(re.escape(PRIOR), chain)]
    bounds = [0] + idx + [len(chain)]
    for a, b in zip(bounds, bounds[1:]):
        pieces.append(chain[a:b])
    entries: list[Entry] = []
    for k, piece in enumerate(pieces):
        if k == 0:
            m = HEAD_HDR_RE.match(piece)
            if not m:
                raise SplitError("INDEX: chain head does not start with the Last-updated marker")
            hdr = m.group(0)
            rest = piece[len(hdr):]
            dm = DATE_RE.match(rest)
            date = dm.group(0) if dm else None
            qual = ""
        else:
            m = PRIOR_HDR_RE.match(piece)
            if not m:
                raise SplitError(f"INDEX: unparseable chain header: {piece[:60]!r}")
            hdr = m.group(0)
            rest = piece[len(hdr):]
            date = m.group(1)
            qual = (m.group(2) or "").strip(" ,")
        # tail
        tail = ""
        tm = TAIL_RE.search(rest)
        if tm:
            tail = rest[tm.start(1):]
            rest = rest[:tm.start(1)]
        else:
            bm = BARE_CLOSE_RE.search(rest)
            if bm and k > 0:
                tail = rest[bm.start(1):]
                rest = rest[:bm.start(1)]
        # trailing whitespace between entries is syntax
        stripped = rest.rstrip()
        tail = rest[len(stripped):] + tail
        rest = stripped
        entries.append(Entry("chain", 0, rest, hdr, tail, date, qual))
    # newest first in the source → chronological rank by reversal
    n = len(entries)
    for i, e in enumerate(entries):
        e.seq = n - 1 - i
    return entries


BULLET_RE = re.compile(r"(?m)^- ")


def parse_current_work(section: str) -> tuple[str, list[Entry]]:
    """Return (header syntax, bullets). `section` starts at '## Current work'."""
    starts = [m.start() for m in BULLET_RE.finditer(section)]
    if not starts:
        raise SplitError("INDEX: no bullets found in the Current work section")
    header = section[:starts[0]]
    bounds = starts + [len(section)]
    entries: list[Entry] = []
    for k, (a, b) in enumerate(zip(bounds, bounds[1:])):
        raw = section[a:b]
        marker = raw[:2]  # "- "
        body = raw[2:]
        stripped = body.rstrip()
        ws = body[len(stripped):]
        body = stripped
        # date: first full date in the bold title, else in the first 400 chars
        title_m = re.match(r"\*\*(.*?)\*\*", body, re.S)
        probe = title_m.group(0) if title_m else body[:400]
        dm = DATE_RE.search(probe) or DATE_RE.search(body[:400])
        date = dm.group(0) if dm else None
        om = re.search(r"\b(first|second|third|fourth|fifth|sixth)\s+(session|entry|wrap)", probe, re.I)
        ordinal = ORDINALS[om.group(1).lower()] if om else 0
        entries.append(Entry("bullet", k, body, marker, ws, date, "", ordinal))
    return header, entries


def apply_overrides(entries: list[Entry]) -> list[Entry]:
    for e in entries:
        if e.date is None:
            key = e.body[:40]
            if key in OVERRIDES:
                e.date = OVERRIDES[key]
                e.qualifier = (e.qualifier + " " if e.qualifier else "") + "(date by override)"
    return entries


def phase_for(date: str) -> str | None:
    for stem, _title, a, b in PHASES:
        if a <= date <= b:
            return stem
    return None


def heading(e: Entry) -> str:
    q = f" — {e.qualifier}" if e.qualifier else ""
    return f"### {e.date}{q}\n\n"


DUP_MARK = "<!-- DUPLICATE CANDIDATE: a chain entry records this date too; keep, merge or drop before the first commit -->\n"


def chronicle_header(stem: str, title: str, today: str) -> str:
    return (
        f"# {title} — session chronicle\n\n"
        f"**Tier B (append-only once committed).** Session narrative moved verbatim out of `INDEX.md` on {today} "
        f"(system hygiene roadmap, Phase 1): one entry per session, oldest first, headed by its date; the entry text is "
        f"exactly as the INDEX carried it. Entries come from two INDEX records — the \"Last updated\" chain and the "
        f"\"Current work\" bullets — interleaved by date; where both recorded one date the bullet carries a "
        f"`DUPLICATE CANDIDATE` comment for resolution before this file's first commit. Handoff paths cited in entries "
        f"resolve under `handoffs/used/` (D-089). British spellings in frozen text are left as written (D-088).\n\n---\n\n"
    )


def recent_section(window: list[Entry], window_n: int) -> tuple[str, list[str]]:
    """Return the Recent-sessions section text and the list of bodies it embeds."""
    intro = (
        f"## Recent sessions\n\n"
        f"*The last {window_n} sessions, newest first. Older entries are in `journal/chronicles/`, one file per phase. "
        f"At wrap: add this session's entry at the top (≤ 2,500 characters), move the oldest entry here to its phase's "
        f"chronicle in the same edit, and check this file's size (target 30,000 characters, ceiling 60,000).*\n\n"
    )
    parts = [intro]
    bodies = []
    for e in window:  # newest first
        parts.append(heading(e))
        parts.append(e.body)
        parts.append("\n\n")
        bodies.append(e.body)
    # the kept text that follows begins with "\n", so end on a single newline
    return "".join(parts)[:-1], bodies


def process_index(src: str, window_n: int, today: str) -> tuple[Accounting, dict[str, str], list[Entry], list[str]]:
    """Return (accounting, outputs{relpath: text}, all entries, notes)."""
    acc = Accounting("INDEX.md", len(src))
    notes: list[str] = []

    # --- locate regions
    c0 = _locate(src, CHAIN_START, "chain start")
    c_end_mark = src.find(CHAIN_END_MARK, c0)
    if c_end_mark < 0:
        raise SplitError("INDEX: chain end marker (**Convention:**) not found after the chain")
    c1 = c_end_mark  # chain = src[c0:c1]; the '\n' before Convention stays in the kept text
    w0 = _locate(src, CW_START, "Current work start") + 1  # keep the preceding '\n' in kept text
    w1 = _locate(src, CW_END, "Current work end") + 1       # '\n## Archived' — keep the '\n' with the removed region? no:
    # Removed region is src[w0:w1] = '## Current work ... <bullets> ...\n' ending right before '## Archived'.
    if not (c0 < c1 < w0 < w1):
        raise SplitError("INDEX: regions out of order")

    chain = src[c0:c1]
    cw = src[w0:w1]

    # --- parse
    chain_entries = apply_overrides(parse_chain(chain))
    cw_header, bullets = parse_current_work(cw)
    bullets = apply_overrides(bullets)
    for e in chain_entries:
        acc.discarded += [e.header_syntax, e.tail_syntax]
    acc.discarded.append(cw_header)
    for e in bullets:
        acc.discarded += [e.header_syntax, e.tail_syntax]

    # --- window vs chronicle
    newest_first = sorted(chain_entries, key=lambda e: -e.seq)
    window = newest_first[:window_n]
    rest_chain = newest_first[window_n:]

    # --- duplicates
    chain_dates = {e.date for e in rest_chain + window if e.date}
    for b in bullets:
        if b.date in chain_dates:
            b.duplicate = True

    # --- kept fragments
    frag_a = src[:c0]
    frag_b = src[c1:w0]
    frag_c = src[w1:]
    acc.kept += [frag_a, frag_b, frag_c]

    recent, window_bodies = recent_section(window, window_n)
    acc.moved += window_bodies  # embedded verbatim in the proposal

    proposed = frag_a + recent + frag_b + frag_c

    # --- chronicles
    outputs: dict[str, str] = {"INDEX.proposed.md": proposed}
    to_chron = rest_chain + bullets
    undated = [e for e in to_chron if e.date is None]
    unassigned = [e for e in to_chron if e.date and phase_for(e.date) is None]
    by_phase: dict[str, list[Entry]] = {stem: [] for stem, *_ in PHASES}
    for e in to_chron:
        if e.date and phase_for(e.date):
            by_phase[phase_for(e.date)].append(e)

    def sort_key(e: Entry):
        # within a date: chain entries first (chronological), then bullets by ordinal then file order
        return (e.date, 0 if e.source == "chain" else 1, e.seq if e.source == "chain" else e.ordinal, e.seq)

    for stem, title, _a, _b in PHASES:
        es = sorted(by_phase[stem], key=sort_key)
        if not es:
            continue
        parts = [chronicle_header(stem, title, today)]
        for e in es:
            parts.append(heading(e))
            if e.duplicate:
                parts.append(DUP_MARK)
            parts.append(e.body)
            parts.append("\n\n")
            acc.moved.append(e.body)
        outputs[f"journal/chronicles/{stem}.md"] = "".join(parts)

    # --- notes for the report
    notes.append(f"chain entries: {len(chain_entries)} (window {len(window)}, to chronicles {len(rest_chain)})")
    notes.append(f"current-work bullets: {len(bullets)}")
    dup_dates = sorted({b.date for b in bullets if b.duplicate and b.date})
    notes.append(f"dates with both a chain entry and a bullet: {len(dup_dates)} → {', '.join(dup_dates)}")
    notes.append(f"undated after overrides: {len(undated)}")
    notes.append(f"unassigned to any phase: {len(unassigned)}")
    for stem, *_ in PHASES:
        notes.append(f"  {stem}: {len(by_phase[stem])} entries")
    notes.append(f"proposed INDEX.md size: {len(proposed):,} characters")
    if undated or unassigned:
        notes.append("BLOCKING: undated or unassigned entries remain (see the entry list)")

    return acc, outputs, chain_entries + bullets, notes


# ----------------------------------------------------------------------------
# DECISION_LOG.md
# ----------------------------------------------------------------------------

DEC_HDR_RE = re.compile(r"(?m)^### D-(\d{3}):")
REG_ROW_RE = re.compile(r"(?m)^\| D-\d{3} \|")
PENDING_ROW_RE = re.compile(r"(?m)^\| (P-\d{3}) \| (.*?) \|.*$")


def process_log(src: str, split_at: list[int], today: str) -> tuple[Accounting, dict[str, str], list[str]]:
    """`split_at` lists the first decision of each successive file: [58] makes one
    volume D-001..D-057 and keeps D-058 onward; [58, 73] makes two volumes."""
    acc = Accounting("DECISION_LOG.md", len(src))
    notes: list[str] = []

    heads = list(DEC_HDR_RE.finditer(src))
    nums = [int(m.group(1)) for m in heads]
    if nums != list(range(1, len(nums) + 1)):
        raise SplitError(f"LOG: decision headers are not D-001..D-{len(nums):03d} in order: {nums[:5]}...")
    n_dec = len(nums)
    n_reg = len(REG_ROW_RE.findall(src))
    if split_at != sorted(split_at) or len(set(split_at)) != len(split_at) or split_at[0] < 2:
        raise SplitError(f"LOG: --split-at must be increasing decision numbers ≥ 2: {split_at}")

    pos = {int(m.group(1)): m.start() for m in heads}
    for s in split_at:
        if s not in pos:
            raise SplitError(f"LOG: split decision D-{s:03d} not found")
    first = heads[0].start()
    bounds = [1] + split_at                       # first decision of each file
    volumes: list[tuple[int, int, str]] = []      # (first, last, body)
    for a, b in zip(bounds, bounds[1:]):
        volumes.append((a, b - 1, src[pos[a]:pos[b]]))
    split_i = pos[split_at[-1]]
    keep_from = split_at[-1]

    # pending rows
    closed_rows: list[str] = []
    kept_after = src[split_i:]
    lines = kept_after.split("\n")
    kept_lines: list[str] = []
    header_row = None
    sep_row = None
    in_table = False
    for ln in lines:
        if ln.startswith("| ID | Topic |"):
            header_row, in_table = ln, True
            kept_lines.append(ln)
            continue
        if in_table and ln.startswith("|----"):
            sep_row = ln
            kept_lines.append(ln)
            continue
        m = PENDING_ROW_RE.match(ln)
        if in_table and m and m.group(2).lstrip().startswith("**CLOSED"):
            closed_rows.append(ln + "\n")
            continue
        if in_table and not ln.startswith("|"):
            in_table = False
        kept_lines.append(ln)
    kept_after_rows = "\n".join(kept_lines)

    # accounting: kept = header..D-001 + (D-keep..end minus the moved rows); moved = volumes + rows
    frag_head = src[:first]
    acc.kept += [frag_head, kept_after_rows]
    acc.moved += [v[2] for v in volumes] + closed_rows

    outputs: dict[str, str] = {}
    vol_names = []
    for k, (a, b, body) in enumerate(volumes, start=1):
        name = f"decisions/volumes/DECISION_LOG_D{a:03d}-D{b:03d}.md"
        vol_names.append((a, b, name))
        vol_hdr = (
            f"# Decision Log — Volume {k}: D-{a:03d} … D-{b:03d}\n\n"
            f"**Frozen (Tier B).** Entries moved verbatim from `decisions/DECISION_LOG.md` on {today} "
            f"(system hygiene roadmap, Phase 4). The register for every decision, and entries D-{keep_from:03d} onward, "
            f"stay in `DECISION_LOG.md`. IDs are global and never renumbered.\n\n---\n\n"
        )
        outputs[name] = vol_hdr + body

    vol_list = "; ".join(f"D-{a:03d} … D-{b:03d} in `{n}`" for a, b, n in vol_names)
    pointer = (
        f"*Earlier entries are in frozen volumes (moved {today}): {vol_list}. This file holds D-{keep_from:03d} onward. "
        f"Rollover: a new volume at a phase close or when this file passes 100,000 characters.*\n\n"
    )
    closed_section = ""
    if closed_rows:
        closed_section = (
            "\n\n## Closed pending items\n\n"
            "*Rows relocated verbatim from the Pending Decisions table; kept for their history.*\n\n"
            + (header_row or "| ID | Topic | Blocking | Target Resolution |") + "\n"
            + (sep_row or "|----|-------|----------|-------------------|") + "\n"
            + "".join(closed_rows)
        )
    proposed = frag_head + pointer + kept_after_rows.rstrip("\n") + closed_section + "\n"
    # The rstrip/added newline above changes whitespace at the very end; account for it exactly:
    # kept_after_rows is counted in full in acc.kept, so the proposal's trailing whitespace is generated text.

    outputs["DECISION_LOG.proposed.md"] = proposed
    notes.append(f"decision entries: {n_dec}; register rows: {n_reg}; current log keeps D-{keep_from:03d}..D-{n_dec:03d}")
    for a, b, n in vol_names:
        notes.append(f"  volume D-{a:03d}..D-{b:03d} ({b - a + 1} entries): {len(outputs[n]):,} characters")
    notes.append(f"pending rows relocated as closed: {len(closed_rows)} → "
                 + ", ".join(PENDING_ROW_RE.match(r).group(1) for r in closed_rows))
    open_rows = [ln for ln in kept_lines if PENDING_ROW_RE.match(ln)]
    notes.append(f"pending rows left open: {len(open_rows)}")
    notes.append(f"proposed DECISION_LOG.md size: {len(proposed):,} characters")
    return acc, outputs, notes


# ----------------------------------------------------------------------------
# Checks and report
# ----------------------------------------------------------------------------

def verify_bodies(acc: Accounting, outputs: dict[str, str]) -> list[str]:
    """Each moved body must appear verbatim in exactly one output."""
    problems = []
    for body in acc.moved:
        if not body:
            continue
        hits = [p for p, t in outputs.items() if body in t]
        if len(hits) != 1:
            problems.append(f"body {body[:50]!r}... found in {len(hits)} outputs: {hits}")
    return problems


def report_index(acc: Accounting, entries: list[Entry], notes: list[str]) -> str:
    out = ["== INDEX.md =="]
    out += [f"  {n}" for n in notes]
    out.append(f"  conservation: source {acc.source_len:,} = kept {sum(map(len, acc.kept)):,} "
               f"+ moved {sum(map(len, acc.moved)):,} + discarded {sum(map(len, acc.discarded)):,} "
               f"→ discrepancy {acc.source_len - acc.total()}  {'OK' if acc.ok() else 'FAIL'}")
    out.append("  -- entries (date | source | phase | qualifier | first 80 chars) --")
    def key(e):
        return (e.date or "0000-00-00", 0 if e.source == "chain" else 1, e.seq if e.source == "chain" else e.ordinal, e.seq)
    for e in sorted(entries, key=key):
        ph = phase_for(e.date) if e.date else None
        flag = "UNDATED" if e.date is None else ("UNASSIGNED" if ph is None else ph.replace("_CHRONICLE", ""))
        dup = " DUP" if e.duplicate else ""
        out.append(f"  {e.date or '----------'} | {e.source:6} | {flag:22} | {e.qualifier[:24]:24} |{dup} {e.first_line}")
    out.append("  -- discarded syntax (verbatim, repr) --")
    for d in acc.discarded:
        if d:
            out.append(f"  {d!r}")
    return "\n".join(out)


def report_log(acc: Accounting, notes: list[str]) -> str:
    out = ["== DECISION_LOG.md =="]
    out += [f"  {n}" for n in notes]
    out.append(f"  conservation: source {acc.source_len:,} = kept {sum(map(len, acc.kept)):,} "
               f"+ moved {sum(map(len, acc.moved)):,} + discarded {sum(map(len, acc.discarded)):,} "
               f"→ discrepancy {acc.source_len - acc.total()}  {'OK' if acc.ok() else 'FAIL'}")
    return "\n".join(out)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def read(path: Path) -> str:
    with open(path, encoding="utf-8", newline="") as f:
        return f.read()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--notes", required=True, help="path to the drl-amr-notes repo")
    ap.add_argument("--only", choices=["index", "log", "both"], default="both")
    ap.add_argument("--window", type=int, default=5, help="chain entries kept in the INDEX (default 5)")
    ap.add_argument("--split-at", default="58", help="comma-separated first decisions of each successive file; "
                    "'58' = one volume D-001..D-057; '58,73' = two volumes (default 58)")
    ap.add_argument("--write", action="store_true", help="write the new files (default: dry run, report only)")
    ap.add_argument("--today", default=dt.date.today().isoformat())
    args = ap.parse_args(argv)

    notes_root = Path(args.notes).expanduser().resolve()
    outputs: dict[str, str] = {}
    blocking: list[str] = []
    reports: list[str] = []

    try:
        if args.only in ("index", "both"):
            src = read(notes_root / "INDEX.md")
            acc, outs, entries, notes = process_index(src, args.window, args.today)
            reports.append(report_index(acc, entries, notes))
            if not acc.ok():
                blocking.append("INDEX conservation failed")
            probs = verify_bodies(acc, outs)
            if probs:
                blocking.append("INDEX body check failed: " + "; ".join(probs[:3]))
            if any("BLOCKING" in n for n in notes):
                blocking.append("INDEX has undated or unassigned entries")
            outputs.update(outs)
        if args.only in ("log", "both"):
            src = read(notes_root / "decisions" / "DECISION_LOG.md")
            acc, outs, notes = process_log(src, [int(x) for x in args.split_at.split(",")], args.today)
            reports.append(report_log(acc, notes))
            if not acc.ok():
                blocking.append("LOG conservation failed")
            probs = verify_bodies(acc, outs)
            if probs:
                blocking.append("LOG body check failed: " + "; ".join(probs[:3]))
            outputs.update(outs)
    except SplitError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    print("\n\n".join(reports))
    print("\n== outputs ==")
    for rel, text in outputs.items():
        exists = (notes_root / rel).exists()
        print(f"  {rel}: {len(text):,} characters{'  (EXISTS — would refuse)' if exists else ''}")
        if exists:
            blocking.append(f"target exists: {rel}")

    if blocking:
        print("\nBLOCKED: " + "; ".join(blocking))
        if args.write:
            print("Nothing written.")
            return 1
        return 0

    if not args.write:
        print("\nDry run — nothing written. Re-run with --write to create the files above.")
        return 0

    for rel, text in outputs.items():
        p = notes_root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "x", encoding="utf-8", newline="") as f:  # "x": never overwrite
            f.write(text)
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
