"""System-map picture, view 1: the document system.

Writes view1_document_system.dot into the current directory: a literal Graphviz file with every
position pinned on the grid set below, so the .dot can also be edited by hand.
The content is drawn from drl-amr-notes/SYSTEM_MAP.md; when the map changes, change the
content lists here, update STAMP, and re-render. Usage:

    cd ~/projects/drl-amr-notes/presentations/system_map
    python ~/projects/drl-amr/tools/notes/system_map_view1.py
    neato -n2 -Tsvg view1_document_system.dot -o view1_document_system.svg
    neato -n2 -Tpng -Gdpi=150 view1_document_system.dot -o view1_document_system.png

The grammar and the method: ~/projects/system-builder/handoffs/
HANDOFF_2026-11-03_System_Map_Drawing_Method.md (and handoffs/used/ once retired).
"""

STAMP = "as of 2026-11-03 · drawn from SYSTEM_MAP.md"   # the snapshot date: change it on every redraw

IN = 72.0  # points per inch

# ---------- grid ----------
COL = [3.9, 6.55, 9.2, 11.85, 14.5]          # box centers, inches
LABEL_X = 1.0                                  # left edge of the layer-label column
BAND_X0, BAND_X1 = 0.7, 15.85
BOX_W, BOX_H = 2.35, 0.62
ROW_H = 0.95          # one row of boxes inside a band
GAP = 0.42            # between bands (room for arrow labels)
TOP = 1.0             # below the title

LIVING = 'style="rounded,filled", fillcolor="#eef5ff", color="#2b5ea8", penwidth=1.5'
OPEN   = 'style="filled", fillcolor="#f4f4f4", color="#555555", penwidth=1.5'
CLOSED = 'style="filled", fillcolor="#e6e6e6", color="#555555", peripheries=2'
EPHEM  = 'style="dashed,filled", fillcolor="#fffbe6", color="#b8860b", penwidth=1.5'
GEN    = 'style="dotted,filled", fillcolor="#f0fff0", color="#2e7d32", penwidth=1.7'

# layer: (key, name, gloss, rows) ; rows: list of [(col, id, title, detail, tier)]
LAYERS = [
  ("orientation", "ORIENTATION", "the state", [
    [(0, "INDEX", "INDEX.md", "hq report · five-session window<br/>target 30K · ceiling 60K", LIVING),
     (1, "README", "README.md", "points to INDEX, conventions, code", LIVING)]]),
  ("permanent", "PERMANENT", "the record", [
    [(0, "CHR", "journal/chronicles/", "one file per track phase<br/>closed at its phase close", OPEN),
     (1, "EXPL", "journal/ experiment logs", "research journal · EXP_LOG_*", OPEN),
     (2, "LOG", "DECISION_LOG.md", "the register of every decision<br/>+ the current epoch · ≤ 100K", OPEN),
     (4, "VOL", "decisions/volumes/", "frozen epochs", CLOSED)]]),
  ("planning", "PLANNING", "what is being built", [
    [(0, "RM", "roadmaps/", "one per track · active or paused<br/>≤ 100K each", LIVING),
     (2, "RMA", "roadmaps/archive/", "done or retired tracks", CLOSED)]]),
  ("teaching", "TEACHING", "why it works", [
    [(0, "EXP", "knowledge/explanations/", "one file per phase or topic<br/>two beats per item", LIVING),
     (1, "FIG", "explanations/figures/", "plates, prefixed by topic", LIVING)]]),
  ("operational", "OPERATIONAL", "how the solver<br/>stack runs<br/>(files prefixed<br/>JEXPRESSO_)", [
    [(0, "ENV", "JEXPRESSO_ENVIRONMENT", "setup · launch · blockers · git rules<br/>≤ 100K", LIVING),
     (1, "SM", "SOURCE_MAP", "the upstream code, mapped", LIVING),
     (2, "SR", "SOURCE_READS", "its closed read record", CLOSED),
     (4, "SYM", "SYMBOL_MAP.txt", "generated from the solver source", GEN)],
    [(0, "CRP", "CODE_AND_REPO_<br/>MANAGEMENT_PRACTICES", "conventions · teach-and-build", LIVING),
     (1, "REC", "RUN_RECIPES", "standing instruments +<br/>the open phase's working section", LIVING),
     (3, "RECB", "RUN_RECIPES_PHASE&lt;N&gt;", "one per closed phase", CLOSED)]]),
  ("program", "PROGRAM", "the research spine", [
    [(0, "PHD", "PhD proposal", "the six-stage plan", LIVING),
     (1, "HIST", "RESEARCH_HISTORY", "the program narrative", LIVING),
     (2, "LAND", "COMPETITIVE_LANDSCAPE", "the novelty claims", LIVING),
     (3, "LIT", "literature/", "external-method digests", LIVING),
     (4, "PROP", "proposals/", "write-once, never authoritative", CLOSED)]]),
  ("outward", "OUTWARD", "what leaves<br/>the project<br/>(files prefixed<br/>JEXPRESSO_)", [
    [(0, "LED", "DG_CHANGESET (the ledger)", "the upstream record", LIVING),
     (1, "PRD", "DG_AMR_PR_DESCRIPTION", "the PR text", LIVING),
     (2, "WALK", "DG_AMR_WALKTHROUGH", "final text, shipped", CLOSED),
     (4, "PRES", "presentations/", "decks and renders,<br/>rebuilt from source", GEN)]]),
  ("ephemeral", "EPHEMERAL", "between sessions;<br/>never a source of truth", [
    [(0, "SP", "SIDE_PROJECTS.md", "ideas parked with a revisit", LIVING),
     (1, "MTG", "meetings/", "prep and notes", EPHEM),
     (2, "HIN", "handoffs/ (inbox)", "not yet started from", EPHEM),
     (4, "HUS", "handoffs/used/", "started from, retired", EPHEM)]]),
  ("configuration", "CONFIGURATION", "the rules under<br/>everything above", [
    [(0, "PROMPT", "system prompt", "claude_project/ · the only copy", LIVING),
     (1, "PKM", "PK_MANIFEST", "the project-knowledge upload set", LIVING),
     (2, "DM", "DOC_MANAGEMENT", "the conventions · authoritative", LIVING),
     (3, "MAP", "SYSTEM_MAP", "the system described · ≤ 15K", LIVING)]]),
]

out = []
w = out.append
pos = {}       # id -> (x, y) inches, y measured downward from the top
band_box = {}  # key -> (y_top, y_bot)

y = TOP
for key, name, gloss, rows in LAYERS:
    h = 0.2 + ROW_H * len(rows)
    band_box[key] = (y, y + h)
    for r, row in enumerate(rows):
        cy = y + 0.1 + ROW_H * r + ROW_H / 2
        for col, nid, title, detail, tier in row:
            pos[nid] = (COL[col], cy, title, detail, tier)
    y += h + GAP
LEG_TOP = y - GAP + 0.35
HEIGHT = LEG_TOP + 1.25

def P(x, ydown):  # inches (y down) -> pos string in points (y up)
    return f'"{x*IN:.1f},{(HEIGHT-ydown)*IN:.1f}!"'

w('// View 1 — the document system. Rendered with:  neato -n2 -Tsvg view1_document_system.dot -o view1_document_system.svg')
w('// Generated by system_map_view1.py (drl-amr tools/notes/); without it, edit by the grid below.')
w('// Every position is pinned (pos="x,y!" in points, y up). Grid: box centers at x = ' +
  ", ".join(f"{c*IN:.0f}" for c in COL) + ' pt; one row per band, two in OPERATIONAL.')
w('// Grammar: border = tier (rounded blue living; square gray open record; double gray closed; dashed amber ephemeral; dotted green generated).')
w('//          thick dark-blue line = lifecycle move; thin gray = propagation; dashed light gray = pairing without flow.')
w('digraph document_system {')
w('  graph [splines=false, outputorder=nodesfirst, bgcolor="white", pad=0.25,')
w(f'         bb="0,0,{(BAND_X1+0.3)*IN:.0f},{HEIGHT*IN:.0f}"];')
w('  node  [fontname="Helvetica", fontsize=11, shape=box, margin="0.10,0.04"];')
w('  edge  [arrowsize=0.75];')
w('')
w('  // ---- title ----')
w(f'  title [shape=plaintext, pos={P((BAND_X0+BAND_X1)/2, 0.42)}, label=<<b><font point-size="19">drl-amr-notes — the document system</font></b><br/>'
  '<font point-size="11" color="#555555">nine layers · within a layer, living documents on the left, closed records on the right · '
  'thick arrows are the only ways text changes homes</font><br/><font point-size="9" color="#888888">' + STAMP + '</font>>];')
w('')
w('  // ---- bands (drawn first, behind the boxes) ----')
for i, (key, name, gloss, rows) in enumerate(LAYERS):
    t, b = band_box[key]
    fill = "#f7f9fc" if i % 2 == 0 else "#ffffff"
    bw, bh = BAND_X1 - BAND_X0, b - t
    w(f'  band_{key} [label="", style="filled", fillcolor="{fill}", color="#d5dbe6", penwidth=1,'
      f' width={bw:.2f}, height={bh:.2f}, fixedsize=true, pos={P((BAND_X0+BAND_X1)/2, (t+b)/2)}];')
    w(f'  lab_{key} [shape=plaintext, fontsize=11, pos={P(LABEL_X+0.72, (t+b)/2)},'
      f' label=<<table border="0" cellpadding="0" cellspacing="1"><tr><td align="left" width="120"><b><font color="#2b3a55">{name}</font></b></td></tr>'
      f'<tr><td align="left" width="120"><font point-size="9" color="#555555">{gloss.replace('<br/>', '<br align="left"/>')}<br align="left"/></font></td></tr></table>>];')
w('')
w('  // ---- documents ----')
for key, name, gloss, rows in LAYERS:
    w(f'  // {name.lower()}')
    for row in rows:
        for col, nid, title, detail, tier in row:
            x, cy, *_ = pos[nid]
            w(f'  {nid} [{tier}, width={BOX_W}, height={BOX_H}, pos={P(x, cy)},'
              f' label=<<b>{title}</b><br/><font point-size="8.5" color="#555555">{detail}</font>>];')
w('')

def xy(nid): return pos[nid][0], pos[nid][1]
half_w, half_h = BOX_W / 2, BOX_H / 2

# ---- lifecycle moves ----
w('  // ---- lifecycle moves (thick) ----')
w('  edge [penwidth=3.2, color="#1f3f73"];')
for a, b in [("RM", "RMA"), ("LOG", "VOL"), ("REC", "RECB"), ("HIN", "HUS")]:
    w(f'  {a} -> {b};')
# vertical straight drops between adjacent bands
w('  INDEX -> CHR;')
w('  RM -> EXP;')
w('')
w('  // ---- propagation (thin): routed through the gap between TEACHING and OPERATIONAL ----')
ex, ey = xy("EXP"); sx, sy = xy("SM")
gap_y = band_box["teaching"][1] + GAP / 2
w(f'  wp1 [shape=point, width=0.01, style=invis, pos={P(ex+0.55, gap_y)}];')
w(f'  wp2 [shape=point, width=0.01, style=invis, pos={P(sx, gap_y)}];')
w(f'  e_src [shape=point, width=0.01, style=invis, pos={P(ex+0.55, ey+half_h)}];')
w('  edge [penwidth=1.1, color="#555555"];')
w('  e_src -> wp1 [arrowhead=none];')
w('  wp1 -> wp2 [arrowhead=none];')
w('  wp2 -> SM;')
w('')
w('  // ---- pairing without flow ----')
w('  SM -> SR [style=dashed, penwidth=0.9, color="#9a9a9a", arrowhead=none];')
w('')

# ---- arrow labels as pinned plaintext ----
def lab(nid, x, ydown, text, color="#1f3f73", align="left"):
    w(f'  {nid} [shape=plaintext, fontsize=8.5, fontcolor="{color}", pos={P(x, ydown)}, label=<{text}>];')
w('  // ---- arrow labels ----')
ix, iy = xy("INDEX"); rx, ry = xy("RM")
lab("lb_wrap", ix + 0.15 + 0.037*52, band_box["orientation"][1] + GAP / 2, "<b>wrap:</b> the entry leaving the five-session window moves")
lab("lb_trim", rx + 0.15 + 0.037*56, band_box["planning"][1] + GAP / 2, "<b>phase close:</b> facts move first, then the section is trimmed")
lab("lb_arch", (COL[1]+COL[2]) / 2 - 0.5, ry - 0.17, "track done")
lx, ly = xy("LOG")
lab("lb_roll", (COL[3]+COL[4]) / 2 - 0.5, ly - 0.17, "phase close")
rcx, rcy = xy("REC")
lab("lb_bank", (COL[2]+COL[3]) / 2 - 0.5, rcy - 0.17, "phase close")
hx, hy = xy("HIN")
lab("lb_retire", (COL[3]+COL[4]) / 2 - 0.5, hy - 0.17, "session start")
lab("lb_fold", (ex + 0.55 + sx) / 2 + 0.25, gap_y - 0.12, "<b>item close:</b> a durable source fact folds in", color="#555555")
w('')

# ---- legend ----
w('  // ---- legend ----')
lt = LEG_TOP
w(f'  leg_box [label="", style="rounded", color="#bbbbbb", width={BAND_X1-BAND_X0:.2f}, height=1.05, fixedsize=true,'
  f' pos={P((BAND_X0+BAND_X1)/2, lt + 0.55)}];')
w(f'  leg_t [shape=plaintext, fontsize=11, pos={P(LABEL_X+0.72, lt+0.55)}, label=<<b>Legend</b><br/><font point-size="9" color="#555555">border = tier<br/>line = kind of flow</font>>];')
items = [("living", "edited in place (Tier A)", LIVING), ("open record", "appended only (Tier B)", OPEN),
         ("closed record", "frozen", CLOSED), ("ephemeral", "never a source of truth", EPHEM),
         ("generated", "never hand-edited", GEN)]
for i, (t, d, tier) in enumerate(items):
    w(f'  leg{i} [{tier}, width=2.2, height=0.42, fontsize=9, pos={P(COL[i], lt+0.33)},'
      f' label=<<b>{t}</b><br/><font point-size="8" color="#555555">{d}</font>>];')
y2 = lt + 0.82
lines = [("lifecycle move", 'penwidth=3.2, color="#1f3f73"', "normal"),
         ("propagation", 'penwidth=1.1, color="#555555"', "normal"),
         ("pairing, no flow", 'style=dashed, penwidth=0.9, color="#9a9a9a"', "none")]
for i, (t, sty, head) in enumerate(lines):
    x0 = COL[i] - 1.05
    w(f'  la{i} [shape=point, width=0.01, style=invis, pos={P(x0, y2)}];')
    w(f'  lb{i} [shape=point, width=0.01, style=invis, pos={P(x0+0.55, y2)}];')
    w(f'  la{i} -> lb{i} [{sty}, arrowhead={head}];')
    w(f'  lt{i} [shape=plaintext, fontsize=9, pos={P(x0+1.2, y2)}, label="{t}"];')
w(f'  leg_note [shape=plaintext, fontsize=9, fontcolor="#555555", pos={P((COL[3]+COL[4])/2, y2)},'
  ' label="the session\'s reads and hand-overs: view 2 · the repo\'s neighbors: view 3"];')
w('}')

open("view1_document_system.dot", "w").write("\n".join(out) + "\n")

