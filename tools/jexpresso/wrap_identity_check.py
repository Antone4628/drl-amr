"""The three-number wrap identity, read off a Jexpresso DG log on a natively
periodic mesh (Phase-6 item 8d-i, JEXPRESSO_ENVIRONMENT.md section 8.11).

Every DG build on a native model prints three numbers that describe one
forest from three places:

    Natively periodic model: true (G grid nodes, V topology vertices)
        -- Gridap's glued model (mesh.jl predicate)
    build_dg_faces_2D!: N of M interior faces matched across the wrap
        -- our shift-aware pairer
    PROBE8B5: ... one-cell facets: periodic-tagged=N1 ...
        -- the one-cell facet census (probe8b5 patch, JX_PROBE_8B5=1):
           a wrap mortar's coarse facet keeps its periodic tag, the hanging
           halves carry none, so N1 = wrap mortar parent faces

With N2 = base wrap edges refined on both sides (a midpoint vertex that
exists on both sides is two grid nodes glued to one topology vertex; a
midpoint only one side carries is one of each):

    G - V = gap0 + N2        gap0 = (nex+1)(ney+1) - nex*ney
    N     = W0 - N1 + N2     W0   = nex + ney  (base wrap faces)

so N2 is determined twice per build and the two must agree. A wrong wrap
glue -- a vertex glued that should not be, a child face matched to the wrong
partner -- breaks the identity before it breaks a flux. The same pass checks
the census (N/N), zero periodic one-cell facets and zero walls, mortar
one-cell facets = 3 x mortar parent faces, and length(edge_type) =
length(facet_cell_ids).

Run on the run's log (or the grep digest of it) with the probes on:

    cd ~/projects/drl-amr
    python -m tools.jexpresso.wrap_identity_check --nex 4 --ney 4 \
        ~/projects/jx/sessions/2026-10-17/s8d_insolve_native.log

Exit 0 when every build passes, 1 otherwise. Builds are delimited by the
'Done p4est_refine' line each one starts with; a log from a run without
JX_PROBE_8B5 has no N1 and is refused.

Named regression: the 2026-10-17 8d-i log -- 41 builds, 0 violations, gap
9 -> 11 -> 14 -> 13 -> 12 -> 11 -> 9, N2 up to 5.
"""

import argparse
import re
import sys

_PAT = {
    "quads": re.compile(r"Done p4est_refine with (\d+) total"),
    "GV": re.compile(r"Natively periodic model: true \((\d+) grid nodes, (\d+) topology"),
    "NM": re.compile(r"(\d+) of (\d+) interior faces matched across the wrap"),
    "P": re.compile(r"(\d+) mortar parent faces \((\d+) child"),
    "census": re.compile(r"census (\d+)/(\d+)"),
    "onecell": re.compile(r"one-cell facets: (\d+) periodic, (\d+) tagged walls, "
                          r"(\d+) untagged walls, (\d+) mortar"),
    "probe": re.compile(r"one-cell facets: periodic-tagged=(\d+)\s+untagged=(\d+)\s+"
                        r"length\(edge_type\)=(\d+)\s+length\(facet_cell_ids\)=(\d+)"),
}


def parse_builds(lines):
    """One dict per build, in log order. A build starts at 'Done p4est_refine'."""
    builds, cur = [], None
    for line in lines:
        m = _PAT["quads"].search(line)
        if m:
            if cur is not None:
                builds.append(cur)
            cur = {"quads": int(m.group(1))}
            continue
        if cur is None:
            continue
        for key in ("GV", "NM", "P", "census", "onecell", "probe"):
            m = _PAT[key].search(line)
            if m:
                cur[key] = tuple(int(g) for g in m.groups())
    if cur is not None:
        builds.append(cur)
    return builds


def check_build(b, gap0, w0):
    """(ok, N2 from the gap, N2 from the pairer, list of failure names)."""
    fails = []
    for key in ("GV", "NM", "P", "census", "onecell", "probe"):
        if key not in b:
            fails.append(f"no {key} line")
    if fails:
        return False, None, None, fails
    G, V = b["GV"]
    N, _M = b["NM"]
    n1 = b["probe"][0]
    n2_gap = G - V - gap0
    n2_pair = N - w0 + n1
    if n2_gap != n2_pair:
        fails.append(f"identity: gap gives N2={n2_gap}, pairer gives N2={n2_pair}")
    if n2_gap < 0:
        fails.append(f"gap {G - V} below the base gap {gap0}")
    if b["census"][0] != b["census"][1]:
        fails.append(f"census {b['census'][0]}/{b['census'][1]}")
    per, tw, uw, mort = b["onecell"]
    if per or tw or uw:
        fails.append(f"one-cell facets: {per} periodic, {tw} tagged walls, {uw} untagged walls")
    if mort != 3 * b["P"][0]:
        fails.append(f"mortar one-cell facets {mort} != 3 x {b['P'][0]} parent faces")
    if b["probe"][2] != b["probe"][3]:
        fails.append(f"edge_type {b['probe'][2]} != facet_cell_ids {b['probe'][3]}")
    return not fails, n2_gap, n2_pair, fails


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("log", help="a Jexpresso run log, or a grep digest that keeps the build lines")
    ap.add_argument("--nex", type=int, required=True, help="base-grid elements in x")
    ap.add_argument("--ney", type=int, required=True, help="base-grid elements in y")
    args = ap.parse_args(argv)

    gap0 = (args.nex + 1) * (args.ney + 1) - args.nex * args.ney
    w0 = args.nex + args.ney
    with open(args.log, encoding="utf-8", errors="replace") as f:
        builds = parse_builds(f)
    if not builds:
        raise SystemExit(f"{args.log}: no 'Done p4est_refine' line -- not a Jexpresso AMR log")
    if not any("probe" in b for b in builds):
        raise SystemExit(f"{args.log}: no PROBE8B5 line -- the run needs JX_PROBE_8B5=1 "
                         f"(probe8b5 patch) for N1")

    print(f"wrap identity: base {args.nex}x{args.ney}, gap0 = {gap0}, W0 = {w0}; "
          f"{len(builds)} builds in {args.log}")
    print(f"{'build':>5s} {'quads':>5s} {'G/V':>7s} {'gap':>3s} {'N/M':>7s} {'N1':>2s} "
          f"{'N2':>5s} {'P':>2s} {'census':>9s}")
    nbad = 0
    for i, b in enumerate(builds, 1):
        ok, n2g, n2p, fails = check_build(b, gap0, w0)
        nbad += 0 if ok else 1
        gv = f"{b['GV'][0]}/{b['GV'][1]}" if "GV" in b else "-"
        gap = f"{b['GV'][0] - b['GV'][1]}" if "GV" in b else "-"
        nm = f"{b['NM'][0]}/{b['NM'][1]}" if "NM" in b else "-"
        n1 = f"{b['probe'][0]}" if "probe" in b else "-"
        n2 = f"{n2g}|{n2p}" if n2g is not None else "-"
        p = f"{b['P'][0]}" if "P" in b else "-"
        cen = f"{b['census'][0]}/{b['census'][1]}" if "census" in b else "-"
        print(f"{i:5d} {b['quads']:5d} {gv:>7s} {gap:>3s} {nm:>7s} {n1:>2s} {n2:>5s} {p:>2s} "
              f"{cen:>9s}  {'ok' if ok else 'FAIL: ' + '; '.join(fails)}")
    print(f"VERDICT: {'PASS' if nbad == 0 else 'FAIL'} -- {nbad} of {len(builds)} builds violate")
    return 0 if nbad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
