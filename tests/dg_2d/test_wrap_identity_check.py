"""Tests for tools/jexpresso/wrap_identity_check.py -- the three-number wrap
identity read off a Jexpresso DG log on a natively periodic mesh (item 8d-i).

The synthetic builds below are the 8d-i run's own lines (4x4 base: gap0 = 9,
W0 = 8) at t = 2.0 (N1 = 2, N2 = 0) and t = 10.5 (N1 = 0, N2 = 5), then the
same lines with one number moved, so the test pins what each failure looks
like rather than only that a correct log passes. The 8d-ii scripted leg's six
builds (8x8 base: gap0 = 17, W0 = 16, a level-2 family on the wrap) pin the
level-2 form of the identity and the centroid-based N1.
"""

import pytest

from tools.jexpresso.wrap_identity_check import check_build, main, parse_builds

GAP0, W0 = 9, 8   # 4x4 base: (5*5 - 16), (4 + 4)


def _build(G, V, N, M, P, census, mort, n1, et, fc, per=0, tw=0, uw=0, n1c=None):
    b = dict(quads=34, GV=(G, V), NM=(N, M), P=(P, 2 * P), census=(census, census),
             onecell=(per, tw, uw, mort), probe=(n1, mort - n1, et, fc))
    if n1c is not None:
        b["centroid"] = (n1c,)
    return b


def _log(G, V, N, M, P, census, mort, n1, et, fc, per=0, census_hi=None, n1c=None):
    census_hi = census if census_hi is None else census_hi
    tail = [] if n1c is None else [f" # PROBE8B5: wrap mortar parent faces by centroid jump = {n1c}"]
    return "\n".join([
        " # Done p4est_refine with 34 total quadrants",
        f" # Natively periodic model: true ({G} grid nodes, {V} topology vertices)",
        f" # build_dg_faces_2D!: {N} of {M} interior faces matched across the wrap",
        f" # build_dg_faces_2D!: {M} interior + 0 periodic = {M} conforming faces; {P} mortar "
        f"parent faces ({2 * P} child halves); 0 wall sides (0 with a boundary flux, "
        f"0 free/transmissive); census {census}/{census_hi} element sides claimed once",
        f" # build_dg_faces_2D!: one-cell facets: {per} periodic, 0 tagged walls, "
        f"0 untagged walls, {mort} mortar",
        f" # PROBE8B5: native_periodic=true  one-cell facets: periodic-tagged={n1}  "
        f"untagged={mort - n1}  length(edge_type)={et}  length(facet_cell_ids)={fc}",
    ] + tail) + "\n"


T20 = (48, 39, 6, 53, 10, 136, 30, 2, 83, 83)      # 8d-i t = 2.0: N1 = 2, N2 = 0
T105 = (53, 39, 13, 53, 10, 136, 30, 0, 83, 83)    # 8d-i t = 10.5: N1 = 0, N2 = 5


def test_measured_builds_pass():
    for args, n2 in ((T20, 0), (T105, 5)):
        ok, n2_gap, n2_pair, fails = check_build(_build(*args), GAP0, W0)
        assert ok, fails
        assert n2_gap == n2_pair == n2


def test_identity_breaks_on_a_moved_vertex():
    # one topology vertex too few: Gridap glued a pair it should not have
    ok, n2_gap, n2_pair, fails = check_build(_build(48, 38, 6, 53, 10, 136, 30, 2, 83, 83), GAP0, W0)
    assert not ok and n2_gap == 1 and n2_pair == 0
    assert any(f.startswith("identity") for f in fails)


def test_identity_breaks_on_a_moved_face():
    # the pairer matched one wrap face too many for the mortars the census counts
    ok, n2_gap, n2_pair, fails = check_build(_build(48, 39, 7, 53, 10, 136, 30, 2, 83, 83), GAP0, W0)
    assert not ok and n2_gap == 0 and n2_pair == 1


def test_identity_breaks_on_a_lost_tag():
    # a wrap mortar whose parent facet lost its periodic tag: N1 undercounted
    ok, n2_gap, n2_pair, fails = check_build(_build(48, 39, 6, 53, 10, 136, 30, 1, 83, 83), GAP0, W0)
    assert not ok and n2_pair == -1


def test_other_checks():
    ok, *_ , fails = check_build(_build(48, 39, 6, 53, 10, 136, 27, 2, 83, 83), GAP0, W0)
    assert not ok and any("3 x" in f for f in fails)
    ok, *_, fails = check_build(_build(48, 39, 6, 53, 10, 136, 30, 2, 83, 82), GAP0, W0)
    assert not ok and any("edge_type" in f for f in fails)
    ok, *_, fails = check_build(_build(48, 39, 6, 53, 10, 136, 30, 2, 83, 83, uw=1), GAP0, W0)
    assert not ok and any("untagged walls" in f for f in fails)
    b = _build(48, 39, 6, 53, 10, 136, 30, 2, 83, 83)
    b["census"] = (135, 136)
    ok, *_, fails = check_build(b, GAP0, W0)
    assert not ok and any(f.startswith("census") for f in fails)


def test_missing_line_is_a_failure_not_a_pass():
    b = _build(*T20)
    del b["probe"]
    ok, *_, fails = check_build(b, GAP0, W0)
    assert not ok and fails == ["no probe line"]


def test_parse_and_main_round_trip(tmp_path, capsys):
    log = tmp_path / "run.log"
    log.write_text(" # noise\n" + _log(*T20) + " # more noise\n" + _log(*T105))
    builds = parse_builds(log.read_text().splitlines())
    assert len(builds) == 2 and builds[0]["probe"][0] == 2 and builds[1]["GV"] == (53, 39)
    assert main([str(log), "--nex", "4", "--ney", "4"]) == 0
    out = capsys.readouterr().out
    assert "PASS -- 0 of 2 builds violate" in out
    log.write_text(_log(*T20) + _log(53, 38, 13, 53, 10, 136, 30, 0, 83, 83))
    assert main([str(log), "--nex", "4", "--ney", "4"]) == 1
    assert "1 of 2 builds violate" in capsys.readouterr().out


def test_refuses_a_log_without_the_probe(tmp_path):
    log = tmp_path / "noprobe.log"
    log.write_text("\n".join(_log(*T20).splitlines()[:-1]) + "\n")
    with pytest.raises(SystemExit, match="PROBE8B5"):
        main([str(log), "--nex", "4", "--ney", "4"])


# The 8d-ii scripted leg (2026-10-17, 8x8 native): preadapt, A1..A5.
GAP0_8, W0_8 = 17, 16
SCRIPTED = (
    (90, 73, 16, 131, 6, 280, 18, 0, 149, 149, 0),
    (95, 78, 15, 131, 10, 292, 30, 1, 161, 161, 1),
    (113, 94, 17, 143, 18, 340, 54, 1, 197, 197, 1),
    (113, 94, 17, 143, 18, 340, 54, 1, 197, 197, 1),
    (112, 93, 17, 146, 16, 340, 48, 1, 194, 194, 1),
    (108, 90, 15, 140, 16, 328, 48, 2, 188, 188, 2),
)


def test_scripted_leg_level_2_both_sources():
    want_n2 = (0, 0, 2, 2, 2, 1)
    for args, n2 in zip(SCRIPTED, want_n2, strict=True):
        b = _build(*args[:10], n1c=args[10])
        for src in ("tag", "centroid"):
            ok, n2_gap, n2_pair, fails = check_build(b, GAP0_8, W0_8, src)
            assert ok, (src, fails)
            assert n2_gap == n2_pair == n2


def test_centroid_source_is_independent_of_the_tag():
    # a lost tag breaks the tag reading and leaves the centroid reading intact
    b = _build(113, 94, 17, 143, 18, 340, 54, 0, 197, 197, n1c=1)
    assert not check_build(b, GAP0_8, W0_8, "tag")[0]
    assert check_build(b, GAP0_8, W0_8, "centroid")[0]
    b = _build(*SCRIPTED[2][:10])
    ok, *_, fails = check_build(b, GAP0_8, W0_8, "centroid")
    assert not ok and fails == ["no centroid line"]


def test_main_centroid_option(tmp_path, capsys):
    log = tmp_path / "l2.log"
    log.write_text("".join(_log(*a[:10], n1c=a[10]) for a in SCRIPTED))
    assert main([str(log), "--nex", "8", "--ney", "8", "--n1-from", "centroid"]) == 0
    assert "N1 from centroid; 6 builds" in capsys.readouterr().out
    log.write_text(_log(*SCRIPTED[0][:10]))
    with pytest.raises(SystemExit, match="centroid jump"):
        main([str(log), "--nex", "8", "--ney", "8", "--n1-from", "centroid"])


def test_base_gap_follows_the_grid():
    # 8x8: gap0 = 81 - 64 = 17, W0 = 16 -- the native 8x8 base build of 2026-10-17
    ok, n2_gap, n2_pair, fails = check_build(
        _build(81, 64, 16, 128, 0, 256, 0, 0, 128, 128), 17, 16)
    assert ok and n2_gap == n2_pair == 0
