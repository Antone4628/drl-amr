"""Pins tools/gate_2d_permute_compare: a renumbered copy of a dump set must
compare as identical, and a changed entry must fail.

The synthetic case is two elements of a 3x3 LGL-like lattice that share an
edge, so three node pairs carry identical coordinates across the two elements
-- exactly the ambiguity that defeats a coordinate-only match and that the
per-element matching must resolve.
"""

import os

import numpy as np
import pytest

from tools.gate_2d_permute_compare import main


def _lattice(x0, y0, h, ngl):
    s = np.array([0.0, 0.5, 1.0]) if ngl == 3 else np.linspace(0.0, 1.0, ngl)
    pts = [(x0 + h * si, y0 + h * sj) for sj in s for si in s]
    return np.array(pts)


def _write(d, L, xy, M, lv, tag):
    os.makedirs(d, exist_ok=True)
    np.savetxt(os.path.join(d, f"GATE2D_operator_{tag}.txt"), L, fmt="%.17g")
    np.savetxt(os.path.join(d, f"GATE2D_nodes_{tag}.txt"), xy, fmt="%.17g")
    np.savetxt(os.path.join(d, f"GATE2D_mass_{tag}.txt"), M, fmt="%.17g")
    np.savetxt(os.path.join(d, f"GATE2D_adlvl_{tag}.txt"), lv, fmt="%d")


@pytest.fixture
def dumps(tmp_path):
    rng = np.random.default_rng(7)
    ngl, k, ne = 3, 9, 2
    n = ne * k
    xy = np.vstack([_lattice(0.0, 0.0, 1.0, ngl), _lattice(1.0, 0.0, 1.0, ngl)])
    L = rng.normal(size=(n, n))
    L[rng.random((n, n)) < 0.6] = 0.0
    M = rng.random(n) + 0.5
    lv = np.array([0, 1])
    tag = "nop2_2elem"
    ref = tmp_path / "ref"
    _write(ref, L, xy, M, lv, tag)

    # renumber: swap the two elements, and permute the nodes inside each
    pe = np.array([1, 0])
    pin = [rng.permutation(k), rng.permutation(k)]
    P = np.concatenate([pe[e] * k + pin[e] for e in range(ne)])  # new ip = P[ref ip]
    Q = np.argsort(P)                                             # ref ip = Q[new ip]
    new = tmp_path / "new"
    _write(new, L[np.ix_(Q, Q)], xy[Q], M[Q], lv[pe], tag)
    return str(ref), str(new), tag, L, xy, M, lv, Q


def test_renumbered_copy_passes(dumps, capsys):
    ref, new, *_ = dumps
    assert main(["--ref", ref, "--new", new, "--tol", "1e-12"]) == 0
    out = capsys.readouterr().out
    assert "permutation is non-trivial" in out
    assert "max |L_new[P,P] - L_ref| = 0.000e+00" in out
    assert "support identical" in out
    assert "0 element(s) differ" in out
    assert "VERDICT: PASS" in out


def test_identity_is_reported(dumps, capsys):
    ref, *_ = dumps
    assert main(["--ref", ref, "--new", ref, "--tol", "1e-12"]) == 0
    assert "permutation is the identity" in capsys.readouterr().out


def test_changed_entry_fails(dumps, capsys):
    ref, new, tag, L, xy, M, lv, Q = dumps
    Lq = L[np.ix_(Q, Q)].copy()
    Lq[3, 5] += 1e-6
    bad = os.path.join(os.path.dirname(new), "bad")
    _write(bad, Lq, xy[Q], M[Q], lv[[1, 0]], tag)
    assert main(["--ref", ref, "--new", bad, "--tol", "1e-10"]) == 1
    assert "VERDICT: FAIL" in capsys.readouterr().out


def test_level_mismatch_fails(dumps, capsys):
    ref, new, tag, L, xy, M, lv, Q = dumps
    bad = os.path.join(os.path.dirname(new), "badlvl")
    _write(bad, L[np.ix_(Q, Q)], xy[Q], M[Q], np.array([0, 0]), tag)
    assert main(["--ref", ref, "--new", bad, "--tol", "1e-12"]) == 1
    assert "1 element(s) differ" in capsys.readouterr().out
