"""Dump the Python 1D DG reference operator for entrywise cross-validation.

Companion to tools/convergence_study.py (DG roadmap Phase 2F). Reuses that
module's verified `make_solver` so the discretization is identical to the one
the order study certified. `solver.Dhat` is the semi-discrete operator:
step() advances via R = Dhat @ q, so no probing is required.
"""

import argparse
import numpy as np

from tools.convergence_study import make_solver

def build_operator(icase=8, nop=4, nelem=50, nq_mode="collocated"):
    """Return (solver, L) where L is the assembled semi-discrete operator.

    Raises if the periodic wrap is welded: step() contains a conditional
    `qp[-1] = qp[0]` living OUTSIDE Dhat, so a clean operator cannot expose
    it -- the same defect shape found in Jexpresso 2026-07-27 (fixed in
    586e7ba3). It must be asserted separately, here, every time.
    """
    s = make_solver(icase, nop, nelem, nq_mode)
    if s.periodicity[-1] == s.periodicity[0]:
        raise AssertionError(
            "PERIODIC WRAP IS WELDED -- periodicity[0] == periodicity[-1], "
            "so step() would overwrite the last DOF with the first. The "
            "Python reference has the Jexpresso wrap defect."
        )
    return s, np.asarray(s.Dhat, dtype=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nop", type=int, default=4)
    ap.add_argument("--nelem", type=int, default=50)
    ap.add_argument("--nq-mode", choices=["overint", "collocated"],
                    default="collocated")
    ap.add_argument("--icase", type=int, default=8)
    ap.add_argument("--outdir", default="tools")
    a = ap.parse_args()

    
    s, L = build_operator(a.icase, a.nop, a.nelem, a.nq_mode)
    print(f"periodicity[0]  = {s.periodicity[0]}")
    print(f"periodicity[-1] = {s.periodicity[-1]}   welded = False")

    # step() contains `qp[-1] = qp[0]` guarded by this condition -- a welded
    # periodic wrap, the same defect found in Jexpresso 2026-07-27. For DG it
    # must be False. Asserted, not assumed: it lives outside Dhat, so a clean
    # operator would not expose it.


    print(f"Dhat shape = {L.shape}   npoin_dg = {s.npoin_dg}   "
          f"nelem*ngl = {s.nelem * s.ngl}")
    print(f"wave_speed = {s.wave_speed}   ngl = {s.ngl}   nq = {s.nq}")

    tag = f"nop{a.nop}_nelem{a.nelem}_{a.nq_mode}"
    np.savetxt(f"{a.outdir}/PY_operator_{tag}.txt", L)
    np.savetxt(f"{a.outdir}/PY_coords_{tag}.txt", s.coord)
    print(f"wrote {a.outdir}/PY_operator_{tag}.txt")
    print(f"wrote {a.outdir}/PY_coords_{tag}.txt")


if __name__ == "__main__":
    main()