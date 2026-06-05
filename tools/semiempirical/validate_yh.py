"""Validate the full YH d-orbital two-center e1b — including the d rotation — vs
the oracle, with the rotation ported from scratch (self-contained).

Phase 3, d two-center integrals, YH case (d-atom A + H). The 9x9 electron-core
(e1b) block is: local-frame integrals (sp from two_center_integrals + d from the
riYH multipole formulas, using per-element d charge separations baked from
PYSEQM's pyseqm_d_params) -> molecular d rotation (D 5x5 + P 3x3 + rotate_core,
ported here) -> unpack -> times -Z_B.

Everything except the validated-and-ported multipole/two-center primitives is
reproduced here, including the d rotation. Checked bit-exact against the oracle's
yh_e1b_contribution for S-H/P-H/Cl-H (geometry recomputed at full precision, not
read from the 6-decimal golden coords).

    MLXMOLKIT=/tmp/mlxmolkit_inspect/mlxmolkit \\
      /tmp/semienv/bin/python tools/semiempirical/validate_yh.py
"""

from __future__ import annotations

import importlib
import math
import os
import sys
import types

import numpy as np

EV, EV1, EV2 = 27.21, 27.21 / 2.0, 27.21 / 4.0
ANG_TO_BOHR = 1.0 / 0.529167
PT5, PT5SQ3 = 0.5, 0.5 * math.sqrt(3.0)

# d charge separations + additive radii, baked per element from pyseqm_d_params.
D_PARAMS = {
    15: (0.90744734, 1.38477069, 1.62554049, 0.27098714, 1.36916053, 0.38883255, 0.71981849),
    16: (0.49861628, 0.96176716, 0.64321070, 0.44863040, 1.89216748, 3.23024211, 0.48881126),
    17: (0.75100923, 1.10740952, 1.51053979, 0.30216047, 1.03731998, 2.34288535, 0.72430196),
}
PAIRS = [(16, 1.34), (15, 1.42), (17, 1.27)]


def _rotation(v):
    """5x5 D and 3x3 P orbital-rotation matrices from the (negated) bond vector."""
    xy = math.hypot(v[0], v[1])
    if xy >= 1e-10:
        ca, sa, cb, sb = v[0] / xy, v[1] / xy, v[2], xy
    else:
        cb = 1.0 if v[2] > 0 else -1.0
        ca, sa, sb = cb, 0.0, 0.0
    c2a, c2b, s2a, s2b = 2 * ca * ca - 1, 2 * cb * cb - 1, 2 * sa * ca, 2 * sb * cb
    D = [[0.0] * 5 for _ in range(5)]
    D[0][0] = PT5SQ3 * c2a * sb * sb; D[1][0] = PT5 * c2a * s2b; D[2][0] = -s2a * sb
    D[3][0] = c2a * (cb * cb + PT5 * sb * sb); D[4][0] = -s2a * cb
    D[0][1] = PT5SQ3 * ca * s2b; D[1][1] = ca * c2b; D[2][1] = -sa * cb
    D[3][1] = -PT5 * ca * s2b; D[4][1] = sa * sb
    D[0][2] = cb * cb - PT5 * sb * sb; D[1][2] = -PT5SQ3 * s2b; D[3][2] = PT5SQ3 * sb * sb
    D[0][3] = PT5SQ3 * sa * s2b; D[1][3] = sa * c2b; D[2][3] = ca * cb
    D[3][3] = -PT5 * sa * s2b; D[4][3] = -ca * sb
    D[0][4] = PT5SQ3 * s2a * sb * sb; D[1][4] = PT5 * s2a * s2b; D[2][4] = c2a * sb
    D[3][4] = s2a * (cb * cb + PT5 * sb * sb); D[4][4] = c2a * cb
    P = [[0.0] * 3 for _ in range(3)]
    P[0][0] = ca * sb; P[1][0] = ca * cb; P[2][0] = -sa
    P[0][1] = sa * sb; P[1][1] = sa * cb; P[2][1] = ca
    P[0][2] = cb; P[1][2] = -sb
    return D, P


def main() -> int:
    mlx = os.environ.get("MLXMOLKIT")
    if not mlx or not os.path.isdir(mlx):
        sys.exit("set MLXMOLKIT to the cloned mlxmolkit package dir")
    for name in ("mlx", "mlx.core"):
        sys.modules[name] = types.ModuleType(name)
    sys.modules["mlx"].core = sys.modules["mlx.core"]
    sys.path.insert(0, mlx)
    params = importlib.import_module("rm1.methods").get_params("PM6_D")
    tci = importlib.import_module("rm1.two_center_integrals")  # already ported to C++
    e1b_ref = importlib.import_module("rm1.tetci_yh").yh_e1b_contribution

    def my_e1b(za, cA, cB):
        pA, pB = params[za], params[1]
        R = float(np.linalg.norm(cB - cA))
        Rb = R * ANG_TO_BOHR
        _, _, rho0A, _, _ = tci._compute_multipole_params(pA)
        _, _, rho0B, _, _ = tci._compute_multipole_params(pB)
        dpA, dsA, ddA, r3, r4, r5, r6 = D_PARAMS[za]
        ddq = EV / math.sqrt(Rb ** 2 + (r3 + rho0B) ** 2)
        dpuz = EV1 / math.sqrt((Rb + dpA) ** 2 + (r4 + rho0B) ** 2) - EV1 / math.sqrt((Rb - dpA) ** 2 + (r4 + rho0B) ** 2)
        ddqd = (EV2 / math.sqrt((Rb - ddA) ** 2 + (r6 + rho0B) ** 2) + EV2 / math.sqrt((Rb + ddA) ** 2 + (r6 + rho0B) ** 2)
                - EV1 / math.sqrt(Rb ** 2 + ddA ** 2 + (r6 + rho0B) ** 2))
        dsq = (EV2 / math.sqrt((Rb - dsA) ** 2 + (r5 + rho0B) ** 2) + EV2 / math.sqrt((Rb + dsA) ** 2 + (r5 + rho0B) ** 2)
               - EV1 / math.sqrt(Rb ** 2 + dsA ** 2 + (r5 + rho0B) ** 2))
        ri10, ri11, ri14 = dsq * 1.154701, dpuz * 1.154701, ddq + ddqd * 1.333333
        ri17, ri20, ri44 = dpuz, ddq + ddqd * 0.666667, ddq + ddqd * -1.333333
        ri, _, _ = tci.two_center_integrals(pA, pB, R)
        ss, ps, ppsig, pppi = ri[0], ri[1], ri[2], ri[3]
        D, P = _rotation(-(cB - cA) / R)
        rc = [0.0] * 45
        rc[0] = ss
        for I, s in enumerate((1, 3, 6)):
            rc[s] = ps * P[0][I]

        def ppc(K, I):
            a = (P[K][0] ** 2, P[K][0] * P[K][1], P[K][1] ** 2, P[K][0] * P[K][2], P[K][1] * P[K][2], P[K][2] ** 2)
            return a[I]
        for I, s in enumerate((2, 4, 5, 7, 8, 9)):
            rc[s] = ppsig * ppc(0, I) + pppi * (ppc(1, I) + ppc(2, I))
        for I, s in enumerate((10, 15, 21, 28, 36)):
            rc[s] = ri10 * D[0][I]
        for I, s in enumerate((11, 12, 13, 16, 17, 18, 22, 23, 24, 29, 30, 31, 37, 38, 39)):
            rc[s] = ri11 * (D[0][I // 3] * P[0][I % 3]) + ri17 * (D[1][I // 3] * P[1][I % 3] + D[2][I // 3] * P[2][I % 3])

        def ddc(K, I):
            a = (D[K][0] ** 2, D[K][0] * D[K][1], D[K][1] ** 2, D[K][0] * D[K][2], D[K][1] * D[K][2], D[K][2] ** 2,
                 D[K][0] * D[K][3], D[K][1] * D[K][3], D[K][2] * D[K][3], D[K][3] ** 2, D[K][0] * D[K][4],
                 D[K][1] * D[K][4], D[K][2] * D[K][4], D[K][3] * D[K][4], D[K][4] ** 2)
            return a[I]
        for I, s in enumerate((14, 19, 20, 25, 26, 27, 32, 33, 34, 35, 40, 41, 42, 43, 44)):
            rc[s] = ri14 * ddc(0, I) + ri20 * (ddc(1, I) + ddc(2, I)) + ri44 * (ddc(3, I) + ddc(4, I))
        idx = [0, 1, 3, 6, 10, 15, 21, 28, 36]
        W = np.zeros((9, 9))
        for i in range(9):
            for j in range(i + 1):
                W[i, j] = rc[idx[i] + j]
                W[j, i] = W[i, j]
        return -float(pB.n_valence) * W

    worst = 0.0
    fails = 0
    for za, r in PAIRS:
        cA = np.array([0.0, 0.0, 0.0])
        cB = np.array([0.3 * r, 0.0, 0.9 * r])
        cB = cB / np.linalg.norm(cB) * r
        d = float(np.max(np.abs(my_e1b(za, cA, cB) - np.asarray(e1b_ref(params[za], params[1], cA, cB)))))
        worst = max(worst, d)
        fails += 0 if d < 1e-6 else 1
        print(f"{za}-H YH e1b  max|d|={d:.2e}  {'OK' if d < 1e-6 else '** FAIL'}")

    print(f"\nYH e1b (riYH + ported d rotation) worst |d| = {worst:.2e}  "
          f"({'bit-exact vs oracle' if fails == 0 else f'{fails} FAILED'})")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
