"""Validate the YH d-orbital two-center local integrals (riYH) against the oracle.

Phase 3, d two-center integrals, YH case (d-atom A + H). The 9x9 electron-core
(e1b) block is: build the local-frame integrals (sp from two_center_integrals,
already ported; d from the riYH multipole formulas here, using the per-element d
charge separations baked from PYSEQM's pyseqm_d_params), then rotate to the
molecular frame and unpack.

This script reproduces the riYH local integrals (the NEW d physics) and checks
the full e1b bit-exact against the oracle's yh_e1b_contribution. The molecular
rotation (generate_rotation_matrix, ~256 LOC) is still taken from the oracle here
— porting it to scalar C++ is the remaining piece for a self-contained YH e1b.

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

EV = 27.21
EV1 = EV / 2.0
EV2 = EV / 4.0
ANG_TO_BOHR = 1.0 / 0.529167

# d charge separations (dp, ds, dorbdorb) + additive radii (rho3..rho6), baked
# per element from PYSEQM's pyseqm_d_params (Slater-Condon — element constants).
D_PARAMS = {
    15: (0.90744734, 1.38477069, 1.62554049, 0.27098714, 1.36916053, 0.38883255, 0.71981849),
    16: (0.49861628, 0.96176716, 0.64321070, 0.44863040, 1.89216748, 3.23024211, 0.48881126),
    17: (0.75100923, 1.10740952, 1.51053979, 0.30216047, 1.03731998, 2.34288535, 0.72430196),
}
PAIRS = [(16, 1.34), (15, 1.42), (17, 1.27)]


def main() -> int:
    mlx = os.environ.get("MLXMOLKIT")
    if not mlx or not os.path.isdir(mlx):
        sys.exit("set MLXMOLKIT to the cloned mlxmolkit package dir")
    for name in ("mlx", "mlx.core"):
        sys.modules[name] = types.ModuleType(name)
    sys.modules["mlx"].core = sys.modules["mlx.core"]
    sys.path.insert(0, mlx)
    params = importlib.import_module("rm1.methods").get_params("PM6_D")
    tci = importlib.import_module("rm1.two_center_integrals")
    rmd = importlib.import_module("rm1.rotation_matrix_d")
    e1b_ref = importlib.import_module("rm1.tetci_yh").yh_e1b_contribution

    def my_e1b(za, cA, cB):
        pA, pB = params[za], params[1]
        R = float(np.linalg.norm(cB - cA))
        Rb = R * ANG_TO_BOHR
        da, qa, rho0A, _, _ = tci._compute_multipole_params(pA)
        _, _, rho0B, _, _ = tci._compute_multipole_params(pB)
        dpA, dsA, ddA, rho3A, rho4A, rho5A, rho6A = D_PARAMS[za]
        r_sdd0 = (rho3A + rho0B) ** 2
        r_spd = (rho4A + rho0B) ** 2
        r_ssd = (rho5A + rho0B) ** 2
        r_sdd = (rho6A + rho0B) ** 2
        ddq = EV / math.sqrt(Rb ** 2 + r_sdd0)
        dpuz = EV1 / math.sqrt((Rb + dpA) ** 2 + r_spd) - EV1 / math.sqrt((Rb - dpA) ** 2 + r_spd)
        ddqd = (EV2 / math.sqrt((Rb - ddA) ** 2 + r_sdd) + EV2 / math.sqrt((Rb + ddA) ** 2 + r_sdd)
                - EV1 / math.sqrt(Rb ** 2 + ddA ** 2 + r_sdd))
        dsq = (EV2 / math.sqrt((Rb - dsA) ** 2 + r_ssd) + EV2 / math.sqrt((Rb + dsA) ** 2 + r_ssd)
               - EV1 / math.sqrt(Rb ** 2 + dsA ** 2 + r_ssd))
        riyh = np.zeros(45)
        riyh[10] = dsq * 1.154701
        riyh[11] = dpuz * 1.154701
        riyh[14] = ddq + ddqd * 1.333333
        riyh[17] = dpuz
        riyh[20] = ddq + ddqd * 0.666667
        riyh[44] = ddq + ddqd * -1.333333
        ri_xh, _, _ = tci.two_center_integrals(pA, pB, R)
        cl = np.zeros((1, 46))
        cl[0, 1] = ri_xh[0]; cl[0, 3] = ri_xh[3]; cl[0, 7] = ri_xh[1]; cl[0, 10] = ri_xh[2]
        cl[0, 15] = riyh[44]; cl[0, 17] = riyh[17]; cl[0, 21] = riyh[20]
        cl[0, 22] = riyh[10]; cl[0, 25] = riyh[11]; cl[0, 28] = riyh[14]
        xij = (cB - cA)[None, :]
        xij = xij / np.linalg.norm(xij)
        cm = rmd.rotate_core(cl[:, 1:], rmd.generate_rotation_matrix(xij), 3)[0]
        idx = [0, 1, 3, 6, 10, 15, 21, 28, 36]
        W = np.zeros((9, 9))
        for i in range(9):
            for j in range(i + 1):
                W[i, j] = cm[idx[i] + j]
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

    print(f"\nYH riYH worst |d| = {worst:.2e}  "
          f"({'bit-exact vs oracle' if fails == 0 else f'{fails} FAILED'})")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
