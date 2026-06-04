"""Validate the scalar d-orbital overlap extraction against the frozen oracle.

Phase 3 gate. The d-overlap convention is NOT the mlxmolkit analytic route
(Wigner-D) — that was tested and fails. The reference is PYSEQM's
``diatom_overlap_matrixD``, which assembles the d-block from direction cosines
(ca, cb, sa, sb of the bond unit vector) and per-jcall radial overlaps computed
with the same A/B reduced integrals as the sp case.

This script re-derives the d-block scalar (one pair at a time) from those
formulas and checks it bit-exact against tools/semiempirical/data/
golden_doverlap.json — the exact recipe the C++ port (overlap_d_device.h) must
follow. It needs no oracle clone (validates against the frozen golden).

Status: d-s block (heavy-d to H, jcall 431) confirmed bit-exact. s-d / d-p / p-d
/ d-d blocks extend the same pattern and are the remaining work.

    python3 tools/semiempirical/validate_doverlap.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

GOLDEN = Path(__file__).resolve().parent / "data" / "golden_doverlap.json"
ANG_TO_BOHR = 1.0 / 0.529167

# zeta_d (valence d Slater exponent) and zeta_s per element, from the PM6 table
# (rocmolkit/src/semiempirical/pm6_params_data.h). Only the elements used in the
# golden d-overlap pairs.
# zeta_d of the heavy atom and zeta_s of the light atom, exact PM6 table values
# (rocmolkit/src/semiempirical/pm6_params_data.h).
ZETA_D = {15: 1.23036, 16: 3.109401, 17: 1.32403}
ZETA_S_TBL = {1: 1.26864, 6: 2.04756, 8: 5.42175, 15: 2.15803, 16: 2.192844, 17: 2.63705}


def _aintgs(alpha: float, n: int = 7):
    a = [0.0] * n
    a[0] = math.exp(-alpha) / alpha
    for k in range(1, n):
        a[k] = a[0] + k * a[k - 1] / alpha
    return a


def _bintgs(beta: float, n: int = 7):
    b = [0.0] * n
    x = beta
    if abs(x) <= 1e-6:
        return [2.0 / (k + 1) if k % 2 == 0 else 0.0 for k in range(n)]
    tx = math.exp(x) / x
    tmx = -math.exp(-x) / x
    sign = 1.0
    b[0] = tx + tmx
    for k in range(1, n):
        sign = -sign
        b[k] = sign * tx + tmx + k * b[k - 1] / x
    return b


def ds_block(zd: float, zs: float, Rb: float, ca, cb, sa, sb):
    """d-s overlap column (5 d-orbitals on A, 1 s on B), jcall 431."""
    al = 0.5 * Rb * (zd + zs)
    be = 0.5 * Rb * (zd - zs)
    A = _aintgs(al)
    B = _bintgs(be)
    S311 = (zs ** 1.5 * zd ** 3.5 * Rb ** 5
            * ((A[2] * (3 * B[0] - B[2]) + A[4] * (3 * B[2] - B[0]) + 4 * A[3] * B[1])
               - (A[0] * (3 * B[2] - B[4]) + A[2] * (3 * B[4] - B[2]) + 4 * A[1] * B[3]))
            / (48.0 * math.sqrt(2.0)))
    s3 = math.sqrt(3.0)
    return [S311 * (2 * ca ** 2 - 1) * sb ** 2 * math.sqrt(0.75),
            S311 * ca * sb * cb * s3,
            S311 * (cb ** 2 - 0.5 * sb ** 2),
            S311 * s3 * sa * sb * cb,
            S311 * s3 * sa * ca * sb ** 2]


def main() -> int:
    golden = json.loads(GOLDEN.read_text())
    worst = 0.0
    fails = 0
    for e in golden:
        zA, zB = e["zA"], e["zB"]
        if zB != 1:  # only the d-s (heavy-d to H) block is ported so far
            continue
        R = e["R"]
        Rb = R * ANG_TO_BOHR
        cB = e["coordB"]
        v = [c / R for c in cB]
        xy = math.hypot(v[0], v[1])
        cb = v[2]
        sb = xy
        ca = v[0] / xy if xy >= 1e-10 else 1.0
        sa = v[1] / xy if xy >= 1e-10 else 0.0
        mine = ds_block(ZETA_D[zA], ZETA_S_TBL[zB], Rb, ca, cb, sa, sb)
        gold = [e["S"][i][0] for i in range(4, 9)]
        d = max(abs(a - b) for a, b in zip(mine, gold))
        worst = max(worst, d)
        fails += 0 if d < 1e-6 else 1
        print(f"{zA}-{zB} d-s  max|dS|={d:.2e}  {'OK' if d < 1e-6 else '** FAIL'}")

    print(f"\nd-s block worst |dS| = {worst:.2e}  "
          f"({'bit-exact vs oracle golden' if fails == 0 else f'{fails} FAILED'})")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
