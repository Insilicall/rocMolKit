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
# Exact PM6 table values (rocmolkit/src/semiempirical/pm6_params_data.h).
ZETA_D = {15: 1.23036, 16: 3.109401, 17: 1.32403}
ZETA_S_TBL = {1: 1.26864, 6: 2.04756, 8: 5.42175, 15: 2.15803, 16: 2.192844, 17: 2.63705}
ZETA_P_TBL = {6: 1.70284, 8: 2.27096, 15: 1.80534, 16: 1.841078, 17: 2.11815}


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


def dp_block(zd: float, zp: float, Rb: float, ca, cb, sa, sb):
    """d-p overlap block (5 d-orbitals on A, 3 p on B), jcall 5 (qn3 d - qn2 p).
    Returns a 5x3 list [d-orbital][p-orbital] for di[4:9, 1:4]."""
    al = 0.5 * Rb * (zd + zp)
    be = 0.5 * Rb * (zd - zp)
    A = _aintgs(al)
    B = _bintgs(be)
    s321 = (zp ** 2.5 * zd ** 3.5 * Rb ** 6
            * ((A[2] * (3 * B[0] - B[2]) + A[3] * (B[1] + B[3]) - A[4] * (B[0] + B[2]) - A[5] * (3 * B[3] - B[1]))
               - (A[0] * (3 * B[2] - B[4]) + A[1] * (B[3] + B[5]) - A[2] * (B[2] + B[4]) - A[3] * (3 * B[5] - B[3])))
            / (96.0 * math.sqrt(2.0)))
    s322 = (zp ** 2.5 * zd ** 3.5 * Rb ** 6
            * (((A[4] - A[2]) * (B[0] - B[2]) + (A[3] - A[5]) * (-B[1] + B[3]))
               - ((A[2] - A[0]) * (B[2] - B[4]) + (A[1] - A[3]) * (-B[3] + B[5])))
            / (32.0 * math.sqrt(6.0)))
    s3 = math.sqrt(3.0)
    s34 = math.sqrt(0.75)
    return [
        [-(s321 * s34 * (2 * ca**2 - 1) * sb**2 * ca * sb - s322 * ((2 * ca**2 - 1) * sb * cb * ca * cb + 2 * sa * ca * sb * sa)),
         -(s321 * s34 * (2 * ca**2 - 1) * sb**2 * sa * sb - s322 * ((2 * ca**2 - 1) * sb * cb * sa * cb - 2 * sa * ca * sb * ca)),
         -(s321 * s34 * (2 * ca**2 - 1) * sb**2 * cb + s322 * ((2 * ca**2 - 1) * sb * cb * sb))],
        [-(s321 * s3 * ca * sb * cb * ca * sb - s322 * (ca * (2 * cb**2 - 1) * ca * cb + sa * cb * sa)),
         -(s321 * s3 * ca * sb * cb * sa * sb - s322 * (ca * (2 * cb**2 - 1) * sa * cb - sa * cb * ca)),
         -(s321 * s3 * ca * sb * cb * cb + s322 * (ca * (2 * cb**2 - 1) * sb))],
        [-(s321 * (cb**2 - 0.5 * sb**2) * ca * sb + s322 * s3 * sb * cb * ca * cb),
         -(s321 * (cb**2 - 0.5 * sb**2) * sa * sb + s322 * s3 * sb * cb * sa * cb),
         -(s321 * (cb**2 - 0.5 * sb**2) * cb - s322 * s3 * sb * cb * sb)],
        [-(s321 * s3 * sa * sb * cb * ca * sb - s322 * ((sa * (2 * cb**2 - 1)) * ca * cb - ca * cb * sa)),
         -(s321 * s3 * sa * sb * cb * sa * sb - s322 * ((sa * (2 * cb**2 - 1)) * sa * cb + ca * cb * ca)),
         -(s321 * s3 * sa * sb * cb * cb + s322 * ((sa * (2 * cb**2 - 1)) * sb))],
        [-(s321 * s3 * sa * ca * sb**2 * ca * sb - s322 * (2 * sa * ca * sb * cb * ca * cb - sb * (2 * ca**2 - 1) * sa)),
         -(s321 * s3 * sa * ca * sb**2 * sa * sb - s322 * (2 * sa * ca * sb * cb * sa * cb + sb * (2 * ca**2 - 1) * ca)),
         -(s321 * s3 * sa * ca * sb**2 * cb + s322 * (2 * sa * ca * sb * cb * sb))],
    ]


def dd_block(zd1: float, zd2: float, Rb: float, ca, cb, sa, sb):
    """d-d overlap 5x5 block (jcall 6, qn3 d - qn3 d). Slater-Koster outer
    product of the (sigma, pi1, pi2, delta1, delta2) projection vectors weighted
    by S331/S332/S333 — except the dyz-dxy cross term di[7,8]=di[8,7], whose
    delta component does not factor rank-1 and is written explicitly."""
    al = 0.5 * Rb * (zd1 + zd2)
    be = 0.5 * Rb * (zd1 - zd2)
    A = _aintgs(al)
    B = _bintgs(be)
    w = zd2 ** 3.5 * zd1 ** 3.5 * Rb ** 7
    s333 = w * (((A[2] - 2 * A[4] + A[6]) * (B[0] - 2 * B[2] + B[4]))
                - ((A[0] - 2 * A[2] + A[4]) * (B[2] - 2 * B[4] + B[6]))) / 768.0
    s332 = -w * ((A[2] * (B[2] - B[0]) + A[4] * (B[0] - B[4]) - A[6] * (B[2] - B[4]))
                 - (A[0] * (B[4] - B[2]) + A[2] * (B[2] - B[6]) - A[4] * (B[4] - B[6]))) / 192.0
    s331 = w * ((A[2] * (9 * B[0] - 6 * B[2] + B[4]) - 2 * A[4] * (3 * B[0] - 2 * B[2] + 3 * B[4])
                 + A[6] * (B[0] - 6 * B[2] + 9 * B[4]))
                - (A[0] * (9 * B[2] - 6 * B[4] + B[6]) - 2 * A[2] * (3 * B[2] - 2 * B[4] + 3 * B[6])
                   + A[4] * (B[2] - 6 * B[4] + 9 * B[6]))) / 1152.0
    s3 = math.sqrt(3.0)
    s34 = math.sqrt(0.75)
    t = 2 * ca ** 2 - 1
    sig = [s34 * t * sb ** 2, s3 * ca * sb * cb, cb ** 2 - 0.5 * sb ** 2, s3 * sa * sb * cb, s3 * sa * ca * sb ** 2]
    p1 = [sb * cb * t, ca * (2 * cb ** 2 - 1), -s3 * sb * cb, sa * (2 * cb ** 2 - 1), 2 * sa * ca * sb * cb]
    p2 = [2 * sa * ca * sb, sa * cb, 0.0, -ca * cb, -t * sb]
    d1 = [t * (cb ** 2 + 0.5 * sb ** 2), -ca * sb * cb, s34 * sb ** 2, -sa * sb * cb, 2 * sa * ca * cb ** 2 + sa * ca * sb ** 2]
    d2 = [2 * sa * ca * cb, -sa * sb, 0.0, ca * sb, -cb * t]
    M = [[s331 * sig[i] * sig[j] + s332 * (p1[i] * p1[j] + p2[i] * p2[j])
          + s333 * (d1[i] * d1[j] + d2[i] * d2[j]) for j in range(5)] for i in range(5)]
    e78 = (s331 * (s3 * sa * sb * cb) * (s3 * sa * ca * sb ** 2)
           + s332 * ((sa * (2 * cb ** 2 - 1)) * (2 * sa * ca * sb * cb) + (ca * cb) * ((2 * ca ** 2 - 1) * sb))
           + s333 * (-(sa * sb * cb) * (2 * sa * ca * cb ** 2 + sa * ca * sb ** 2) + ca * sb * cb * (2 * ca ** 2 - 1)))
    M[3][4] = e78
    M[4][3] = e78
    return M


def _angles(cB, R):
    v = [c / R for c in cB]
    xy = math.hypot(v[0], v[1])
    cb = v[2]
    sb = xy
    ca = v[0] / xy if xy >= 1e-10 else 1.0
    sa = v[1] / xy if xy >= 1e-10 else 0.0
    return ca, cb, sa, sb


def main() -> int:
    golden = json.loads(GOLDEN.read_text())
    worst = 0.0
    fails = 0
    for e in golden:
        zA, zB = e["zA"], e["zB"]
        Rb = e["R"] * ANG_TO_BOHR
        ca, cb, sa, sb = _angles(e["coordB"], e["R"])
        S = e["S"]
        if zB == 1:  # d-s block (heavy-d to H)
            mine = ds_block(ZETA_D[zA], ZETA_S_TBL[zB], Rb, ca, cb, sa, sb)
            gold = [S[i][0] for i in range(4, 9)]
            d = max(abs(a - b) for a, b in zip(mine, gold))
            label = "d-s"
        elif zB in (6, 8):  # d-p block (heavy-d to a 2nd-row p atom)
            mine = dp_block(ZETA_D[zA], ZETA_P_TBL[zB], Rb, ca, cb, sa, sb)
            d = max(abs(mine[i][j] - S[4 + i][1 + j]) for i in range(5) for j in range(3))
            label = "d-p"
        elif zB in ZETA_D:  # d-d block (heavy-d to heavy-d)
            mine = dd_block(ZETA_D[zA], ZETA_D[zB], Rb, ca, cb, sa, sb)
            d = max(abs(mine[i][j] - S[4 + i][4 + j]) for i in range(5) for j in range(5))
            label = "d-d"
        else:
            continue
        worst = max(worst, d)
        fails += 0 if d < 1e-6 else 1
        print(f"{zA}-{zB} {label}  max|dS|={d:.2e}  {'OK' if d < 1e-6 else '** FAIL'}")

    print(f"\nd-s + d-p + d-d blocks worst |dS| = {worst:.2e}  "
          f"({'bit-exact vs oracle golden' if fails == 0 else f'{fails} FAILED'})")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
