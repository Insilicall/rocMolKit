"""Freeze PM6_D Mulliken charges for YH-scope molecules from the oracle integrals.

Generalizes tools/semiempirical/validate_pm6d.py (which proved H2S bit-exact) to
an arbitrary molecule whose only heavy atom carries d-orbitals and whose other
atoms are hydrogens (H2S, PH3, HCl, ...). Assembles the 9-orbital NDDO/PM6_D SCF
from the oracle's integral functions (overlap_d_molecular_frame, tetci_yh e1b/W,
two_center_integrals, rotation) plus the baked one-center d W integrals + maps
(data/golden_w_onecenter_d.json), runs it to convergence, and writes the
converged per-atom charges to data/golden_pm6d_charges.json. This is the par-by-
par anchor the C++ PM6_D engine validates against — every integral here is
already ported to C++ and validated bit-exact against the oracle.

    MLXMOLKIT=/tmp/mlxmolkit_inspect/mlxmolkit \\
      /tmp/semienv/bin/python tools/semiempirical/gen_pm6d_golden.py
"""

from __future__ import annotations

import importlib
import json
import os
import sys
import types
from pathlib import Path

import numpy as np

WDATA = Path(__file__).resolve().parent / "data" / "golden_w_onecenter_d.json"
OUT = Path(__file__).resolve().parent / "data" / "golden_pm6d_charges.json"

# YH-scope molecules: one d-atom + hydrogens. Geometries in Angstrom.
MOLECULES = [
    ("H2S", [16, 1, 1],
     [[0.0, 0.0, 0.0], [0.9686, 0.0, 0.9269], [-0.9686, 0.0, 0.9269]]),
    ("PH3", [15, 1, 1, 1],
     [[0.0, 0.0, 0.0], [1.1932, 0.0, 0.7700], [-0.5966, 1.0333, 0.7700],
      [-0.5966, -1.0333, 0.7700]]),
    ("HCl", [17, 1],
     [[0.0, 0.0, 0.0], [0.0, 0.0, 1.2746]]),
]


def main() -> int:
    mlx = os.environ.get("MLXMOLKIT")
    if not mlx or not os.path.isdir(mlx):
        sys.exit("set MLXMOLKIT to the cloned mlxmolkit package dir")
    for name in ("mlx", "mlx.core"):
        sys.modules[name] = types.ModuleType(name)
    sys.modules["mlx"].core = sys.modules["mlx.core"]
    sys.path.insert(0, mlx)
    methods = importlib.import_module("rm1.methods")
    ovd = importlib.import_module("rm1.overlap_d")
    tyh = importlib.import_module("rm1.tetci_yh")
    tci = importlib.import_module("rm1.two_center_integrals")
    rot = importlib.import_module("rm1.rotation")
    P = methods.get_params("PM6_D")

    wd = json.loads(WDATA.read_text())
    m = wd["maps"]
    TI, TJ, WT, FL = m["TRIL_I"], m["TRIL_J"], m["WEIGHT_45"], m["FLOCAL"]
    WInt = {int(k): np.array(v) for k, v in wd["W"].items()}
    norb_of = {1: 1, 15: 9, 16: 9, 17: 9}
    valence = {1: 1, 15: 5, 16: 6, 17: 7}

    def beta(p, o):
        return p.beta_s if o == 0 else (p.beta_p if o < 4 else p.beta_d)

    def diagv(p, no):
        return [p.Uss] if no == 1 else [p.Uss] + [p.Upp] * 3 + [p.Udd] * 5

    def run(Z, coords):
        coords = np.asarray(coords, dtype=np.float64)
        n = len(Z)
        norb = [norb_of[z] for z in Z]
        starts, off = [], 0
        for no in norb:
            starts.append(off)
            off += no
        nb = off
        nocc = sum(valence[z] for z in Z) // 2

        H = np.zeros((nb, nb))
        for a in range(n):
            for o, val in enumerate(diagv(P[Z[a]], norb[a])):
                H[starts[a] + o, starts[a] + o] = val
        for i in range(n):
            for j in range(i + 1, n):
                S = np.asarray(ovd.overlap_d_molecular_frame(P[Z[i]], P[Z[j]], coords[i], coords[j]))
                for mo in range(norb[i]):
                    for no in range(norb[j]):
                        h = 0.5 * (beta(P[Z[i]], mo) + beta(P[Z[j]], no)) * S[mo, no]
                        H[starts[i] + mo, starts[j] + no] = h
                        H[starts[j] + no, starts[i] + mo] = h
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if norb[i] == 9 and norb[j] == 1:
                    e = np.asarray(tyh.yh_e1b_contribution(P[Z[i]], P[Z[j]], coords[i], coords[j]))
                    H[starts[i]:starts[i] + 9, starts[i]:starts[i] + 9] += e
                else:
                    R = float(np.linalg.norm(coords[j] - coords[i]))
                    ri, _, _ = tci.two_center_integrals(P[Z[i]], P[Z[j]], R)
                    H[starts[i], starts[i]] += -valence[Z[j]] * ri[0]

        def build_fock(Pm):
            F = H.copy()
            for a in range(n):
                p, s = P[Z[a]], starts[a]
                if norb[a] == 1:
                    F[s, s] += Pm[s, s] * p.gss * 0.5
                    continue
                Pss = Pm[s, s]
                Ppp = Pm[s + 1, s + 1] + Pm[s + 2, s + 2] + Pm[s + 3, s + 3]
                sp1, sp2 = p.gsp - 0.5 * p.hsp, 1.5 * p.hsp - 0.5 * p.gsp
                ppd, ppo = 1.25 * p.gp2 - 0.25 * p.gpp, 0.75 * p.gpp - 1.25 * p.gp2
                F[s, s] += Pss * p.gss * 0.5 + Ppp * sp1
                for k in range(1, 4):
                    pk = s + k
                    F[pk, pk] += Pss * sp1 + Pm[pk, pk] * p.gpp * 0.5 + (Ppp - Pm[pk, pk]) * ppd
                    F[s, pk] += Pm[s, pk] * sp2
                    F[pk, s] += Pm[pk, s] * sp2
                for k in range(1, 4):
                    for l in range(k + 1, 4):
                        F[s + k, s + l] += Pm[s + k, s + l] * ppo
                        F[s + l, s + k] += Pm[s + l, s + k] * ppo
                W = WInt[Z[a]]
                Pp = np.array([Pm[s + TI[k], s + TJ[k]] * WT[k] for k in range(45)])
                for col, wis, pis in FL:
                    fp = sum(W[wi] * Pp[pi] for wi, pi in zip(wis, pis))
                    i2, j2 = TI[col], TJ[col]
                    F[s + i2, s + j2] += fp
                    if i2 != j2:
                        F[s + j2, s + i2] += fp
            for i in range(n):
                for j in range(i + 1, n):
                    sA, sB = starts[i], starts[j]
                    dA = hB = None
                    if norb[i] == 9 and norb[j] == 1:
                        dA, hB = i, j
                    elif norb[i] == 1 and norb[j] == 9:
                        dA, hB = j, i
                    if dA is not None:
                        sD, sH = starts[dA], starts[hB]
                        W = np.asarray(tyh.yh_rotated_integral_matrix(P[Z[dA]], P[Z[hB]], coords[dA], coords[hB]))
                        for mu in range(9):
                            for nu in range(9):
                                F[sD + mu, sD + nu] += Pm[sH, sH] * W[mu, nu]
                        F[sH, sH] += sum(Pm[sD + mu, sD + nu] * W[mu, nu] for mu in range(9) for nu in range(9))
                        for mu in range(9):
                            ks = -0.5 * sum(W[mu, nu] * Pm[sD + nu, sH] for nu in range(9))
                            F[sD + mu, sH] += ks
                            F[sH, sD + mu] += ks
                    else:
                        w, _, _ = rot.rotate_integrals_to_molecular_frame(P[Z[i]], P[Z[j]], coords[i], coords[j])
                        F[sA, sA] += Pm[sB, sB] * w[0, 0, 0, 0]
                        F[sB, sB] += Pm[sA, sA] * w[0, 0, 0, 0]
                        kk = -0.5 * Pm[sA, sB] * w[0, 0, 0, 0]
                        F[sA, sB] += kk
                        F[sB, sA] += kk
            return F

        ev, C = np.linalg.eigh(H)
        Pm = sum(2 * np.outer(C[:, k], C[:, k]) for k in range(nocc))
        for _ in range(800):
            ev, C = np.linalg.eigh(build_fock(Pm))
            Pn = sum(2 * np.outer(C[:, k], C[:, k]) for k in range(nocc))
            if np.sqrt(np.mean((Pn - Pm) ** 2)) < 1e-10:
                Pm = Pn
                break
            Pm = 0.3 * Pn + 0.7 * Pm
        q = [valence[Z[a]] - float(np.trace(Pm[starts[a]:starts[a] + norb[a], starts[a]:starts[a] + norb[a]]))
             for a in range(n)]
        return q

    out = []
    for name, Z, coords in MOLECULES:
        q = run(Z, coords)
        out.append({"name": name, "atoms": Z, "coords": coords, "q": [round(x, 6) for x in q]})
        print(f"{name:5s} q = {[round(x, 4) for x in q]}")

    OUT.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {OUT.relative_to(Path(__file__).resolve().parent.parent.parent)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
