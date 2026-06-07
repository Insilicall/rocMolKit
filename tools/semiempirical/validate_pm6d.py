"""Validate the full PM6_D SCF for H2S end-to-end — the Phase 3 integration.

Assembles the 9-orbital (d-bearing) NDDO/PM6_D SCF from the validated pieces and
checks the converged Mulliken charge against the PYSEQM golden (q_S = -0.3617):

  H_core   : diagonal (Uss/Upp/Udd) + resonance (beta * d-overlap) + YH e1b
  Fock     : one-center sp + one-center d (baked W integrals + packing maps)
             + two-center d J/K (YH W: J on both atoms + the cross-atom K term —
             the piece tetci_yh.yh_fock leaves disabled, supplied here)
  SCF      : 9-orbital closed-shell loop with damped mixing

The d-overlap, YH e1b/W, multipole and two-center primitives are reused from the
oracle here (all already ported to C++ and validated bit-exact); the one-center d
W integrals + packing maps are baked into data/golden_w_onecenter_d.json. This
script proves the assembly + SCF recipe the C++ engine follows.

    MLXMOLKIT=/tmp/mlxmolkit_inspect/mlxmolkit \\
      /tmp/semienv/bin/python tools/semiempirical/validate_pm6d.py
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

    Z = [16, 1, 1]
    coords = np.array([[0, 0, 0.0], [0.9686, 0, 0.9269], [-0.9686, 0, 0.9269]])
    norb = {16: 9, 1: 1}
    nb, starts, nocc = 11, [0, 9, 10], 4

    def beta(p, o):
        return p.beta_s if o == 0 else (p.beta_p if o < 4 else p.beta_d)

    def diagv(p, no):
        return [p.Uss] if no == 1 else [p.Uss] + [p.Upp] * 3 + [p.Udd] * 5

    H = np.zeros((nb, nb))
    for a in range(3):
        for o, val in enumerate(diagv(P[Z[a]], norb[Z[a]])):
            H[starts[a] + o, starts[a] + o] = val
    for i in range(3):
        for j in range(i + 1, 3):
            S = np.asarray(ovd.overlap_d_molecular_frame(P[Z[i]], P[Z[j]], coords[i], coords[j]))
            for mo in range(norb[Z[i]]):
                for no in range(norb[Z[j]]):
                    h = 0.5 * (beta(P[Z[i]], mo) + beta(P[Z[j]], no)) * S[mo, no]
                    H[starts[i] + mo, starts[j] + no] = h
                    H[starts[j] + no, starts[i] + mo] = h
    for i in range(3):
        for j in range(3):
            if i == j:
                continue
            if norb[Z[i]] == 9 and Z[j] == 1:
                e = np.asarray(tyh.yh_e1b_contribution(P[Z[i]], P[Z[j]], coords[i], coords[j]))
                H[starts[i]:starts[i] + 9, starts[i]:starts[i] + 9] += e
            else:
                R = float(np.linalg.norm(coords[j] - coords[i]))
                ri, _, _ = tci.two_center_integrals(P[Z[i]], P[Z[j]], R)
                H[starts[i], starts[i]] += -P[Z[j]].n_valence * ri[0]

    def build_fock(Pm):
        F = H.copy()
        for a in range(3):
            p, s = P[Z[a]], starts[a]
            if norb[Z[a]] == 1:
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
        for i in range(3):
            for j in range(i + 1, 3):
                sA, sB = starts[i], starts[j]
                if norb[Z[i]] == 9 and Z[j] == 1:  # two-center d (YH): J on both + K
                    W = np.asarray(tyh.yh_rotated_integral_matrix(P[Z[i]], P[Z[j]], coords[i], coords[j]))
                    for mu in range(9):
                        for nu in range(9):
                            F[sA + mu, sA + nu] += Pm[sB, sB] * W[mu, nu]
                    F[sB, sB] += sum(Pm[sA + mu, sA + nu] * W[mu, nu] for mu in range(9) for nu in range(9))
                    for mu in range(9):
                        ks = -0.5 * sum(W[mu, nu] * Pm[sA + nu, sB] for nu in range(9))
                        F[sA + mu, sB] += ks
                        F[sB, sA + mu] += ks
                else:  # H-H sp
                    w, _, _ = rot.rotate_integrals_to_molecular_frame(P[Z[i]], P[Z[j]], coords[i], coords[j])
                    F[sA, sA] += Pm[sB, sB] * w[0, 0, 0, 0]
                    F[sB, sB] += Pm[sA, sA] * w[0, 0, 0, 0]
                    kk = -0.5 * Pm[sA, sB] * w[0, 0, 0, 0]
                    F[sA, sB] += kk
                    F[sB, sA] += kk
        return F

    ev, C = np.linalg.eigh(H)
    Pm = sum(2 * np.outer(C[:, k], C[:, k]) for k in range(nocc))
    it = 0
    for it in range(400):
        ev, C = np.linalg.eigh(build_fock(Pm))
        Pn = sum(2 * np.outer(C[:, k], C[:, k]) for k in range(nocc))
        if np.sqrt(np.mean((Pn - Pm) ** 2)) < 1e-9:
            Pm = Pn
            break
        Pm = 0.3 * Pn + 0.7 * Pm

    qS = 6.0 - float(np.trace(Pm[:9, :9]))
    gold = -0.3617
    d = abs(qS - gold)
    print(f"H2S PM6_D: q_S = {qS:.4f}  (golden {gold})  d = {d:.2e}  converged at it={it}")
    print("OK — full PM6_D SCF reproduces the d-orbital golden charge"
          if d < 1e-3 else "** FAIL")
    return 0 if d < 1e-3 else 1


if __name__ == "__main__":
    raise SystemExit(main())
