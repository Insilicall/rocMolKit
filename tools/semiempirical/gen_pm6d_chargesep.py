"""Print the two-center d charge separations (dp, ds, dd, rho3-6) for the PM6_D
d-elements, derived from MOPAC's PM6 tail exponents via PYSEQM's pyseqm_d_params.

These are the constants hand-pasted into dChargeSeparations() in
rocmolkit/src/semiempirical/two_center_d_device.h (the table is tiny, so it is
not auto-generated -- this script only documents the provenance and lets you
reproduce / extend it).

PYSEQM/mlxmolkit only define the tail exponents for P/S/Cl/Br/I, so for Al/Si
(which MOPAC PM6 supports but PYSEQM does not) we inject the s/p/d_orb_exp_tail
columns from data/pm6_params_mopac.csv into PM6_TAIL_EXPONENTS before calling the
oracle. For P/S/Cl this reproduces the already-baked values bit-for-bit (to the
1e-9 precision the table was stored at), which is the regression check.

    MLXMOLKIT=/tmp/mlxmolkit_inspect/mlxmolkit \
      python3 tools/semiempirical/gen_pm6d_chargesep.py
"""
from __future__ import annotations

import csv
import importlib.util
import os
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
CSV = HERE / "data" / "pm6_params_mopac.csv"

# qn_sp / qn_d for the supported d-elements (period 3 -> 3, period 4 -> 4, etc.).
ELEMS = {13: "Al", 14: "Si", 15: "P", 16: "S", 17: "Cl", 35: "Br", 53: "I"}


def load_tail():
    out = {}
    with open(CSV) as f:
        rows = list(csv.reader(f))
    hdr = [h.strip() for h in rows[0]]
    for r in rows[1:]:
        if not r or not r[0].strip():
            continue
        d = {h: v.strip() for h, v in zip(hdr, r)}
        z = int(d["N"])
        out[z] = {
            "zeta_s": float(d["zeta_s"]), "zeta_p": float(d["zeta_p"]),
            "zeta_d": float(d["zeta_d"]), "G2SD": float(d["G2SD"]),
            "tail": (float(d["s_orb_exp_tail"]), float(d["p_orb_exp_tail"]),
                     float(d["d_orb_exp_tail"])),
        }
    return out


def main() -> int:
    mlx = os.environ.get("MLXMOLKIT")
    if not mlx or not os.path.isdir(mlx):
        sys.exit("set MLXMOLKIT to the cloned mlxmolkit package dir")
    rm1 = Path(mlx) / "rm1"
    pkg = types.ModuleType("rm1")
    pkg.__path__ = [str(rm1)]
    sys.modules["rm1"] = pkg
    spec = importlib.util.spec_from_file_location(
        "rm1.tetci_multipole_pyseqm", str(rm1 / "tetci_multipole_pyseqm.py"))
    T = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(T)

    tail = load_tail()

    class A:
        pass

    print(f"{'Z':>3} {'sym':<4}  dp        ds        dd        "
          f"rho3      rho4      rho5      rho6")
    for z, sym in ELEMS.items():
        d = tail[z]
        a = A()
        a.Z = z
        a.zeta_s, a.zeta_p, a.zeta_d = d["zeta_s"], d["zeta_p"], d["zeta_d"]
        a.G2SD = d["G2SD"]
        # Feed MOPAC's CSV tail exponents (PYSEQM lacks Al/Si).
        T.PM6_TAIL_EXPONENTS[z] = d["tail"]
        r = T.pyseqm_d_params(a)
        print(f"{z:>3} {sym:<4}  "
              f"{r['dp']:.8f} {r['ds']:.8f} {r['dorbdorb']:.8f} "
              f"{r['rho3']:.8f} {r['rho4']:.8f} {r['rho5']:.8f} {r['rho6']:.8f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
