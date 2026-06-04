"""Reproducible CPU validation of the semi-empirical engine vs the PYSEQM goldens.

Compiles validate_driver.cpp (plain g++, no HIP), feeds it the golden molecules,
and checks Mulliken charges and heat of formation against the frozen references:
  - Phase 1 (H/C/N/O/F): tools/semiempirical/data/golden_intermediates.json
  - Phase 2 (P/S/Cl):    tools/semiempirical/data/golden_pm6sp_charges.py

The engine is PM6 with sp orbitals (PM6_SP) and the PWCCT core-core repulsion, so
both charges and HoF match PM6_SP. The same __host__ __device__ math runs on the
GPU; this CPU check is the host-side reference. Run from anywhere:

    python3 tools/semiempirical/validate.py
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent.parent / "rocmolkit" / "src" / "semiempirical"
DATA = HERE / "data"

# Phase-1 sp molecules that the engine handles (the heavy d-bearing entries in
# golden_intermediates.json are PM6_D references, out of scope for the sp engine).
PHASE1 = ["H2O", "CH4", "NH3", "HF", "COC", "CNC"]

# H2O/CH4/NH3/HF/COC/CNC geometries (Angstrom) — must match the golden file.
PHASE1_GEOM = {
    "H2O": [(8, 0, 0, 0), (1, 0.7572, 0, 0.5868), (1, -0.7572, 0, 0.5868)],
    "CH4": [(6, 0, 0, 0), (1, 0.629, 0.629, 0.629), (1, -0.629, -0.629, 0.629),
            (1, -0.629, 0.629, -0.629), (1, 0.629, -0.629, -0.629)],
    "NH3": [(7, 0, 0, 0.122), (1, 0, 0.939, -0.286), (1, 0.813, -0.470, -0.286),
            (1, -0.813, -0.470, -0.286)],
    "HF": [(9, 0, 0, 0), (1, 0.917, 0, 0)],
    "COC": [(6, -1.165, 0, 0.183), (8, 0, 0, -0.616), (6, 1.165, 0, 0.183),
            (1, -2.020, 0, -0.498), (1, -1.225, 0.886, 0.821), (1, -1.225, -0.886, 0.821),
            (1, 2.020, 0, -0.498), (1, 1.225, 0.886, 0.821), (1, 1.225, -0.886, 0.821)],
    "CNC": [(6, -1.215, 0.085, 0), (7, 0, -0.612, 0), (6, 1.215, 0.085, 0),
            (1, 0, -1.236, 0.812), (1, -2.080, -0.587, 0), (1, -1.275, 0.726, 0.882),
            (1, -1.275, 0.726, -0.882), (1, 2.080, -0.587, 0), (1, 1.275, 0.726, 0.882),
            (1, 1.275, 0.726, -0.882)],
}


def _load_golden_pm6sp():
    spec = importlib.util.spec_from_file_location("g", DATA / "golden_pm6sp_charges.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.GOLDEN_PM6SP


def main() -> int:
    # Build the molecule list: (name, atoms[(Z,x,y,z)], charges, hof).
    mols = []
    inter = json.loads((DATA / "golden_intermediates.json").read_text())["molecules"]
    for name in PHASE1:
        e = inter[name]
        mols.append((name, PHASE1_GEOM[name], e["charges"], e["heat_of_formation_kcal"]))
    for name, z, coords, q, hof in _load_golden_pm6sp():
        atoms = [(z[i], *coords[i]) for i in range(len(z))]
        mols.append((name, atoms, q, hof))

    out = SRC.parent.parent.parent / "build_validate_driver"
    cmd = ["g++", "-std=c++17", "-O2", f"-I{SRC}", str(HERE / "validate_driver.cpp")] + \
        [str(SRC / f"{m}.cpp") for m in ("scf", "core_hamiltonian", "two_center", "overlap", "pm6_params")] + \
        ["-o", str(out)]
    subprocess.run(cmd, check=True)

    stdin = "".join(
        f"{len(atoms)}\n" + "".join(f"{a[0]} {a[1]} {a[2]} {a[3]}\n" for a in atoms)
        for _, atoms, _, _ in mols
    )
    res = subprocess.run([str(out)], input=stdin, capture_output=True, text=True, check=True)
    lines = [ln for ln in res.stdout.splitlines() if ln.strip()]
    out.unlink(missing_ok=True)

    worst_q = worst_h = 0.0
    fails = 0
    for (name, atoms, gq, ghof), line in zip(mols, lines):
        parts = line.split()
        conv = int(parts[0])
        hof = float(parts[1])
        q = [float(x) for x in parts[2:]]
        dq = max(abs(a - b) for a, b in zip(q, gq))
        dh = abs(hof - ghof)
        worst_q = max(worst_q, dq)
        worst_h = max(worst_h, dh)
        ok = conv == 1 and dq < 1e-4 and dh < 1e-2
        fails += 0 if ok else 1
        print(f"{name:7} conv={conv} max|dq|={dq:.2e} dHoF={dh:.2e} {'OK' if ok else '** FAIL'}")

    print(f"\n{len(mols)} molecules · worst |dq|={worst_q:.2e} · worst |dHoF|={worst_h:.2e} kcal/mol")
    print("ALL BIT-EXACT vs PYSEQM golden" if fails == 0 else f"{fails} FAILED")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
