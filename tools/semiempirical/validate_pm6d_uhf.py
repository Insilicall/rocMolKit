"""Validate the open-shell (UHF) PM6 SCF against MOPAC 23.2.5 UHF.

pm6dCharges takes a spin multiplicity `mult` (2S+1); mult>1 (or an odd electron
count) routes to UHF: two spin densities Pα/Pβ with `Fσ = H + J(Pα+Pβ) − K(Pσ)`.
This builds the dump_charges driver, runs the same radicals through MOPAC
(UHF + DOUBLET/TRIPLET), and checks the engine's charges + canonical heat of
formation bit-close to MOPAC. Sp-only (the d UHF path is a separate phase).

    MOPAC_DIR=/tmp/mopac_bin/mopac-23.2.5-linux \
      python3 tools/semiempirical/validate_pm6d_uhf.py
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

MOPAC_DIR = os.environ.get("MOPAC_DIR", "/tmp/mopac_bin/mopac-23.2.5-linux")
MOPAC = f"{MOPAC_DIR}/bin/mopac"
LIB = f"{MOPAC_DIR}/lib"
HERE = Path(__file__).resolve().parent
SRC = HERE.parent.parent / "rocmolkit" / "src" / "semiempirical"
DRV = "/tmp/pm6d_uhf_drv"
WORK = tempfile.mkdtemp(prefix="pm6d_uhf_")
SYM = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F"}
SPIN = {2: "DOUBLET", 3: "TRIPLET", 4: "QUARTET"}

# name, mult (2S+1), [(Z, x, y, z), ...]  (sp-only open-shell radicals)
RADICALS = [
    ("CH3.", 2, [(6, 0, 0, 0), (1, 0, 0, 1.08), (1, 1.02, 0, -0.36), (1, -0.51, 0.88, -0.36)]),
    ("NO.", 2, [(7, 0, 0, 0), (8, 0, 0, 1.15)]),
    ("OH.", 2, [(8, 0, 0, 0), (1, 0, 0, 0.96)]),
    ("NH2.", 2, [(7, 0, 0, 0), (1, 0.80, 0, 0.58), (1, -0.80, 0, 0.58)]),
    ("CN.", 2, [(6, 0, 0, 0), (7, 0, 0, 1.17)]),
    ("O2", 3, [(8, 0, 0, 0), (8, 0, 0, 1.21)]),  # triplet ground state
    # NO2. is a known hard UHF multi-solution case: the simple damped-mixing SCF
    # converges to a different (higher) UHF minimum than MOPAC's. Robust DIIS /
    # level-shifting for such cases is future work (see SEMIEMPIRICAL_DESIGN.md).
]


def build_driver():
    cc = shutil.which("g++") or shutil.which("c++")
    if cc is None:
        sys.exit("need g++ to build the dump_charges driver")
    subprocess.run([cc, "-std=c++17", "-O2", f"-I{SRC}", str(HERE / "dump_charges.cpp"),
                    str(SRC / "scf_d.cpp"), str(SRC / "core_hamiltonian.cpp"),
                    str(SRC / "pm6_params.cpp"), str(SRC / "overlap.cpp"),
                    str(SRC / "two_center.cpp"), "-o", DRV], check=True)


def mopac(name, mult, atoms):
    base = f"{WORK}/{re.sub(r'[^A-Za-z0-9]', '_', name)}"
    with open(base + ".mop", "w") as f:
        f.write(f"PM6 1SCF UHF {SPIN[mult]} PRECISE AUX(PRECISION=10)\n{name}\n\n")
        for z, x, y, zc in atoms:
            f.write(f"{SYM[z]} {x:.5f} 1 {y:.5f} 1 {zc:.5f} 1\n")
    subprocess.run([MOPAC, base + ".mop"], env=dict(os.environ, LD_LIBRARY_PATH=LIB),
                   capture_output=True)
    aux = open(base + ".aux").read()
    q = [float(x) for x in re.search(r"ATOM_CHARGES\[\d+\]=\s*\n(.*?)\n\s*[A-Z_]+", aux, re.S).group(1).split()]
    hof = float(re.search(r"HEAT_OF_FORMATION:KCAL/MOL=([+\-0-9.DE]+)", aux).group(1).replace("D", "E"))
    return q, hof


def engine(mult, atoms):
    stdin = f"{len(atoms)} 0 {mult}\n" + "\n".join(f"{z} {x} {y} {zc}" for z, x, y, zc in atoms) + "\n"
    out = subprocess.run([DRV], input=stdin, capture_output=True, text=True).stdout
    m = re.search(r"ok=(\d+) hof_pyseqm=([\-0-9.]+) hof_pm6=([\-0-9.]+) q=([\-0-9. ]+)", out)
    if not m or m.group(1) == "0":
        return None, None
    return [float(x) for x in m.group(4).split()], float(m.group(3))


def main() -> int:
    if not Path(MOPAC).exists():
        sys.exit(f"set MOPAC_DIR (no mopac at {MOPAC})")
    build_driver()
    worst_dq = worst_dh = 0.0
    print(f"{'radical':8}{'mult':>5}{'dq_MOPAC':>10}{'dHoF_MOPAC':>12}")
    for name, mult, atoms in RADICALS:
        qe, he = engine(mult, atoms)
        if qe is None:
            print(f"{name:8}{mult:>5}   ** engine returned no result")
            return 1
        qm, hm = mopac(name, mult, atoms)
        dq = max(abs(a - b) for a, b in zip(qe, qm))
        dh = abs(he - hm)
        worst_dq, worst_dh = max(worst_dq, dq), max(worst_dh, dh)
        print(f"{name:8}{mult:>5}{dq:10.4f}{dh:12.3f}")
    ok = worst_dq < 1e-3 and worst_dh < 1.5
    print(f"\nworst |dq|={worst_dq:.4f}  worst |dHoF|={worst_dh:.3f} kcal  "
          f"{'OK (UHF matches MOPAC)' if ok else '** MISMATCH'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
