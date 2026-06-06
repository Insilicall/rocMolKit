"""Validate aluminum (Al, Z=13) and silicon (Si, Z=14) PM6_D d-element halides
against MOPAC 23.2.5.

Al and Si are main-group d-elements (nOrb=9, qnD=3) that MOPAC PM6 supports but
PYSEQM/mlxmolkit do NOT (their PM6_D parameter set raises KeyError for Z=13/14).
The d-orbital machinery here was therefore only ever validated against the
PYSEQM-baked halogens/chalcogens (P/S/Cl/Br/I), and Al/Si silently failed
(pm6dCharges returned ok=0) because their one-center d two-electron W integrals
and their two-center d charge separations were never baked. Both are now derived
straight from MOPAC's PM6 CSV tail exponents (gen_onecenter_d_header.py /
dChargeSeparations), so the SCF converges to MOPAC-bit-exact Mulliken charges.

Only the charges are checked (Al/Si EISOL and the PWCCT core-core HoF reference
are uncalibrated, so the canonical HoF is reported as NaN -- see the report).

    MOPAC_DIR=/tmp/mopac_bin/mopac-23.2.5-linux \
      python3 tools/semiempirical/validate_pm6d_aluminum.py
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
DRV = "/tmp/pm6d_aluminum_drv"
WORK = tempfile.mkdtemp(prefix="pm6d_aluminum_")
SYM = {1: "H", 9: "F", 13: "Al", 14: "Si", 17: "Cl", 35: "Br"}

# name, [(Z, x, y, z), ...]  (closed-shell; geometry need not be optimal -- MOPAC
# and the engine use the same coordinates, so the SCF charges must still agree).
MOLS = [
    ("AlH3",  [(13, 0, 0, 0), (1, 0, 0, 1.59), (1, 1.50, 0, -0.79), (1, -1.50, 0, -0.79)]),
    ("AlF3",  [(13, 0, 0, 0), (9, 0, 0, 1.63), (9, 1.41, 0, -0.81), (9, -1.41, 0, -0.81)]),
    ("AlCl3", [(13, 0, 0, 0), (17, 0, 0, 2.07), (17, 1.79, 0, -1.03), (17, -1.79, 0, -1.03)]),
    ("AlBr3", [(13, 0, 0, 0), (35, 0, 0, 2.22), (35, 1.92, 0, -1.11), (35, -1.92, 0, -1.11)]),
    ("SiH4",  [(14, 0, 0, 0), (1, 0.86, 0.86, 0.86), (1, -0.86, -0.86, 0.86),
               (1, 0.86, -0.86, -0.86), (1, -0.86, 0.86, -0.86)]),
    ("SiCl4", [(14, 0, 0, 0), (17, 1.17, 1.17, 1.17), (17, -1.17, -1.17, 1.17),
               (17, 1.17, -1.17, -1.17), (17, -1.17, 1.17, -1.17)]),
]


def build_driver():
    cc = shutil.which("g++") or shutil.which("c++")
    if cc is None:
        sys.exit("need g++ to build the dump_charges driver")
    subprocess.run([cc, "-std=c++17", "-O2", f"-I{SRC}", str(HERE / "dump_charges.cpp"),
                    str(SRC / "scf_d.cpp"), str(SRC / "core_hamiltonian.cpp"),
                    str(SRC / "pm6_params.cpp"), str(SRC / "overlap.cpp"),
                    str(SRC / "two_center.cpp"), "-o", DRV], check=True)


def mopac(name, atoms):
    base = f"{WORK}/{name}"
    with open(base + ".mop", "w") as f:
        f.write(f"PM6 1SCF PRECISE AUX(PRECISION=10)\n{name}\n\n")
        for z, x, y, zc in atoms:
            f.write(f"{SYM[z]} {x:.5f} 1 {y:.5f} 1 {zc:.5f} 1\n")
    subprocess.run([MOPAC, base + ".mop"], env=dict(os.environ, LD_LIBRARY_PATH=LIB),
                   capture_output=True)
    aux = base + ".aux"
    if not os.path.exists(aux):
        return None
    t = open(aux).read()
    return [float(x) for x in
            re.search(r"ATOM_CHARGES\[\d+\]=\s*\n(.*?)\n\s*[A-Z_]+", t, re.S).group(1).split()]


def engine(atoms):
    stdin = f"{len(atoms)} 0 1\n" + "\n".join(f"{z} {x} {y} {zc}" for z, x, y, zc in atoms) + "\n"
    out = subprocess.run([DRV], input=stdin, capture_output=True, text=True).stdout
    m = re.search(r"ok=(\d+) hof_pyseqm=\S+ hof_pm6=\S+ q=([\-0-9. ]+)", out)
    if not m or m.group(1) == "0":
        return None
    return [float(x) for x in m.group(2).split()]


def main() -> int:
    if not Path(MOPAC).exists():
        sys.exit(f"set MOPAC_DIR (no mopac at {MOPAC})")
    build_driver()
    worst = 0.0
    print(f"{'molecule':9}{'dq_MOPAC':>10}")
    for name, atoms in MOLS:
        qe = engine(atoms)
        if qe is None:
            print(f"{name:9}   ** engine returned no result (ok=0)")
            return 1
        qm = mopac(name, atoms)
        if qm is None:
            print(f"{name:9}   ** MOPAC produced no AUX")
            return 1
        dq = max(abs(a - b) for a, b in zip(qe, qm))
        worst = max(worst, dq)
        print(f"{name:9}{dq:10.4f}")
    ok = worst < 1e-3
    print(f"\nworst |dq| = {worst:.4f}  "
          f"{'OK (Al/Si halides match MOPAC charges)' if ok else '** MISMATCH'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
