"""Validate group-12 metal (Zn/Cd/Hg) halides and the formerly silently-wrong
high-qn-sp x d-ligand main-group cases against MOPAC 23.2.5.

These exercise the general MOPAC Slater overlap (overlap_mopac_device.h): the
metal-sp(qn 4/5/6) x ligand-d(qn 3) block and the qn>=4 sp x d-ligand block that
the old per-jcall overlap couldn't express. Only the Mulliken charges are checked
(group-12 EISOL is uncalibrated, so the canonical HoF is reported as NaN).

    MOPAC_DIR=/tmp/mopac_bin/mopac-23.2.5-linux \
      python3 tools/semiempirical/validate_pm6d_metals.py
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
DRV = "/tmp/pm6d_metals_drv"
WORK = tempfile.mkdtemp(prefix="pm6d_metals_")
SYM = {16: "S", 17: "Cl", 35: "Br", 30: "Zn", 48: "Cd", 80: "Hg", 31: "Ga", 13: "Al"}

# name, [(Z, x, y, z), ...]  (closed-shell; geometry need not be optimal -- MOPAC
# and the engine use the same coordinates, so the SCF charges must still agree).
MOLS = [
    ("ZnCl2", [(30, 0, 0, 0), (17, 0, 0, 2.07), (17, 0, 0, -2.07)]),
    ("ZnBr2", [(30, 0, 0, 0), (35, 0, 0, 2.21), (35, 0, 0, -2.21)]),
    ("CdCl2", [(48, 0, 0, 0), (17, 0, 0, 2.28), (17, 0, 0, -2.28)]),
    ("HgCl2", [(80, 0, 0, 0), (17, 0, 0, 2.25), (17, 0, 0, -2.25)]),
    ("GaCl3", [(31, 0, 0, 0), (17, 0, 0, 2.10), (17, 1.82, 0, -1.05), (17, -1.82, 0, -1.05)]),
    ("ZnClBr", [(30, 0, 0, 0), (17, 0, 0, 2.07), (35, 0, 0, -2.21)]),
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
            print(f"{name:9}   ** engine returned no result")
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
          f"{'OK (group-12 + high-qn-sp x d-ligand match MOPAC)' if ok else '** MISMATCH'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
