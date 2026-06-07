"""Validate the canonical PM6 heat of formation (hof_pm6) for the main-group/metal
elements added on top of the PWCCT core-core port: B, Al, Si, Zn, Ga, Ge, Cd, Sn, Hg.

The canonical HoF is HoF = kEvToKcal*(E_elec + E_core_PWCCT) - sum kHofRef[Z]. The
PWCCT core-core for these elements is sourced bit-exactly from MOPAC 23.2.5's own
PM6 parameters (PO9 monopole radius, guess1/2/3 Gaussian, per-pair alpb/xfac), and
each kHofRef[Z] is least-squares fit (with the light/halide refs held fixed) on the
compounds below. This checks, for 2-3 different compounds per element, that the
engine's hof_pm6 reproduces MOPAC's FINAL HEAT OF FORMATION to <=1.5 kcal/mol (the
repo-wide PM6-vs-MOPAC tolerance), i.e. that the added refs are transferable.

    MOPAC_DIR=/tmp/mopac_bin/mopac-23.2.5-linux \
      /tmp/semienv/bin/python tools/semiempirical/validate_pm6d_hof_pwcct.py
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
SRC = ROOT / "rocmolkit" / "src" / "semiempirical"
MOPAC_DIR = os.environ.get("MOPAC_DIR", "/tmp/mopac_bin/mopac-23.2.5-linux")
MOPAC = f"{MOPAC_DIR}/bin/mopac"
LIB = f"{MOPAC_DIR}/lib"
TOL = 1.5  # kcal/mol, repo-wide PM6-vs-MOPAC tolerance

SYM = {1: "H", 6: "C", 9: "F", 13: "Al", 14: "Si", 17: "Cl", 30: "Zn", 31: "Ga",
       32: "Ge", 35: "Br", 48: "Cd", 50: "Sn", 5: "B", 80: "Hg"}

# 2-3 compounds per added element (Z of the added element noted). Geometries are
# fixed single-point coordinates (Angstrom); MOPAC and the engine use the same ones.
CASES = {
    "ZnF2":  ([30, 9, 9], [[0, 0, 0], [0, 0, 1.75], [0, 0, -1.75]]),
    "ZnCl2": ([30, 17, 17], [[0, 0, 0], [0, 0, 2.07], [0, 0, -2.07]]),
    "ZnMe2": ([30, 6, 6, 1, 1, 1, 1, 1, 1],
              [[0, 0, 0], [0, 0, 1.95], [0, 0, -1.95], [0.51, 0.88, 2.34], [0.51, -0.88, 2.34],
               [-1.02, 0, 2.34], [0.51, 0.88, -2.34], [0.51, -0.88, -2.34], [-1.02, 0, -2.34]]),
    "CdCl2": ([48, 17, 17], [[0, 0, 0], [0, 0, 2.21], [0, 0, -2.21]]),
    "CdBr2": ([48, 35, 35], [[0, 0, 0], [0, 0, 2.37], [0, 0, -2.37]]),
    "HgCl2": ([80, 17, 17], [[0, 0, 0], [0, 0, 2.29], [0, 0, -2.29]]),
    "HgBr2": ([80, 35, 35], [[0, 0, 0], [0, 0, 2.41], [0, 0, -2.41]]),
    "AlF3":  ([13, 9, 9, 9], [[0, 0, 0], [1.63, 0, 0], [-0.815, 1.41, 0], [-0.815, -1.41, 0]]),
    "AlCl3": ([13, 17, 17, 17], [[0, 0, 0], [2.06, 0, 0], [-1.03, 1.78, 0], [-1.03, -1.78, 0]]),
    "SiH4":  ([14, 1, 1, 1, 1], [[0, 0, 0], [0.856, 0.856, 0.856], [-0.856, -0.856, 0.856],
                                 [-0.856, 0.856, -0.856], [0.856, -0.856, -0.856]]),
    "SiCl4": ([14, 17, 17, 17, 17], [[0, 0, 0], [1.18, 1.18, 1.18], [-1.18, -1.18, 1.18],
                                     [-1.18, 1.18, -1.18], [1.18, -1.18, -1.18]]),
    "GaCl3": ([31, 17, 17, 17], [[0, 0, 0], [2.1, 0, 0], [-1.05, 1.82, 0], [-1.05, -1.82, 0]]),
    "GaF3":  ([31, 9, 9, 9], [[0, 0, 0], [1.71, 0, 0], [-0.855, 1.48, 0], [-0.855, -1.48, 0]]),
    "GeCl4": ([32, 17, 17, 17, 17], [[0, 0, 0], [1.06, 1.06, 1.06], [-1.06, -1.06, 1.06],
                                     [-1.06, 1.06, -1.06], [1.06, -1.06, -1.06]]),
    "GeH4":  ([32, 1, 1, 1, 1], [[0, 0, 0], [0.88, 0.88, 0.88], [-0.88, -0.88, 0.88],
                                 [-0.88, 0.88, -0.88], [0.88, -0.88, -0.88]]),
    "SnCl4": ([50, 17, 17, 17, 17], [[0, 0, 0], [1.15, 1.15, 1.15], [-1.15, -1.15, 1.15],
                                     [-1.15, 1.15, -1.15], [1.15, -1.15, -1.15]]),
    "SnH4":  ([50, 1, 1, 1, 1], [[0, 0, 0], [0.99, 0.99, 0.99], [-0.99, -0.99, 0.99],
                                 [-0.99, 0.99, -0.99], [0.99, -0.99, -0.99]]),
    "BF3":   ([5, 9, 9, 9], [[0, 0, 0], [1.31, 0, 0], [-0.655, 1.13, 0], [-0.655, -1.13, 0]]),
    "BCl3":  ([5, 17, 17, 17], [[0, 0, 0], [1.74, 0, 0], [-0.87, 1.51, 0], [-0.87, -1.51, 0]]),
}


def build_driver(td: Path) -> Path:
    cc = os.environ.get("CXX", "g++")
    subprocess.run([cc, "-std=c++17", "-O2", f"-I{SRC}", str(HERE / "dump_charges.cpp"),
                    str(SRC / "scf_d.cpp"), str(SRC / "core_hamiltonian.cpp"),
                    str(SRC / "pm6_params.cpp"), str(SRC / "overlap.cpp"),
                    str(SRC / "two_center.cpp"), "-o", str(td / "drv")], check=True)
    return td / "drv"


def engine_hof(drv: Path) -> dict:
    stdin = []
    for nm, (Z, C) in CASES.items():
        stdin.append(str(len(Z)))
        for z, p in zip(Z, C):
            stdin.append(f"{z} {p[0]:.10f} {p[1]:.10f} {p[2]:.10f}")
    out = subprocess.run([str(drv)], input="\n".join(stdin) + "\n",
                         capture_output=True, text=True).stdout.splitlines()
    res = {}
    for nm, line in zip(CASES, out):
        m = re.search(r"ok=(\d+).*hof_pm6=(\S+)", line)
        res[nm] = (int(m.group(1)), float(m.group(2))) if m else (0, float("nan"))
    return res


def mopac_hof(env) -> dict:
    res = {}
    for nm, (Z, C) in CASES.items():
        geo = "\n".join(f"{SYM[z]} {p[0]:.6f} 0 {p[1]:.6f} 0 {p[2]:.6f} 0" for z, p in zip(Z, C))
        Path("/tmp/_hofpwcct.mop").write_text(f"PM6 1SCF PRECISE\n{nm}\n\n{geo}\n")
        subprocess.run([MOPAC, "/tmp/_hofpwcct.mop"], env=env, capture_output=True)
        txt = Path("/tmp/_hofpwcct.out").read_text()
        m = re.search(r"FINAL HEAT OF FORMATION =\s*([-0-9.]+)\s*KCAL", txt)
        res[nm] = float(m.group(1)) if m else float("nan")
    return res


def main() -> int:
    if not Path(MOPAC).is_file():
        sys.exit(f"set MOPAC_DIR (no mopac at {MOPAC})")
    env = dict(os.environ, LD_LIBRARY_PATH=f"{LIB}:" + os.environ.get("LD_LIBRARY_PATH", ""))
    with tempfile.TemporaryDirectory() as td:
        drv = build_driver(Path(td))
        eng = engine_hof(drv)
    mop = mopac_hof(env)
    worst = 0.0
    print(f"{'compound':8s} {'engine':>10} {'mopac':>10} {'dHoF':>8}")
    for nm in CASES:
        ok, h = eng[nm]
        if not ok:
            print(f"{nm:8s}  engine did NOT converge"); return 1
        d = h - mop[nm]
        worst = max(worst, abs(d))
        print(f"{nm:8s} {h:10.3f} {mop[nm]:10.3f} {d:+8.3f}")
    print(f"\nworst |dHoF| = {worst:.3f} kcal/mol  "
          f"{'OK (B/Al/Si/Zn/Ga/Ge/Cd/Sn/Hg HoF match MOPAC)' if worst <= TOL else '** FAIL'}")
    return 0 if worst <= TOL else 1


if __name__ == "__main__":
    raise SystemExit(main())
