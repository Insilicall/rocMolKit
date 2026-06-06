"""Validate the PM6_D ionic path (net charge != 0) against MOPAC 23.2.5.

pm6dCharges / scfBatchDGpu take a net molecular charge; the electron count is
sum(valence) - charge (so a cation removes electrons, an anion adds them) and the
converged Mulliken charges sum to that net charge. This builds the dump_charges
driver, runs the same closed-shell ions through MOPAC (CHARGE=...), and checks
the engine's charges + canonical heat of formation bit-close to MOPAC.

    MOPAC_DIR=/tmp/mopac_bin/mopac-23.2.5-linux \
      python3 tools/semiempirical/validate_pm6d_ions.py
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
DRV = "/tmp/pm6d_ions_drv"
WORK = tempfile.mkdtemp(prefix="pm6d_ions_")
SYM = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F", 16: "S", 17: "Cl", 35: "Br", 53: "I"}

# name, net charge, [(Z, x, y, z), ...]  (closed-shell ions)
IONS = [
    ("NH4+", 1, [(7, 0, 0, 0), (1, 0, 0, 1.02), (1, 0.96, 0, -0.34),
                 (1, -0.48, 0.83, -0.34), (1, -0.48, -0.83, -0.34)]),
    ("OH-", -1, [(8, 0, 0, 0), (1, 0, 0, 0.96)]),
    ("CN-", -1, [(6, 0, 0, 0), (7, 0, 0, 1.16)]),
    ("Cl-", -1, [(17, 0, 0, 0)]),
    ("CH3NH3+", 1, [(6, 0, 0, 0), (7, 0, 0, 1.50), (1, 1.02, 0, -0.36),
                    (1, -0.51, 0.88, -0.36), (1, -0.51, -0.88, -0.36),
                    (1, 0.48, 0.83, 1.86), (1, 0.48, -0.83, 1.86), (1, -0.96, 0, 1.86)]),
    ("HCOO-", -1, [(6, 0, 0, 0), (8, 1.02, 0, 0.60), (8, -1.02, 0, 0.60), (1, 0, 0, -1.10)]),
]


def build_driver():
    cc = shutil.which("g++") or shutil.which("c++")
    if cc is None:
        sys.exit("need g++ to build the dump_charges driver")
    subprocess.run([cc, "-std=c++17", "-O2", f"-I{SRC}", str(HERE / "dump_charges.cpp"),
                    str(SRC / "scf_d.cpp"), str(SRC / "core_hamiltonian.cpp"),
                    str(SRC / "pm6_params.cpp"), str(SRC / "overlap.cpp"),
                    str(SRC / "two_center.cpp"), "-o", DRV], check=True)


def mopac(name, charge, atoms):
    base = f"{WORK}/{name.replace('+', 'p').replace('-', 'm')}"
    with open(base + ".mop", "w") as f:
        f.write(f"PM6 1SCF CHARGE={charge} PRECISE AUX(PRECISION=10)\n{name}\n\n")
        for z, x, y, zc in atoms:
            f.write(f"{SYM[z]} {x:.5f} 1 {y:.5f} 1 {zc:.5f} 1\n")
    subprocess.run([MOPAC, base + ".mop"], env=dict(os.environ, LD_LIBRARY_PATH=LIB),
                   capture_output=True)
    aux = open(base + ".aux").read()
    q = [float(x) for x in re.search(r"ATOM_CHARGES\[\d+\]=\s*\n(.*?)\n\s*[A-Z_]+", aux, re.S).group(1).split()]
    hof = float(re.search(r"HEAT_OF_FORMATION:KCAL/MOL=([+\-0-9.DE]+)", aux).group(1).replace("D", "E"))
    return q, hof


def engine(charge, atoms):
    stdin = f"{len(atoms)} {charge}\n" + "\n".join(f"{z} {x} {y} {zc}" for z, x, y, zc in atoms) + "\n"
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
    print(f"{'ion':10}{'charge':>7}{'sum_q':>8}{'dq_MOPAC':>10}{'dHoF_MOPAC':>12}")
    for name, charge, atoms in IONS:
        qe, he = engine(charge, atoms)
        if qe is None:
            print(f"{name:10}{charge:>7}   ** engine returned no result")
            return 1
        qm, hm = mopac(name, charge, atoms)
        dq = max(abs(a - b) for a, b in zip(qe, qm))
        dh = abs(he - hm)
        worst_dq, worst_dh = max(worst_dq, dq), max(worst_dh, dh)
        print(f"{name:10}{charge:>7}{sum(qe):8.3f}{dq:10.4f}{dh:12.3f}")
    ok = worst_dq < 1e-3 and worst_dh < 1.5
    print(f"\nworst |dq|={worst_dq:.4f}  worst |dHoF|={worst_dh:.3f} kcal  "
          f"{'OK (ions match MOPAC)' if ok else '** MISMATCH'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
