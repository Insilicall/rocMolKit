"""Validate the PM6 pairwise core-core (PWCCT) bit-close to MOPAC.

The canonical PM6 core-core (Stewart 2007 / MOPAC) is the Pairwise Core-Core Term
that our engine ports in pwcct_device.h. This checks pwcctCoreCoreDev against
MOPAC's own NUCLEAR-NUCLEAR REPULSION (the gold-standard PM6 reference), frozen
below from MOPAC 23.2.5 single-point PM6 runs (keyword: PM6 1SCF LARGE ENPART).
The two agree to MOPAC's printed precision (1e-4 eV), confirming the PWCCT port —
including the H-{N,O,C} and C-C special cases and the heavy/interhalide pairs.

Note: this is the *core-core* term only. Our electronic SCF still follows PYSEQM
PM6_D (charges/eigenvalues match MOPAC); the PWCCT term is what aligns the total
energy / heat of formation with canonical PM6.

    python3 tools/semiempirical/validate_pwcct.py
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
SRC = ROOT / "rocmolkit" / "src" / "semiempirical"

# name, atoms (Z), coords (Angstrom), MOPAC PM6 NUCLEAR-NUCLEAR REPULSION (eV).
CASES = [
    ("H2S", [16, 1, 1], [[0, 0, 0.0], [0.9686, 0, 0.9269], [-0.9686, 0, 0.9269]], 107.8434),
    ("PH3", [15, 1, 1, 1], [[0, 0, 0], [1.1932, 0, 0.77], [-0.5966, 1.0333, 0.77],
                            [-0.5966, -1.0333, 0.77]], 148.2526),
    ("HCl", [17, 1], [[0, 0, 0], [0, 0, 1.2746]], 63.0455),
    ("H2CS", [16, 6, 1, 1], [[0, 0, 0], [0, 0, 1.61], [0, 0.9281, 2.1944],
                             [0, -0.9281, 2.1944]], 329.5233),
    ("CH3Cl", [17, 6, 1, 1, 1], [[0, 0, 1.781], [0, 0, 0], [1.036, 0, -0.337],
                                 [-0.518, 0.897, -0.337], [-0.518, -0.897, -0.337]], 460.1314),
    ("Cl2", [17, 17], [[0, 0, 0], [0, 0, 1.988]], 302.4743),
    ("HBr", [35, 1], [[0, 0, 0], [0, 0, 1.41]], 53.8739),
    ("Br2", [35, 35], [[0, 0, 0], [0, 0, 2.28]], 243.1496),
    ("BrCl", [35, 17], [[0, 0, 0], [0, 0, 1.94]], 288.8301),
    ("ICl", [53, 17], [[0, 0, 0], [0, 0, 2.32]], 251.6721),
    ("CHBrCl2", [6, 35, 17, 17, 1], [[0, 0, 0], [0, 0, 1.94], [1.7, 0, -0.7],
                                     [-0.85, 1.47, -0.7], [-0.6, -1.0, -0.4]], 1312.9883),
    # Main-group / metal elements sourced from MOPAC's own PM6 params (PO9 monopole
    # radius, guess Gaussian, alpb/xfac). MOPAC PM6 LARGE ENPART NUCLEAR-NUCLEAR
    # REPULSION (eV) for the same single-point geometry.
    ("ZnF2", [30, 9, 9], [[0, 0, 0], [0, 0, 1.75], [0, 0, -1.75]], 377.2528),
    ("ZnCl2", [30, 17, 17], [[0, 0, 0], [0, 0, 2.07], [0, 0, -2.07]], 328.9389),
    ("CdCl2", [48, 17, 17], [[0, 0, 0], [0, 0, 2.21], [0, 0, -2.21]], 301.8059),
    ("CdBr2", [48, 35, 35], [[0, 0, 0], [0, 0, 2.37], [0, 0, -2.37]], 273.8646),
    ("HgCl2", [80, 17, 17], [[0, 0, 0], [0, 0, 2.29], [0, 0, -2.29]], 292.7479),
    ("HgBr2", [80, 35, 35], [[0, 0, 0], [0, 0, 2.41], [0, 0, -2.41]], 272.0942),
    ("AlF3", [13, 9, 9, 9], [[0, 0, 0], [1.63, 0, 0], [-0.815, 1.41, 0],
                             [-0.815, -1.41, 0]], 1103.0818),
    ("AlCl3", [13, 17, 17, 17], [[0, 0, 0], [2.06, 0, 0], [-1.03, 1.78, 0],
                                 [-1.03, -1.78, 0]], 914.9420),
    ("SiH4", [14, 1, 1, 1, 1], [[0, 0, 0], [0.856, 0.856, 0.856], [-0.856, -0.856, 0.856],
                                [-0.856, 0.856, -0.856], [0.856, -0.856, -0.856]], 141.2726),
    ("SiCl4", [14, 17, 17, 17, 17], [[0, 0, 0], [1.18, 1.18, 1.18], [-1.18, -1.18, 1.18],
                                     [-1.18, 1.18, -1.18], [1.18, -1.18, -1.18]], 1760.5832),
    ("GaCl3", [31, 17, 17, 17], [[0, 0, 0], [2.1, 0, 0], [-1.05, 1.82, 0],
                                 [-1.05, -1.82, 0]], 923.7378),
    ("GeCl4", [32, 17, 17, 17, 17], [[0, 0, 0], [1.06, 1.06, 1.06], [-1.06, -1.06, 1.06],
                                     [-1.06, 1.06, -1.06], [1.06, -1.06, -1.06]], 1986.7777),
    ("SnCl4", [50, 17, 17, 17, 17], [[0, 0, 0], [1.15, 1.15, 1.15], [-1.15, -1.15, 1.15],
                                     [-1.15, 1.15, -1.15], [1.15, -1.15, -1.15]], 1881.2774),
    ("BF3", [5, 9, 9, 9], [[0, 0, 0], [1.31, 0, 0], [-0.655, 1.13, 0],
                           [-0.655, -1.13, 0]], 1315.5449),
    ("BCl3", [5, 17, 17, 17], [[0, 0, 0], [1.74, 0, 0], [-0.87, 1.51, 0],
                              [-0.87, -1.51, 0]], 1049.5597),
]


def main() -> int:
    lines = ["#include <cstdio>", "#include <cmath>", '#include "pwcct_device.h"',
             "using namespace nvMolKit::semiempirical;", "int main(){ double worst=0;"]
    for nm, Z, c, ref in CASES:
        za = ",".join(str(z) for z in Z)
        ca = ",".join(repr(float(x)) for row in c for x in row)
        lines += [
            f"{{ int z[]={{{za}}}; double co[]={{{ca}}};",
            f"  double e=pwcctCoreCoreDev({len(Z)},z,co);",
            f"  double d=std::fabs(e-({ref!r})); worst=std::fmax(worst,d);",
            f'  std::printf("%-8s PWCCT=%.4f MOPAC=%.4f d=%.2e\\n","{nm}",e,{ref!r},d); }}',
        ]
    lines += ['std::printf("\\nworst |d|=%.2e eV %s\\n",worst,worst<2e-4?"OK":"** FAIL");',
              "return worst<2e-4?0:1;}"]
    with tempfile.TemporaryDirectory() as td:
        cf = Path(td) / "t.cpp"
        cf.write_text("\n".join(lines))
        exe = Path(td) / "t"
        subprocess.run(["g++", "-std=c++17", "-O2", f"-I{SRC}", str(cf),
                        str(SRC / "pm6_params.cpp"), "-o", str(exe)], check=True)
        r = subprocess.run([str(exe)], capture_output=True, text=True)
        print(r.stdout.strip())
        return r.returncode


if __name__ == "__main__":
    raise SystemExit(main())
