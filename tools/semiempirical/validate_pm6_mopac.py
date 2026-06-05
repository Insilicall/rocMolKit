"""Validate the canonical (MOPAC-aligned) PM6 heat of formation against MOPAC.

With the PWCCT core-core (bit-exact to MOPAC) and the MOPAC-calibrated per-element
reference, pm6dCharges' hofPm6Kcal is the canonical PM6 heat of formation. This
compares it to MOPAC 23.2.5's own FINAL HEAT OF FORMATION (frozen below). The
electronic SCF is PYSEQM PM6_D (charges/eigenvalues ~MOPAC), so the agreement is:

  * light + Br molecules: ~1 kcal/mol (tol 1.5);
  * iodine: looser (~kcal, PYSEQM's qn5 d-treatment diverges from MOPAC; tol 8);
  * IBr is excluded -- PYSEQM has an unphysical qn5 s-d overlap (S>1) there that
    the engine faithfully reproduces, so its energy is meaningless (MOPAC is fine).

    g++ -std=c++17 -O2 -I../../rocmolkit/src/semiempirical validate_pm6_mopac.py ...
    python3 tools/semiempirical/validate_pm6_mopac.py
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
SRC = ROOT / "rocmolkit" / "src" / "semiempirical"

# name, atoms, coords, MOPAC PM6 FINAL HEAT OF FORMATION (kcal/mol), tolerance.
CASES = [
    ("H2S", [16, 1, 1], [[0, 0, 0.0], [0.9686, 0, 0.9269], [-0.9686, 0, 0.9269]], -1.5451, 1.5),
    ("PH3", [15, 1, 1, 1], [[0, 0, 0], [1.1932, 0, 0.77], [-0.5966, 1.0333, 0.77],
                            [-0.5966, -1.0333, 0.77]], 7.4497, 1.5),
    ("HCl", [17, 1], [[0, 0, 0], [0, 0, 1.2746]], -31.7850, 1.5),
    ("H2CS", [16, 6, 1, 1], [[0, 0, 0], [0, 0, 1.61], [0, 0.9281, 2.1944],
                             [0, -0.9281, 2.1944]], 38.0768, 1.5),
    ("CH3Cl", [17, 6, 1, 1, 1], [[0, 0, 1.781], [0, 0, 0], [1.036, 0, -0.337],
                                 [-0.518, 0.897, -0.337], [-0.518, -0.897, -0.337]], -14.9318, 1.5),
    ("CH3SH", [16, 6, 1, 1, 1, 1], [[0, 0, 0], [1.819, 0, 0], [-0.14, 1.329, 0],
                                    [2.18, 1.028, 0], [2.18, -0.514, 0.89],
                                    [2.18, -0.514, -0.89]], -2.3590, 1.5),
    ("Cl2", [17, 17], [[0, 0, 0], [0, 0, 1.988]], -0.3996, 1.5),
    ("HSSH", [16, 16, 1, 1], [[0, 0, 0], [2.055, 0, 0], [-0.5, 1.23, 0],
                              [2.555, -0.5, 1.13]], 5.7598, 1.5),
    ("HBr", [35, 1], [[0, 0, 0], [0, 0, 1.41]], -15.2892, 1.5),
    ("Br2", [35, 35], [[0, 0, 0], [0, 0, 2.28]], 2.9297, 1.5),
    ("BrCl", [35, 17], [[0, 0, 0], [0, 0, 1.94]], 20.2277, 1.5),
    ("CHBrCl2", [6, 35, 17, 17, 1], [[0, 0, 0], [0, 0, 1.94], [1.7, 0, -0.7],
                                     [-0.85, 1.47, -0.7], [-0.6, -1.0, -0.4]], 3.2144, 1.5),
    # iodine: looser tolerance (PYSEQM qn5 vs MOPAC)
    ("HI", [53, 1], [[0, 0, 0], [0, 0, 1.609]], 2.2526, 1.5),
    ("CH3I", [53, 6, 1, 1, 1], [[0, 0, 2.139], [0, 0, 0], [1.028, 0, -0.363],
                                [-0.514, 0.89, -0.363], [-0.514, -0.89, -0.363]], 7.5565, 8.0),
    ("ICl", [53, 17], [[0, 0, 0], [0, 0, 2.32]], 3.1249, 8.0),
]


def main() -> int:
    lines = ["#include <cstdio>", "#include <cmath>", "#include <vector>",
             '#include "scf_d.h"', "using namespace nvMolKit::semiempirical;",
             "int main(){ int fail=0;"]
    for nm, Z, c, ref, tol in CASES:
        za = ",".join(str(z) for z in Z)
        ca = ",".join(repr(float(x)) for row in c for x in row)
        lines += [
            f"{{ int z[]={{{za}}}; double co[]={{{ca}}}; double q[{len(Z)}], hof=0;",
            f"  bool ok=pm6dCharges({len(Z)},z,co,q,nullptr,800,1e-10,&hof);",
            f"  double d=std::fabs(hof-({ref!r}));",
            f'  std::printf("%-8s ok=%d PM6=%.3f MOPAC=%.3f d=%.3f %s\\n","{nm}",ok,hof,'
            f'{ref!r},d, (ok&&d<{tol!r})?"OK":"** FAIL");',
            f"  if(!(ok&&d<{tol!r})) fail++; }}",
        ]
    lines += ['std::printf("\\n%s\\n", fail==0?"PM6 HoF matches MOPAC within tol":"** FAIL");',
              "return fail;}"]
    with tempfile.TemporaryDirectory() as td:
        cf = Path(td) / "t.cpp"
        cf.write_text("\n".join(lines))
        exe = Path(td) / "t"
        subprocess.run(
            ["g++", "-std=c++17", "-O2", f"-I{SRC}", str(cf),
             str(SRC / "core_hamiltonian.cpp"), str(SRC / "pm6_params.cpp"),
             str(SRC / "overlap.cpp"), str(SRC / "scf_d.cpp"), "-o", str(exe)], check=True)
        r = subprocess.run([str(exe)], capture_output=True, text=True)
        print(r.stdout.strip())
        return r.returncode


if __name__ == "__main__":
    raise SystemExit(main())
