"""Validate the PM6-D3H4 post-SCF correction bit-exact against the oracle.

PM6-D3H4 adds, on top of the NDDO heat of formation, three empirical corrections
(kcal/mol): Grimme D3 dispersion (e_disp; with s8=0 only the C6/e6 term), the H4
hydrogen-bond term (e_hb, over donor-H...acceptor N/O triples) and a short-range
H-H repulsion (e_hh). This compares the C++ port (d3_device.h / h4_device.h)
against mlxmolkit's rm1.pm6_d3h4 over a set that exercises each term, including
real hydrogen bonds.

    MLXMOLKIT=/tmp/mlxmolkit_inspect/mlxmolkit \\
      /tmp/semienv/bin/python tools/semiempirical/validate_d3h4.py
"""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
import tempfile
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
SRC = ROOT / "rocmolkit" / "src" / "semiempirical"

MOLECULES = [
    ("H2S", [16, 1, 1], [[0, 0, 0.0], [0.9686, 0, 0.9269], [-0.9686, 0, 0.9269]]),
    ("CH3I", [53, 6, 1, 1, 1],
     [[0, 0, 2.139], [0, 0, 0], [1.028, 0, -0.363], [-0.514, 0.890, -0.363],
      [-0.514, -0.890, -0.363]]),
    # linear H-bonded water dimer (exercises e_hb) + H-H repulsion
    ("H2O_dimer", [8, 1, 1, 8, 1, 1],
     [[0, 0, 0.0], [0.96, 0, 0], [-0.24, 0.93, 0], [2.85, 0, 0], [3.13, 0.90, 0],
      [3.13, -0.90, 0]]),
    ("methanol", [6, 8, 1, 1, 1, 1],
     [[0, 0, 0.0], [1.41, 0, 0], [-0.36, 1.03, 0], [-0.36, -0.51, 0.89],
      [-0.36, -0.51, -0.89], [1.78, 0.86, 0]]),
]


def main() -> int:
    mlx = os.environ.get("MLXMOLKIT")
    if not mlx or not os.path.isdir(mlx):
        sys.exit("set MLXMOLKIT to the cloned mlxmolkit package dir")
    for name in ("mlx", "mlx.core"):
        sys.modules[name] = types.ModuleType(name)
    sys.modules["mlx"].core = sys.modules["mlx.core"]
    sys.path.insert(0, mlx)
    D = importlib.import_module("rm1.pm6_d3h4")

    lines = ["#include <cstdio>", "#include <cmath>", '#include "h4_device.h"',
             "using namespace nvMolKit::semiempirical;", "int main(){", "double worst=0;"]
    for nm, Z, c in MOLECULES:
        r = D.pm6_d3h4_correction(Z, np.asarray(c, float))
        za = ",".join(str(z) for z in Z)
        ca = ",".join(repr(float(x)) for row in c for x in row)
        lines += [
            f"{{ int z[]={{{za}}}; double co[]={{{ca}}};",
            f"  double T=pm6dD3H4Correction({len(Z)},z,co);",
            f"  double d=std::fabs(T-({float(r['e_total'])!r})); worst=std::fmax(worst,d);",
            f'  std::printf("%-10s D3H4 mine=%.5f orac=%.5f d=%.2e\\n","{nm}",T,'
            f"{float(r['e_total'])!r},d); }}",
        ]
    lines += ['std::printf("\\nworst |d|=%.2e %s\\n",worst,worst<1e-6?"OK":"** FAIL");',
              "return worst<1e-6?0:1;}"]
    with tempfile.TemporaryDirectory() as td:
        cf = Path(td) / "t.cpp"
        cf.write_text("\n".join(lines))
        exe = Path(td) / "t"
        subprocess.run(["g++", "-std=c++17", "-O2", f"-I{SRC}", str(cf), "-o", str(exe)],
                       check=True)
        r = subprocess.run([str(exe)], capture_output=True, text=True)
        print(r.stdout.strip())
        return r.returncode


if __name__ == "__main__":
    raise SystemExit(main())
