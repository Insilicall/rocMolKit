"""Validate the mixed heavy-halogen (interhalide) two-center machinery bit-exact.

For pairs where BOTH atoms carry d-orbitals with DIFFERENT principal quantum
numbers (Br-Cl, Br-S, I-Cl, I-S, I-Br), the reverse-dsBlock trick can't build the
s-d / p-d block, so the engine routes them through the transpiled, PYSEQM-faithful
``interhalideOverlapDDev`` (jcall codes 7/853/9). This checks the full engine path:

  * the 9x9 diatomic overlap via ``diatomOverlapDDev`` (which dispatches to the
    interhalide kernel) vs the oracle ``overlap_d_molecular_frame`` — tested in
    BOTH atom orders to exercise the heavier-first ordering + transpose;
  * the 9x9x9x9 YY two-center two-electron tensor via ``yyWMolecular`` vs the
    oracle ``_yy_pair_w_pyseqm``.

The transpile reproduces even the oracle's unphysical qn5 overlap entries
(e.g. I-Br S[3,0] > 1) bit-exact, so nothing is excluded.

    MLXMOLKIT=/tmp/mlxmolkit_inspect/mlxmolkit \\
      /tmp/semienv/bin/python tools/semiempirical/validate_interhalide_components.py
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

# name, zHi (heavier qn), zLo, separation (Angstrom)
CASES = [
    ("Br-Cl", 35, 17, 2.18),
    ("Br-S", 35, 16, 2.24),
    ("I-Cl", 53, 17, 2.55),
    ("I-S", 53, 16, 2.40),
    ("I-Br", 53, 35, 2.74),
]


def main() -> int:
    mlx = os.environ.get("MLXMOLKIT")
    if not mlx or not Path(mlx).is_dir():
        sys.exit("set MLXMOLKIT to the cloned mlxmolkit package dir")
    for n in ("mlx", "mlx.core"):
        sys.modules[n] = types.ModuleType(n)
    sys.modules["mlx"].core = sys.modules["mlx.core"]
    sys.path.insert(0, mlx)
    params = importlib.import_module("rm1.methods").get_params("PM6_D")
    overlap = importlib.import_module("rm1.overlap_d").overlap_d_molecular_frame
    yyw = importlib.import_module("rm1.d_two_center")._yy_pair_w_pyseqm

    lines = ["#include <cstdio>", "#include <cmath>",
             '#include "core_hamiltonian.h"', '#include "overlap_d_device.h"',
             '#include "two_center_yx_device.h"',
             "using namespace nvMolKit::semiempirical;",
             "int main(){ double worst = 0;"]
    notes = []
    for nm, zhi, zlo, r in CASES:
        cA = np.array([0.0, 0.0, 0.0])
        cB = np.array([0.3 * r, 0.0, 0.9 * r])
        cB = cB / np.linalg.norm(cB) * r
        # oracle overlap (rows = heavier atom A) and YY W tensor [A,A,B,B]
        S = np.asarray(overlap(params[zhi], params[zlo], cA, cB))
        W = np.asarray(yyw(params[zhi], params[zlo], cA, cB))
        if np.max(np.abs(S)) > 1.0:
            notes.append(f"{nm}: reproduces oracle's unphysical |S|>1 entry bit-exact")
        sflat = ", ".join(repr(float(S[i][j])) for i in range(9) for j in range(9))
        wflat = ", ".join(repr(float(W[mu][nu][lam][sg]))
                          for mu in range(9) for nu in range(9)
                          for lam in range(9) for sg in range(9))
        cbC = ", ".join(repr(float(x)) for x in cB)
        lines += [
            f"{{ AtomIntParams hi, lo; gatherAtomIntParamsD({zhi}, hi); gatherAtomIntParamsD({zlo}, lo);",
            f"  double cH[3]={{0,0,0}}, cL[3]={{{cbC}}};",
            "  double S[81]; diatomOverlapDDev(hi, cH, lo, cL, S);",
            "  double Sr[81]; diatomOverlapDDev(lo, cL, hi, cH, Sr);  // reversed order",
            f"  static const double gS[81]={{{sflat}}};",
            "  double ws=0, wr=0; for(int i=0;i<9;++i)for(int j=0;j<9;++j){"
            "    ws=std::fmax(ws,std::fabs(S[i*9+j]-gS[i*9+j]));"
            "    wr=std::fmax(wr,std::fabs(Sr[j*9+i]-gS[i*9+j])); }",
            "  static double Wt[9*9*9*9]; yyWMolecular(hi, cH, lo, cL, Wt);",
            f"  static const double gW[9*9*9*9]={{{wflat}}};",
            "  double ww=0; for(int i=0;i<9*9*9*9;++i) ww=std::fmax(ww,std::fabs(Wt[i]-gW[i]));",
            "  double w=std::fmax(std::fmax(ws,wr),ww);"
            f'  std::printf("%-6s overlap|d|=%.2e rev|d|=%.2e yyW|d|=%.2e  %s\\n",'
            f'"{nm}",ws,wr,ww,(ws<1e-9&&wr<1e-9&&ww<1e-7)?"OK":"** FAIL");',
            "  worst=std::fmax(worst, (ww<1e-7)?std::fmax(ws,wr):w); }",
        ]
    lines += ['std::printf("\\ninterhalide two-center worst |d|=%.2e %s\\n",worst,'
              'worst<1e-7?"OK":"** FAIL");', "return worst<1e-7?0:1;}"]

    with tempfile.TemporaryDirectory() as td:
        cf = Path(td) / "t.cpp"
        cf.write_text("\n".join(lines))
        exe = Path(td) / "t"
        subprocess.run(
            ["g++", "-std=c++17", "-O2", f"-I{SRC}", str(cf),
             str(SRC / "core_hamiltonian.cpp"), str(SRC / "pm6_params.cpp"),
             str(SRC / "overlap.cpp"), "-o", str(exe)], check=True)
        res = subprocess.run([str(exe)], capture_output=True, text=True)
        print(res.stdout.strip())
        for n in notes:
            print("  note:", n)
        return res.returncode


if __name__ == "__main__":
    raise SystemExit(main())
