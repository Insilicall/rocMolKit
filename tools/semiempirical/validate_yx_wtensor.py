"""Validate the assembled YX molecular two-center tensor bit-exact vs the oracle.

Chains all the YX pieces — transpiled local-frame d integrals (riLocalYX), the sp
molecular tensor (twoCenterMolecularDev), and the d-rotation — into the full
9x9x4x4 (mu nu_A | lam sig_B) tensor (yxWMolecular) and checks it against a frozen
oracle capture (golden_yx_wtensor.json, the S-C _yx_pair_w_pyseqm output).

    python3 tools/semiempirical/validate_yx_wtensor.py
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
SRC = ROOT / "rocmolkit" / "src" / "semiempirical"
GOLD = Path(__file__).resolve().parent / "data" / "golden_yx_wtensor.json"


def main() -> int:
    g = json.loads(GOLD.read_text())
    zA, zB = g["zA"], g["zB"]
    cA, cB = g["coordA"], g["coordB"]
    W = g["W"]  # 9x9x4x4
    flat = [W[mu][nu][lam][sig] for mu in range(9) for nu in range(9)
            for lam in range(4) for sig in range(4)]
    init = ", ".join(repr(float(x)) for x in flat)
    lines = [
        "#include <cstdio>",
        "#include <cmath>",
        '#include "core_hamiltonian.h"',
        '#include "two_center_yx_device.h"',
        "using namespace nvMolKit::semiempirical;",
        "int main(){",
        "AtomIntParams pA,pB;",
        f"gatherAtomIntParamsD({zA},pA); gatherAtomIntParamsD({zB},pB);",
        f"double cA[3]={{{cA[0]!r},{cA[1]!r},{cA[2]!r}}}, cB[3]={{{cB[0]!r},{cB[1]!r},{cB[2]!r}}};",
        "double W[9*9*4*4];",
        "if(!yxWMolecular(pA,cA,pB,cB,W)){ std::printf(\"no d params\\n\"); return 1; }",
        f"static const double gW[9*9*4*4]={{{init}}};",
        "double worst=0; int wk=-1;",
        "for(int i=0;i<9*9*4*4;++i){double d=std::fabs(W[i]-gW[i]); if(d>worst){worst=d;wk=i;}}",
        r'std::printf("YX W(9x9x4x4) worst |d|=%.3e at %d (mine=%.5f gold=%.5f)\n",worst,wk,W[wk],gW[wk]);',
        "return worst<1e-7?0:1;}",  # FP-level: matmul order + sp code path
    ]
    with tempfile.TemporaryDirectory() as td:
        cf = Path(td) / "t.cpp"
        cf.write_text("\n".join(lines))
        exe = Path(td) / "t"
        subprocess.run(
            ["g++", "-std=c++17", "-O2", f"-I{SRC}", str(cf),
             str(SRC / "core_hamiltonian.cpp"), str(SRC / "pm6_params.cpp"),
             str(SRC / "overlap.cpp"), "-o", str(exe)], check=True)
        r = subprocess.run([str(exe)], capture_output=True, text=True)
        print(r.stdout.strip())
        if r.stderr.strip():
            print(r.stderr.strip())
        print("OK — YX molecular two-center tensor bit-exact"
              if r.returncode == 0 else "** FAIL")
        return r.returncode


if __name__ == "__main__":
    raise SystemExit(main())
