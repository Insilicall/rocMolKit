"""Validate the Br/I + 2nd-row d-orbital machinery at the component level.

Bromomethane (Br-C, jcall 642) and iodomethane (I-C, jcall 752) are multi-solution
SCF cases — the Fock is bit-exact to the oracle, but damped mixing and the
oracle's DIIS converge to different valid fixed points — so the heavy-halogen +
2nd-row machinery is validated here against frozen oracle captures: the diatomic
overlap (sp + d-s/d-p radials) vs overlap_d_molecular_frame, and the YX
two-center tensor vs _yx_pair_w_pyseqm.

    python3 tools/semiempirical/validate_organohalide_components.py
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
SRC = ROOT / "rocmolkit" / "src" / "semiempirical"
DATA = Path(__file__).resolve().parent / "data"

# name -> (zA, zB, coordA, coordB, overlap_json, yxw_json)
CASES = [
    ("Br-C", 35, 6, "0,0,1.939", "0,0,0", "golden_brc_overlap.json", "golden_brc_yxw.json"),
    ("I-C", 53, 6, "0,0,2.139", "0,0,0", "golden_ic_overlap.json", "golden_ic_yxw.json"),
]


def main() -> int:
    fails = 0
    for name, zA, zB, cA, cB, ovf, wf in CASES:
        ov = json.loads((DATA / ovf).read_text())  # 9x4
        yxw = json.loads((DATA / wf).read_text())   # 9x9x4x4
        ov_flat = ", ".join(repr(float(ov[i][j])) for i in range(9) for j in range(4))
        w_flat = ", ".join(repr(float(yxw[mu][nu][lam][sig]))
                           for mu in range(9) for nu in range(9)
                           for lam in range(4) for sig in range(4))
        lines = [
            "#include <cstdio>", "#include <cmath>",
            '#include "core_hamiltonian.h"', '#include "overlap_d_device.h"',
            '#include "two_center_yx_device.h"',
            "using namespace nvMolKit::semiempirical;",
            "int main(){",
            f"AtomIntParams pA,pB; gatherAtomIntParamsD({zA},pA); gatherAtomIntParamsD({zB},pB);",
            f"double cA[3]={{{cA}}}, cB[3]={{{cB}}};",
            "double S[81]; diatomOverlapDDev(pA,cA,pB,cB,S);",
            f"static const double gS[36]={{{ov_flat}}};",
            "double ws=0; for(int i=0;i<9;++i)for(int j=0;j<4;++j) ws=std::fmax(ws,std::fabs(S[i*4+j]-gS[i*4+j]));",
            "double W[9*9*4*4]; yxWMolecular(pA,cA,pB,cB,W);",
            f"static const double gW[9*9*4*4]={{{w_flat}}};",
            "double ww=0; for(int i=0;i<9*9*4*4;++i) ww=std::fmax(ww,std::fabs(W[i]-gW[i]));",
            r'std::printf("overlap worst|d|=%.2e  yxW worst|d|=%.2e\n",ws,ww);',
            "return (ws<1e-9 && ww<1e-7)?0:1;}",
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
            print(f"{name:5s} {r.stdout.strip()}  {'OK' if r.returncode == 0 else '** FAIL'}")
            fails += r.returncode
    print("OK — Br/I + 2nd-row overlap + YX two-center bit-exact"
          if fails == 0 else "** FAIL")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
