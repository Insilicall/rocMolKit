"""Validate the Br qn4-qn2 d-orbital machinery (jcall 642) at the component level.

CH3Br is a multi-solution SCF case (the Fock is bit-exact to the oracle, but
damped mixing and the oracle's DIIS land on different valid fixed points), so the
Br + 2nd-row machinery is validated here against frozen oracle captures instead of
a full-molecule charge:

  - the Br-C diatomic overlap (sp jcall 642 + d-s/d-p jcall 642 radials)
    vs overlap_d_molecular_frame (golden_brc_overlap.json), and
  - the Br-C YX two-center tensor (yxWMolecular) vs _yx_pair_w_pyseqm
    (golden_brc_yxw.json).

    python3 tools/semiempirical/validate_brc_components.py
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
SRC = ROOT / "rocmolkit" / "src" / "semiempirical"
DATA = Path(__file__).resolve().parent / "data"
CA, CB = "0,0,1.939", "0,0,0"  # Br, C (the capture geometry)


def main() -> int:
    ov = json.loads((DATA / "golden_brc_overlap.json").read_text())  # 9x4
    yxw = json.loads((DATA / "golden_brc_yxw.json").read_text())     # 9x9x4x4
    ov_flat = ", ".join(repr(float(ov[i][j])) for i in range(9) for j in range(4))
    w_flat = ", ".join(repr(float(yxw[mu][nu][lam][sig]))
                       for mu in range(9) for nu in range(9)
                       for lam in range(4) for sig in range(4))
    lines = [
        "#include <cstdio>",
        "#include <cmath>",
        '#include "core_hamiltonian.h"',
        '#include "overlap_d_device.h"',
        '#include "two_center_yx_device.h"',
        "using namespace nvMolKit::semiempirical;",
        "int main(){",
        "AtomIntParams pA,pB; gatherAtomIntParamsD(35,pA); gatherAtomIntParamsD(6,pB);",
        f"double cA[3]={{{CA}}}, cB[3]={{{CB}}};",
        "double S[81]; diatomOverlapDDev(pA,cA,pB,cB,S);",
        f"static const double gS[36]={{{ov_flat}}};",
        "double ws=0; for(int i=0;i<9;++i)for(int j=0;j<4;++j) ws=std::fmax(ws,std::fabs(S[i*4+j]-gS[i*4+j]));",
        "double W[9*9*4*4]; yxWMolecular(pA,cA,pB,cB,W);",
        f"static const double gW[9*9*4*4]={{{w_flat}}};",
        "double ww=0; for(int i=0;i<9*9*4*4;++i) ww=std::fmax(ww,std::fabs(W[i]-gW[i]));",
        r'std::printf("Br-C overlap worst|d|=%.2e  yxW worst|d|=%.2e\n",ws,ww);',
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
        print(r.stdout.strip())
        print("OK — Br qn4-qn2 overlap + YX two-center bit-exact"
              if r.returncode == 0 else "** FAIL")
        return r.returncode


if __name__ == "__main__":
    raise SystemExit(main())
