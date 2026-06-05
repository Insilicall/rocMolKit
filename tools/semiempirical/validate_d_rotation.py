"""Validate the ported d-orbital two-center rotation bit-exact against the oracle.

The molecular-frame two-center w-tensor for a d-bearing pair is the local-frame
45x45 block transformed by the YM similarity rotation. This checks the C++ port
(generateRotationMatrixD + rotate2Center2ElectronD) against a frozen oracle
capture (data/golden_d_rotation.json: the WW input and the rotated output of
PYSEQM's Rotate2Center2Electron for an S-C pair, bond along z). Feeding the same
WW + bond, the C++ must reproduce the rotated tensor.

    python3 tools/semiempirical/validate_d_rotation.py
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
SRC = ROOT / "rocmolkit" / "src" / "semiempirical"
GOLD = Path(__file__).resolve().parent / "data" / "golden_d_rotation.json"


def main() -> int:
    g = json.loads(GOLD.read_text())
    ww = ",".join(repr(float(v)) for row in g["WW"] for v in row)
    out = ",".join(repr(float(v)) for row in g["out"] for v in row)
    lines = [
        "#include <cstdio>",
        "#include <cmath>",
        '#include "d_rotation_device.h"',
        "using namespace nvMolKit::semiempirical;",
        "int main(){",
        f"static const double WW[2025]={{{ww}}};",
        f"static const double GO[2025]={{{out}}};",
        "double v[3]={0,0,1.0};",  # S-C bond along +z (the capture geometry)
        "double mat[675]; drot::generateRotationMatrixD(v,mat);",
        "double w[2025]; drot::rotate2Center2ElectronD(WW,mat,w);",
        "double worst=0; int wk=-1;",
        "for(int i=0;i<2025;++i){double d=std::fabs(w[i]-GO[i]); if(d>worst){worst=d;wk=i;}}",
        r'std::printf("d-rotation worst |d|=%.3e at (%d,%d)\n",worst,wk/45,wk%45);',
        "return worst<1e-9?0:1;}",
    ]
    with tempfile.TemporaryDirectory() as td:
        cf = Path(td) / "t.cpp"
        cf.write_text("\n".join(lines))
        exe = Path(td) / "t"
        subprocess.run(["g++", "-std=c++17", "-O2", f"-I{SRC}", str(cf), "-o", str(exe)],
                       check=True)
        r = subprocess.run([str(exe)], capture_output=True, text=True)
        print(r.stdout.strip())
        print("OK — d-orbital two-center rotation bit-exact"
              if r.returncode == 0 else "** FAIL")
        return r.returncode


if __name__ == "__main__":
    raise SystemExit(main())
