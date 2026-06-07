"""Validate the transpiled local-frame d-orbital reduced integrals bit-exact.

Feeds the frozen oracle inputs (data/golden_yx_local.json — captured from PYSEQM's
two_elec_two_center_int_local_frame_d_orbitals) to the generated C++ riLocalYX and
checks the 450-slot reduced-integral vector against the oracle output. Isolates
the formula transcription (same inputs in, compare outputs).

    python3 tools/semiempirical/gen_d_localframe.py
    python3 tools/semiempirical/validate_d_localframe.py
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
SRC = ROOT / "rocmolkit" / "src" / "semiempirical"
GOLD = Path(__file__).resolve().parent / "data" / "golden_yx_local.json"

YX_INPUTS = ["r0", "da0", "db0", "qa0", "qb0", "dpa0", "dsa0", "dda0",
             "rho0a", "rho0b", "rho1a", "rho1b", "rho2a", "rho2b",
             "rho3a", "rho4a", "rho5a", "rho6a"]
YY_INPUTS = ["r0", "da0", "db0", "qa0", "qb0", "dpa0", "dpb0", "dsa0", "dsb0",
             "dda0", "ddb0", "rho0a", "rho0b", "rho1a", "rho1b", "rho2a", "rho2b",
             "rho3a", "rho3b", "rho4a", "rho4b", "rho5a", "rho5b", "rho6a", "rho6b"]

CASES = [
    ("YX", "golden_yx_local.json", "riYX", 450, YX_INPUTS),
    ("YY", "golden_yy_local.json", "riYY", 2025, YY_INPUTS),
]


def main() -> int:
    fails = 0
    for case, fname, riname, size, inputs in CASES:
        g = json.loads((GOLD.parent / fname).read_text())
        args = ", ".join(repr(float(g[k])) for k in inputs)
        init = ", ".join(repr(float(x)) for x in g[riname])
        cpp = f"""
#include <cstdio>
#include <cmath>
#include "d_localframe_generated.h"
using namespace nvMolKit::semiempirical::dlocal;
int main() {{
  double ri[{size}];
  riLocal{case}({args}, ri);
  static const double exp[{size}] = {{{init}}};
  double worst = 0.0; int worstk = -1;
  for (int k = 0; k < {size}; ++k) {{
    double d = std::fabs(ri[k] - exp[k]);
    if (d > worst) {{ worst = d; worstk = k; }}
  }}
  std::printf("riLocal{case} worst |d|=%.3e at k=%d\\n", worst, worstk);
  return worst < 1e-9 ? 0 : 1;
}}
"""
        with tempfile.TemporaryDirectory() as td:
            cf = Path(td) / "t.cpp"
            cf.write_text(cpp)
            exe = Path(td) / "t"
            subprocess.run(["g++", "-std=c++17", "-O2", f"-I{SRC}", str(cf), "-o", str(exe)],
                           check=True)
            r = subprocess.run([str(exe)], capture_output=True, text=True)
            print(r.stdout.strip(), "OK" if r.returncode == 0 else "** FAIL")
            fails += r.returncode
    print("OK — local-frame d reduced integrals bit-exact"
          if fails == 0 else "** FAIL")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
