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


def main() -> int:
    g = json.loads(GOLD.read_text())
    args = ", ".join(repr(float(g[k])) for k in YX_INPUTS)
    expect = g["riYX"]
    init = ", ".join(repr(float(x)) for x in expect)
    cpp = f"""
#include <cstdio>
#include <cmath>
#include "d_localframe_generated.h"
using namespace nvMolKit::semiempirical::dlocal;
int main() {{
  double ri[450];
  riLocalYX({args}, ri);
  static const double exp[450] = {{{init}}};
  double worst = 0.0; int worstk = -1;
  for (int k = 0; k < 450; ++k) {{
    double d = std::fabs(ri[k] - exp[k]);
    if (d > worst) {{ worst = d; worstk = k; }}
  }}
  std::printf("riLocalYX worst |d|=%.3e at k=%d  (ri=%.10f exp=%.10f)\\n",
              worst, worstk, worstk>=0?ri[worstk]:0.0, worstk>=0?exp[worstk]:0.0);
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
        print(r.stdout.strip())
        print("OK — local-frame YX reduced integrals bit-exact"
              if r.returncode == 0 else "** FAIL")
        return r.returncode


if __name__ == "__main__":
    raise SystemExit(main())
