"""Freeze d-orbital overlap matrices from the PYSEQM oracle (Phase 3 anchor).

The d-orbital overlap (PM6_D) is the gate for Phase 3. The reference is PYSEQM's
``diatom_overlap_matrixD`` (LANL, BSD-3), vendored in guillaume-osmo/mlxmolkit
(MIT) as a pure-NumPy port that runs here. A from-scratch analytic route
(overlap_d_local + a Wigner-D rotation) was tested and does NOT reproduce the
oracle's d-block — the exact d-orbital frame convention lives in the PYSEQM code —
so the C++ port must reproduce ``diatom_overlap_matrixD``, validated par-by-par
against the frozen matrices here.

Setup (one-time, outside the repo):

    python3 -m venv /tmp/semienv && /tmp/semienv/bin/pip install numpy
    git clone https://github.com/guillaume-osmo/mlxmolkit /tmp/mlxmolkit_inspect

Run:

    MLXMOLKIT=/tmp/mlxmolkit_inspect/mlxmolkit \\
      /tmp/semienv/bin/python tools/semiempirical/gen_doverlap_golden.py

Rewrites tools/semiempirical/data/golden_doverlap.json. Orbital order per atom:
[s, px, py, pz, dz2, dxz, dyz, dx2-y2, dxy].
"""

from __future__ import annotations

import importlib
import json
import os
import sys
import types
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "data" / "golden_doverlap.json"

# Representative pairs (heavy d-bearing with H / C / O, and heavy-heavy) at
# chemical separations, placed off-axis to exercise the d-orbital rotation.
PAIRS = [(16, 1, 1.34), (16, 6, 1.81), (16, 16, 2.05), (17, 1, 1.27),
         (17, 6, 1.78), (15, 1, 1.42), (16, 8, 1.44), (17, 17, 1.99)]


def main() -> None:
    mlx = os.environ.get("MLXMOLKIT")
    if not mlx or not Path(mlx).is_dir():
        sys.exit("set MLXMOLKIT to the cloned mlxmolkit package dir (see module docstring)")
    for name in ("mlx", "mlx.core"):
        sys.modules[name] = types.ModuleType(name)
    sys.modules["mlx"].core = sys.modules["mlx.core"]
    sys.path.insert(0, mlx)
    params = importlib.import_module("rm1.methods").get_params("PM6_D")
    overlap = importlib.import_module("rm1.overlap_d").overlap_d_molecular_frame

    out = []
    for za, zb, r in PAIRS:
        cA = np.array([0.0, 0.0, 0.0])
        cB = np.array([0.3 * r, 0.0, 0.9 * r])
        cB = cB / np.linalg.norm(cB) * r
        S = np.asarray(overlap(params[za], params[zb], cA, cB))
        out.append({
            "zA": za, "zB": zb, "R": round(r, 4),
            "coordB": [round(float(x), 6) for x in cB],
            "S": [[round(float(v), 8) for v in row] for row in S],
        })
        print(f"{za}-{zb}: {S.shape}  max|S|={np.max(np.abs(S)):.4f}")

    OUT.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {OUT.relative_to(Path(__file__).resolve().parent.parent.parent)}")


if __name__ == "__main__":
    main()
