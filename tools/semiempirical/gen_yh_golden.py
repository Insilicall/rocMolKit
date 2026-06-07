"""Freeze the YH d-orbital two-center H_core (e1b) contribution from the oracle.

Phase 3 next gate: the d-orbital two-center integrals. The YH case (atom A has
d-orbitals, atom B = H) is the simplest — it yields the 9x9 electron-core
attraction block that fills atom A's H_core diagonal block. The reference is
PYSEQM's d-orbital two-center integrals + the d rotation (rm1/tetci_yh.py,
rotation_matrix_d.py), vendored in mlxmolkit. This freezes that 9x9 block for the
YH pairs (S-H/P-H/Cl-H) as the par-by-par validation anchor for the C++ port,
the same discipline used for the sp integrals and the d-overlap.

Setup (one-time):
    python3 -m venv /tmp/semienv && /tmp/semienv/bin/pip install numpy
    git clone https://github.com/guillaume-osmo/mlxmolkit /tmp/mlxmolkit_inspect

Run:
    MLXMOLKIT=/tmp/mlxmolkit_inspect/mlxmolkit \\
      /tmp/semienv/bin/python tools/semiempirical/gen_yh_golden.py
"""

from __future__ import annotations

import importlib
import json
import os
import sys
import types
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "data" / "golden_yh_e1b.json"
PAIRS = [(16, 1.34), (15, 1.42), (17, 1.27)]  # (d-atom Z, S-H/P-H/Cl-H distance)


def main() -> None:
    mlx = os.environ.get("MLXMOLKIT")
    if not mlx or not Path(mlx).is_dir():
        sys.exit("set MLXMOLKIT to the cloned mlxmolkit package dir (see module docstring)")
    for name in ("mlx", "mlx.core"):
        sys.modules[name] = types.ModuleType(name)
    sys.modules["mlx"].core = sys.modules["mlx.core"]
    sys.path.insert(0, mlx)
    params = importlib.import_module("rm1.methods").get_params("PM6_D")
    e1b_fn = importlib.import_module("rm1.tetci_yh").yh_e1b_contribution

    out = []
    for za, r in PAIRS:
        cA = np.array([0.0, 0.0, 0.0])
        cB = np.array([0.3 * r, 0.0, 0.9 * r])
        cB = cB / np.linalg.norm(cB) * r
        e1b = np.asarray(e1b_fn(params[za], params[1], cA, cB))
        out.append({
            "zA": za, "zB": 1, "R": round(r, 4),
            "coordB": [round(float(x), 6) for x in cB],
            "e1b": [[round(float(v), 8) for v in row] for row in e1b],
        })
        print(f"{za}-H e1b 9x9  max|e1b|={np.max(np.abs(e1b)):.4f}")

    OUT.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {OUT.relative_to(Path(__file__).resolve().parent.parent.parent)}")


if __name__ == "__main__":
    main()
