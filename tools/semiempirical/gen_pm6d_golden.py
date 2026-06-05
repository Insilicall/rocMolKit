"""Freeze PM6_D Mulliken charges + heat of formation from the oracle engine.

Runs the oracle's own native PM6_D SCF (rm1.scf.nddo_energy, a vendored NumPy
port of PYSEQM's d-orbital NDDO) on each YH-scope molecule (one d-atom +
hydrogens: H2S, PH3, HCl, ...) and freezes the converged per-atom charges and the
heat of formation to data/golden_pm6d_charges.json. This is the independent
par-by-par anchor the C++ PM6_D engine validates against — a different code path
than the hand-assembled validate_pm6d.py recipe, so agreement cross-checks both.

    MLXMOLKIT=/tmp/mlxmolkit_inspect/mlxmolkit \\
      /tmp/semienv/bin/python tools/semiempirical/gen_pm6d_golden.py
"""

from __future__ import annotations

import importlib
import json
import os
import sys
import types
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "data" / "golden_pm6d_charges.json"

# YH-scope molecules: one d-atom + hydrogens. Geometries in Angstrom.
MOLECULES = [
    ("H2S", [16, 1, 1],
     [[0.0, 0.0, 0.0], [0.9686, 0.0, 0.9269], [-0.9686, 0.0, 0.9269]]),
    ("PH3", [15, 1, 1, 1],
     [[0.0, 0.0, 0.0], [1.1932, 0.0, 0.7700], [-0.5966, 1.0333, 0.7700],
      [-0.5966, -1.0333, 0.7700]]),
    ("HCl", [17, 1],
     [[0.0, 0.0, 0.0], [0.0, 0.0, 1.2746]]),
]


def main() -> int:
    mlx = os.environ.get("MLXMOLKIT")
    if not mlx or not os.path.isdir(mlx):
        sys.exit("set MLXMOLKIT to the cloned mlxmolkit package dir")
    for name in ("mlx", "mlx.core"):
        sys.modules[name] = types.ModuleType(name)
    sys.modules["mlx"].core = sys.modules["mlx.core"]
    sys.path.insert(0, mlx)
    scf = importlib.import_module("rm1.scf")
    P = importlib.import_module("rm1.methods").get_params("PM6_D")

    out = []
    for name, Z, coords in MOLECULES:
        r = scf.nddo_energy(Z, np.asarray(coords, dtype=np.float64), method="PM6_D",
                            max_iter=400, conv_tol=1e-10)
        Pm = np.asarray(r["density"])
        info = scf._build_basis_info(Z, P)
        starts = info["atom_basis_start"]
        nb = [P[z].n_basis for z in Z]
        q = [P[Z[a]].n_valence
             - float(np.trace(Pm[starts[a]:starts[a] + nb[a], starts[a]:starts[a] + nb[a]]))
             for a in range(len(Z))]
        out.append({
            "name": name, "atoms": Z, "coords": coords,
            "q": [round(x, 6) for x in q],
            "hof_kcal": round(float(r["heat_of_formation_kcal"]), 6),
        })
        print(f"{name:5s} conv={r['converged']} HoF={r['heat_of_formation_kcal']:.4f} kcal  "
              f"q={[round(x, 4) for x in q]}")

    OUT.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {OUT.relative_to(Path(__file__).resolve().parent.parent.parent)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
