"""Freeze PM6_D oracle intermediates as bit-exactness targets for the HIP port.

The HIP semi-empirical SCF engine (rocmolkit/src/semiempirical/) is developed
against a CPU oracle that is itself bit-exact to PYSEQM. Rather than depend on
that oracle at build/test time, we freeze its outputs — charges, heat of
formation, eigenvalues, and (for the simplest molecules) the H_core and density
matrices — into data/golden_intermediates.json. The HIP code validates against
those frozen numbers stage by stage; see docs/SEMIEMPIRICAL_DESIGN.md.

The oracle is guillaume-osmo/mlxmolkit's native PM6_D path (MIT), which calls a
vendored NumPy port of PYSEQM (BSD-3-Clause, LANL). We run only its pure-NumPy
`native=True` path (no MLX/Metal, no torch) — verified to reproduce every charge
in golden_pm6_charges.py to <=1e-3 e.

Setup (one-time, outside the repo — we do NOT vendor the external code):

    python3 -m venv /tmp/semienv && /tmp/semienv/bin/pip install numpy
    git clone https://github.com/guillaume-osmo/mlxmolkit /tmp/mlxmolkit_inspect

Run:

    MLXMOLKIT=/tmp/mlxmolkit_inspect/mlxmolkit \\
      /tmp/semienv/bin/python tools/semiempirical/oracle_dump.py

This rewrites tools/semiempirical/data/golden_intermediates.json.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import sys
import types
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
GOLDEN_CHARGES = HERE / "data" / "golden_pm6_charges.py"
OUT = HERE / "data" / "golden_intermediates.json"

# Valence (core) charge and PM6_D valence-orbital count per element.
N_VALENCE = {1: 1, 6: 4, 7: 5, 8: 6, 9: 7, 15: 5, 16: 6, 17: 7, 35: 7, 53: 7}
N_ORBITAL = {1: 1, 6: 4, 7: 4, 8: 4, 9: 4, 15: 9, 16: 9, 17: 9, 35: 9, 53: 9}

# Molecules whose full H_core + density we freeze (smallest sp cases, for
# stage-by-stage HIP validation). The rest get scalars only (charges/HoF).
FULL_MATRIX = {"H2O", "CH4", "NH3", "HF"}


def _load_oracle(mlx_path: str):
    """Import mlxmolkit's rm1 oracle with MLX stubbed (pure-NumPy native path).

    Returns (scf, get_params, overlap_d_molecular_frame).
    """
    for name in ("mlx", "mlx.core"):
        sys.modules[name] = types.ModuleType(name)
    sys.modules["mlx"].core = sys.modules["mlx.core"]
    sys.path.insert(0, mlx_path)
    scf = importlib.import_module("rm1.scf")
    methods = importlib.import_module("rm1.methods")
    overlap_d = importlib.import_module("rm1.overlap_d")
    return scf, methods.get_params, overlap_d.overlap_d_molecular_frame


def _overlap_matrix(z_list, coords, params, overlap_fn):
    """Full molecular overlap S (n_basis x n_basis): identity on the diagonal
    atom blocks (ZDO), overlap_d_molecular_frame on the off-diagonal pairs."""
    starts, off = [], 0
    for z in z_list:
        starts.append(off)
        off += N_ORBITAL[z]
    n = off
    S = np.eye(n)
    for i, zi in enumerate(z_list):
        for j, zj in enumerate(z_list):
            if j <= i:
                continue
            block = overlap_fn(params[zi], params[zj],
                               np.array(coords[i], float), np.array(coords[j], float))
            block = np.asarray(block)[:N_ORBITAL[zi], :N_ORBITAL[zj]]
            si, sj = starts[i], starts[j]
            S[si:si + N_ORBITAL[zi], sj:sj + N_ORBITAL[zj]] = block
            S[sj:sj + N_ORBITAL[zj], si:si + N_ORBITAL[zi]] = block.T
    return S


def _load_golden():
    spec = importlib.util.spec_from_file_location("golden", GOLDEN_CHARGES)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.GOLDEN_PM6


def _mulliken(density: np.ndarray, z_list: list[int]) -> list[float]:
    q, off = [], 0
    for z in z_list:
        nb = N_ORBITAL[z]
        q.append(N_VALENCE[z] - float(np.trace(density[off:off + nb, off:off + nb])))
        off += nb
    return q


def main() -> None:
    mlx_path = os.environ.get("MLXMOLKIT")
    if not mlx_path or not Path(mlx_path).is_dir():
        sys.exit("set MLXMOLKIT to the cloned mlxmolkit package dir (see module docstring)")
    scf, get_params, overlap_fn = _load_oracle(mlx_path)
    params = get_params("PM6_D")

    out: dict[str, object] = {
        "_provenance": "mlxmolkit native PM6_D (MIT) -> PYSEQM numpy port (BSD-3, LANL); "
                       "method=PM6_D, native=True, pure NumPy. Frozen by oracle_dump.py.",
        "method": "PM6_D",
        "molecules": {},
    }
    worst = 0.0
    for name, z, coords, q_heavy, tol, tier in _load_golden():
        r = scf.nddo_energy(z, np.array(coords, float), method="PM6_D", native=True)
        density = np.asarray(r["density"])
        q = _mulliken(density, z)
        heavy = [q[i] for i, zz in enumerate(z) if zz != 1]
        dq = float(np.max(np.abs(np.array(heavy) - np.array(q_heavy))))
        worst = max(worst, dq)

        entry: dict[str, object] = {
            "Z": z,
            "tier": tier,
            "charges": [round(v, 6) for v in q],
            "heat_of_formation_kcal": round(float(r["heat_of_formation_kcal"]), 4),
            "energy_eV": round(float(r["energy_eV"]), 6),
            "n_iter": int(r["n_iter"]),
            "eigenvalues_eV": [round(float(v), 6) for v in np.asarray(r["eigenvalues"]).ravel()],
            "n_basis": int(r["n_basis"]),
        }
        if name in FULL_MATRIX:
            entry["density"] = [[round(float(v), 8) for v in row] for row in density]
            S = _overlap_matrix(z, coords, params, overlap_fn)
            entry["overlap"] = [[round(float(v), 8) for v in row] for row in S]
        out["molecules"][name] = entry
        print(f"{name:7} {tier:4} max|dq|={dq:.5f}")

    OUT.write_text(json.dumps(out, indent=1))
    print(f"\nworst |dq| = {worst:.5f}  ->  wrote {OUT.relative_to(HERE.parent.parent)}")


if __name__ == "__main__":
    main()
