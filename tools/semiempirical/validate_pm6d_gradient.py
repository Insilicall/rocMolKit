"""Validate the C++ PM6_D frozen-density gradient bit-exact against the oracle.

The engine's ``pm6dGradient`` is a frozen-density (Hellmann-Feynman) gradient:
solve the SCF once for the density P, then central-finite-difference
E = 0.5 tr(P (H + F)) + E_nuc rebuilt at each displaced geometry without
re-solving. This is exactly the oracle's ``anal_grad.analytical_gradient`` (same
delta = 1e-5, same frozen-density energy E = 0.5 tr(P(H+F)) + E_nuc, validated
bit-exact element by element). The driver pins the oracle's SCF settings
(max_iter=200, conv_tol=1e-8) so the converged density coincides.

The two agree to the **finite-difference floor**, not to machine zero: the central
difference subtracts two total NDDO energies ~O(1e4 eV) to recover a ~1e-4 eV
difference at delta=1e-5, so catastrophic cancellation leaves ~eps*|E|/delta of
round-off (~1e-9 eV/A for small molecules, growing to ~5e-7 eV/A for multi-heavy-
atom systems), and NumPy vs C++ sum in different orders. Feeding the *same*
density to both still leaves this residual -- it is the FD method's floor, shared
by the oracle, not an algorithm difference. The threshold (1e-6 eV/A) sits far
below any chemically-meaningful force (~1e-3 eV/A).

    MLXMOLKIT=/tmp/mlxmolkit_inspect/mlxmolkit \\
      /tmp/semienv/bin/python tools/semiempirical/validate_pm6d_gradient.py /tmp/pm6d_grad_drv
"""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent

# A spread of bonding modes incl. d-block + an interhalide (Br...Cl) pair.
MOLECULES = [
    ("H2S", [16, 1, 1], [[0, 0, 0.0], [0.9686, 0, 0.9269], [-0.9686, 0, 0.9269]]),
    ("HCl", [17, 1], [[0, 0, 0.0], [0, 0, 1.2746]]),
    ("CH3Cl", [17, 6, 1, 1, 1],
     [[0, 0, 1.781], [0, 0, 0], [1.036, 0, -0.337], [-0.518, 0.897, -0.337],
      [-0.518, -0.897, -0.337]]),
    ("Cl2", [17, 17], [[0, 0, 0], [0, 0, 1.988]]),
    ("HBr", [35, 1], [[0, 0, 0], [0, 0, 1.41]]),
    ("CH3I", [53, 6, 1, 1, 1],
     [[0, 0, 2.139], [0, 0, 0], [1.028, 0, -0.363], [-0.514, 0.890, -0.363],
      [-0.514, -0.890, -0.363]]),
    ("BrCl", [35, 17], [[0, 0, 0], [0, 0, 1.94]]),
    ("CHBrCl2", [6, 35, 17, 17, 1],
     [[0, 0, 0], [0, 0, 1.94], [1.7, 0, -0.7], [-0.85, 1.47, -0.7], [-0.6, -1.0, -0.4]]),
]


def main() -> int:
    drv = sys.argv[1] if len(sys.argv) > 1 else "/tmp/pm6d_grad_drv"
    mlx = os.environ.get("MLXMOLKIT")
    if not mlx or not os.path.isdir(mlx):
        sys.exit("set MLXMOLKIT to the cloned mlxmolkit package dir")
    for n in ("mlx", "mlx.core"):
        sys.modules[n] = types.ModuleType(n)
    sys.modules["mlx"].core = sys.modules["mlx.core"]
    sys.path.insert(0, mlx)
    anal = importlib.import_module("rm1.anal_grad").analytical_gradient

    stdin = []
    for _, Z, c in MOLECULES:
        stdin.append(str(len(Z)))
        for z, row in zip(Z, c):
            stdin.append(f"{z} {row[0]:.10f} {row[1]:.10f} {row[2]:.10f}")
    out = subprocess.run([drv], input="\n".join(stdin) + "\n", capture_output=True,
                         text=True, check=True).stdout.strip().splitlines()

    worst = 0.0
    fails = 0
    for (nm, Z, c), line in zip(MOLECULES, out):
        tok = line.split()
        if tok[0] != "1":
            print(f"{nm:8s} ** engine did not converge")
            fails += 1
            continue
        mine = np.asarray([float(x) for x in tok[2:]], dtype=np.float64).reshape(len(Z), 3)
        _, gold = anal(Z, np.asarray(c, dtype=np.float64), method="PM6_D")
        gold = np.asarray(gold, dtype=np.float64)
        d = float(np.max(np.abs(mine - gold)))
        worst = max(worst, d)
        ok = d < 1e-6  # finite-difference round-off floor (see module docstring)
        fails += 0 if ok else 1
        print(f"{nm:8s} max|d(grad)|={d:.2e} eV/A  |grad|max={np.max(np.abs(gold)):.4f}  "
              f"{'OK' if ok else '** FAIL'}")

    print(f"\nPM6_D gradient worst |d| = {worst:.2e} eV/A vs oracle analytical_gradient  "
          f"({'matches to the FD round-off floor' if fails == 0 else f'{fails} FAILED'})")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
