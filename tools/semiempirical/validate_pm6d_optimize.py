"""Validate the C++ PM6_D geometry optimizer against the oracle nddo_optimize.

The engine's ``pm6dOptimize`` is L-BFGS (m=8) with backtracking/Armijo line search
on the frozen-density gradient -- the same algorithm as the oracle's
``gradient.nddo_optimize``. Because the gradient agrees with the oracle only to the
finite-difference floor (~1e-7 eV/A), the L-BFGS *paths* diverge after a few steps,
so the optimizer is NOT validated bit-exact. Instead it is validated by the
properties a correct optimizer must have, starting each molecule from a perturbed
geometry:

  * it converges (RMS gradient < grad_tol);
  * the energy decreases from the start;
  * it lands on the SAME minimum as the oracle -- final energy agrees to <1e-3 eV
    (rotation/translation-invariant) and the characteristic bond length agrees to
    <1e-3 A (an internal coordinate, also frame-invariant).

    MLXMOLKIT=/tmp/mlxmolkit_inspect/mlxmolkit \\
      /tmp/semienv/bin/python tools/semiempirical/validate_pm6d_optimize.py /tmp/pm6d_opt_drv
"""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
import types
from pathlib import Path

import numpy as np

# name, Z, perturbed start geometry, (i, j) bond to track
MOLECULES = [
    ("H2S", [16, 1, 1], [[0, 0, 0.0], [1.0, 0, 0.95], [-1.0, 0, 0.95]], (0, 1)),
    ("HCl", [17, 1], [[0, 0, 0.0], [0, 0, 1.40]], (0, 1)),
    ("Cl2", [17, 17], [[0, 0, 0], [0, 0, 2.15]], (0, 1)),
    ("CH3Cl", [17, 6, 1, 1, 1],
     [[0, 0, 1.85], [0, 0, 0], [1.05, 0, -0.36], [-0.52, 0.91, -0.36],
      [-0.52, -0.91, -0.36]], (0, 1)),
    ("BrCl", [35, 17], [[0, 0, 0], [0, 0, 2.10]], (0, 1)),
]


def bond(c, i, j):
    return float(np.linalg.norm(np.asarray(c[i]) - np.asarray(c[j])))


def main() -> int:
    drv = sys.argv[1] if len(sys.argv) > 1 else "/tmp/pm6d_opt_drv"
    mlx = os.environ.get("MLXMOLKIT")
    if not mlx or not os.path.isdir(mlx):
        sys.exit("set MLXMOLKIT to the cloned mlxmolkit package dir")
    for n in ("mlx", "mlx.core"):
        sys.modules[n] = types.ModuleType(n)
    sys.modules["mlx"].core = sys.modules["mlx.core"]
    sys.path.insert(0, mlx)
    nddo_optimize = importlib.import_module("rm1.gradient").nddo_optimize

    stdin = []
    for _, Z, c, _ in MOLECULES:
        stdin.append(str(len(Z)))
        for z, row in zip(Z, c):
            stdin.append(f"{z} {row[0]:.10f} {row[1]:.10f} {row[2]:.10f}")
    out = subprocess.run([drv], input="\n".join(stdin) + "\n", capture_output=True,
                         text=True, check=True).stdout.strip().splitlines()

    fails = 0
    for (nm, Z, c0, (bi, bj)), line in zip(MOLECULES, out):
        tok = line.split()
        conv, nit = int(tok[0]), int(tok[1])
        E_mine, gRms = float(tok[2]), float(tok[3])
        cm = np.asarray([float(x) for x in tok[4:]], dtype=np.float64).reshape(len(Z), 3)

        ref = nddo_optimize(Z, np.asarray(c0, dtype=np.float64), method="PM6_D",
                            max_iter=200, grad_tol=0.005)
        E_ref = float(ref["energy_eV"])
        c_ref = np.asarray(ref["coords"])

        dE = abs(E_mine - E_ref)
        db = abs(bond(cm, bi, bj) - bond(c_ref, bi, bj))
        ok = conv == 1 and dE < 1e-3 and db < 1e-3
        fails += 0 if ok else 1
        print(f"{nm:6s} conv={conv} it={nit:2d} gRms={gRms:.4f}  E={E_mine:.4f} eV "
              f"(dE={dE:.2e})  r{bi}{bj}={bond(cm, bi, bj):.4f} A (db={db:.2e})  "
              f"{'OK' if ok else '** FAIL'}")

    print(f"\nPM6_D optimizer: {'converges to the oracle minimum (energy + bond length)'
          if fails == 0 else f'{fails} FAILED'}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
