"""Validate the C++ PM6_D engine against the frozen oracle charges.

Feeds each YH-scope molecule from data/golden_pm6d_charges.json to the compiled
validate_pm6d_driver and checks the converged Mulliken charges bit-close to the
golden. No oracle clone needed (validates against the frozen golden, whose every
integral was validated bit-exact against the PYSEQM-derived oracle).

    g++ -std=c++17 -O2 -I../../rocmolkit/src/semiempirical validate_pm6d_driver.cpp \\
        ../../rocmolkit/src/semiempirical/{core_hamiltonian,pm6_params,overlap}.cpp \\
        -o /tmp/pm6d_drv
    python3 tools/semiempirical/validate_pm6d_cpp.py /tmp/pm6d_drv
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

GOLDEN = Path(__file__).resolve().parent / "data" / "golden_pm6d_charges.json"


def main() -> int:
    drv = sys.argv[1] if len(sys.argv) > 1 else "/tmp/pm6d_drv"
    mols = json.loads(GOLDEN.read_text())

    stdin = []
    for mol in mols:
        stdin.append(str(len(mol["atoms"])))
        for z, c in zip(mol["atoms"], mol["coords"]):
            stdin.append(f"{z} {c[0]:.10f} {c[1]:.10f} {c[2]:.10f}")
    out = subprocess.run([drv], input="\n".join(stdin) + "\n", capture_output=True,
                         text=True, check=True).stdout.strip().splitlines()

    worst = 0.0
    worstH = 0.0
    fails = 0
    for mol, line in zip(mols, out):
        tok = line.split()
        conv = int(tok[0])
        hof = float(tok[1])
        q = [float(x) for x in tok[2:]]
        gold = mol["q"]
        d = max(abs(a - b) for a, b in zip(q, gold)) if conv and len(q) == len(gold) else float("inf")
        dH = abs(hof - mol["hof_kcal"]) if conv else float("inf")
        ok = conv == 1 and d < 1e-4 and dH < 1e-3
        fails += 0 if ok else 1
        if d != float("inf"):
            worst = max(worst, d)
            worstH = max(worstH, dH)
        print(f"{mol['name']:5s} conv={conv} max|dq|={d:.2e} |dHoF|={dH:.2e} kcal  "
              f"HoF={hof:.4f}  q={[round(x, 4) for x in q]}  {'OK' if ok else '** FAIL'}")

    print(f"\nPM6_D C++ engine worst |dq| = {worst:.2e}, worst |dHoF| = {worstH:.2e} kcal  "
          f"({'matches oracle golden' if fails == 0 else f'{fails} FAILED'})")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
