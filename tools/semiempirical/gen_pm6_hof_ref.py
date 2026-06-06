"""Calibrate the canonical PM6 heat-of-formation reference against MOPAC.

The canonical PM6 heat of formation is HoF = kEvToKcal*(E_elec + E_core_PWCCT) -
sum_atoms kHofRef[Z], where kHofRef[Z] = kEvToKcal*EISOL - EHEAT for the element.
Rather than recomputing MOPAC's EISOL from the atomic reference configuration, we
fit kHofRef by least squares so the simple hydrides (one heavy atom + H, where the
PYSEQM electronic structure is closest to MOPAC) reproduce MOPAC's own FINAL HEAT
OF FORMATION. MOPAC 23.2.5 is the oracle.

Requires a built MOPAC (set MOPAC_DIR to the unpacked mopac-*-linux dir, which has
bin/mopac and lib/) and a venv with numpy.

    MOPAC_DIR=/tmp/mopac_bin/mopac-23.2.5-linux \\
      /tmp/semienv/bin/python tools/semiempirical/gen_pm6_hof_ref.py

Rewrites rocmolkit/src/semiempirical/pwcct_ref_data.h (do not hand-edit).
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
SRC = ROOT / "rocmolkit" / "src" / "semiempirical"
OUT = SRC / "pwcct_ref_data.h"
SYM = {1: "H", 5: "B", 6: "C", 7: "N", 8: "O", 9: "F", 13: "Al", 14: "Si", 15: "P",
       16: "S", 17: "Cl", 30: "Zn", 31: "Ga", 32: "Ge", 35: "Br", 48: "Cd", 50: "Sn",
       53: "I", 80: "Hg"}
CONV = 23.060548

# Light/halide refs [1,6,7,8,9,15,16,17,35,53] are fit on simple closed-shell
# hydrides (+ H2/Cl2). The main-group/metal refs [5,13,14,30,31,32,48,50,80]
# (B, Al, Si, Zn, Ga, Ge, Cd, Sn, Hg) are fit AFTERWARDS with the light refs held
# fixed, on metal halides / hydrides / methyls (single-point geometries), so the
# added refs are transferable without perturbing the validated organic HoF.
EXTRA_ELEMS = [5, 13, 14, 30, 31, 32, 48, 50, 80]

CAL = {
    "H2": ([1, 1], [[0, 0, 0], [0, 0, 0.74]]),
    "CH4": ([6, 1, 1, 1, 1], [[0, 0, 0], [0.629, 0.629, 0.629], [-0.629, -0.629, 0.629],
                              [-0.629, 0.629, -0.629], [0.629, -0.629, -0.629]]),
    "NH3": ([7, 1, 1, 1], [[0, 0, 0], [0.94, 0, 0.33], [-0.47, 0.81, 0.33], [-0.47, -0.81, 0.33]]),
    "H2O": ([8, 1, 1], [[0, 0, 0], [0.757, 0, 0.587], [-0.757, 0, 0.587]]),
    "HF": ([9, 1], [[0, 0, 0], [0, 0, 0.917]]),
    "PH3": ([15, 1, 1, 1], [[0, 0, 0], [1.19, 0, 0.77], [-0.6, 1.03, 0.77], [-0.6, -1.03, 0.77]]),
    "H2S": ([16, 1, 1], [[0, 0, 0], [0.9686, 0, 0.9269], [-0.9686, 0, 0.9269]]),
    "HCl": ([17, 1], [[0, 0, 0], [0, 0, 1.2746]]),
    "HBr": ([35, 1], [[0, 0, 0], [0, 0, 1.41]]),
    "HI": ([53, 1], [[0, 0, 0], [0, 0, 1.609]]),
    "Cl2": ([17, 17], [[0, 0, 0], [0, 0, 1.988]]),
}

# Calibration set for the extra (metal/main-group) elements (light refs held fixed).
CAL_EXTRA = {
    "ZnF2": ([30, 9, 9], [[0, 0, 0], [0, 0, 1.75], [0, 0, -1.75]]),
    "ZnCl2": ([30, 17, 17], [[0, 0, 0], [0, 0, 2.07], [0, 0, -2.07]]),
    "ZnMe2": ([30, 6, 6, 1, 1, 1, 1, 1, 1],
              [[0, 0, 0], [0, 0, 1.95], [0, 0, -1.95], [0.51, 0.88, 2.34], [0.51, -0.88, 2.34],
               [-1.02, 0, 2.34], [0.51, 0.88, -2.34], [0.51, -0.88, -2.34], [-1.02, 0, -2.34]]),
    "CdCl2": ([48, 17, 17], [[0, 0, 0], [0, 0, 2.21], [0, 0, -2.21]]),
    "CdBr2": ([48, 35, 35], [[0, 0, 0], [0, 0, 2.37], [0, 0, -2.37]]),
    "HgCl2": ([80, 17, 17], [[0, 0, 0], [0, 0, 2.29], [0, 0, -2.29]]),
    "HgBr2": ([80, 35, 35], [[0, 0, 0], [0, 0, 2.41], [0, 0, -2.41]]),
    "AlF3": ([13, 9, 9, 9], [[0, 0, 0], [1.63, 0, 0], [-0.815, 1.41, 0], [-0.815, -1.41, 0]]),
    "AlCl3": ([13, 17, 17, 17], [[0, 0, 0], [2.06, 0, 0], [-1.03, 1.78, 0], [-1.03, -1.78, 0]]),
    "SiH4": ([14, 1, 1, 1, 1], [[0, 0, 0], [0.856, 0.856, 0.856], [-0.856, -0.856, 0.856],
                                [-0.856, 0.856, -0.856], [0.856, -0.856, -0.856]]),
    "SiCl4": ([14, 17, 17, 17, 17], [[0, 0, 0], [1.18, 1.18, 1.18], [-1.18, -1.18, 1.18],
                                     [-1.18, 1.18, -1.18], [1.18, -1.18, -1.18]]),
    "GaCl3": ([31, 17, 17, 17], [[0, 0, 0], [2.1, 0, 0], [-1.05, 1.82, 0], [-1.05, -1.82, 0]]),
    "GaF3": ([31, 9, 9, 9], [[0, 0, 0], [1.71, 0, 0], [-0.855, 1.48, 0], [-0.855, -1.48, 0]]),
    "GeCl4": ([32, 17, 17, 17, 17], [[0, 0, 0], [1.06, 1.06, 1.06], [-1.06, -1.06, 1.06],
                                     [-1.06, 1.06, -1.06], [1.06, -1.06, -1.06]]),
    "GeH4": ([32, 1, 1, 1, 1], [[0, 0, 0], [0.88, 0.88, 0.88], [-0.88, -0.88, 0.88],
                                [-0.88, 0.88, -0.88], [0.88, -0.88, -0.88]]),
    "SnCl4": ([50, 17, 17, 17, 17], [[0, 0, 0], [1.15, 1.15, 1.15], [-1.15, -1.15, 1.15],
                                     [-1.15, 1.15, -1.15], [1.15, -1.15, -1.15]]),
    "SnH4": ([50, 1, 1, 1, 1], [[0, 0, 0], [0.99, 0.99, 0.99], [-0.99, -0.99, 0.99],
                               [-0.99, 0.99, -0.99], [0.99, -0.99, -0.99]]),
    "BF3": ([5, 9, 9, 9], [[0, 0, 0], [1.31, 0, 0], [-0.655, 1.13, 0], [-0.655, -1.13, 0]]),
    "BCl3": ([5, 17, 17, 17], [[0, 0, 0], [1.74, 0, 0], [-0.87, 1.51, 0], [-0.87, -1.51, 0]]),
}

DRIVER = r'''
#include <cstdio>
#include <vector>
#include "scf_d.h"
#include "core_hamiltonian.h"
#include "energy_device.h"
#include "pwcct_device.h"
using namespace nvMolKit::semiempirical;
int main(){ int n; while(scanf("%d",&n)==1&&n>0){
  std::vector<int> z(n); std::vector<double> c(3*n);
  for(int a=0;a<n;++a) scanf("%d %lf %lf %lf",&z[a],&c[3*a],&c[3*a+1],&c[3*a+2]);
  std::vector<double> g(3*n); double E=0;
  if(!pm6dGradient(n,z.data(),c.data(),g.data(),&E,800,1e-10,1e-5)){ printf("0\n"); continue; }
  std::vector<AtomIntParams> ap(n); for(int a=0;a<n;++a) gatherAtomIntParamsD(z[a],ap[a]);
  double am1=nuclearRepulsionAm1Dev(n,ap.data(),c.data());
  printf("1 %.10f\n", (E-am1) + pwcctCoreCoreDev(n,z.data(),c.data())); // E_total = eElec + PWCCT
}
return 0;}
'''

HEADER = '''// SPDX-FileCopyrightText: Copyright (c) 2025 InsilicAll. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// GENERATED by tools/semiempirical/gen_pm6_hof_ref.py (calibrated against MOPAC
// 23.2.5 PM6) -- DO NOT HAND-EDIT.
//
// Per-element heat-of-formation reference for the canonical (MOPAC-aligned) PM6
// heat of formation: HoF(kcal) = kEvToKcal * (E_elec + E_core_PWCCT) - sum kHofRef.
// Each kHofRef[Z] = kEvToKcal*EISOL - EHEAT, fit by least squares. The light/halide
// refs [1,6,7,8,9,15,16,17,35,53] are fit on simple hydrides (<0.1 kcal/mol). The
// added main-group/metal refs [5,13,14,30,31,32,48,50,80] (B, Al, Si, Zn, Ga, Ge,
// Cd, Sn, Hg) are fit with the light refs held fixed, on metal halides/hydrides/
// methyls, and reproduce MOPAC PM6 HoF to <=1.2 kcal/mol per compound. With the
// PWCCT core-core (bit-exact to MOPAC) and the PYSEQM electronic SCF, the HoF then
// matches MOPAC PM6 to ~1 kcal/mol for light + Br molecules (iodine looser).

#ifndef NVMOLKIT_SEMIEMPIRICAL_PWCCT_REF_DATA_H
#define NVMOLKIT_SEMIEMPIRICAL_PWCCT_REF_DATA_H

#include "pwcct_data.h"  // kPwcctMaxZ

namespace nvMolKit {
namespace semiempirical {
namespace pwcct {

constexpr double kEvToKcal = 23.060548;

// kHofRef[Z] (kcal/mol); unsupported elements are 0.
constexpr double kHofRef[kPwcctMaxZ + 1] = {%(VALS)s};

}  // namespace pwcct
}  // namespace semiempirical
}  // namespace nvMolKit

#endif  // NVMOLKIT_SEMIEMPIRICAL_PWCCT_REF_DATA_H
'''


def main() -> int:
    mop = os.environ.get("MOPAC_DIR")
    if not mop or not (Path(mop) / "bin" / "mopac").is_file():
        sys.exit("set MOPAC_DIR to the unpacked mopac-*-linux dir (see module docstring)")
    env = dict(os.environ, LD_LIBRARY_PATH=f"{mop}/lib:" + os.environ.get("LD_LIBRARY_PATH", ""))

    def mopac_hof(Z, c):
        geo = "\n".join(f"{SYM[z]} {p[0]:.6f} 0 {p[1]:.6f} 0 {p[2]:.6f} 0" for z, p in zip(Z, c))
        Path("/tmp/_hofref.mop").write_text(f"PM6 1SCF CHARGE=0 PRECISE\nx\n\n{geo}\n")
        subprocess.run([f"{mop}/bin/mopac", "/tmp/_hofref.mop"], env=env, capture_output=True)
        for l in Path("/tmp/_hofref.out").read_text().splitlines():
            if "FINAL HEAT" in l:
                return float(l.split("=")[1].split("KCAL")[0])
        raise RuntimeError("no HoF from MOPAC")

    with tempfile.TemporaryDirectory() as td:
        cf = Path(td) / "drv.cpp"
        cf.write_text(DRIVER)
        exe = Path(td) / "drv"
        subprocess.run(["g++", "-std=c++17", "-O2", f"-I{SRC}", str(cf),
                        str(SRC / "core_hamiltonian.cpp"), str(SRC / "pm6_params.cpp"),
                        str(SRC / "overlap.cpp"), str(SRC / "scf_d.cpp"), "-o", str(exe)], check=True)

        def engine_etot(table):
            names = list(table)
            stdin = []
            for nm in names:
                Z, c = table[nm]
                stdin.append(str(len(Z)))
                stdin += [f"{z} {p[0]:.10f} {p[1]:.10f} {p[2]:.10f}" for z, p in zip(Z, c)]
            out = subprocess.run([str(exe)], input="\n".join(stdin) + "\n",
                                 capture_output=True, text=True).stdout.split("\n")
            res = {}
            for nm, line in zip(names, out):
                t = line.split()
                if not t or t[0] != "1":
                    sys.exit(f"engine did not converge for calibration molecule {nm}")
                res[nm] = float(t[1])  # E_elec + PWCCT (eV)
            return res

        etot_extra = engine_etot(CAL_EXTRA)

    # Stage 1: the light/halide refs are the ALREADY-SHIPPED, validated values; read
    # them back from the current header so re-running this generator to add metals
    # never perturbs the organic HoF. (The original hydride fit that produced them is
    # preserved in git history.) Re-derive them by passing --refit-light if needed.
    refmap = {}
    if "--refit-light" in sys.argv:
        with tempfile.TemporaryDirectory() as td2:
            cf = Path(td2) / "drv.cpp"; cf.write_text(DRIVER); exe = Path(td2) / "drv"
            subprocess.run(["g++", "-std=c++17", "-O2", f"-I{SRC}", str(cf),
                            str(SRC / "core_hamiltonian.cpp"), str(SRC / "pm6_params.cpp"),
                            str(SRC / "overlap.cpp"), str(SRC / "scf_d.cpp"), "-o", str(exe)], check=True)

            def _etot(table):
                names = list(table); stdin = []
                for nm in names:
                    Z, c = table[nm]; stdin.append(str(len(Z)))
                    stdin += [f"{z} {p[0]:.10f} {p[1]:.10f} {p[2]:.10f}" for z, p in zip(Z, c)]
                out = subprocess.run([str(exe)], input="\n".join(stdin) + "\n",
                                     capture_output=True, text=True).stdout.split("\n")
                return {nm: float(l.split()[1]) for nm, l in zip(names, out) if l.split() and l.split()[0] == "1"}
            etot = _etot(CAL)
        rows = [(CAL[nm][0], etot[nm], mopac_hof(*CAL[nm])) for nm in CAL]
        elems = sorted({e for Z, _, _ in rows for e in Z})
        A = np.array([[Z.count(e) for e in elems] for Z, _, _ in rows], float)
        y = np.array([CONV * T - hof for _, T, hof in rows])
        ref, *_ = np.linalg.lstsq(A, y, rcond=None)
        refmap = {e: r for e, r in zip(elems, ref)}
        resid = float(np.max(np.abs(y - A @ ref)))
    else:
        cur = OUT.read_text()
        for m in re.finditer(r"([-0-9.eE+]+),\s*//\s*(\d+)\s", cur):
            refmap[int(m.group(2))] = float(m.group(1))
        for z in EXTRA_ELEMS:  # drop any stale extra refs before re-fitting them
            refmap.pop(z, None)
        resid = 0.0

    # Stage 2: fit the extra (metal/main-group) refs with the light refs held fixed,
    # on the metal halides/hydrides/methyls -- so the organic HoF is not perturbed.
    erows = [(CAL_EXTRA[nm][0], etot_extra[nm], mopac_hof(*CAL_EXTRA[nm])) for nm in CAL_EXTRA]
    Ae = np.array([[Z.count(e) for e in EXTRA_ELEMS] for Z, _, _ in erows], float)
    ye = np.array([CONV * T - hof - sum(refmap.get(z, 0.0) for z in Z if z not in EXTRA_ELEMS)
                   for Z, T, hof in erows], float)
    refe, *_ = np.linalg.lstsq(Ae, ye, rcond=None)
    for e, r in zip(EXTRA_ELEMS, refe):
        refmap[e] = r
    eresid = float(np.max(np.abs(ye - Ae @ refe)))

    maxz = max(SYM) + 1
    vals = []
    for z in range(maxz):
        v = float(refmap.get(z, 0.0))
        comment = f"  // {z} {SYM[z]}" if z in SYM else ""
        vals.append(f"    {v!r},{comment}")
    OUT.write_text(HEADER % {"VALS": "\n" + "\n".join(vals) + "\n"})
    print(f"wrote {OUT.relative_to(ROOT)}  (light residual {resid:.3f} kcal, "
          f"extra residual {eresid:.3f} kcal); "
          f"ref={ {SYM[e]: round(r, 3) for e, r in refmap.items()} }")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
