// SPDX-FileCopyrightText: Copyright (c) 2025 InsilicAll. All rights reserved.
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
// CPU validation driver for the PM6_D (d-orbital) semi-empirical SCF. Reads
// YH-scope molecules (one d-atom + hydrogens) from stdin and prints the
// converged Mulliken charges, for tools/semiempirical/validate_pm6d_cpp.py to
// compare against data/golden_pm6d_charges.json. Build with plain g++ (no HIP):
//
//   g++ -std=c++17 -O2 -I../../rocmolkit/src/semiempirical \
//       validate_pm6d_driver.cpp \
//       ../../rocmolkit/src/semiempirical/{core_hamiltonian,pm6_params,overlap}.cpp
//
// stdin format: per molecule "<nAtoms>" then nAtoms lines "<Z> <x> <y> <z>".
// Output per molecule: "<conv> <q0> <q1> ...".

#include <cmath>
#include <cstdio>
#include <vector>

#include "core_hamiltonian.h"        // gatherAtomIntParamsD
#include "core_hamiltonian_d_device.h"
#include "energy_device.h"           // nuclearRepulsionDev, heatOfFormationKcalDev
#include "pm6_params.h"              // pm6ValenceElectrons
#include "scf_d_device.h"

using namespace nvMolKit::semiempirical;

int main() {
  int nAtoms = 0;
  while (std::scanf("%d", &nAtoms) == 1 && nAtoms > 0) {
    std::vector<int> z(nAtoms);
    std::vector<double> coords(3 * nAtoms);
    for (int a = 0; a < nAtoms; ++a) {
      if (std::scanf("%d %lf %lf %lf", &z[a], &coords[3 * a], &coords[3 * a + 1],
                     &coords[3 * a + 2]) != 4) {
        return 1;
      }
    }

    std::vector<AtomIntParams> ap(nAtoms);
    std::vector<int> start(nAtoms), norb(nAtoms);
    int nBasis = 0, nElec = 0;
    bool ok = true;
    for (int a = 0; a < nAtoms; ++a) {
      if (!gatherAtomIntParamsD(z[a], ap[a])) { ok = false; break; }
      start[a] = nBasis;
      norb[a] = ap[a].nOrb;
      nBasis += norb[a];
      nElec += pm6ValenceElectrons(z[a]);
    }
    if (!ok || nBasis == 0) {
      std::printf("0\n");
      std::fflush(stdout);
      continue;
    }

    const int n2 = nBasis * nBasis;
    std::vector<double> H(n2), density(n2), eval(nBasis), F(n2), eigA(n2), C(n2), Pnew(n2);
    buildCoreHamiltonianDDev(nBasis, nAtoms, ap.data(), start.data(), norb.data(), coords.data(),
                             H.data());
    int conv = 0, niter = 0;
    double eElec = 0.0;
    scfLoopDDev(nBasis, nAtoms, ap.data(), start.data(), norb.data(), coords.data(), H.data(),
                nElec / 2, 800, 1e-10, density.data(), eval.data(), F.data(), eigA.data(),
                C.data(), Pnew.data(), &conv, &niter, &eElec);

    const double eNuc = nuclearRepulsionAm1Dev(nAtoms, ap.data(), coords.data());
    const double hof = heatOfFormationKcalDev(eElec, eNuc, nAtoms, ap.data());

    std::printf("%d %.6f", conv, hof);
    for (int a = 0; a < nAtoms; ++a) {
      double pop = 0.0;
      for (int o = 0; o < norb[a]; ++o) pop += density[(start[a] + o) * nBasis + (start[a] + o)];
      std::printf(" %.6f", static_cast<double>(pm6ValenceElectrons(z[a])) - pop);
    }
    std::printf("\n");
    std::fflush(stdout);
  }
  return 0;
}
