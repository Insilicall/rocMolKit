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
// Host CPU driver for the public PM6_D API (scf_d.h). Gathers the per-atom
// parameters, builds H_core, runs the d-orbital SCF (DIIS), and emits Mulliken
// charges + heat of formation — the same device-callable code the GPU kernel
// runs, so the two are bit-exact. See docs/SEMIEMPIRICAL_DESIGN.md.

#include "scf_d.h"

#include <vector>

#include "core_hamiltonian.h"  // gatherAtomIntParamsD
#include "core_hamiltonian_d_device.h"
#include "energy_device.h"  // nuclearRepulsionAm1Dev, heatOfFormationKcalDev
#include "pm6_params.h"     // pm6ValenceElectrons
#include "scf_d_device.h"

namespace nvMolKit {
namespace semiempirical {

bool pm6dCharges(int nAtoms, const int* atoms, const double* coords, double* q,
                 double* hofKcal, int maxIter, double convTol) {
  if (nAtoms <= 0) return false;

  std::vector<AtomIntParams> ap(nAtoms);
  std::vector<int> start(nAtoms), norb(nAtoms);
  int nBasis = 0, nElec = 0;
  for (int a = 0; a < nAtoms; ++a) {
    if (!gatherAtomIntParamsD(atoms[a], ap[a])) return false;
    start[a] = nBasis;
    norb[a] = ap[a].nOrb;
    nBasis += norb[a];
    nElec += pm6ValenceElectrons(atoms[a]);
  }
  if (nBasis == 0 || nElec % 2 != 0) return false;  // unsupported / open shell

  const int n2 = nBasis * nBasis;
  std::vector<double> H(n2), density(n2), eval(nBasis), F(n2), eigA(n2), C(n2), Pnew(n2),
      ecom(n2), diisF(kScfDiisMax * n2), diisE(kScfDiisMax * n2);
  buildCoreHamiltonianDDev(nBasis, nAtoms, ap.data(), start.data(), norb.data(), coords, H.data());

  int conv = 0, niter = 0;
  double eElec = 0.0;
  scfLoopDDev(nBasis, nAtoms, ap.data(), start.data(), norb.data(), coords, H.data(), nElec / 2,
              maxIter, convTol, density.data(), eval.data(), F.data(), eigA.data(), C.data(),
              Pnew.data(), ecom.data(), diisF.data(), diisE.data(), &conv, &niter, &eElec);
  if (!conv) return false;

  for (int a = 0; a < nAtoms; ++a) {
    double pop = 0.0;
    for (int o = 0; o < norb[a]; ++o) pop += density[(start[a] + o) * nBasis + (start[a] + o)];
    q[a] = static_cast<double>(pm6ValenceElectrons(atoms[a])) - pop;
  }
  if (hofKcal != nullptr) {
    const double eNuc = nuclearRepulsionAm1Dev(nAtoms, ap.data(), coords);
    *hofKcal = heatOfFormationKcalDev(eElec, eNuc, nAtoms, ap.data());
  }
  return true;
}

}  // namespace semiempirical
}  // namespace nvMolKit
