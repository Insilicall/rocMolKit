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
// Device-callable closed-shell NDDO/PM6_D SCF loop (9-orbital, YH scope). One
// thread runs a whole molecule's SCF: d Fock build (fock_d_device.h) -> Jacobi
// diagonalization -> density -> damped mixing, mirroring the validated
// tools/semiempirical/gen_pm6d_golden.py loop so it converges to the same
// charges. Reuses jacobiEigenDev / buildDensityDev from scf_device.h.

#ifndef NVMOLKIT_SEMIEMPIRICAL_SCF_D_DEVICE_H
#define NVMOLKIT_SEMIEMPIRICAL_SCF_D_DEVICE_H

#include <cmath>

#include "device_macros.h"
#include "fock_d_device.h"
#include "scf_device.h"  // jacobiEigenDev, buildDensityDev

namespace nvMolKit {
namespace semiempirical {

// Full PM6_D SCF for one molecule. H is the (host- or device-built) core
// Hamiltonian. Scratch F/eigA/C/Pnew are each nBasis*nBasis; eval is nBasis;
// density (output) is nBasis*nBasis. On return density holds the converged P,
// eval the final orbital energies, *conv whether it converged, *niter the count.
NVMOLKIT_HD inline void scfLoopDDev(int nBasis, int nAtoms, const AtomIntParams* ap,
                                    const int* start, const int* norb, const double* coords,
                                    const double* H, int nOcc, int maxIter, double convTol,
                                    double* density, double* eval, double* F, double* eigA,
                                    double* C, double* Pnew, int* conv, int* niter, double* eElec) {
  const int n2 = nBasis * nBasis;
  for (int i = 0; i < n2; ++i) eigA[i] = H[i];
  jacobiEigenDev(eigA, nBasis, eval, C);
  buildDensityDev(C, nBasis, nOcc, density);

  bool converged = false;
  int it = 0;
  for (it = 0; it < maxIter; ++it) {
    buildFockDDev(nBasis, nAtoms, ap, start, norb, coords, H, density, F);
    for (int i = 0; i < n2; ++i) eigA[i] = F[i];
    jacobiEigenDev(eigA, nBasis, eval, C);
    buildDensityDev(C, nBasis, nOcc, Pnew);

    double ss = 0.0;
    for (int i = 0; i < n2; ++i) {
      const double d = Pnew[i] - density[i];
      ss += d * d;
    }
    if (std::sqrt(ss / static_cast<double>(n2)) < convTol) {
      for (int i = 0; i < n2; ++i) density[i] = Pnew[i];
      converged = true;
      break;
    }
    for (int i = 0; i < n2; ++i) density[i] = 0.3 * Pnew[i] + 0.7 * density[i];
  }
  *conv = converged ? 1 : 0;
  *niter = it + 1;

  // Final Fock + electronic energy E_elec = 0.5 sum(P .* (H + F)).
  buildFockDDev(nBasis, nAtoms, ap, start, norb, coords, H, density, F);
  double e = 0.0;
  for (int i = 0; i < n2; ++i) e += 0.5 * density[i] * (H[i] + F[i]);
  *eElec = e;
}

}  // namespace semiempirical
}  // namespace nvMolKit

#endif  // NVMOLKIT_SEMIEMPIRICAL_SCF_D_DEVICE_H
