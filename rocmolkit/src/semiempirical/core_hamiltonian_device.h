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
// Device-callable NDDO/PM6 sp core Hamiltonian assembly. Shared __host__
// __device__ code so H_core can be built on the GPU (closing the 100%-GPU SCF)
// or on the CPU reference (core_hamiltonian.cpp), identically.

#ifndef NVMOLKIT_SEMIEMPIRICAL_CORE_HAMILTONIAN_DEVICE_H
#define NVMOLKIT_SEMIEMPIRICAL_CORE_HAMILTONIAN_DEVICE_H

#include "device_macros.h"
#include "overlap_device.h"
#include "two_center_device.h"

namespace nvMolKit {
namespace semiempirical {

// Build H_core for one molecule into H (nBasis*nBasis row-major): diagonal
// one-center (uss/upp), two-center resonance 1/2 (beta_u+beta_v) S_uv, and
// electron-core attraction (e1b) summed over the other atoms.
NVMOLKIT_HD inline void buildCoreHamiltonianDev(int nBasis, int nAtoms, const AtomIntParams* ap,
                                                const int* start, const int* norb,
                                                const double* coords, double* H) {
  for (int i = 0; i < nBasis * nBasis; ++i) H[i] = 0.0;

  // Diagonal: uss on s, upp on the three p.
  for (int a = 0; a < nAtoms; ++a) {
    const int s = start[a];
    for (int o = 0; o < norb[a]; ++o) {
      const int mu = s + o;
      H[mu * nBasis + mu] = (o == 0) ? ap[a].uss : ap[a].upp;
    }
  }

  // Two-center resonance H_uv = 1/2 (beta_u + beta_v) S_uv.
  for (int i = 0; i < nAtoms; ++i) {
    for (int j = i + 1; j < nAtoms; ++j) {
      double blk[16];
      diatomOverlapSpDev(ap[i], &coords[3 * i], ap[j], &coords[3 * j], blk);
      for (int mo = 0; mo < norb[i]; ++mo) {
        const double bmu = (mo == 0) ? ap[i].betaS : ap[i].betaP;
        for (int no = 0; no < norb[j]; ++no) {
          const double bnu = (no == 0) ? ap[j].betaS : ap[j].betaP;
          const double h = 0.5 * (bmu + bnu) * blk[mo * norb[j] + no];
          const int mu = start[i] + mo, nu = start[j] + no;
          H[mu * nBasis + nu] = h;
          H[nu * nBasis + mu] = h;
        }
      }
    }
  }

  // Electron-core attraction: e1b (electron on i attracted to core j).
  for (int i = 0; i < nAtoms; ++i) {
    for (int j = 0; j < nAtoms; ++j) {
      if (i == j) continue;
      double w[256], e1b[16], e2a[16];
      twoCenterMolecularDev(ap[i], &coords[3 * i], ap[j], &coords[3 * j], w, e1b, e2a);
      for (int mo = 0; mo < norb[i]; ++mo)
        for (int no = 0; no < norb[i]; ++no)
          H[(start[i] + mo) * nBasis + (start[i] + no)] += e1b[mo * 4 + no];
    }
  }
}

}  // namespace semiempirical
}  // namespace nvMolKit

#endif  // NVMOLKIT_SEMIEMPIRICAL_CORE_HAMILTONIAN_DEVICE_H
