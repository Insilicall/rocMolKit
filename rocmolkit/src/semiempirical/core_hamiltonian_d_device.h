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
// Device-callable NDDO/PM6_D core Hamiltonian assembly for molecules whose only
// heavy atom carries d-orbitals and whose remaining atoms are hydrogens (the YH
// scope: H2S, PH3, HCl, ...). Mirrors the validated tools/semiempirical/
// validate_pm6d.py recipe exactly: diagonal Uss/Upp/Udd, resonance
// 1/2(beta_u+beta_v) S_uv with the full d-overlap, and electron-core attraction
// (the 9x9 YH e1b on the d-atom, the sp monopole on each H). Same __host__
// __device__ code feeds the CPU reference and the GPU.

#ifndef NVMOLKIT_SEMIEMPIRICAL_CORE_HAMILTONIAN_D_DEVICE_H
#define NVMOLKIT_SEMIEMPIRICAL_CORE_HAMILTONIAN_D_DEVICE_H

#include "device_macros.h"
#include "overlap_d_device.h"
#include "two_center_d_device.h"  // yhWMolecular
#include "two_center_device.h"    // twoCenterMolecularDev

namespace nvMolKit {
namespace semiempirical {

// Resonance beta for orbital o within an atom: s (o==0), p (1..3), d (4..8).
NVMOLKIT_HD inline double resonanceBetaD(const AtomIntParams& p, int o) {
  return (o == 0) ? p.betaS : (o < 4 ? p.betaP : p.betaD);
}

// A copy of p capped to its sp shell (nOrb <= 4), for the sp-only two-center
// monopole path (e1b on a hydrogen from a d-atom core uses sp multipoles only).
NVMOLKIT_HD inline AtomIntParams spCapped(const AtomIntParams& p) {
  AtomIntParams q = p;
  if (q.nOrb > 4) q.nOrb = 4;
  return q;
}

// Build H_core (nBasis*nBasis, row-major) for a PM6_D YH-scope molecule.
NVMOLKIT_HD inline void buildCoreHamiltonianDDev(int nBasis, int nAtoms, const AtomIntParams* ap,
                                                 const int* start, const int* norb,
                                                 const double* coords, double* H) {
  for (int i = 0; i < nBasis * nBasis; ++i) H[i] = 0.0;

  // Diagonal one-center one-electron: Uss on s, Upp on p, Udd on d.
  for (int a = 0; a < nAtoms; ++a) {
    const int s = start[a];
    for (int o = 0; o < norb[a]; ++o) {
      const int mu = s + o;
      H[mu * nBasis + mu] = (o == 0) ? ap[a].uss : (o < 4 ? ap[a].upp : ap[a].udd);
    }
  }

  // Two-center resonance H_uv = 1/2 (beta_u + beta_v) S_uv (full d-overlap).
  for (int i = 0; i < nAtoms; ++i) {
    for (int j = i + 1; j < nAtoms; ++j) {
      double S[81];
      diatomOverlapDDev(ap[i], &coords[3 * i], ap[j], &coords[3 * j], S);
      for (int mo = 0; mo < norb[i]; ++mo) {
        const double bmu = resonanceBetaD(ap[i], mo);
        for (int no = 0; no < norb[j]; ++no) {
          const double h = 0.5 * (bmu + resonanceBetaD(ap[j], no)) * S[mo * norb[j] + no];
          const int mu = start[i] + mo, nu = start[j] + no;
          H[mu * nBasis + nu] = h;
          H[nu * nBasis + mu] = h;
        }
      }
    }
  }

  // Electron-core attraction (e1b): electron on atom i attracted to core j.
  for (int i = 0; i < nAtoms; ++i) {
    for (int j = 0; j < nAtoms; ++j) {
      if (i == j) continue;
      if (norb[i] == 9) {  // d-atom A: full 9x9 e1b from core j (YH/YX/YY).
        // The electron-core attraction (mu nu_A | s_B s_B) depends on B only
        // through Z_B and rho0_B, so the YH machinery handles any core j once B
        // is reduced to its monopole (sp cap avoids a d-orbital twoCenterLocal).
        const AtomIntParams pj = spCapped(ap[j]);
        double W[81];
        yhWMolecular(ap[i], &coords[3 * i], pj, &coords[3 * j], W);
        const double e = -static_cast<double>(ap[j].valence);
        for (int mo = 0; mo < 9; ++mo)
          for (int no = 0; no < 9; ++no)
            H[(start[i] + mo) * nBasis + (start[i] + no)] += e * W[mo * 9 + no];
      } else {  // sp monopole/multipole on atom i's sp orbitals from core j
        const AtomIntParams pi = spCapped(ap[i]);
        const AtomIntParams pj = spCapped(ap[j]);
        double w[256], e1b[16], e2a[16];
        twoCenterMolecularDev(pi, &coords[3 * i], pj, &coords[3 * j], w, e1b, e2a);
        const int ni = pi.nOrb;
        for (int mo = 0; mo < ni; ++mo)
          for (int no = 0; no < ni; ++no)
            H[(start[i] + mo) * nBasis + (start[i] + no)] += e1b[mo * 4 + no];
      }
    }
  }
}

}  // namespace semiempirical
}  // namespace nvMolKit

#endif  // NVMOLKIT_SEMIEMPIRICAL_CORE_HAMILTONIAN_D_DEVICE_H
