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
// Device-callable NDDO/PM6 Fock build (sp basis): F = H + G(P), one-center
// Slater-Condon + two-center Coulomb/exchange. The SAME inline __host__
// __device__ code feeds the CPU reference (scf.cpp) and the HIP kernel, so the
// bit-exact validation covers both. Operates on raw arrays — no std::vector.

#ifndef NVMOLKIT_SEMIEMPIRICAL_FOCK_DEVICE_H
#define NVMOLKIT_SEMIEMPIRICAL_FOCK_DEVICE_H

#include "device_macros.h"
#include "two_center_device.h"

namespace nvMolKit {
namespace semiempirical {

// Build the Fock matrix for one molecule.
//   nBasis   : total sp basis size
//   nAtoms   : atom count
//   ap       : per-atom gathered parameters (length nAtoms)
//   start    : first basis index of each atom (length nAtoms)
//   norb     : sp orbital count of each atom, 1 or 4 (length nAtoms)
//   coords   : nAtoms*3 (Angstrom)
//   H, P     : nBasis*nBasis row-major (core Hamiltonian, density)
//   F        : nBasis*nBasis row-major output
NVMOLKIT_HD inline void buildFockDev(int nBasis, int nAtoms, const AtomIntParams* ap,
                                     const int* start, const int* norb, const double* coords,
                                     const double* H, const double* P, double* F) {
  for (int t = 0; t < nBasis * nBasis; ++t) F[t] = H[t];

  // One-center two-electron (Slater-Condon).
  for (int a = 0; a < nAtoms; ++a) {
    const AtomIntParams& p = ap[a];
    const int s = start[a];
    const double Pss = P[s * nBasis + s];
    if (norb[a] == 1) {
      F[s * nBasis + s] += Pss * p.gss * 0.5;
      continue;
    }
    const int pk0 = s + 1, pk1 = s + 2, pk2 = s + 3;
    const double Ppp = P[pk0 * nBasis + pk0] + P[pk1 * nBasis + pk1] + P[pk2 * nBasis + pk2];
    const double sp1 = p.gsp - 0.5 * p.hsp;
    const double sp2 = 1.5 * p.hsp - 0.5 * p.gsp;
    const double ppd = 1.25 * p.gp2 - 0.25 * p.gpp;
    const double ppoff = 0.75 * p.gpp - 1.25 * p.gp2;

    F[s * nBasis + s] += Pss * p.gss * 0.5 + Ppp * sp1;
    for (int k = 1; k <= 3; ++k) {
      const int pk = s + k;
      F[pk * nBasis + pk] += Pss * sp1 + P[pk * nBasis + pk] * p.gpp * 0.5
                             + (Ppp - P[pk * nBasis + pk]) * ppd;
      F[s * nBasis + pk] += P[s * nBasis + pk] * sp2;
      F[pk * nBasis + s] += P[pk * nBasis + s] * sp2;
    }
    for (int k = 1; k <= 3; ++k)
      for (int l = k + 1; l <= 3; ++l) {
        const int pk = s + k, pl = s + l;
        F[pk * nBasis + pl] += P[pk * nBasis + pl] * ppoff;
        F[pl * nBasis + pk] += P[pl * nBasis + pk] * ppoff;
      }
  }

  // Two-center Coulomb/exchange from the rotated w tensor.
  for (int i = 0; i < nAtoms; ++i) {
    for (int j = i + 1; j < nAtoms; ++j) {
      double w[256], e1b[16], e2a[16];
      twoCenterMolecularDev(ap[i], &coords[3 * i], ap[j], &coords[3 * j], w, e1b, e2a);
      const int nA = norb[i], nB = norb[j], sA = start[i], sB = start[j];
      for (int mu = 0; mu < nA; ++mu)
        for (int nu = 0; nu < nA; ++nu) {
          double acc = 0.0;
          for (int lam = 0; lam < nB; ++lam)
            for (int sig = 0; sig < nB; ++sig)
              acc += P[(sB + lam) * nBasis + (sB + sig)] * w[detail::wIdxDev(mu, nu, lam, sig)];
          F[(sA + mu) * nBasis + (sA + nu)] += acc;
        }
      for (int lam = 0; lam < nB; ++lam)
        for (int sig = 0; sig < nB; ++sig) {
          double acc = 0.0;
          for (int mu = 0; mu < nA; ++mu)
            for (int nu = 0; nu < nA; ++nu)
              acc += P[(sA + mu) * nBasis + (sA + nu)] * w[detail::wIdxDev(mu, nu, lam, sig)];
          F[(sB + lam) * nBasis + (sB + sig)] += acc;
        }
      for (int mu = 0; mu < nA; ++mu)
        for (int lam = 0; lam < nB; ++lam) {
          double acc = 0.0;
          for (int nu = 0; nu < nA; ++nu)
            for (int sig = 0; sig < nB; ++sig)
              acc += w[detail::wIdxDev(mu, nu, lam, sig)] * P[(sA + nu) * nBasis + (sB + sig)];
          acc *= -0.5;
          F[(sA + mu) * nBasis + (sB + lam)] += acc;
          F[(sB + lam) * nBasis + (sA + mu)] += acc;
        }
    }
  }
}

// GPU batch: build the Fock matrix for nMol molecules (one thread per molecule)
// from concatenated inputs. molNAtoms/molNBasis are per-molecule counts;
// atomsAll (sum nAtoms) and coordsAll (3*sum nAtoms) are the concatenated atoms;
// Hall/Pall/Fall are the concatenated nBasis*nBasis matrices. Defined in
// fock_kernels.hip.cpp. Returns false on an unsupported element.
bool buildFockBatchGpu(int nMol, const int* molNAtoms, const int* molNBasis,
                       const int* atomsAll, const double* coordsAll,
                       const double* Hall, const double* Pall, double* Fall);

}  // namespace semiempirical
}  // namespace nvMolKit

#endif  // NVMOLKIT_SEMIEMPIRICAL_FOCK_DEVICE_H
