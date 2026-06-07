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
// Device-callable PM6-D3 dispersion correction (Grimme D3, zero-damping). With
// PM6-D3H4's s8 = 0 only the C6/e6 term is active. Returns the dispersion energy
// in kcal/mol, to be added to the PM6_D heat of formation. Reference data baked
// in d3_data.h; ported from mlxmolkit's rm1.pm6_d3h4.d3_energy (Grimme, BSD-3).

#ifndef NVMOLKIT_SEMIEMPIRICAL_D3_DEVICE_H
#define NVMOLKIT_SEMIEMPIRICAL_D3_DEVICE_H

#include <cmath>

#include "d3_data.h"
#include "device_macros.h"

namespace nvMolKit {
namespace semiempirical {

// Gaussian-weighted reference-C6 interpolation from the coordination numbers,
// mirroring MOPAC getc6. zi/zj are 1-based atomic numbers (<= kD3MaxZ).
NVMOLKIT_HD inline double d3GetC6(int zi, int zj, double cni, double cnj) {
  using namespace d3;
  const int nz = kD3MaxZ + 1;
  const double* tab = &kD3C6ab[(zi * nz + zj) * 75];  // [mxc][mxc][3]
  double c6mem = -1e99, rsum = 0.0, csum = 0.0;
  for (int i = 0; i < kD3Mxc[zi]; ++i)
    for (int j = 0; j < kD3Mxc[zj]; ++j) {
      const double c6 = tab[(i * kD3MaxC + j) * 3 + 0];
      if (c6 > 0.0) {
        c6mem = c6;
        const double cn1 = tab[(i * kD3MaxC + j) * 3 + 1];
        const double cn2 = tab[(i * kD3MaxC + j) * 3 + 2];
        const double r = (cn1 - cni) * (cn1 - cni) + (cn2 - cnj) * (cn2 - cnj);
        const double w = std::exp(-4.0 * r);
        rsum += w;
        csum += w * c6;
      }
    }
  return (rsum > 0.0) ? csum / rsum : c6mem;
}

// PM6-D3 dispersion energy (kcal/mol) for one molecule. atoms: 1-based Z (each
// <= kD3MaxZ and supported); coords: nAtoms*3 (Angstrom). rthr cutoff in Angstrom.
NVMOLKIT_HD inline double pm6dD3Energy(int nAtoms, const int* atoms, const double* coords,
                                       double rthr = 15.0) {
  using namespace d3;
  if (nAtoms < 2) return 0.0;
  const double rthrBohr = rthr / kD3Bohr;
  const double rthrBohr2 = rthrBohr * rthrBohr;
  const int nz = kD3MaxZ + 1;

  // Coordination numbers cn[i] (Pauling, k1 = 16; rcov scaled by 4/3, in bohr).
  double cn[64];
  for (int i = 0; i < nAtoms; ++i) {
    double s = 0.0;
    const double rci = (4.0 / 3.0) * kD3Rcov[atoms[i]] / kD3Bohr;
    for (int j = 0; j < nAtoms; ++j) {
      if (i == j) continue;
      const double dx = (coords[3 * j] - coords[3 * i]) / kD3Bohr;
      const double dy = (coords[3 * j + 1] - coords[3 * i + 1]) / kD3Bohr;
      const double dz = (coords[3 * j + 2] - coords[3 * i + 2]) / kD3Bohr;
      const double rij = std::sqrt(dx * dx + dy * dy + dz * dz);
      if (rij < 1e-12) continue;
      const double rco = rci + (4.0 / 3.0) * kD3Rcov[atoms[j]] / kD3Bohr;
      s += 1.0 / (1.0 + std::exp(-16.0 * (rco / rij - 1.0)));
    }
    cn[i] = s;
  }

  double e6 = 0.0;
  for (int i = 0; i < nAtoms - 1; ++i) {
    const int zi = atoms[i];
    for (int j = i + 1; j < nAtoms; ++j) {
      const int zj = atoms[j];
      const double dx = (coords[3 * j] - coords[3 * i]) / kD3Bohr;
      const double dy = (coords[3 * j + 1] - coords[3 * i + 1]) / kD3Bohr;
      const double dz = (coords[3 * j + 2] - coords[3 * i + 2]) / kD3Bohr;
      const double rij2 = dx * dx + dy * dy + dz * dz;
      if (rij2 > rthrBohr2) continue;
      const double rij = std::sqrt(rij2);
      const double rr = kD3R0ab[zj * nz + zi] / rij;
      const double tmp6 = kPM6D3_rs6 * rr;
      const double damp6 = 1.0 / (1.0 + 6.0 * std::pow(tmp6, kPM6D3_alp6));
      const double c6 = d3GetC6(zi, zj, cn[i], cn[j]);
      const double r6 = rij2 * rij2 * rij2;
      e6 += c6 * damp6 / r6;
    }
  }
  return -kPM6D3_s6 * e6 * kD3AuToKcal;
}

}  // namespace semiempirical
}  // namespace nvMolKit

#endif  // NVMOLKIT_SEMIEMPIRICAL_D3_DEVICE_H
