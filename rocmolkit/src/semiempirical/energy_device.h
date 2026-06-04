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
// Device-callable PM6_D core-core (nuclear) repulsion and heat of formation.
// Shared __host__ __device__ code so HoF can be computed on the GPU (closing the
// SCF energy path on-device) or on the CPU reference, identically.

#ifndef NVMOLKIT_SEMIEMPIRICAL_ENERGY_DEVICE_H
#define NVMOLKIT_SEMIEMPIRICAL_ENERGY_DEVICE_H

#include <cmath>

#include "device_macros.h"

namespace nvMolKit {
namespace semiempirical {

// PM6_D core-core repulsion (eV): (ss|ss) Coulomb with per-element exponential +
// Gaussian corrections and the N-H/O-H special case. coords in Angstrom.
NVMOLKIT_HD inline double nuclearRepulsionDev(int nAtoms, const AtomIntParams* ap,
                                              const double* coords) {
  constexpr double kEV = 27.21;
  constexpr double kAngToBohr = 1.0 / 0.529167;
  double eNuc = 0.0;
  for (int i = 0; i < nAtoms; ++i) {
    for (int j = i + 1; j < nAtoms; ++j) {
      const double dx = coords[3 * i] - coords[3 * j];
      const double dy = coords[3 * i + 1] - coords[3 * j + 1];
      const double dz = coords[3 * i + 2] - coords[3 * j + 2];
      const double R = std::sqrt(dx * dx + dy * dy + dz * dz);
      const double Rb = R * kAngToBohr;

      const double rho0A = (ap[i].gss > 0.0) ? 0.5 * kEV / ap[i].gss : 0.0;
      const double rho0B = (ap[j].gss > 0.0) ? 0.5 * kEV / ap[j].gss : 0.0;
      const double aee = (rho0A + rho0B) * (rho0A + rho0B);
      const double ssss = kEV / std::sqrt(Rb * Rb + aee);

      const double ZA = static_cast<double>(ap[i].valence);
      const double ZB = static_cast<double>(ap[j].valence);
      const double t1 = ZA * ZB * ssss;

      const bool nhohA = (ap[i].z == 7 || ap[i].z == 8) && ap[j].z == 1;
      const bool nhohB = (ap[j].z == 7 || ap[j].z == 8) && ap[i].z == 1;
      double t2 = std::exp(-ap[i].alpha * R);
      if (nhohA) t2 *= R;
      double t3 = std::exp(-ap[j].alpha * R);
      if (nhohB) t3 *= R;

      const double t4 = ZA * ZB / R;
      double t5 = 0.0, t6 = 0.0;
      for (int k = 0; k < 4; ++k) {
        if (ap[i].gaussK[k] != 0.0)
          t5 += ap[i].gaussK[k] * std::exp(-ap[i].gaussL[k] * (R - ap[i].gaussM[k]) * (R - ap[i].gaussM[k]));
        if (ap[j].gaussK[k] != 0.0)
          t6 += ap[j].gaussK[k] * std::exp(-ap[j].gaussL[k] * (R - ap[j].gaussM[k]) * (R - ap[j].gaussM[k]));
      }
      eNuc += t1 * (1.0 + t2 + t3) + t4 * (t5 + t6);
    }
  }
  return eNuc;
}

// Heat of formation (kcal/mol): (E_elec + E_nuc - sum eisol) * eV->kcal + sum
// eheat. eElec/eNuc in eV.
NVMOLKIT_HD inline double heatOfFormationKcalDev(double eElec, double eNuc, int nAtoms,
                                                 const AtomIntParams* ap) {
  constexpr double kEvToKcal = 23.061;
  double eisol = 0.0, eheat = 0.0;
  for (int a = 0; a < nAtoms; ++a) {
    eisol += ap[a].eisol;
    eheat += ap[a].eheat;
  }
  return (eElec + eNuc - eisol) * kEvToKcal + eheat;
}

}  // namespace semiempirical
}  // namespace nvMolKit

#endif  // NVMOLKIT_SEMIEMPIRICAL_ENERGY_DEVICE_H
