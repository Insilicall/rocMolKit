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

// Pairwise core-core terms (PWCCT) for PM6: (chi, alpha) per element pair (public
// Stewart PM6 / MOPAC). Symmetric; pairs not listed contribute (0, 0). Covers the
// sp elements H/C/N/O/F/P/S/Cl.
NVMOLKIT_HD inline void getPwcct(int z1, int z2, double& chi, double& alp) {
  const int a = z1 < z2 ? z1 : z2;
  const int b = z1 < z2 ? z2 : z1;
  // {min, max, chi, alpha}
  const double kTab[36][4] = {
      {1, 1, 2.24359, 3.54094}, {1, 6, 0.21651, 1.02781}, {6, 6, 0.81351, 2.61371},
      {1, 7, 0.17551, 0.96941}, {6, 7, 0.85995, 2.68611}, {7, 7, 0.67531, 2.5745},
      {1, 8, 0.19229, 1.26094}, {6, 8, 0.99021, 2.88961}, {7, 8, 0.76476, 2.78429},
      {8, 8, 0.535112, 2.623998}, {1, 9, 0.8158, 3.13674}, {6, 9, 0.73297, 3.0276},
      {7, 9, 0.63585, 2.85665}, {8, 9, 0.67425, 3.01544}, {9, 9, 0.68134, 3.17576},
      {1, 15, 1.23499, 1.92654}, {6, 15, 0.97951, 1.99465}, {7, 15, 0.97215, 2.14704},
      {8, 15, 0.8787, 2.22077}, {9, 15, 0.51458, 2.23436}, {15, 15, 0.9025, 1.50579},
      {1, 16, 0.84971, 2.21597}, {6, 16, 0.66685, 2.2103}, {7, 16, 0.73871, 2.28999},
      {8, 16, 0.74721, 2.38329}, {9, 16, 0.37525, 2.18719}, {15, 16, 0.56227, 1.59533},
      {16, 16, 0.473856, 1.794556}, {1, 17, 0.75483, 2.40289}, {6, 17, 0.51579, 2.1622},
      {7, 17, 0.52075, 2.17213}, {8, 17, 0.58551, 2.32324}, {9, 17, 0.41112, 2.31327},
      {15, 17, 0.35236, 1.46831}, {16, 17, 0.35697, 1.71544}, {17, 17, 0.33292, 1.82324}};
  for (int i = 0; i < 36; ++i) {
    if (static_cast<int>(kTab[i][0]) == a && static_cast<int>(kTab[i][1]) == b) {
      chi = kTab[i][2];
      alp = kTab[i][3];
      return;
    }
  }
  chi = 0.0;
  alp = 0.0;
}

// PM6 core-core (nuclear) repulsion (eV): (ss|ss) Coulomb with the pairwise PWCCT
// term, the unpolarized-core repulsion, the C/N/O-H and C-C special cases, and
// the per-element Gaussian corrections. coords in Angstrom. Exact PYSEQM PM6.
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
      const double gam = kEV / std::sqrt(Rb * Rb + (rho0A + rho0B) * (rho0A + rho0B));

      const double ZA = static_cast<double>(ap[i].valence);
      const double ZB = static_cast<double>(ap[j].valence);
      const int zA = ap[i].z, zB = ap[j].z;

      const double cbrt = std::pow(static_cast<double>(zA), 1.0 / 3.0)
                          + std::pow(static_cast<double>(zB), 1.0 / 3.0);
      double ratio = cbrt / R;
      const double r6 = ratio * ratio * ratio;
      const double unpolcore = 1e-8 * (r6 * r6) * (r6 * r6);  // (cbrt/R)^12

      double chi, alp;
      getPwcct(zA, zB, chi, alp);
      const bool isXH = ((zA == 6 || zA == 7 || zA == 8) && zB == 1)
                        || ((zB == 6 || zB == 7 || zB == 8) && zA == 1);
      double expo2;
      if (isXH) {
        expo2 = unpolcore + ZA * ZB * gam * (1.0 + 2.0 * chi * std::exp(-alp * R * R));
      } else {
        const double f = R + 0.0003 * std::pow(R, 6);
        expo2 = unpolcore + ZA * ZB * gam * (1.0 + 2.0 * chi * std::exp(-alp * f));
      }
      if (zA == 6 && zB == 6) {
        expo2 += ZA * ZB * gam * 9.28 * std::exp(-5.98 * R);
      }

      const double t4 = ZA * ZB / R;
      double t5 = 0.0, t6 = 0.0;
      for (int k = 0; k < 4; ++k) {
        if (ap[i].gaussK[k] != 0.0)
          t5 += ap[i].gaussK[k] * std::exp(-ap[i].gaussL[k] * (R - ap[i].gaussM[k]) * (R - ap[i].gaussM[k]));
        if (ap[j].gaussK[k] != 0.0)
          t6 += ap[j].gaussK[k] * std::exp(-ap[j].gaussL[k] * (R - ap[j].gaussM[k]) * (R - ap[j].gaussM[k]));
      }
      eNuc += expo2 + t4 * (t5 + t6);
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
