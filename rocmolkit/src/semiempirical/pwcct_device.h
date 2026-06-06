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
// Device-callable PM6 pairwise core-core repulsion (PWCCT, Stewart 2007 / MOPAC),
// the canonical PM6 core-core that replaces the AM1-style term. Per atom pair
// (eV, distances in Angstrom):
//   E = Z_A Z_B (s_A s_A|s_B s_B) * parenthesis + 1e-8*((Z_A^1/3+Z_B^1/3)/R)^12
//       + Z_A Z_B / R * (g_A(R) + g_B(R))
// with (s_A s_A|s_B s_B) = e2 / sqrt(R^2 + (rho0_A+rho0_B)^2) the monopole integral
// (rho0 = pcore in Angstrom), the parenthesis carrying the per-pair x/alpha
// (special-cased for H-{N,O,C} and C-C), and g the per-element unpolarizable-core
// Gaussian a*exp(-b*(R-c)^2). Ported from SCINE Sparrow's PM6PairwiseRepulsion
// (BSD-3-Clause) in the paper's eV/Angstrom convention; validated bit-close to
// MOPAC's NUCLEAR-NUCLEAR REPULSION. The electronic SCF is unchanged (the
// core-core does not enter the Fock), so charges/eigenvalues stay bit-exact to
// PYSEQM while the energy / heat of formation aligns with canonical PM6.

#ifndef NVMOLKIT_SEMIEMPIRICAL_PWCCT_DEVICE_H
#define NVMOLKIT_SEMIEMPIRICAL_PWCCT_DEVICE_H

#include <cmath>

#include "device_macros.h"   // AtomIntParams
#include "pm6_params.h"   // pm6ValenceElectrons (core charge)
#include "pwcct_data.h"
#include "pwcct_ref_data.h"  // kHofRef, kEvToKcal

namespace nvMolKit {
namespace semiempirical {

namespace pwcctdetail {
constexpr double kE2 = 14.399645352;          // e^2 in eV*Angstrom (Hartree*bohr_in_A)
constexpr double kBohrToAng = 0.5291772109;   // pcore (bohr) -> rho0 (Angstrom)
constexpr double kExpCoef = 0.0003;           // Angstrom^-5 (paper)
constexpr double kFurtherExpCC = 5.98;        // Angstrom^-1 (paper)
constexpr double kFactorCC = 9.28;
constexpr double kCAdd = 1.0e-8;              // eV*Angstrom^12 (paper)
}  // namespace pwcctdetail

// Per-element unpolarizable-core Gaussian g(z, R) = a exp(-b (R-c)^2) (R in Angstrom).
NVMOLKIT_HD inline double pwcctGaussianDev(int z, double R) {
  using namespace pwcct;
  if (z > kPwcctMaxZ || kGaussA[z] == 0.0) return 0.0;
  const double rmc = R - kGaussC[z];
  return kGaussA[z] * std::exp(-kGaussB[z] * rmc * rmc);
}

// PM6 pairwise core-core energy between atoms zi, zj (with core charges zci, zcj =
// valence electrons) at distance R (Angstrom), eV. The core charge is passed in so
// the function is device-callable (the host-only pm6ValenceElectrons is not used
// here); on the GPU it comes from AtomIntParams.valence.
NVMOLKIT_HD inline double pwcctPairDev(int zi, int zj, double zci, double zcj, double R) {
  using namespace pwcct;
  using namespace pwcctdetail;
  const int nz = kPwcctMaxZ + 1;

  // Monopole (s_A s_A | s_B s_B) integral (eV), rho0 from pcore (bohr -> Angstrom).
  const double rho = (kPcore[zi] + kPcore[zj]) * kBohrToAng;
  const double ssss = kE2 / std::sqrt(R * R + rho * rho);

  // Pairwise parenthesis with the per-pair x / alpha and PM6 special cases.
  const double x = kPairX[zi * nz + zj], alpha = kPairAlpha[zi * nz + zj];
  const bool hNOC = (zi == 1 && (zj == 6 || zj == 7 || zj == 8)) ||
                    (zj == 1 && (zi == 6 || zi == 7 || zi == 8));
  double paren;
  if (hNOC) {
    paren = 1.0 + 2.0 * x * std::exp(-alpha * (R * R));
  } else {
    const double R2 = R * R, R6 = R2 * R2 * R2;
    paren = 1.0 + 2.0 * x * std::exp(-alpha * (R + kExpCoef * R6));
    if (zi == 6 && zj == 6) paren += kFactorCC * std::exp(-kFurtherExpCC * R);
    // (Si-O special case omitted: silicon is outside the supported set.)
  }

  const double za = zci, zb = zcj;
  const double baseTerm = za * zb * ssss * paren;
  const double v = (std::cbrt(static_cast<double>(zi)) + std::cbrt(static_cast<double>(zj))) / R;
  const double v2 = v * v, v6 = v2 * v2 * v2;
  const double addl = kCAdd * v6 * v6;
  const double gauss = za * zb / R * (pwcctGaussianDev(zi, R) + pwcctGaussianDev(zj, R));
  return baseTerm + addl + gauss;  // eV
}

// Total PM6 core-core (PWCCT) energy for a molecule (eV). atoms: 1-based Z;
// coords: nAtoms*3 row-major (Angstrom). Host-only (uses pm6ValenceElectrons for
// the core charges); the device/GPU path uses heatOfFormationPm6KcalAp.
inline double pwcctCoreCoreDev(int nAtoms, const int* atoms, const double* coords) {
  double e = 0.0;
  for (int i = 0; i < nAtoms - 1; ++i) {
    for (int j = i + 1; j < nAtoms; ++j) {
      const double dx = coords[3 * j] - coords[3 * i];
      const double dy = coords[3 * j + 1] - coords[3 * i + 1];
      const double dz = coords[3 * j + 2] - coords[3 * i + 2];
      e += pwcctPairDev(atoms[i], atoms[j], pm6ValenceElectrons(atoms[i]),
                        pm6ValenceElectrons(atoms[j]), std::sqrt(dx * dx + dy * dy + dz * dz));
    }
  }
  return e;
}

// Canonical (MOPAC-aligned) PM6 heat of formation (kcal/mol) from the converged
// electronic energy eElec (eV): HoF = kEvToKcal*(eElec + E_core_PWCCT) - sum ref.
// Matches MOPAC PM6 to ~1 kcal/mol for light + Br molecules (iodine looser).
// Host-only (uses pwcctCoreCoreDev); the GPU path uses heatOfFormationPm6KcalAp.
inline double heatOfFormationPm6Kcal(double eElec, int nAtoms, const int* atoms,
                                     const double* coords) {
  using namespace pwcct;
  double ref = 0.0;
  for (int a = 0; a < nAtoms; ++a) {
    // kHofRef is calibrated (and the PWCCT core-core is faithful) only for the
    // validated element set; for any other element the absolute energetics are
    // uncalibrated, so the HoF is unavailable (NaN) rather than silently wrong.
    if (kHofRef[atoms[a]] == 0.0) return NAN;
    ref += kHofRef[atoms[a]];
  }
  return kEvToKcal * (eElec + pwcctCoreCoreDev(nAtoms, atoms, coords)) - ref;
}

// Same, sourcing the atomic numbers from gathered AtomIntParams (.z) -- for the
// GPU batch kernel, which carries ap rather than a separate Z array.
NVMOLKIT_HD inline double heatOfFormationPm6KcalAp(double eElec, int nAtoms,
                                                   const AtomIntParams* ap, const double* coords) {
  using namespace pwcct;
  double ref = 0.0, ecc = 0.0;
  for (int a = 0; a < nAtoms; ++a) {
    if (kHofRef[ap[a].z] == 0.0) return NAN;  // uncalibrated element -> HoF unavailable
    ref += kHofRef[ap[a].z];
  }
  for (int i = 0; i < nAtoms - 1; ++i)
    for (int j = i + 1; j < nAtoms; ++j) {
      const double dx = coords[3 * j] - coords[3 * i];
      const double dy = coords[3 * j + 1] - coords[3 * i + 1];
      const double dz = coords[3 * j + 2] - coords[3 * i + 2];
      ecc += pwcctPairDev(ap[i].z, ap[j].z, ap[i].valence, ap[j].valence,
                          std::sqrt(dx * dx + dy * dy + dz * dz));
    }
  return kEvToKcal * (eElec + ecc) - ref;
}

}  // namespace semiempirical
}  // namespace nvMolKit

#endif  // NVMOLKIT_SEMIEMPIRICAL_PWCCT_DEVICE_H
