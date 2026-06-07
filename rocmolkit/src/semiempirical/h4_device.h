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
// Device-callable PM6-D3H4 post-SCF corrections: the H4 hydrogen-bond term
// (Rezac-Hobza 2012) over (donor-H...acceptor) N/O triples, the short-range H-H
// repulsion, and the combined D3+H4+HH correction (kcal/mol) added to the PM6_D
// heat of formation. Ported bit-exact from mlxmolkit's rm1.pm6_d3h4 (BSD-3).

#ifndef NVMOLKIT_SEMIEMPIRICAL_H4_DEVICE_H
#define NVMOLKIT_SEMIEMPIRICAL_H4_DEVICE_H

#include <cmath>

#include "d3_device.h"  // pm6dD3Energy
#include "device_macros.h"

namespace nvMolKit {
namespace semiempirical {
namespace h4detail {

// PM6-D3H4 H4 parameters and Rezac covalent radii (Angstrom), indexed by Z.
constexpr double kParaOHO = 2.32, kParaOHN = 3.1, kParaNHO = 1.07, kParaNHN = 2.01;
constexpr double kMultWHO = 0.42, kMultNH4 = 3.61, kMultCOO = 1.41;

NVMOLKIT_HD inline double covRadiusH4(int z) {
  switch (z) {
    case 1: return 0.37;  case 6: return 0.77;  case 7: return 0.75;  case 8: return 0.73;
    case 9: return 0.71;  case 15: return 1.06; case 16: return 1.02; case 17: return 0.99;
    case 35: return 1.14; case 53: return 1.33; default: return 0.0;
  }
}

NVMOLKIT_HD inline double dist(const double* c, int a, int b) {
  const double dx = c[3 * a] - c[3 * b], dy = c[3 * a + 1] - c[3 * b + 1],
               dz = c[3 * a + 2] - c[3 * b + 2];
  return std::sqrt(dx * dx + dy * dy + dz * dz);
}

// Rezac smooth-step valence contribution of the (a,b) bond (1 covalent -> 0).
NVMOLKIT_HD inline double cvalence(const int* z, const double* c, int a, int b) {
  const double r0 = covRadiusH4(z[a]) + covRadiusH4(z[b]);
  const double r1 = r0 * 1.6;
  const double r = dist(c, a, b);
  if (r == 0.0 || r >= r1) return 0.0;
  if (r <= r0) return 1.0;
  const double x = (r - r0) / (r1 - r0);
  return 1.0 - (-20.0 * std::pow(x, 7) + 70.0 * std::pow(x, 6) - 84.0 * std::pow(x, 5)
                + 35.0 * std::pow(x, 4));
}

// H4 contribution of one (atom_i - h - atom_j) triple (kcal/mol).
NVMOLKIT_HD inline double h4Triple(int nAtoms, const int* z, const double* c, int h, int ai,
                                   int aj) {
  const double rih = dist(c, h, ai), rjh = dist(c, h, aj);
  const double v1x = c[3 * ai] - c[3 * h], v1y = c[3 * ai + 1] - c[3 * h + 1],
               v1z = c[3 * ai + 2] - c[3 * h + 2];
  const double v2x = c[3 * aj] - c[3 * h], v2y = c[3 * aj + 1] - c[3 * h + 1],
               v2z = c[3 * aj + 2] - c[3 * h + 2];
  const double n1 = std::sqrt(v1x * v1x + v1y * v1y + v1z * v1z);
  const double n2 = std::sqrt(v2x * v2x + v2y * v2y + v2z * v2z);
  double cosA = (v1x * v2x + v1y * v2y + v1z * v2z) / (n1 * n2 > 1e-12 ? n1 * n2 : 1e-12);
  cosA = cosA > 1.0 ? 1.0 : (cosA < -1.0 ? -1.0 : cosA);
  const double angle = M_PI - std::acos(cosA);
  if (angle >= M_PI / 2.0) return 0.0;

  int di, a_i;
  double rdh, rah;
  if (rih < rjh) { di = ai; a_i = aj; rdh = rih; rah = rjh; }
  else { di = aj; a_i = ai; rdh = rjh; rah = rih; }

  const double rda = dist(c, ai, aj);
  const double eRadial = -0.00303407407407313510 * std::pow(rda, 7)
      + 0.07357629629627092382 * std::pow(rda, 6) - 0.70087111111082800452 * std::pow(rda, 5)
      + 3.25309629629461749545 * std::pow(rda, 4) - 7.20687407406838786983 * std::pow(rda, 3)
      + 5.31754666665572184314 * rda * rda + 3.40736000001102778967 * rda - 4.68512000000450434811;

  const double a = angle / (M_PI / 2.0);
  const double xa = -20.0 * std::pow(a, 7) + 70.0 * std::pow(a, 6) - 84.0 * std::pow(a, 5)
                    + 35.0 * std::pow(a, 4);
  const double eAngular = 1.0 - xa * xa;

  const int zd = z[di], za = z[a_i];
  double ePara = 0.0;
  if (zd == 8 && za == 8) ePara = kParaOHO;
  else if (zd == 8 && za == 7) ePara = kParaOHN;
  else if (zd == 7 && za == 8) ePara = kParaNHO;
  else if (zd == 7 && za == 7) ePara = kParaNHN;
  if (ePara == 0.0) return 0.0;

  double eBondSwitch = 1.0;
  if (rdh > 1.15) {
    const double rdhs = rdh - 1.15;
    const double ravgs = 0.5 * rdh + 0.5 * rah - 1.15;
    const double x = rdhs / (ravgs > 1e-12 ? ravgs : 1e-12);
    eBondSwitch = 1.0 - (-20.0 * std::pow(x, 7) + 70.0 * std::pow(x, 6) - 84.0 * std::pow(x, 5)
                         + 35.0 * std::pow(x, 4));
  }

  double eScaleW = 1.0;
  if (zd == 8 && za == 8) {
    double hyd = 0.0, oth = 0.0;
    for (int k = 0; k < nAtoms; ++k) {
      if (z[k] == 1) hyd += cvalence(z, c, di, k);
      else oth += cvalence(z, c, di, k);
    }
    if (hyd >= 1.0) {
      const double slope = kMultWHO - 1.0;
      double fv = 0.0;
      if (hyd > 1.0 && hyd <= 2.0) fv = hyd - 1.0;
      if (hyd > 2.0 && hyd < 3.0) fv = 3.0 - hyd;
      const double fv2 = (1.0 - oth) > 0.0 ? (1.0 - oth) : 0.0;
      eScaleW = 1.0 + slope * fv * fv2;
    }
  }

  double eScaleChd = 1.0;
  if (zd == 7) {
    const double slope = kMultNH4 - 1.0;
    double v = 0.0;
    for (int k = 0; k < nAtoms; ++k) v += cvalence(z, c, di, k);
    v = (v > 3.0) ? (v - 3.0) : 0.0;
    eScaleChd = 1.0 + slope * v;
  }

  double eScaleCha = 1.0;
  if (za == 8) {
    const double slope = kMultCOO - 1.0;
    double cdist = 1e300, cvO1 = 0.0;
    int cc = -1;
    for (int k = 0; k < nAtoms; ++k) {
      const double v = cvalence(z, c, a_i, k);
      cvO1 += v;
      if (v > 0.0 && z[k] == 6) {
        const double d = dist(c, k, a_i);
        if (d < cdist) { cdist = d; cc = k; }
      }
    }
    if (cc != -1) {
      double odist = 1e300, cvCC = 0.0;
      int o2 = -1;
      for (int k = 0; k < nAtoms; ++k) {
        const double v = cvalence(z, c, cc, k);
        cvCC += v;
        if (v > 0.0 && k != a_i && z[k] == 8) {
          const double d = dist(c, k, cc);
          if (d < odist) { odist = d; o2 = k; }
        }
      }
      if (o2 != -1) {
        double cvO2 = 0.0;
        for (int k = 0; k < nAtoms; ++k) cvO2 += cvalence(z, c, o2, k);
        const double fO1 = (1.0 - std::fabs(1.0 - cvO1)) > 0.0 ? (1.0 - std::fabs(1.0 - cvO1)) : 0.0;
        const double fO2 = (1.0 - std::fabs(1.0 - cvO2)) > 0.0 ? (1.0 - std::fabs(1.0 - cvO2)) : 0.0;
        const double fCC = (1.0 - std::fabs(3.0 - cvCC)) > 0.0 ? (1.0 - std::fabs(3.0 - cvCC)) : 0.0;
        eScaleCha = 1.0 + slope * fO1 * fO2 * fCC;
      }
    }
  }
  return ePara * eRadial * eAngular * eBondSwitch * eScaleW * eScaleChd * eScaleCha;
}

NVMOLKIT_HD inline double polyHH(double r) {
  if (r <= 1.0) return 25.46293603147693;
  if (r < 1.5)
    return -2714.952351603469651 * std::pow(r, 5) + 17103.650110591705015 * std::pow(r, 4)
           - 42511.857982217959943 * std::pow(r, 3) + 52063.196799138342612 * r * r
           - 31430.658335972289933 * r + 7516.084696095140316;
  return 118.7326 * std::exp(-1.53965 * std::pow(r, 1.72905));
}

}  // namespace h4detail

// PM6-D3H4 H4 hydrogen-bond correction (kcal/mol).
NVMOLKIT_HD inline double pm6dH4Energy(int nAtoms, const int* atoms, const double* coords) {
  using namespace h4detail;
  double e = 0.0;
  for (int h = 0; h < nAtoms; ++h) {
    if (atoms[h] != 1) continue;
    for (int ai = 0; ai < nAtoms; ++ai) {
      if (atoms[ai] != 7 && atoms[ai] != 8) continue;
      for (int aj = ai + 1; aj < nAtoms; ++aj) {
        if (atoms[aj] != 7 && atoms[aj] != 8) continue;
        e += h4Triple(nAtoms, atoms, coords, h, ai, aj);
      }
    }
  }
  return e;
}

// Short-range H-H repulsion (kcal/mol).
NVMOLKIT_HD inline double pm6dHHRepulsion(int nAtoms, const int* atoms, const double* coords) {
  using namespace h4detail;
  double e = 0.0;
  for (int i = 0; i < nAtoms; ++i) {
    if (atoms[i] != 1) continue;
    for (int j = 0; j < i; ++j) {
      if (atoms[j] != 1) continue;
      e += polyHH(dist(coords, i, j));
    }
  }
  return e;
}

// Full PM6-D3H4 post-SCF correction (kcal/mol): D3 dispersion + H4 H-bond + H-H.
NVMOLKIT_HD inline double pm6dD3H4Correction(int nAtoms, const int* atoms, const double* coords) {
  return pm6dD3Energy(nAtoms, atoms, coords) + pm6dH4Energy(nAtoms, atoms, coords)
         + pm6dHHRepulsion(nAtoms, atoms, coords);
}

}  // namespace semiempirical
}  // namespace nvMolKit

#endif  // NVMOLKIT_SEMIEMPIRICAL_H4_DEVICE_H
