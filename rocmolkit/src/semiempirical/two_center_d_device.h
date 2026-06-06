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
// Device-callable d-orbital two-center integrals, YH case (d-atom A + H). Builds
// the 9x9 (mu nu_A | s_B s_B) matrix W: sp part from the validated two-center
// integrals, d part from the riYH multipole formulas (using per-element d charge
// separations baked from PYSEQM's pyseqm_d_params), then the molecular d rotation
// (D 5x5 + P 3x3 + packed rotate_core) — all ported from PYSEQM and validated
// bit-exact vs the oracle (tools/semiempirical/validate_yh.py + golden_yh_e1b).
// The H_core electron-core attraction is e1b = -Z_B * W.

#ifndef NVMOLKIT_SEMIEMPIRICAL_TWO_CENTER_D_DEVICE_H
#define NVMOLKIT_SEMIEMPIRICAL_TWO_CENTER_D_DEVICE_H

#include <cmath>

#include "device_macros.h"
#include "two_center_device.h"  // computeMultipoleParamsDev, twoCenterLocalDev

namespace nvMolKit {
namespace semiempirical {

// Per-element d charge separations (dp, ds, dorbdorb) + additive radii
// (rho3..rho6), baked from PYSEQM's pyseqm_d_params (Slater-Condon constants).
// Returns false if z has no d parameters in this set.
NVMOLKIT_HD inline bool dChargeSeparations(int z, double& dp, double& ds, double& dd,
                                           double& rho3, double& rho4, double& rho5, double& rho6) {
  if (z == 13) {  // Al (qn3 sp, qnD3 d) — derived from the MOPAC PM6 CSV tail
                  // exponents (4.74234, 4.66963, 7.13114) via pyseqm_d_params;
                  // PYSEQM/mlxmolkit have no Al entry, so these come straight from
                  // MOPAC's parameter set (see tools/semiempirical/gen_pm6d_chargesep.py).
    dp = 0.94828440; ds = 1.25286620; dd = 1.57557233;
    rho3 = 0.27161832; rho4 = 0.57631046; rho5 = 0.44796295; rho6 = 0.70819641;
  } else if (z == 14) {  // Si (qn3 sp, qnD3 d) — MOPAC PM6 CSV tail exponents
                         // (8.38811, 1.84305, 0.70860). rho5 hits the POIJ ceiling (5.0).
    dp = 0.70768770; ds = 1.36284126; dd = 0.93958912;
    rho3 = 2.73348617; rho4 = 1.60785370; rho5 = 5.00000000; rho6 = 1.07941668;
  } else if (z == 15) {  // P
    dp = 0.90744734; ds = 1.38477069; dd = 1.62554049;
    rho3 = 0.27098714; rho4 = 1.36916053; rho5 = 0.38883255; rho6 = 0.71981849;
  } else if (z == 16) {  // S
    dp = 0.49861628; ds = 0.96176716; dd = 0.64321070;
    rho3 = 0.44863040; rho4 = 1.89216748; rho5 = 3.23024211; rho6 = 0.48881126;
  } else if (z == 17) {  // Cl
    dp = 0.75100923; ds = 1.10740952; dd = 1.51053979;
    rho3 = 0.30216047; rho4 = 1.03731998; rho5 = 2.34288535; rho6 = 0.72430196;
  } else if (z == 35) {  // Br (qn4 sp, qnD4 d)
    dp = 1.02888957; ds = 0.56131331; dd = 1.66693804;
    rho3 = 0.88253204; rho4 = 0.82718856; rho5 = 0.36545910; rho6 = 1.13300353;
  } else if (z == 53) {  // I (qn5 sp, qnD5 d) -- canonical zeta_d=1.87518 (was a
                         // corrupted 2.72301 inherited from mlxmolkit's hardcoded
                         // pm6_params.py; PYSEQM CSV + MOPAC both use 1.87518)
    dp = 1.29634098; ds = 0.77747287; dd = 1.63749501;
    rho3 = 0.80039306; rho4 = 1.20097689; rho5 = 0.71635330; rho6 = 1.06583498;
  } else {
    return false;
  }
  return true;
}

// d charge separations for the two-center two-electron tensors (YX/YY). These
// match dChargeSeparations (pyseqm_d_params) for P/S/Cl/Br, but for iodine the
// oracle's YX/YY path uses cal_par's qn5 values, which differ from the (qn3-like)
// pyseqm_d_params that the YH electron-core path uses for iodine. Returns false
// if z has no d parameters.
NVMOLKIT_HD inline bool dChargeSeparationsTwoCenter(int z, double& dp, double& ds, double& dd,
                                                    double& rho3, double& rho4, double& rho5,
                                                    double& rho6) {
  if (z == 53) {  // I — cal_par (qn5) values, distinct from pyseqm_d_params
    dp = 1.29634100; ds = 0.77747300; dd = 1.63749500;
    rho3 = 0.80039300; rho4 = 1.20097700; rho5 = 0.71635300; rho6 = 1.06583500;
    return true;
  }
  return dChargeSeparations(z, dp, ds, dd, rho3, rho4, rho5, rho6);
}

// 9x9 (mu nu_A | s_B s_B) matrix for the YH case (zA has d, zB = H). Writes W
// row-major into out (81 doubles). Returns false if zA has no d parameters.
NVMOLKIT_HD inline bool yhWMolecular(const AtomIntParams& pA, const double coordA[3],
                                     const AtomIntParams& pB, const double coordB[3], double* out) {
  using namespace detail;
  double dp, ds, dd, rho3, rho4, rho5, rho6;
  if (!dChargeSeparations(pA.z, dp, ds, dd, rho3, rho4, rho5, rho6)) return false;

  const double Rvec[3] = {coordB[0] - coordA[0], coordB[1] - coordA[1], coordB[2] - coordA[2]};
  const double R = std::sqrt(Rvec[0] * Rvec[0] + Rvec[1] * Rvec[1] + Rvec[2] * Rvec[2]);
  const double Rb = R * kSemiAngToBohr;

  double daA, qaA, rho0A, rho1A, rho2A, daB, qaB, rho0B, rho1B, rho2B;
  computeMultipoleParamsDev(pA, daA, qaA, rho0A, rho1A, rho2A);
  computeMultipoleParamsDev(pB, daB, qaB, rho0B, rho1B, rho2B);

  const double ev1 = kSemiEV / 2.0, ev2 = kSemiEV / 4.0;
  auto sq = [](double x) { return x * x; };
  const double ddq = kSemiEV / std::sqrt(Rb * Rb + sq(rho3 + rho0B));
  const double dpuz = ev1 / std::sqrt(sq(Rb + dp) + sq(rho4 + rho0B))
                      - ev1 / std::sqrt(sq(Rb - dp) + sq(rho4 + rho0B));
  const double ddqd = ev2 / std::sqrt(sq(Rb - dd) + sq(rho6 + rho0B))
                      + ev2 / std::sqrt(sq(Rb + dd) + sq(rho6 + rho0B))
                      - ev1 / std::sqrt(Rb * Rb + dd * dd + sq(rho6 + rho0B));
  const double dsq = ev2 / std::sqrt(sq(Rb - ds) + sq(rho5 + rho0B))
                     + ev2 / std::sqrt(sq(Rb + ds) + sq(rho5 + rho0B))
                     - ev1 / std::sqrt(Rb * Rb + ds * ds + sq(rho5 + rho0B));
  const double ri10 = dsq * 1.154701, ri11 = dpuz * 1.154701, ri14 = ddq + ddqd * 1.333333;
  const double ri17 = dpuz, ri20 = ddq + ddqd * 0.666667, ri44 = ddq + ddqd * -1.333333;

  // sp part of (mu nu_A | s_B s_B): twoCenterLocal XH gives [ss, ssigma, sigma2, pi2].
  double riXh[22], coreXh[8];
  int pt;
  if (twoCenterLocalDev(pA, pB, R, riXh, coreXh, &pt) == 0) return false;
  const double ss = riXh[0], ps = riXh[1], ppsig = riXh[2], pppi = riXh[3];

  // Orbital-rotation matrices D (5x5) and P (3x3) from the negated bond vector.
  const double inv = 1.0 / R;
  double v[3] = {-Rvec[0] * inv, -Rvec[1] * inv, -Rvec[2] * inv};
  const double xy = std::sqrt(v[0] * v[0] + v[1] * v[1]);
  double ca, sa, cb, sb;
  if (xy >= 1e-10) { ca = v[0] / xy; sa = v[1] / xy; cb = v[2]; sb = xy; }
  else { cb = (v[2] > 0) ? 1.0 : -1.0; ca = cb; sa = 0.0; sb = 0.0; }
  const double c2a = 2 * ca * ca - 1, c2b = 2 * cb * cb - 1, s2a = 2 * sa * ca, s2b = 2 * sb * cb;
  const double PT5 = 0.5, PT5SQ3 = 0.5 * std::sqrt(3.0);
  double D[5][5] = {{0}};
  D[0][0] = PT5SQ3 * c2a * sb * sb; D[1][0] = PT5 * c2a * s2b; D[2][0] = -s2a * sb;
  D[3][0] = c2a * (cb * cb + PT5 * sb * sb); D[4][0] = -s2a * cb;
  D[0][1] = PT5SQ3 * ca * s2b; D[1][1] = ca * c2b; D[2][1] = -sa * cb;
  D[3][1] = -PT5 * ca * s2b; D[4][1] = sa * sb;
  D[0][2] = cb * cb - PT5 * sb * sb; D[1][2] = -PT5SQ3 * s2b; D[3][2] = PT5SQ3 * sb * sb;
  D[0][3] = PT5SQ3 * sa * s2b; D[1][3] = sa * c2b; D[2][3] = ca * cb;
  D[3][3] = -PT5 * sa * s2b; D[4][3] = -ca * sb;
  D[0][4] = PT5SQ3 * s2a * sb * sb; D[1][4] = PT5 * s2a * s2b; D[2][4] = c2a * sb;
  D[3][4] = s2a * (cb * cb + PT5 * sb * sb); D[4][4] = c2a * cb;
  double Pm[3][3] = {{0}};
  Pm[0][0] = ca * sb; Pm[1][0] = ca * cb; Pm[2][0] = -sa;
  Pm[0][1] = sa * sb; Pm[1][1] = sa * cb; Pm[2][1] = ca;
  Pm[0][2] = cb; Pm[1][2] = -sb;

  double rc[45];
  for (int i = 0; i < 45; ++i) rc[i] = 0.0;
  rc[0] = ss;
  const int psr[3] = {1, 3, 6};
  for (int i = 0; i < 3; ++i) rc[psr[i]] = ps * Pm[0][i];
  const int ppr[6] = {2, 4, 5, 7, 8, 9};
  auto ppc = [&](int K, int I) {
    const double a[6] = {Pm[K][0] * Pm[K][0], Pm[K][0] * Pm[K][1], Pm[K][1] * Pm[K][1],
                         Pm[K][0] * Pm[K][2], Pm[K][1] * Pm[K][2], Pm[K][2] * Pm[K][2]};
    return a[I];
  };
  for (int i = 0; i < 6; ++i) rc[ppr[i]] = ppsig * ppc(0, i) + pppi * (ppc(1, i) + ppc(2, i));
  const int dsr[5] = {10, 15, 21, 28, 36};
  for (int i = 0; i < 5; ++i) rc[dsr[i]] = ri10 * D[0][i];
  const int dpr[15] = {11, 12, 13, 16, 17, 18, 22, 23, 24, 29, 30, 31, 37, 38, 39};
  for (int i = 0; i < 15; ++i)
    rc[dpr[i]] = ri11 * (D[0][i / 3] * Pm[0][i % 3])
                 + ri17 * (D[1][i / 3] * Pm[1][i % 3] + D[2][i / 3] * Pm[2][i % 3]);
  const int ddr[15] = {14, 19, 20, 25, 26, 27, 32, 33, 34, 35, 40, 41, 42, 43, 44};
  auto ddc = [&](int K, int I) {
    const double a[15] = {D[K][0] * D[K][0], D[K][0] * D[K][1], D[K][1] * D[K][1],
                          D[K][0] * D[K][2], D[K][1] * D[K][2], D[K][2] * D[K][2],
                          D[K][0] * D[K][3], D[K][1] * D[K][3], D[K][2] * D[K][3], D[K][3] * D[K][3],
                          D[K][0] * D[K][4], D[K][1] * D[K][4], D[K][2] * D[K][4], D[K][3] * D[K][4],
                          D[K][4] * D[K][4]};
    return a[I];
  };
  for (int i = 0; i < 15; ++i)
    rc[ddr[i]] = ri14 * ddc(0, i) + ri20 * (ddc(1, i) + ddc(2, i)) + ri44 * (ddc(3, i) + ddc(4, i));

  const int indx[9] = {0, 1, 3, 6, 10, 15, 21, 28, 36};
  for (int i = 0; i < 9; ++i)
    for (int j = 0; j <= i; ++j) {
      out[i * 9 + j] = rc[indx[i] + j];
      out[j * 9 + i] = out[i * 9 + j];
    }
  return true;
}

}  // namespace semiempirical
}  // namespace nvMolKit

#endif  // NVMOLKIT_SEMIEMPIRICAL_TWO_CENTER_D_DEVICE_H
