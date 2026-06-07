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
// Device-callable two-electron two-center integral math (Dewar-Thiel multipole
// model). The SAME inline __host__ __device__ code feeds the CPU reference
// (two_center.cpp wrappers) and the HIP kernels, so the bit-exact validation
// covers both. Operates on gathered AtomIntParams — no parameter-table lookup.

#ifndef NVMOLKIT_SEMIEMPIRICAL_TWO_CENTER_DEVICE_H
#define NVMOLKIT_SEMIEMPIRICAL_TWO_CENTER_DEVICE_H

#include <cmath>

#include "device_macros.h"

namespace nvMolKit {
namespace semiempirical {

// pairType codes returned by twoCenterLocalDev.
enum { kPairHH = 0, kPairXH = 1, kPairXX = 2 };

namespace detail {

constexpr double kSemiEV = 27.21;                 // Hartree -> eV (MOPAC)
constexpr double kSemiAngToBohr = 1.0 / 0.529167;

NVMOLKIT_HD inline void computeMultipoleParamsDev(const AtomIntParams& p, double& da, double& qa,
                                                  double& rho0, double& rho1, double& rho2) {
  rho0 = (p.gss > 1e-10) ? 0.5 * kSemiEV / p.gss : 0.0;
  if (p.nOrb == 1) {
    da = qa = rho1 = rho2 = 0.0;
    return;
  }
  const double qn = static_cast<double>(p.qn);
  const double zs = p.zetaS, zp = p.zetaP;

  da = (2.0 * qn + 1.0) * std::pow(4.0 * zs * zp, qn + 0.5)
       / std::pow(zs + zp, 2.0 * qn + 2.0) / std::sqrt(3.0);
  qa = std::sqrt((4.0 * qn * qn + 6.0 * qn + 2.0) / 20.0) / zp;

  if (p.hsp > 0.0) {
    const double hspAu = p.hsp / kSemiEV;
    const double D1 = da;
    double d1 = std::pow(std::fabs(hspAu / (D1 * D1)), 1.0 / 3.0);
    if (hspAu < 0.0) d1 = -d1;
    double d2 = d1 + 0.04;
    for (int it = 0; it < 5; ++it) {
      const double h1 = 0.5 * d1 - 0.5 / std::sqrt(4.0 * D1 * D1 + 1.0 / (d1 * d1));
      const double h2 = 0.5 * d2 - 0.5 / std::sqrt(4.0 * D1 * D1 + 1.0 / (d2 * d2));
      const double d3 = (std::fabs(h2 - h1) > 1e-16) ? d1 + (d2 - d1) * (hspAu - h1) / (h2 - h1) : d2;
      d1 = d2;
      d2 = d3;
    }
    rho1 = 0.5 / d2;
  } else {
    rho1 = 0.0;
  }

  double hpp = 0.5 * (p.gpp - p.gp2);
  if (hpp < 0.1) hpp = 0.1;
  const double hppAu = hpp / kSemiEV;
  const double D2 = qa;
  double q1 = std::pow(std::fabs(hppAu / 3.0 / (D2 * D2 * D2 * D2)), 0.2);
  if (hppAu < 0.0) q1 = -q1;
  double q2 = q1 + 0.04;
  for (int it = 0; it < 5; ++it) {
    const double p1 = 0.25 * q1 - 0.5 / std::sqrt(4.0 * D2 * D2 + 1.0 / (q1 * q1))
                      + 0.25 / std::sqrt(8.0 * D2 * D2 + 1.0 / (q1 * q1));
    const double p2 = 0.25 * q2 - 0.5 / std::sqrt(4.0 * D2 * D2 + 1.0 / (q2 * q2))
                      + 0.25 / std::sqrt(8.0 * D2 * D2 + 1.0 / (q2 * q2));
    const double q3 = (std::fabs(p2 - p1) > 1e-16) ? q1 + (q2 - q1) * (hppAu - p1) / (p2 - p1) : q2;
    q1 = q2;
    q2 = q3;
  }
  rho2 = 0.5 / q2;
}

NVMOLKIT_HD inline void rotationMatrixDev(const double v[3], double rot[3][3]) {
  const double vx = v[0], vy = v[1], vz = v[2];
  const double w = 1.0 + vx;
  if (std::fabs(w) < 1e-7) {
    rot[0][0] = -1; rot[0][1] = 0; rot[0][2] = 0;
    rot[1][0] = 0;  rot[1][1] = -1; rot[1][2] = 0;
    rot[2][0] = 0;  rot[2][1] = 0;  rot[2][2] = 1;
    return;
  }
  double qy = vz, qz = -vy, qw = w;
  const double norm = std::sqrt(qy * qy + qz * qz + qw * qw);
  qy /= norm; qz /= norm; qw /= norm;
  rot[0][0] = 1 - 2 * (qy * qy + qz * qz);
  rot[0][1] = -2 * qz * qw;
  rot[0][2] = 2 * qy * qw;
  rot[1][0] = 2 * qz * qw;
  rot[1][1] = 1 - 2 * qz * qz;
  rot[1][2] = 2 * qy * qz;
  rot[2][0] = -2 * qy * qw;
  rot[2][1] = 2 * qy * qz;
  rot[2][2] = 1 - 2 * qy * qy;
}

NVMOLKIT_HD inline int wIdxDev(int mu, int nu, int lam, int sig) {
  return ((mu * 4 + nu) * 4 + lam) * 4 + sig;
}

}  // namespace detail

// Local-frame two-electron integrals. For XH the heavy atom must be A. Writes ri
// (1/4/22) and core (2/5/8); returns the ri count, or 0 on a misuse/empty shell.
NVMOLKIT_HD inline int twoCenterLocalDev(const AtomIntParams& pA, const AtomIntParams& pB,
                                         double R_ang, double* ri, double* core, int* pairType) {
  using namespace detail;
  const int nA = pA.nOrb, nB = pB.nOrb;
  if (nA == 0 || nB == 0) return 0;
  if (nA == 1 && nB > 1) return 0;

  const double R = R_ang * kSemiAngToBohr;
  double daA, qaA, rho0A, rho1A, rho2A, daB, qaB, rho0B, rho1B, rho2B;
  computeMultipoleParamsDev(pA, daA, qaA, rho0A, rho1A, rho2A);
  computeMultipoleParamsDev(pB, daB, qaB, rho0B, rho1B, rho2B);

  const double ev1 = kSemiEV / 2.0, ev2 = kSemiEV / 4.0, ev3 = kSemiEV / 8.0, ev4 = kSemiEV / 16.0;
  const double ZA = static_cast<double>(pA.valence);
  const double ZB = static_cast<double>(pB.valence);

  if (nA == 1 && nB == 1) {
    const double aee = (rho0A + rho0B) * (rho0A + rho0B);
    ri[0] = kSemiEV / std::sqrt(R * R + aee);
    core[0] = ZB * ri[0];
    core[1] = ZA * ri[0];
    *pairType = kPairHH;
    return 1;
  }

  if (nA > 1 && nB == 1) {
    const double da = daA;
    const double qa = qaA * 2.0;
    const double aee = (rho0A + rho0B) * (rho0A + rho0B);
    const double ade = (rho1A + rho0B) * (rho1A + rho0B);
    const double aqe = (rho2A + rho0B) * (rho2A + rho0B);
    const double ee = kSemiEV / std::sqrt(R * R + aee);
    ri[0] = ee;
    ri[1] = ev1 / std::sqrt((R + da) * (R + da) + ade) - ev1 / std::sqrt((R - da) * (R - da) + ade);
    const double ev1dsqr6 = ev1 / std::sqrt(R * R + aqe);
    ri[2] = ee + ev2 / std::sqrt((R + qa) * (R + qa) + aqe)
            + ev2 / std::sqrt((R - qa) * (R - qa) + aqe) - ev1dsqr6;
    ri[3] = ee + ev1 / std::sqrt(R * R + qa * qa + aqe) - ev1dsqr6;
    core[0] = ZB * ri[0];
    core[1] = ZB * ri[1];
    core[2] = ZB * ri[2];
    core[3] = ZB * ri[3];
    core[4] = ZA * ri[0];
    *pairType = kPairXH;
    return 4;
  }

  const double da = daA, db = daB;
  const double qa = qaA * 2.0, qb = qaB * 2.0;
  const double qa1 = qaA, qb1 = qaB;
  const double aee = (rho0A + rho0B) * (rho0A + rho0B);
  const double ade = (rho1A + rho0B) * (rho1A + rho0B);
  const double aqe = (rho2A + rho0B) * (rho2A + rho0B);
  const double aed = (rho0A + rho1B) * (rho0A + rho1B);
  const double aeq = (rho0A + rho2B) * (rho0A + rho2B);
  const double axx = (rho1A + rho1B) * (rho1A + rho1B);
  const double adq = (rho1A + rho2B) * (rho1A + rho2B);
  const double aqd = (rho2A + rho1B) * (rho2A + rho1B);
  const double aqq = (rho2A + rho2B) * (rho2A + rho2B);

  const double ee = kSemiEV / std::sqrt(R * R + aee);
  const double dze = -ev1 / std::sqrt((R + da) * (R + da) + ade) + ev1 / std::sqrt((R - da) * (R - da) + ade);
  const double ev1dsqr6 = ev1 / std::sqrt(R * R + aqe);
  const double qzze = ev2 / std::sqrt((R - qa) * (R - qa) + aqe)
                      + ev2 / std::sqrt((R + qa) * (R + qa) + aqe) - ev1dsqr6;
  const double qxxe = ev1 / std::sqrt(R * R + qa * qa + aqe) - ev1dsqr6;
  const double edz = -ev1 / std::sqrt((R - db) * (R - db) + aed) + ev1 / std::sqrt((R + db) * (R + db) + aed);
  const double ev1dsqr12 = ev1 / std::sqrt(R * R + aeq);
  const double eqzz = ev2 / std::sqrt((R - qb) * (R - qb) + aeq)
                      + ev2 / std::sqrt((R + qb) * (R + qb) + aeq) - ev1dsqr12;
  const double eqxx = ev1 / std::sqrt(R * R + qb * qb + aeq) - ev1dsqr12;

  const double ev2dsqr20 = ev2 / std::sqrt((R + da) * (R + da) + adq);
  const double ev2dsqr22 = ev2 / std::sqrt((R - da) * (R - da) + adq);
  const double ev2dsqr24 = ev2 / std::sqrt((R - db) * (R - db) + aqd);
  const double ev2dsqr26 = ev2 / std::sqrt((R + db) * (R + db) + aqd);
  const double ev2dsqr36 = ev2 / std::sqrt(R * R + aqq);
  const double ev2dsqr39 = ev2 / std::sqrt(R * R + qa * qa + aqq);
  const double ev2dsqr40 = ev2 / std::sqrt(R * R + qb * qb + aqq);
  const double ev3dsqr42 = ev3 / std::sqrt((R - qb) * (R - qb) + aqq);
  const double ev3dsqr44 = ev3 / std::sqrt((R + qb) * (R + qb) + aqq);
  const double ev3dsqr46 = ev3 / std::sqrt((R + qa) * (R + qa) + aqq);
  const double ev3dsqr48 = ev3 / std::sqrt((R - qa) * (R - qa) + aqq);

  ri[0] = ee;
  ri[1] = -dze;
  ri[2] = ee + qzze;
  ri[3] = ee + qxxe;
  ri[4] = -edz;
  ri[5] = ev2 / std::sqrt((R + da - db) * (R + da - db) + axx)
          + ev2 / std::sqrt((R - da + db) * (R - da + db) + axx)
          - ev2 / std::sqrt((R - da - db) * (R - da - db) + axx)
          - ev2 / std::sqrt((R + da + db) * (R + da + db) + axx);
  ri[6] = ev1 / std::sqrt(R * R + (da - db) * (da - db) + axx)
          - ev1 / std::sqrt(R * R + (da + db) * (da + db) + axx);
  ri[7] = -edz + ev3 / std::sqrt((R + qa - db) * (R + qa - db) + aqd)
          - ev3 / std::sqrt((R + qa + db) * (R + qa + db) + aqd)
          + ev3 / std::sqrt((R - qa - db) * (R - qa - db) + aqd)
          - ev3 / std::sqrt((R - qa + db) * (R - qa + db) + aqd)
          - ev2dsqr24 + ev2dsqr26;
  ri[8] = -edz - ev2dsqr24
          + ev2 / std::sqrt((R - db) * (R - db) + qa * qa + aqd)
          + ev2dsqr26
          - ev2 / std::sqrt((R + db) * (R + db) + qa * qa + aqd);
  ri[9] = ev2 / std::sqrt((qa1 - db) * (qa1 - db) + (R + qa1) * (R + qa1) + aqd)
          - ev2 / std::sqrt((qa1 - db) * (qa1 - db) + (R - qa1) * (R - qa1) + aqd)
          - ev2 / std::sqrt((qa1 + db) * (qa1 + db) + (R + qa1) * (R + qa1) + aqd)
          + ev2 / std::sqrt((qa1 + db) * (qa1 + db) + (R - qa1) * (R - qa1) + aqd);
  ri[10] = ee + eqzz;
  ri[11] = ee + eqxx;
  ri[12] = -dze + ev3 / std::sqrt((R + da - qb) * (R + da - qb) + adq)
           - ev3 / std::sqrt((R - da - qb) * (R - da - qb) + adq)
           + ev3 / std::sqrt((R + da + qb) * (R + da + qb) + adq)
           - ev3 / std::sqrt((R - da + qb) * (R - da + qb) + adq)
           + ev2dsqr22 - ev2dsqr20;
  ri[13] = -dze - ev2dsqr20
           + ev2 / std::sqrt((R + da) * (R + da) + qb * qb + adq)
           + ev2dsqr22
           - ev2 / std::sqrt((R - da) * (R - da) + qb * qb + adq);
  ri[14] = ev2 / std::sqrt((da - qb1) * (da - qb1) + (R - qb1) * (R - qb1) + adq)
           - ev2 / std::sqrt((da - qb1) * (da - qb1) + (R + qb1) * (R + qb1) + adq)
           - ev2 / std::sqrt((da + qb1) * (da + qb1) + (R - qb1) * (R - qb1) + adq)
           + ev2 / std::sqrt((da + qb1) * (da + qb1) + (R + qb1) * (R + qb1) + adq);
  ri[15] = ee + eqzz + qzze
           + ev4 / std::sqrt((R + qa - qb) * (R + qa - qb) + aqq)
           + ev4 / std::sqrt((R + qa + qb) * (R + qa + qb) + aqq)
           + ev4 / std::sqrt((R - qa - qb) * (R - qa - qb) + aqq)
           + ev4 / std::sqrt((R - qa + qb) * (R - qa + qb) + aqq)
           - ev3dsqr48 - ev3dsqr46 - ev3dsqr42 - ev3dsqr44 + ev2dsqr36;
  ri[16] = ee + eqzz + qxxe
           + ev3 / std::sqrt((R - qb) * (R - qb) + qa * qa + aqq)
           + ev3 / std::sqrt((R + qb) * (R + qb) + qa * qa + aqq)
           - ev3dsqr42 - ev3dsqr44 - ev2dsqr39 + ev2dsqr36;
  ri[17] = ee + eqxx + qzze
           + ev3 / std::sqrt((R + qa) * (R + qa) + qb * qb + aqq)
           + ev3 / std::sqrt((R - qa) * (R - qa) + qb * qb + aqq)
           - ev3dsqr46 - ev3dsqr48 - ev2dsqr40 + ev2dsqr36;
  const double qxxqxx = ev3 / std::sqrt(R * R + (qa - qb) * (qa - qb) + aqq)
                        + ev3 / std::sqrt(R * R + (qa + qb) * (qa + qb) + aqq)
                        - ev2dsqr39 - ev2dsqr40 + ev2dsqr36;
  ri[18] = ee + eqxx + qxxe + qxxqxx;
  ri[19] = ev3 / std::sqrt((R + qa1 - qb1) * (R + qa1 - qb1) + (qa1 - qb1) * (qa1 - qb1) + aqq)
           - ev3 / std::sqrt((R + qa1 + qb1) * (R + qa1 + qb1) + (qa1 - qb1) * (qa1 - qb1) + aqq)
           - ev3 / std::sqrt((R - qa1 - qb1) * (R - qa1 - qb1) + (qa1 - qb1) * (qa1 - qb1) + aqq)
           + ev3 / std::sqrt((R - qa1 + qb1) * (R - qa1 + qb1) + (qa1 - qb1) * (qa1 - qb1) + aqq)
           - ev3 / std::sqrt((R + qa1 - qb1) * (R + qa1 - qb1) + (qa1 + qb1) * (qa1 + qb1) + aqq)
           + ev3 / std::sqrt((R + qa1 + qb1) * (R + qa1 + qb1) + (qa1 + qb1) * (qa1 + qb1) + aqq)
           + ev3 / std::sqrt((R - qa1 - qb1) * (R - qa1 - qb1) + (qa1 + qb1) * (qa1 + qb1) + aqq)
           - ev3 / std::sqrt((R - qa1 + qb1) * (R - qa1 + qb1) + (qa1 + qb1) * (qa1 + qb1) + aqq);
  const double qxxqyy = ev2 / std::sqrt(R * R + qa * qa + qb * qb + aqq)
                        - ev2dsqr39 - ev2dsqr40 + ev2dsqr36;
  ri[20] = ee + eqxx + qxxe + qxxqyy;
  ri[21] = 0.5 * (qxxqxx - qxxqyy);

  core[0] = ZB * ri[0];
  core[1] = ZB * ri[1];
  core[2] = ZB * ri[2];
  core[3] = ZB * ri[3];
  core[4] = ZA * ri[0];
  core[5] = ZA * ri[4];
  core[6] = ZA * ri[10];
  core[7] = ZA * ri[11];
  *pairType = kPairXX;
  return 22;
}

// MOPAC-faithful electron-core attraction monopole/multipole radii for the e1b/e2a
// terms (mndod spcore): the CORE atom contributes its special additive radius
// po(9)=pocord (AtomIntParams::rhoCore) in place of the regular monopole rho0 when
// pocord is defined. Returns the local-frame core-attraction reduced integrals
// riA[0..3] (A's [s,sigma,pi-sigma,pi-pi] orbitals attracted to the B core) using
// B's core radius, mirroring how twoCenterLocalDev fills ri[0..3] but with the
// core monopole. coreRhoB is B's monopole radius (rhoCore if set, else rho0B).
// When no atom has pocord this reproduces the standard ri[0..3] bit-for-bit, so
// callers can always route through it without changing existing elements.
NVMOLKIT_HD inline void detailCoreAttractionRiDev(const AtomIntParams& pA, double R,
                                                  double coreRhoB, double* riA) {
  using namespace detail;
  double daA, qaA, rho0A, rho1A, rho2A;
  computeMultipoleParamsDev(pA, daA, qaA, rho0A, rho1A, rho2A);
  const double ev1 = kSemiEV / 2.0, ev2 = kSemiEV / 4.0;
  // Same multipole-of-A vs monopole-of-B(core) structure as twoCenterLocalDev XH.
  const double da = daA, qa = qaA * 2.0;
  const double aee = (rho0A + coreRhoB) * (rho0A + coreRhoB);
  const double ade = (rho1A + coreRhoB) * (rho1A + coreRhoB);
  const double aqe = (rho2A + coreRhoB) * (rho2A + coreRhoB);
  const double ee = kSemiEV / std::sqrt(R * R + aee);
  riA[0] = ee;
  riA[1] = ev1 / std::sqrt((R + da) * (R + da) + ade) - ev1 / std::sqrt((R - da) * (R - da) + ade);
  const double ev1dsqr6 = ev1 / std::sqrt(R * R + aqe);
  riA[2] = ee + ev2 / std::sqrt((R + qa) * (R + qa) + aqe)
           + ev2 / std::sqrt((R - qa) * (R - qa) + aqe) - ev1dsqr6;
  riA[3] = ee + ev1 / std::sqrt(R * R + qa * qa + aqe) - ev1dsqr6;
}

// Molecular-frame electron-core attraction e1b (4x4) for atom A's sp orbitals
// attracted to the core of atom B, using B's special core radius coreRhoB
// (MOPAC pocord). Mirrors the e1b rotation in twoCenterMolecularDev but sources
// the reduced integrals from detailCoreAttractionRiDev (B monopole = coreRhoB).
NVMOLKIT_HD inline void coreAttractionE1bDev(const AtomIntParams& pA, const double cA[3],
                                             const AtomIntParams& pB, const double cB[3],
                                             double coreRhoB, double* e1b) {
  using namespace detail;
  for (int i = 0; i < 16; ++i) e1b[i] = 0.0;
  const double Rvec[3] = {cB[0] - cA[0], cB[1] - cA[1], cB[2] - cA[2]};
  const double R = std::sqrt(Rvec[0] * Rvec[0] + Rvec[1] * Rvec[1] + Rvec[2] * Rvec[2]);
  if (R < 1e-10) return;
  const double Rb = R * kSemiAngToBohr;
  double riA[4];
  detailCoreAttractionRiDev(pA, Rb, coreRhoB, riA);
  const double ZB = static_cast<double>(pB.valence);
  const double v[3] = {-Rvec[0] / R, -Rvec[1] / R, -Rvec[2] / R};
  double rot[3][3];
  rotationMatrixDev(v, rot);
  const double* r0 = rot[0];
  const double* r1 = rot[1];
  const double* r2 = rot[2];
  if (pA.nOrb == 1) { e1b[0] = -ZB * riA[0]; return; }
  e1b[0] = -ZB * riA[0];
  for (int k = 0; k < 3; ++k) {
    const double e = -ZB * riA[1] * r0[k];
    e1b[(k + 1) * 4 + 0] = e;
    e1b[0 * 4 + (k + 1)] = e;
  }
  for (int k = 0; k < 3; ++k) {
    e1b[(k + 1) * 4 + (k + 1)] =
        -ZB * (riA[2] * r0[k] * r0[k] + riA[3] * (r1[k] * r1[k] + r2[k] * r2[k]));
    for (int l = k + 1; l < 3; ++l) {
      const double e = -ZB * (riA[2] * r0[k] * r0[l] + riA[3] * (r1[k] * r1[l] + r2[k] * r2[l]));
      e1b[(k + 1) * 4 + (l + 1)] = e;
      e1b[(l + 1) * 4 + (k + 1)] = e;
    }
  }
}

// Molecular-frame w tensor (flattened 4x4x4x4) + e1b/e2a (4x4) for an sp pair.
NVMOLKIT_HD inline bool twoCenterMolecularDev(const AtomIntParams& pA, const double cA[3],
                                              const AtomIntParams& pB, const double cB[3],
                                              double* w, double* e1b, double* e2a) {
  using namespace detail;
  const int nA = pA.nOrb, nB = pB.nOrb;
  if (nA == 0 || nB == 0) return false;

  for (int i = 0; i < 256; ++i) w[i] = 0.0;
  for (int i = 0; i < 16; ++i) { e1b[i] = 0.0; e2a[i] = 0.0; }

  const double Rvec[3] = {cB[0] - cA[0], cB[1] - cA[1], cB[2] - cA[2]};
  const double R = std::sqrt(Rvec[0] * Rvec[0] + Rvec[1] * Rvec[1] + Rvec[2] * Rvec[2]);
  if (R < 1e-10) return true;

  if (nA == 1 && nB > 1) {
    double ws[256], e1s[16], e2s[16];
    if (!twoCenterMolecularDev(pB, cB, pA, cA, ws, e1s, e2s)) return false;
    for (int mu = 0; mu < 4; ++mu)
      for (int nu = 0; nu < 4; ++nu)
        for (int lam = 0; lam < 4; ++lam)
          for (int sig = 0; sig < 4; ++sig)
            w[wIdxDev(mu, nu, lam, sig)] = ws[wIdxDev(lam, sig, mu, nu)];
    for (int i = 0; i < 16; ++i) { e1b[i] = e2s[i]; e2a[i] = e1s[i]; }
    return true;
  }

  double ri[22], core[8];
  int ptype;
  if (twoCenterLocalDev(pA, pB, R, ri, core, &ptype) == 0) return false;

  const double v[3] = {-Rvec[0] / R, -Rvec[1] / R, -Rvec[2] / R};
  double rot[3][3];
  rotationMatrixDev(v, rot);
  const double* r0 = rot[0];
  const double* r1 = rot[1];
  const double* r2 = rot[2];
  const double ZA = static_cast<double>(pA.valence);
  const double ZB = static_cast<double>(pB.valence);

  if (ptype == kPairHH) {
    w[wIdxDev(0, 0, 0, 0)] = ri[0];
    e1b[0] = -ZB * ri[0];
    e2a[0] = -ZA * ri[0];
    return true;
  }

  if (ptype == kPairXH) {
    w[wIdxDev(0, 0, 0, 0)] = ri[0];
    for (int k = 0; k < 3; ++k) {
      const double v_ps = ri[1] * r0[k];
      w[wIdxDev(k + 1, 0, 0, 0)] = v_ps;
      w[wIdxDev(0, k + 1, 0, 0)] = v_ps;
    }
    for (int k = 0; k < 3; ++k)
      for (int l = 0; l < 3; ++l)
        w[wIdxDev(k + 1, l + 1, 0, 0)] =
            ri[2] * r0[k] * r0[l] + ri[3] * (r1[k] * r1[l] + r2[k] * r2[l]);
    e1b[0] = -ZB * ri[0];
    for (int k = 0; k < 3; ++k) {
      const double e = -ZB * ri[1] * r0[k];
      e1b[(k + 1) * 4 + 0] = e;
      e1b[0 * 4 + (k + 1)] = e;
    }
    for (int k = 0; k < 3; ++k) {
      e1b[(k + 1) * 4 + (k + 1)] =
          -ZB * (ri[2] * r0[k] * r0[k] + ri[3] * (r1[k] * r1[k] + r2[k] * r2[k]));
      for (int l = k + 1; l < 3; ++l) {
        const double e = -ZB * (ri[2] * r0[k] * r0[l] + ri[3] * (r1[k] * r1[l] + r2[k] * r2[l]));
        e1b[(k + 1) * 4 + (l + 1)] = e;
        e1b[(l + 1) * 4 + (k + 1)] = e;
      }
    }
    e2a[0] = -ZA * ri[0];
    return true;
  }

  for (int kk = 0; kk < 4; ++kk) {
    for (int ll = 0; ll <= kk; ++ll) {
      for (int mm = 0; mm < 4; ++mm) {
        for (int nn = 0; nn <= mm; ++nn) {
          const int k = kk - 1, l = ll - 1, m = mm - 1, n = nn - 1;
          double val = 0.0;
          if (kk == 0) {
            if (mm == 0) {
              val = ri[0];
            } else if (nn == 0) {
              val = ri[4] * r0[m];
            } else {
              val = ri[10] * r0[m] * r0[n] + ri[11] * (r1[m] * r1[n] + r2[m] * r2[n]);
            }
          } else if (ll == 0) {
            if (mm == 0) {
              val = ri[1] * r0[k];
            } else if (nn == 0) {
              val = ri[5] * r0[k] * r0[m] + ri[6] * (r1[k] * r1[m] + r2[k] * r2[m]);
            } else {
              const double t0 = r0[k] * r0[m] * r0[n];
              const double t1 = (r1[m] * r1[n] + r2[m] * r2[n]) * r0[k];
              const double mix = r1[k] * (r1[n] * r0[m] + r1[m] * r0[n])
                                 + r2[k] * (r2[m] * r0[n] + r2[n] * r0[m]);
              val = ri[12] * t0 + ri[13] * t1 + ri[14] * mix;
            }
          } else {
            if (mm == 0) {
              val = ri[2] * r0[k] * r0[l] + ri[3] * (r1[k] * r1[l] + r2[k] * r2[l]);
            } else if (nn == 0) {
              const double t0 = r0[k] * r0[l] * r0[m];
              const double t1 = (r1[k] * r1[l] + r2[k] * r2[l]) * r0[m];
              const double t2a = r1[l] * r1[m] + r2[l] * r2[m];
              const double t2b = r1[k] * r1[m] + r2[k] * r2[m];
              val = ri[7] * t0 + ri[8] * t1 + ri[9] * (r0[k] * t2a + r0[l] * t2b);
            } else {
              const double t0 = r0[k] * r0[l] * r0[m] * r0[n];
              const double t1 = (r1[k] * r1[l] + r2[k] * r2[l]) * r0[m] * r0[n];
              const double t2 = (r1[m] * r1[n] + r2[m] * r2[n]) * r0[k] * r0[l];
              const double quad = r1[k] * r1[l] * r1[m] * r1[n] + r2[k] * r2[l] * r2[m] * r2[n];
              const double mix1 = r0[m] * (r1[l] * r1[n] + r2[l] * r2[n]);
              const double mix2 = r0[n] * (r1[l] * r1[m] + r2[l] * r2[m]);
              const double val5 = r0[k] * (mix1 + mix2)
                                  + r0[l] * (r0[m] * (r1[k] * r1[n] + r2[k] * r2[n])
                                             + r0[n] * (r1[k] * r1[m] + r2[k] * r2[m]));
              const double mix3 = r1[k] * r1[l] * r2[m] * r2[n] + r2[k] * r2[l] * r1[m] * r1[n];
              const double cross = (r1[k] * r2[l] + r2[k] * r1[l]) * (r1[m] * r2[n] + r2[m] * r1[n]);
              val = ri[15] * t0 + ri[16] * t1 + ri[17] * t2 + ri[18] * quad
                    + ri[19] * val5 + ri[20] * mix3 + ri[21] * cross;
            }
          }
          w[wIdxDev(kk, ll, mm, nn)] = val;
          w[wIdxDev(ll, kk, mm, nn)] = val;
          w[wIdxDev(kk, ll, nn, mm)] = val;
          w[wIdxDev(ll, kk, nn, mm)] = val;
        }
      }
    }
  }

  for (int mu = 0; mu < nA; ++mu)
    for (int nu = 0; nu <= mu; ++nu) {
      e1b[mu * 4 + nu] = -ZB * w[wIdxDev(mu, nu, 0, 0)];
      e1b[nu * 4 + mu] = e1b[mu * 4 + nu];
    }
  for (int mu = 0; mu < nB; ++mu)
    for (int nu = 0; nu <= mu; ++nu) {
      e2a[mu * 4 + nu] = -ZA * w[wIdxDev(0, 0, mu, nu)];
      e2a[nu * 4 + mu] = e2a[mu * 4 + nu];
    }
  return true;
}

}  // namespace semiempirical
}  // namespace nvMolKit

#endif  // NVMOLKIT_SEMIEMPIRICAL_TWO_CENTER_DEVICE_H
