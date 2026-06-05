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
// Device-callable diatomic STO overlap (NDDO/PM6_SP, A/B reduced-integral
// method). The SAME inline __host__ __device__ code feeds the CPU reference
// (overlap.cpp) and the HIP kernels. Operates on gathered AtomIntParams.

#ifndef NVMOLKIT_SEMIEMPIRICAL_OVERLAP_DEVICE_H
#define NVMOLKIT_SEMIEMPIRICAL_OVERLAP_DEVICE_H

#include <cmath>

#include "device_macros.h"

namespace nvMolKit {
namespace semiempirical {
namespace ovdetail {

constexpr double kOvAngToBohr = 1.0 / 0.529167;
constexpr int kOvNMax = 11;

NVMOLKIT_HD inline void aintgs(double alpha, double* a) {
  if (std::fabs(alpha) < 1e-10) {
    for (int k = 0; k < kOvNMax; ++k) a[k] = 0.0;
    return;
  }
  a[0] = std::exp(-alpha) / alpha;
  for (int k = 1; k < kOvNMax; ++k) a[k] = a[0] + k * a[k - 1] / alpha;
}

NVMOLKIT_HD inline void bintgs(double beta, double* b) {
  const double x = beta;
  if (std::fabs(x) <= 1e-6) {
    for (int k = 0; k < kOvNMax; ++k) b[k] = (k % 2 == 0) ? 2.0 / (k + 1) : 0.0;
    return;
  }
  if (std::fabs(x) <= 0.5) {
    const double even[7][4] = {
        {2.0, 1.0 / 3, 1.0 / 60, 1.0 / 2520}, {2.0 / 3, 1.0 / 5, 1.0 / 84, 1.0 / 3240},
        {2.0 / 5, 1.0 / 7, 1.0 / 108, 1.0 / 3960}, {2.0 / 7, 1.0 / 9, 1.0 / 132, 1.0 / 4680},
        {2.0 / 9, 1.0 / 11, 1.0 / 156, 1.0 / 5400}, {2.0 / 11, 1.0 / 13, 1.0 / 180, 1.0 / 6120},
        {2.0 / 13, 1.0 / 15, 1.0 / 204, 1.0 / 6840}};
    const double odd[6][3] = {
        {-2.0 / 3, -1.0 / 15, -1.0 / 420}, {-2.0 / 5, -1.0 / 21, -1.0 / 540},
        {-2.0 / 7, -1.0 / 27, -1.0 / 660}, {-2.0 / 9, -1.0 / 33, -1.0 / 780},
        {-2.0 / 11, -1.0 / 39, -1.0 / 900}, {-2.0 / 13, -1.0 / 45, -1.0 / 1020}};
    const double x2 = x * x, x3 = x * x2, x4 = x2 * x2, x5 = x2 * x3, x6 = x2 * x4;
    for (int k = 0; k < kOvNMax; ++k) {
      if (k % 2 == 0) {
        const double* c = even[k / 2];
        b[k] = c[0] + c[1] * x2 + c[2] * x4 + c[3] * x6;
      } else {
        const double* c = odd[k / 2];
        b[k] = c[0] * x + c[1] * x3 + c[2] * x5;
      }
    }
    return;
  }
  const double xc = (x > 500.0) ? 500.0 : (x < -500.0 ? -500.0 : x);
  const double tx = std::exp(xc) / x;
  const double tmx = -std::exp(-xc) / x;
  double sign = 1.0;
  b[0] = tx + tmx;
  for (int k = 1; k < kOvNMax; ++k) {
    sign = -sign;
    b[k] = sign * tx + tmx + k * b[k - 1] / x;
  }
}

NVMOLKIT_HD inline void ovRotation(const double v[3], double rot[3][3]) {
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

}  // namespace ovdetail

// (nA x nB) molecular-frame overlap block between atoms A and B (sp, qn in {1,2}).
// Orbital order [s, px, py, pz]; coords in Angstrom. Heavier shells (qn>=3) and
// coincident centers write zeros. Returns nA*nB.
NVMOLKIT_HD inline int diatomOverlapSpDev(const AtomIntParams& pA, const double coordA[3],
                                          const AtomIntParams& pB, const double coordB[3],
                                          double* outBlock) {
  using namespace ovdetail;
  const int nA = pA.nOrb, nB = pB.nOrb;
  if (nA == 0 || nB == 0) return 0;

  const double Rvec[3] = {coordB[0] - coordA[0], coordB[1] - coordA[1], coordB[2] - coordA[2]};
  const double R = std::sqrt(Rvec[0] * Rvec[0] + Rvec[1] * Rvec[1] + Rvec[2] * Rvec[2]);
  if (R < 1e-10) {
    for (int i = 0; i < nA * nB; ++i) outBlock[i] = 0.0;
    return nA * nB;
  }

  if (pA.qn < pB.qn) {  // PYSEQM convention: heavier (higher qn) atom first.
    double tmp[16];
    diatomOverlapSpDev(pB, coordB, pA, coordA, tmp);
    for (int i = 0; i < nA; ++i)
      for (int j = 0; j < nB; ++j) outBlock[i * nB + j] = tmp[j * nA + i];
    return nA * nB;
  }

  int jcall = 0;
  if (pA.qn == 1 && pB.qn == 1) jcall = 2;
  else if (pA.qn == 2 && pB.qn == 1) jcall = 3;
  else if (pA.qn == 2 && pB.qn == 2) jcall = 4;
  else if (pA.qn == 3 && pB.qn == 1) jcall = 431;
  else if (pA.qn == 3 && pB.qn == 2) jcall = 5;
  else if (pA.qn == 3 && pB.qn == 3) jcall = 6;
  else if (pA.qn == 4 && pB.qn == 1) jcall = 541;  // Br + H
  else if (pA.qn == 4 && pB.qn == 2) jcall = 642;  // Br + 2nd-row (C/N/O/F)
  else if (pA.qn == 5 && pB.qn == 1) jcall = 651;  // I + H
  else if (pA.qn == 5 && pB.qn == 2) jcall = 752;  // I + 2nd-row (C/N/O/F)
  else if (pA.qn == 4 && pB.qn == 4) jcall = 8;    // Br + Br
  else if (pA.qn == 5 && pB.qn == 5) jcall = 10;   // I + I
  else {  // mixed heavy (Br-I etc.) sp overlap formulas not yet ported
    for (int i = 0; i < nA * nB; ++i) outBlock[i] = 0.0;
    return nA * nB;
  }

  const double Rb = R * kOvAngToBohr;
  const double zsA = pA.zetaS, zpA = pA.zetaP, zsB = pB.zetaS, zpB = pB.zetaP;

  double A111[kOvNMax], B111[kOvNMax];
  aintgs(0.5 * Rb * (zsA + zsB), A111);
  bintgs(0.5 * Rb * (zsA - zsB), B111);
  double A211[kOvNMax] = {0}, B211[kOvNMax] = {0};
  if (nA > 1) { aintgs(0.5 * Rb * (zpA + zsB), A211); bintgs(0.5 * Rb * (zpA - zsB), B211); }
  double A121[kOvNMax] = {0}, B121[kOvNMax] = {0};
  if (nB > 1) { aintgs(0.5 * Rb * (zsA + zpB), A121); bintgs(0.5 * Rb * (zsA - zpB), B121); }
  double A22[kOvNMax] = {0}, B22[kOvNMax] = {0};
  if (nA > 1 && nB > 1) { aintgs(0.5 * Rb * (zpA + zpB), A22); bintgs(0.5 * Rb * (zpA - zpB), B22); }

  const double sqrt3 = std::sqrt(3.0);
  double S111 = 0.0, S211 = 0.0, S121 = 0.0, S221 = 0.0, S222 = 0.0;

  if (jcall == 2) {
    S111 = std::pow(zsA * zsB * Rb * Rb, 1.5) * (A111[2] * B111[0] - B111[2] * A111[0]) / 4.0;
  } else if (jcall == 3) {
    S111 = std::pow(zsB, 1.5) * std::pow(zsA, 2.5) * std::pow(Rb, 4)
           * (A111[3] * B111[0] - B111[3] * A111[0] + A111[2] * B111[1] - B111[2] * A111[1])
           / (sqrt3 * 8.0);
    if (nA > 1)
      S211 = std::pow(zsB, 1.5) * std::pow(zpA, 2.5) * std::pow(Rb, 4)
             * (A211[2] * B211[0] - B211[2] * A211[0] + A211[3] * B211[1] - B211[3] * A211[1])
             / 8.0;
  } else if (jcall == 4) {
    S111 = std::pow(zsA * zsB, 2.5) * std::pow(Rb, 5)
           * (A111[4] * B111[0] + B111[4] * A111[0] - 2.0 * A111[2] * B111[2]) / 48.0;
    if (nA > 1 && nB > 1) {
      S211 = std::pow(zsB * zpA, 2.5) * std::pow(Rb, 5)
             * (A211[3] * (B211[0] - B211[2]) - A211[1] * (B211[2] - B211[4])
                + B211[3] * (A211[0] - A211[2]) - B211[1] * (A211[2] - A211[4])) / (16.0 * sqrt3);
      S121 = std::pow(zpB * zsA, 2.5) * std::pow(Rb, 5)
             * (A121[3] * (B121[0] - B121[2]) - A121[1] * (B121[2] - B121[4])
                - B121[3] * (A121[0] - A121[2]) + B121[1] * (A121[2] - A121[4])) / (16.0 * sqrt3);
      const double w = std::pow(zpB * zpA, 2.5) * std::pow(Rb, 5) / 16.0;
      S221 = -w * (B22[2] * (A22[4] + A22[0]) - A22[2] * (B22[4] + B22[0]));
      S222 = 0.5 * w * (A22[4] * (B22[0] - B22[2]) - B22[4] * (A22[0] - A22[2])
                        - A22[2] * B22[0] + B22[2] * A22[0]);
    }
  } else if (jcall == 431) {  // heavy qn=3 + H
    const double sqrt10 = std::sqrt(10.0), sqrt30 = std::sqrt(30.0);
    S111 = std::pow(zsB, 1.5) * std::pow(zsA, 3.5) * std::pow(Rb, 5)
           * (A111[4] * B111[0] + 2.0 * B111[1] * A111[3] - 2.0 * A111[1] * B111[3]
              - B111[4] * A111[0]) / (sqrt10 * 24.0);
    if (nA > 1)
      S211 = std::pow(zsB, 1.5) * std::pow(zpA, 3.5) * std::pow(Rb, 5)
             * (A211[3] * (B211[0] + B211[2]) - A211[1] * (B211[4] + B211[2])
                + B211[1] * (A211[2] + A211[4]) - B211[3] * (A211[2] + A211[0])) / (8.0 * sqrt30);
  } else if (jcall == 5) {  // qn=3 - qn=2
    const double sqrt10 = std::sqrt(10.0), sqrt30 = std::sqrt(30.0);
    S111 = std::pow(zsB, 2.5) * std::pow(zsA, 3.5) * std::pow(Rb, 6)
           * (A111[5] * B111[0] + B111[1] * A111[4] - 2.0 * B111[2] * A111[3]
              - 2.0 * A111[2] * B111[3] + B111[4] * A111[1] + B111[5] * A111[0]) / (sqrt30 * 48.0);
    if (nA > 1 && nB > 1) {
      S211 = std::pow(zsB, 2.5) * std::pow(zpA, 3.5) * std::pow(Rb, 6)
             * (A211[4] * B211[0] + B211[1] * A211[5] - 2.0 * B211[3] * A211[3]
                - 2.0 * A211[2] * B211[2] + A211[1] * B211[5] + A211[0] * B211[4]) / (48.0 * sqrt10);
      S121 = std::pow(zpB, 2.5) * std::pow(zsA, 3.5) * std::pow(Rb, 6)
             * ((A121[4] * B121[0] - A121[5] * B121[1]) + 2.0 * (A121[3] * B121[1] - A121[4] * B121[2])
                - 2.0 * (A121[1] * B121[3] - A121[2] * B121[4])
                - (A121[0] * B121[4] - A121[1] * B121[5])) / (48.0 * sqrt10);
      S221 = std::pow(zpB, 2.5) * std::pow(zpA, 3.5) * std::pow(Rb, 6)
             * ((A22[3] * B22[0] - A22[5] * B22[2]) + (A22[2] * B22[1] - A22[4] * B22[3])
                - (A22[1] * B22[2] - A22[3] * B22[4]) - (A22[0] * B22[3] - A22[2] * B22[5]))
             / (16.0 * sqrt30);
      S222 = std::pow(zpB, 2.5) * std::pow(zpA, 3.5) * std::pow(Rb, 6)
             * ((A22[5] - A22[3]) * (B22[0] - B22[2]) + (A22[4] - A22[2]) * (B22[1] - B22[3])
                - (A22[3] - A22[1]) * (B22[2] - B22[4]) - (A22[2] - A22[0]) * (B22[3] - B22[5]))
             / (32.0 * sqrt30);
    }
  } else if (jcall == 6) {  // qn=3 - qn=3
    S111 = std::pow(zsA * zsB, 3.5) * std::pow(Rb, 7)
           * (A111[6] * B111[0] - 3.0 * B111[2] * A111[4] + 3.0 * A111[2] * B111[4]
              - A111[0] * B111[6]) / 1440.0;
    if (nA > 1 && nB > 1) {
      S211 = std::pow(zsB * zpA, 3.5) * std::pow(Rb, 7)
             * ((A211[5] * B211[0] + A211[6] * B211[1]) + (-A211[4] * B211[1] - A211[5] * B211[2])
                - 2.0 * (A211[3] * B211[2] + A211[4] * B211[3])
                - 2.0 * (-A211[2] * B211[3] - A211[3] * B211[4])
                + (A211[1] * B211[4] + A211[2] * B211[5])
                + (-A211[0] * B211[5] - A211[1] * B211[6])) / (480.0 * sqrt3);
      S121 = std::pow(zpB * zsA, 3.5) * std::pow(Rb, 7)
             * ((A121[5] * B121[0] - A121[6] * B121[1]) + (A121[4] * B121[1] - A121[5] * B121[2])
                - 2.0 * (A121[3] * B121[2] - A121[4] * B121[3])
                - 2.0 * (A121[2] * B121[3] - A121[3] * B121[4])
                + (A121[1] * B121[4] - A121[2] * B121[5])
                + (A121[0] * B121[5] - A121[1] * B121[6])) / (480.0 * sqrt3);
      S221 = std::pow(zpB * zpA, 3.5) * std::pow(Rb, 7)
             * ((A22[4] * B22[0] - A22[6] * B22[2]) - 2.0 * (A22[2] * B22[2] - A22[4] * B22[4])
                + (A22[0] * B22[4] - A22[2] * B22[6])) / 480.0;
      S222 = std::pow(zpB * zpA, 3.5) * std::pow(Rb, 7)
             * ((A22[6] - A22[4]) * (B22[0] - B22[2]) - 2.0 * (A22[4] - A22[2]) * (B22[2] - B22[4])
                + (A22[2] - A22[0]) * (B22[4] - B22[6])) / 960.0;
    }
  } else if (jcall == 541) {  // heavy qn=4 + H (Br + H)
    S111 = std::pow(zsB, 1.5) * std::pow(zsA, 4.5) * std::pow(Rb, 6)
           * (A111[5] * B111[0] + 3.0 * B111[1] * A111[4] + 2.0 * B111[2] * A111[3]
              - 2.0 * B111[3] * A111[2] - 3.0 * A111[1] * B111[4] - B111[5] * A111[0])
           / (std::sqrt(35.0) * 96.0);
    if (nA > 1)
      S211 = std::pow(zsB, 1.5) * std::pow(zpA, 4.5) * std::pow(Rb, 6)
             * ((A211[4] * B211[0] + A211[5] * B211[1])
                - 2.0 * (-A211[3] * B211[1] - A211[4] * B211[2])
                + 2.0 * (-A211[1] * B211[3] - A211[2] * B211[4])
                - (A211[0] * B211[4] + A211[1] * B211[5])) / (32.0 * std::sqrt(105.0));
  } else if (jcall == 642) {  // heavy qn=4 + 2nd-row qn=2 (Br + C/N/O/F)
    S111 = std::pow(zsB, 2.5) * std::pow(zsA, 4.5) * std::pow(Rb, 7)
           * (A111[6] * B111[0] + 2.0 * B111[1] * A111[5] - B111[2] * A111[4] - 4.0 * B111[3] * A111[3]
              - A111[2] * B111[4] + 2.0 * B111[5] * A111[1] + A111[0] * B111[6])
           / (std::sqrt(105.0) * 192.0);
    if (nA > 1 && nB > 1) {
      S211 = std::pow(zsB, 2.5) * std::pow(zpA, 4.5) * std::pow(Rb, 7)
             * ((A211[5] * B211[0] + A211[6] * B211[1]) - (-A211[4] * B211[1] - A211[5] * B211[2])
                - 2.0 * (A211[3] * B211[2] + A211[4] * B211[3])
                + 2.0 * (-A211[2] * B211[3] - A211[3] * B211[4])
                + (A211[1] * B211[4] + A211[2] * B211[5]) - (-A211[0] * B211[5] - A211[1] * B211[6]))
             / (192.0 * std::sqrt(35.0));
      S121 = std::pow(zpB, 2.5) * std::pow(zsA, 4.5) * std::pow(Rb, 7)
             * ((A121[5] * B121[0] - A121[6] * B121[1]) + 3.0 * (A121[4] * B121[1] - A121[5] * B121[2])
                + 2.0 * (A121[3] * B121[2] - A121[4] * B121[3]) - 2.0 * (A121[2] * B121[3] - A121[3] * B121[4])
                - 3.0 * (A121[1] * B121[4] - A121[2] * B121[5]) - (A121[0] * B121[5] - A121[1] * B121[6]))
             / (192.0 * std::sqrt(35.0));
      S221 = std::pow(zpB, 2.5) * std::pow(zpA, 4.5) * std::pow(Rb, 7)
             * ((A22[4] * B22[0] - A22[6] * B22[2]) + 2.0 * (A22[3] * B22[1] - A22[5] * B22[3])
                - 2.0 * (A22[1] * B22[3] - A22[3] * B22[5]) - (A22[0] * B22[4] - A22[2] * B22[6]))
             / (64.0 * std::sqrt(105.0));
      S222 = std::pow(zpB, 2.5) * std::pow(zpA, 4.5) * std::pow(Rb, 7)
             * ((A22[6] - A22[4]) * (B22[0] - B22[2]) + 2.0 * (A22[5] - A22[3]) * (B22[1] - B22[3])
                - 2.0 * (A22[3] - A22[1]) * (B22[3] - B22[5]) - (A22[2] - A22[0]) * (B22[4] - B22[6]))
             / (128.0 * std::sqrt(105.0));
    }
  } else if (jcall == 651) {  // heavy qn=5 + H (I + H)
    S111 = std::pow(zsB, 1.5) * std::pow(zsA, 5.5) * std::pow(Rb, 7)
           * (A111[6] * B111[0] + 4.0 * B111[1] * A111[5] + 5.0 * B111[2] * A111[4]
              - 5.0 * B111[4] * A111[2] - 4.0 * A111[1] * B111[5] - B111[6] * A111[0])
           / (std::sqrt(14.0) * 1440.0);
    if (nA > 1)
      S211 = std::pow(zsB, 1.5) * std::pow(zpA, 5.5) * std::pow(Rb, 7)
             * ((A211[5] * B211[0] + A211[6] * B211[1]) - 3.0 * (-A211[4] * B211[1] - A211[5] * B211[2])
                + 2.0 * (A211[3] * B211[2] + A211[4] * B211[3]) + 2.0 * (-A211[2] * B211[3] - A211[3] * B211[4])
                - 3.0 * (A211[1] * B211[4] + A211[2] * B211[5]) + (-A211[0] * B211[5] - A211[1] * B211[6]))
             / (480.0 * std::sqrt(42.0));
  } else if (jcall == 752) {  // heavy qn=5 + 2nd-row qn=2 (I + C/N/O/F)
    S111 = std::pow(zsB, 2.5) * std::pow(zsA, 5.5) * std::pow(Rb, 8)
           * (A111[7] * B111[0] + 3.0 * B111[1] * A111[6] + B111[2] * A111[5] - 5.0 * B111[3] * A111[4]
              - 5.0 * A111[3] * B111[4] + B111[5] * A111[2] + 3.0 * B111[6] * A111[1] + B111[7] * A111[0])
           / (std::sqrt(42.0) * 2880.0);
    if (nA > 1 && nB > 1) {
      S211 = std::pow(zsB, 2.5) * std::pow(zpA, 5.5) * std::pow(Rb, 8)
             * ((A211[6] * B211[0] + A211[7] * B211[1]) - 2.0 * (-A211[5] * B211[1] - A211[6] * B211[2])
                - (A211[4] * B211[2] + A211[5] * B211[3]) + 4.0 * (-A211[3] * B211[3] - A211[4] * B211[4])
                - (A211[2] * B211[4] + A211[3] * B211[5]) - 2.0 * (-A211[1] * B211[5] - A211[2] * B211[6])
                + (A211[0] * B211[6] + A211[1] * B211[7])) / (2880.0 * std::sqrt(14.0));
      S121 = std::pow(zpB, 2.5) * std::pow(zsA, 5.5) * std::pow(Rb, 8)
             * ((A121[6] * B121[0] - A121[7] * B121[1]) + 4.0 * (A121[5] * B121[1] - A121[6] * B121[2])
                + 5.0 * (A121[4] * B121[2] - A121[5] * B121[3]) - 5.0 * (A121[2] * B121[4] - A121[3] * B121[5])
                - 4.0 * (A121[1] * B121[5] - A121[2] * B121[6]) - (A121[0] * B121[6] - A121[1] * B121[7]))
             / (2880.0 * std::sqrt(14.0));
      S221 = std::pow(zpB, 2.5) * std::pow(zpA, 5.5) * std::pow(Rb, 8)
             * ((A22[5] * B22[0] - A22[7] * B22[2]) + 3.0 * (A22[4] * B22[1] - A22[6] * B22[3])
                + 2.0 * (A22[3] * B22[2] - A22[5] * B22[4]) - 2.0 * (A22[2] * B22[3] - A22[4] * B22[5])
                - 3.0 * (A22[1] * B22[4] - A22[3] * B22[6]) - (A22[0] * B22[5] - A22[2] * B22[7]))
             / (960.0 * std::sqrt(42.0));
      S222 = std::pow(zpB, 2.5) * std::pow(zpA, 5.5) * std::pow(Rb, 8)
             * ((A22[7] - A22[5]) * (B22[0] - B22[2]) + 3.0 * (A22[6] - A22[4]) * (B22[1] - B22[3])
                + 2.0 * (A22[5] - A22[3]) * (B22[2] - B22[4]) - 2.0 * (A22[4] - A22[2]) * (B22[3] - B22[5])
                - 3.0 * (A22[3] - A22[1]) * (B22[4] - B22[6]) - (A22[2] - A22[0]) * (B22[5] - B22[7]))
             / (1920.0 * std::sqrt(42.0));
    }
  } else if (jcall == 8) {  // heavy qn=4 + qn=4 (Br + Br)
    S111 = std::pow(zsA * zsB, 4.5) * std::pow(Rb, 9)
           * (A111[8] * B111[0] - 4.0 * B111[2] * A111[6] + 6.0 * A111[4] * B111[4]
              - 4.0 * A111[2] * B111[6] + A111[0] * B111[8]) / 80640.0;
    if (nA > 1 && nB > 1) {
      S211 = std::pow(zsB, 4.5) * std::pow(zpA, 4.5) * std::pow(Rb, 9)
             * ((A211[7] * B211[0] + A211[8] * B211[1]) + (-A211[6] * B211[1] - A211[7] * B211[2])
                - 3.0 * (A211[5] * B211[2] + A211[6] * B211[3]) - 3.0 * (-A211[4] * B211[3] - A211[5] * B211[4])
                + 3.0 * (A211[3] * B211[4] + A211[4] * B211[5]) + 3.0 * (-A211[2] * B211[5] - A211[3] * B211[6])
                - (A211[1] * B211[6] + A211[2] * B211[7]) - (-A211[0] * B211[7] - A211[1] * B211[8]))
             / (26880.0 * sqrt3);
      S121 = std::pow(zpB, 4.5) * std::pow(zsA, 4.5) * std::pow(Rb, 9)
             * ((A121[7] * B121[0] - A121[8] * B121[1]) + (A121[6] * B121[1] - A121[7] * B121[2])
                - 3.0 * (A121[5] * B121[2] - A121[6] * B121[3]) - 3.0 * (A121[4] * B121[3] - A121[5] * B121[4])
                + 3.0 * (A121[3] * B121[4] - A121[4] * B121[5]) + 3.0 * (A121[2] * B121[5] - A121[3] * B121[6])
                - (A121[1] * B121[6] - A121[2] * B121[7]) - (A121[0] * B121[7] - A121[1] * B121[8]))
             / (26880.0 * sqrt3);
      S221 = std::pow(zpB, 4.5) * std::pow(zpA, 4.5) * std::pow(Rb, 9)
             * ((A22[6] * B22[0] - A22[8] * B22[2]) - 3.0 * (A22[4] * B22[2] - A22[6] * B22[4])
                + 3.0 * (A22[2] * B22[4] - A22[4] * B22[6]) - (A22[0] * B22[6] - A22[2] * B22[8])) / 26880.0;
      S222 = std::pow(zpB, 4.5) * std::pow(zpA, 4.5) * std::pow(Rb, 9)
             * ((A22[8] - A22[6]) * (B22[0] - B22[2]) - 3.0 * (A22[6] - A22[4]) * (B22[2] - B22[4])
                + 3.0 * (A22[4] - A22[2]) * (B22[4] - B22[6]) - (A22[2] - A22[0]) * (B22[6] - B22[8]))
             / 53760.0;
    }
  } else if (jcall == 10) {  // heavy qn=5 + qn=5 (I + I)
    S111 = std::pow(zsA * zsB, 5.5) * std::pow(Rb, 11)
           * (A111[10] * B111[0] - 5.0 * B111[2] * A111[8] + 10.0 * A111[6] * B111[4]
              - 10.0 * A111[4] * B111[6] + 5.0 * A111[2] * B111[8] - A111[0] * B111[10]) / 7257600.0;
    if (nA > 1 && nB > 1) {
      S211 = std::pow(zsB, 5.5) * std::pow(zpA, 5.5) * std::pow(Rb, 11)
             * ((A211[9] * B211[0] + A211[10] * B211[1]) + (-A211[8] * B211[1] - A211[9] * B211[2])
                - 4.0 * (A211[7] * B211[2] + A211[8] * B211[3]) - 4.0 * (-A211[6] * B211[3] - A211[7] * B211[4])
                + 6.0 * (A211[5] * B211[4] + A211[6] * B211[5]) + 6.0 * (-A211[4] * B211[5] - A211[5] * B211[6])
                - 4.0 * (A211[3] * B211[6] + A211[4] * B211[7]) - 4.0 * (-A211[2] * B211[7] - A211[3] * B211[8])
                + (A211[1] * B211[8] + A211[2] * B211[9]) + (-A211[0] * B211[9] - A211[1] * B211[10]))
             / (2419200.0 * sqrt3);
      S121 = std::pow(zpB, 5.5) * std::pow(zsA, 5.5) * std::pow(Rb, 11)
             * ((A121[9] * B121[0] - A121[10] * B121[1]) + (A121[8] * B121[1] - A121[9] * B121[2])
                - 4.0 * (A121[7] * B121[2] - A121[8] * B121[3]) - 4.0 * (A121[6] * B121[3] - A121[7] * B121[4])
                + 6.0 * (A121[5] * B121[4] - A121[6] * B121[5]) + 6.0 * (A121[4] * B121[5] - A121[5] * B121[6])
                - 4.0 * (A121[3] * B121[6] - A121[4] * B121[7]) - 4.0 * (A121[2] * B121[7] - A121[3] * B121[8])
                + (A121[1] * B121[8] - A121[2] * B121[9]) + (A121[0] * B121[9] - A121[1] * B121[10]))
             / (2419200.0 * sqrt3);
      S221 = std::pow(zpB, 5.5) * std::pow(zpA, 5.5) * std::pow(Rb, 11)
             * ((A22[8] * B22[0] - A22[10] * B22[2]) - 4.0 * (A22[6] * B22[2] - A22[8] * B22[4])
                + 5.0 * (A22[4] * B22[4] - A22[6] * B22[6]) - 4.0 * (A22[2] * B22[6] - A22[4] * B22[8])
                + (A22[0] * B22[8] - A22[2] * B22[10])) / 2419200.0;
      S222 = std::pow(zpB, 5.5) * std::pow(zpA, 5.5) * std::pow(Rb, 11)
             * ((A22[10] - A22[8]) * (B22[0] - B22[2]) - 4.0 * (A22[8] - A22[6]) * (B22[2] - B22[4])
                + 5.0 * (A22[6] - A22[4]) * (B22[4] - B22[6]) - 4.0 * (A22[4] - A22[2]) * (B22[6] - B22[8])
                + (A22[2] - A22[0]) * (B22[8] - B22[10])) / 4838400.0;
    }
  }

  const double v[3] = {Rvec[0] / R, Rvec[1] / R, Rvec[2] / R};
  double rot[3][3];
  ovRotation(v, rot);
  const double* r0 = rot[0];
  const double* r1 = rot[1];
  const double* r2 = rot[2];

  for (int i = 0; i < nA * nB; ++i) outBlock[i] = 0.0;
  outBlock[0] = S111;
  if (jcall == 3 || jcall == 431 || jcall == 541 || jcall == 651) {
    if (nA > 1)
      for (int k = 0; k < 3; ++k) outBlock[(k + 1) * nB + 0] = S211 * r0[k];
  } else if (jcall == 4 || jcall == 5 || jcall == 6 || jcall == 642 || jcall == 752
             || jcall == 8 || jcall == 10) {
    if (nA > 1 && nB > 1) {
      for (int k = 0; k < 3; ++k) {
        outBlock[(k + 1) * nB + 0] = S211 * r0[k];
        outBlock[0 * nB + (k + 1)] = -S121 * r0[k];
      }
      for (int k = 0; k < 3; ++k)
        for (int l = 0; l < 3; ++l)
          outBlock[(k + 1) * nB + (l + 1)] =
              -S221 * r0[k] * r0[l] + S222 * (r1[k] * r1[l] + r2[k] * r2[l]);
    }
  }
  return nA * nB;
}

}  // namespace semiempirical
}  // namespace nvMolKit

#endif  // NVMOLKIT_SEMIEMPIRICAL_OVERLAP_DEVICE_H
