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
// Diatomic STO overlap for NDDO/PM6, ported faithfully from the PYSEQM
// PM6_SP formulation (the A/B reduced-integral method of Mulliken-Roothaan)
// so that the result is bit-exact to the validated CPU oracle. See
// docs/SEMIEMPIRICAL_DESIGN.md and tools/semiempirical/golden_intermediates.json.

#include "overlap.h"

#include <cmath>

#include "pm6_params.h"

namespace nvMolKit {
namespace semiempirical {

namespace {

// 1 bohr = 0.529167 Angstrom (MOPAC/PYSEQM convention used by the oracle).
constexpr double kAngToBohr = 1.0 / 0.529167;

constexpr int kNMax = 7;  // covers jcall 2/3/4 (needs A/B up to index 4)

// A-integrals: a[0] = exp(-alpha)/alpha ; a[k] = a[0] + k*a[k-1]/alpha.
void aintgs(double alpha, double* a) {
  if (std::fabs(alpha) < 1e-10) {
    for (int k = 0; k < kNMax; ++k) {
      a[k] = 0.0;
    }
    return;
  }
  a[0] = std::exp(-alpha) / alpha;
  for (int k = 1; k < kNMax; ++k) {
    a[k] = a[0] + k * a[k - 1] / alpha;
  }
}

// B-integrals b[k](beta), three regimes matching PYSEQM PM6_SP.
void bintgs(double beta, double* b) {
  const double x = beta;
  if (std::fabs(x) <= 1e-6) {  // even -> 2/(k+1), odd -> 0
    for (int k = 0; k < kNMax; ++k) {
      b[k] = (k % 2 == 0) ? 2.0 / (k + 1) : 0.0;
    }
    return;
  }
  if (std::fabs(x) <= 0.5) {  // Taylor series
    // even_coeffs[k/2] dotted with (1, x^2, x^4, x^6);
    // odd_factors[k/2] dotted with (x, x^3, x^5).
    static const double even[7][4] = {
        {2.0, 1.0 / 3, 1.0 / 60, 1.0 / 2520},
        {2.0 / 3, 1.0 / 5, 1.0 / 84, 1.0 / 3240},
        {2.0 / 5, 1.0 / 7, 1.0 / 108, 1.0 / 3960},
        {2.0 / 7, 1.0 / 9, 1.0 / 132, 1.0 / 4680},
        {2.0 / 9, 1.0 / 11, 1.0 / 156, 1.0 / 5400},
        {2.0 / 11, 1.0 / 13, 1.0 / 180, 1.0 / 6120},
        {2.0 / 13, 1.0 / 15, 1.0 / 204, 1.0 / 6840},
    };
    static const double odd[6][3] = {
        {-2.0 / 3, -1.0 / 15, -1.0 / 420},
        {-2.0 / 5, -1.0 / 21, -1.0 / 540},
        {-2.0 / 7, -1.0 / 27, -1.0 / 660},
        {-2.0 / 9, -1.0 / 33, -1.0 / 780},
        {-2.0 / 11, -1.0 / 39, -1.0 / 900},
        {-2.0 / 13, -1.0 / 45, -1.0 / 1020},
    };
    const double x2 = x * x, x3 = x * x2, x4 = x2 * x2, x5 = x2 * x3, x6 = x2 * x4;
    for (int k = 0; k < kNMax; ++k) {
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
  // Exact recurrence for |beta| > 0.5.
  const double xc = (x > 500.0) ? 500.0 : (x < -500.0 ? -500.0 : x);
  const double tx = std::exp(xc) / x;
  const double tmx = -std::exp(-xc) / x;
  double sign = 1.0;
  b[0] = tx + tmx;
  for (int k = 1; k < kNMax; ++k) {
    sign = -sign;
    b[k] = sign * tx + tmx + k * b[k - 1] / x;
  }
}

// 3x3 rotation taking unit vector v to the x-axis (quaternion (0,vz,-vy,1+vx)).
// rot[row][col]; row 0 = sigma (bond) direction, rows 1,2 = pi directions.
void rotationMatrix(const double v[3], double rot[3][3]) {
  const double vx = v[0], vy = v[1], vz = v[2];
  const double w = 1.0 + vx;
  if (std::fabs(w) < 1e-7) {  // antipodal v ~ (-1,0,0): 180 deg about z
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

int spOrbitalCount(int z) {
  const int n = pm6NumOrbitals(z);
  return (n >= 4) ? 4 : n;  // clamp d-bearing atoms to the sp block
}

}  // namespace

int principalQn(int z) {
  if (z <= 0) return 0;
  if (z <= 2) return 1;
  if (z <= 10) return 2;
  if (z <= 18) return 3;
  if (z <= 36) return 4;
  if (z <= 54) return 5;
  return 6;
}

int diatomOverlapSp(int zA, const double coordA[3],
                    int zB, const double coordB[3],
                    double* outBlock) {
  const Pm6ElementParams* pA = pm6ParamsForZ(zA);
  const Pm6ElementParams* pB = pm6ParamsForZ(zB);
  if (pA == nullptr || pB == nullptr) {
    return 0;
  }
  const int nA = spOrbitalCount(zA);
  const int nB = spOrbitalCount(zB);
  if (nA == 0 || nB == 0) {
    return 0;
  }

  const double Rvec[3] = {coordB[0] - coordA[0], coordB[1] - coordA[1], coordB[2] - coordA[2]};
  const double R = std::sqrt(Rvec[0] * Rvec[0] + Rvec[1] * Rvec[1] + Rvec[2] * Rvec[2]);
  if (R < 1e-10) {
    for (int i = 0; i < nA * nB; ++i) outBlock[i] = 0.0;
    return nA * nB;
  }

  const int qnA = principalQn(zA);
  const int qnB = principalQn(zB);

  // PYSEQM convention: heavier (higher qn) atom first. If A is lighter, compute
  // B-A and transpose into the (nA x nB) output.
  if (qnA < qnB) {
    double tmp[16];  // (nB x nA), nB,nA <= 4
    diatomOverlapSp(zB, coordB, zA, coordA, tmp);
    for (int i = 0; i < nA; ++i) {
      for (int j = 0; j < nB; ++j) {
        outBlock[i * nB + j] = tmp[j * nA + i];  // transpose
      }
    }
    return nA * nB;
  }

  // jcall selects the local-frame formula set. Phase 1 supports qn in {1,2}.
  int jcall = 0;
  if (qnA == 1 && qnB == 1) jcall = 2;
  else if (qnA == 2 && qnB == 1) jcall = 3;
  else if (qnA == 2 && qnB == 2) jcall = 4;
  else {
    for (int i = 0; i < nA * nB; ++i) outBlock[i] = 0.0;  // qn>=3 not yet ported
    return nA * nB;
  }

  const double Rb = R * kAngToBohr;
  const double zsA = pA->zeta_s, zpA = pA->zeta_p;
  const double zsB = pB->zeta_s, zpB = pB->zeta_p;

  // alpha = 0.5 R (z1+z2), beta = 0.5 R (z1-z2) per orbital-pair.
  double A111[kNMax], B111[kNMax];
  aintgs(0.5 * Rb * (zsA + zsB), A111);
  bintgs(0.5 * Rb * (zsA - zsB), B111);

  double A211[kNMax] = {0}, B211[kNMax] = {0};
  if (nA > 1) {
    aintgs(0.5 * Rb * (zpA + zsB), A211);
    bintgs(0.5 * Rb * (zpA - zsB), B211);
  }
  double A121[kNMax] = {0}, B121[kNMax] = {0};
  if (nB > 1) {
    aintgs(0.5 * Rb * (zsA + zpB), A121);
    bintgs(0.5 * Rb * (zsA - zpB), B121);
  }
  double A22[kNMax] = {0}, B22[kNMax] = {0};
  if (nA > 1 && nB > 1) {
    aintgs(0.5 * Rb * (zpA + zpB), A22);
    bintgs(0.5 * Rb * (zpA - zpB), B22);
  }

  const double sqrt3 = std::sqrt(3.0);
  double S111 = 0.0, S211 = 0.0, S121 = 0.0, S221 = 0.0, S222 = 0.0;

  if (jcall == 2) {
    S111 = std::pow(zsA * zsB * Rb * Rb, 1.5)
           * (A111[2] * B111[0] - B111[2] * A111[0]) / 4.0;
  } else if (jcall == 3) {
    S111 = std::pow(zsB, 1.5) * std::pow(zsA, 2.5) * std::pow(Rb, 4)
           * (A111[3] * B111[0] - B111[3] * A111[0]
              + A111[2] * B111[1] - B111[2] * A111[1])
           / (sqrt3 * 8.0);
    if (nA > 1) {
      S211 = std::pow(zsB, 1.5) * std::pow(zpA, 2.5) * std::pow(Rb, 4)
             * (A211[2] * B211[0] - B211[2] * A211[0]
                + A211[3] * B211[1] - B211[3] * A211[1])
             / 8.0;
    }
  } else {  // jcall == 4
    S111 = std::pow(zsA * zsB, 2.5) * std::pow(Rb, 5)
           * (A111[4] * B111[0] + B111[4] * A111[0] - 2.0 * A111[2] * B111[2])
           / 48.0;
    if (nA > 1 && nB > 1) {
      S211 = std::pow(zsB * zpA, 2.5) * std::pow(Rb, 5)
             * (A211[3] * (B211[0] - B211[2]) - A211[1] * (B211[2] - B211[4])
                + B211[3] * (A211[0] - A211[2]) - B211[1] * (A211[2] - A211[4]))
             / (16.0 * sqrt3);
      S121 = std::pow(zpB * zsA, 2.5) * std::pow(Rb, 5)
             * (A121[3] * (B121[0] - B121[2]) - A121[1] * (B121[2] - B121[4])
                - B121[3] * (A121[0] - A121[2]) + B121[1] * (A121[2] - A121[4]))
             / (16.0 * sqrt3);
      const double w = std::pow(zpB * zpA, 2.5) * std::pow(Rb, 5) / 16.0;
      S221 = -w * (B22[2] * (A22[4] + A22[0]) - A22[2] * (B22[4] + B22[0]));
      S222 = 0.5 * w
             * (A22[4] * (B22[0] - B22[2]) - B22[4] * (A22[0] - A22[2])
                - A22[2] * B22[0] + B22[2] * A22[0]);
    }
  }

  // Rotate local-frame overlaps into the molecular frame.
  const double v[3] = {Rvec[0] / R, Rvec[1] / R, Rvec[2] / R};
  double rot[3][3];
  rotationMatrix(v, rot);
  const double* r0 = rot[0];
  const double* r1 = rot[1];
  const double* r2 = rot[2];

  for (int i = 0; i < nA * nB; ++i) outBlock[i] = 0.0;
  outBlock[0] = S111;  // s-s

  if (jcall == 3) {
    if (nA > 1) {
      for (int k = 0; k < 3; ++k) outBlock[(k + 1) * nB + 0] = S211 * r0[k];
    }
  } else if (jcall == 4) {
    if (nA > 1 && nB > 1) {
      for (int k = 0; k < 3; ++k) {
        outBlock[(k + 1) * nB + 0] = S211 * r0[k];          // p_A - s_B
        outBlock[0 * nB + (k + 1)] = -S121 * r0[k];         // s_A - p_B (minus)
      }
      for (int k = 0; k < 3; ++k) {
        for (int l = 0; l < 3; ++l) {
          outBlock[(k + 1) * nB + (l + 1)] =
              -S221 * r0[k] * r0[l] + S222 * (r1[k] * r1[l] + r2[k] * r2[l]);
        }
      }
    }
  }
  return nA * nB;
}

}  // namespace semiempirical
}  // namespace nvMolKit
