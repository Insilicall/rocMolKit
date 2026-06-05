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
// Device-callable d-orbital two-center rotation: maps the local-frame reduced
// two-electron integral block (45x45 packed lower-triangle pairs) into the
// molecular frame. Ported from PYSEQM's RotationMatrixD (BSD-3): build the 15x45
// pair-rotation matrix from the P (3x3) and D (5x5) orbital rotations, then apply
// the YM similarity transform FINAL = YM * WW * YM^T (sp 10x10 block preserved).
// Validated bit-exact against the oracle by validate_d_wtensor.py.

#ifndef NVMOLKIT_SEMIEMPIRICAL_D_ROTATION_DEVICE_H
#define NVMOLKIT_SEMIEMPIRICAL_D_ROTATION_DEVICE_H

#include <cmath>

#include "device_macros.h"

namespace nvMolKit {
namespace semiempirical {
namespace drot {

// Build the 15x45 pair-rotation matrix from the bond unit vector v = (cB-cA)/R
// (negated internally, matching PYSEQM). matrix is row-major [15][45].
NVMOLKIT_HD inline void generateRotationMatrixD(const double v[3], double* matrix) {
  const double PT5SQ3 = 0.8660254037841, PT5 = 0.5;
  const double x = -v[0], y = -v[1], z = -v[2];
  const double xy = std::sqrt(x * x + y * y);
  const double tmp = (z < 0.0) ? -1.0 : (z > 0.0 ? 1.0 : 0.0);
  double CA, CB, SA, SB;
  if (xy >= 1e-10) { CA = x / xy; SA = y / xy; CB = z; SB = xy; }
  else { CA = tmp; CB = tmp; SA = 0.0; SB = 0.0; }
  const double C2A = 2.0 * CA * CA - 1.0, C2B = 2.0 * CB * CB - 1.0;
  const double S2A = 2.0 * SA * CA, S2B = 2.0 * SB * CB;

  double P[3][3] = {{0}};
  P[0][0] = CA * SB; P[1][0] = CA * CB; P[2][0] = -SA;
  P[0][1] = SA * SB; P[1][1] = SA * CB; P[2][1] = CA;
  P[0][2] = CB;      P[1][2] = -SB;

  double D[5][5] = {{0}};
  D[0][0] = PT5SQ3 * C2A * SB * SB; D[1][0] = PT5 * C2A * S2B; D[2][0] = -S2A * SB;
  D[3][0] = C2A * (CB * CB + PT5 * SB * SB); D[4][0] = -S2A * CB;
  D[0][1] = PT5SQ3 * CA * S2B; D[1][1] = CA * C2B; D[2][1] = -SA * CB;
  D[3][1] = -PT5 * CA * S2B; D[4][1] = SA * SB;
  D[0][2] = CB * CB - PT5 * SB * SB; D[1][2] = -PT5SQ3 * S2B; D[3][2] = PT5SQ3 * SB * SB;
  D[0][3] = PT5SQ3 * SA * S2B; D[1][3] = SA * C2B; D[2][3] = CA * CB;
  D[3][3] = -PT5 * SA * S2B; D[4][3] = -CA * SB;
  D[0][4] = PT5SQ3 * S2A * SB * SB; D[1][4] = PT5 * S2A * S2B; D[2][4] = C2A * SB;
  D[3][4] = S2A * (CB * CB + PT5 * SB * SB); D[4][4] = C2A * CB;

  const int INDX[9] = {0, 1, 3, 6, 10, 15, 21, 28, 36};
  for (int i = 0; i < 15 * 45; ++i) matrix[i] = 0.0;
  auto M = [&](int r, int c) -> double& { return matrix[r * 45 + c]; };

  M(0, 0) = 1.0;                                   // S-S
  for (int K = 0; K < 3; ++K) {                    // P-S
    const int KL = INDX[K + 1];
    M(0, KL) = P[K][0]; M(1, KL) = P[K][1]; M(2, KL) = P[K][2];
  }
  for (int K = 0; K < 3; ++K) {                    // P-P diagonal
    const int KL = INDX[K + 1] + K + 1;
    M(0, KL) = P[K][0] * P[K][0]; M(1, KL) = P[K][0] * P[K][1]; M(2, KL) = P[K][1] * P[K][1];
    M(3, KL) = P[K][0] * P[K][2]; M(4, KL) = P[K][1] * P[K][2]; M(5, KL) = P[K][2] * P[K][2];
  }
  for (int K = 1; K < 3; ++K)                      // P-P off-diagonal
    for (int L = 0; L <= K - 1; ++L) {
      const int KL = INDX[K + 1] + L + 1;
      M(0, KL) = P[K][0] * P[L][0] * 2.0;
      M(1, KL) = P[K][0] * P[L][1] + P[K][1] * P[L][0];
      M(2, KL) = P[K][1] * P[L][1] * 2.0;
      M(3, KL) = P[K][0] * P[L][2] + P[K][2] * P[L][0];
      M(4, KL) = P[K][1] * P[L][2] + P[K][2] * P[L][1];
      M(5, KL) = P[K][2] * P[L][2] * 2.0;
    }
  for (int K = 0; K < 5; ++K) {                    // D-S
    const int KL = INDX[K + 4];
    for (int r = 0; r < 5; ++r) M(r, KL) = D[K][r];
  }
  for (int K = 0; K < 5; ++K)                      // D-P
    for (int L = 0; L < 3; ++L) {
      const int KL = INDX[K + 4] + L + 1;
      for (int dr = 0; dr < 5; ++dr)
        for (int pr = 0; pr < 3; ++pr) M(dr * 3 + pr, KL) = D[K][dr] * P[L][pr];
    }
  for (int K = 0; K < 5; ++K) {                    // D-D diagonal
    const int KL = INDX[K + 4] + K + 4;
    M(0, KL) = D[K][0] * D[K][0]; M(1, KL) = D[K][0] * D[K][1]; M(2, KL) = D[K][1] * D[K][1];
    M(3, KL) = D[K][0] * D[K][2]; M(4, KL) = D[K][1] * D[K][2]; M(5, KL) = D[K][2] * D[K][2];
    M(6, KL) = D[K][0] * D[K][3]; M(7, KL) = D[K][1] * D[K][3]; M(8, KL) = D[K][2] * D[K][3];
    M(9, KL) = D[K][3] * D[K][3]; M(10, KL) = D[K][0] * D[K][4]; M(11, KL) = D[K][1] * D[K][4];
    M(12, KL) = D[K][2] * D[K][4]; M(13, KL) = D[K][3] * D[K][4]; M(14, KL) = D[K][4] * D[K][4];
  }
  for (int K = 0; K < 5; ++K)                      // D-D off-diagonal
    for (int L = 0; L < K; ++L) {
      const int KL = INDX[K + 4] + L + 4;
      M(0, KL) = D[K][0] * D[L][0] * 2.0;
      M(1, KL) = D[K][0] * D[L][1] + D[K][1] * D[L][0];
      M(2, KL) = D[K][1] * D[L][1] * 2.0;
      M(3, KL) = D[K][0] * D[L][2] + D[K][2] * D[L][0];
      M(4, KL) = D[K][1] * D[L][2] + D[K][2] * D[L][1];
      M(5, KL) = D[K][2] * D[L][2] * 2.0;
      M(6, KL) = D[K][0] * D[L][3] + D[K][3] * D[L][0];
      M(7, KL) = D[K][1] * D[L][3] + D[K][3] * D[L][1];
      M(8, KL) = D[K][2] * D[L][3] + D[K][3] * D[L][2];
      M(9, KL) = D[K][3] * D[L][3] * 2.0;
      M(10, KL) = D[K][0] * D[L][4] + D[K][4] * D[L][0];
      M(11, KL) = D[K][1] * D[L][4] + D[K][4] * D[L][1];
      M(12, KL) = D[K][2] * D[L][4] + D[K][4] * D[L][2];
      M(13, KL) = D[K][3] * D[L][4] + D[K][4] * D[L][3];
      M(14, KL) = D[K][4] * D[L][4] * 2.0;
    }
}

// Apply the YM similarity transform to the local-frame 45x45 WW, producing the
// molecular-frame 45x45 (row-major). out may not alias WW.
NVMOLKIT_HD inline void rotate2Center2ElectronD(const double* WW, const double* matrix,
                                                double* out) {
  const int MET[45] = {1, 2, 3, 2, 3, 3, 2, 3, 3, 3, 4, 5, 5, 5, 6, 4, 5, 5, 5, 6, 6,
                       4, 5, 5, 5, 6, 6, 6, 4, 5, 5, 5, 6, 6, 6, 6, 4, 5, 5, 5, 6, 6, 6, 6, 6};
  const int META[6] = {1, 2, 5, 11, 16, 31};
  const int METB[6] = {1, 4, 10, 15, 30, 45};
  const int METI[15][6] = {
      {1, 2, 3, 11, 12, 15}, {0, 4, 5, 16, 13, 20}, {0, 7, 6, 22, 14, 21},
      {0, 0, 8, 29, 17, 26}, {0, 0, 9, 37, 18, 27}, {0, 0, 10, 0, 19, 28},
      {0, 0, 0, 0, 23, 33}, {0, 0, 0, 0, 24, 34}, {0, 0, 0, 0, 25, 35},
      {0, 0, 0, 0, 30, 36}, {0, 0, 0, 0, 31, 41}, {0, 0, 0, 0, 32, 42},
      {0, 0, 0, 0, 38, 43}, {0, 0, 0, 0, 39, 44}, {0, 0, 0, 0, 40, 45}};

  double YM[45 * 45];
  for (int i = 0; i < 45 * 45; ++i) YM[i] = 0.0;
  YM[0] = 1.0;  // YM[0][0]
  for (int KL = 1; KL < 45; ++KL) {
    const int mkl = MET[KL] - 1;
    const int NKL = METB[mkl] - META[mkl] + 1;
    for (int I = 0; I < NKL; ++I)
      YM[(METI[I][mkl] - 1) * 45 + KL] = matrix[I * 45 + KL];
  }

  // FINAL[i][j] = sum_{k,l} YM[i][k] * WW[l][k] * YM[j][l].
  // STEP[i][l] = sum_k YM[i][k] * WW[l][k];  FINAL[i][j] = sum_l STEP[i][l]*YM[j][l].
  double STEP[45 * 45];
  for (int i = 0; i < 45; ++i)
    for (int l = 0; l < 45; ++l) {
      double s = 0.0;
      for (int k = 0; k < 45; ++k) s += YM[i * 45 + k] * WW[l * 45 + k];
      STEP[i * 45 + l] = s;
    }
  double FINAL[45 * 45];
  for (int i = 0; i < 45; ++i)
    for (int j = 0; j < 45; ++j) {
      double s = 0.0;
      for (int l = 0; l < 45; ++l) s += STEP[i * 45 + l] * YM[j * 45 + l];
      FINAL[i * 45 + j] = s;
    }
  // sp 10x10 block stays the local value, then transpose (1<->2).
  for (int i = 0; i < 10; ++i)
    for (int j = 0; j < 10; ++j) FINAL[i * 45 + j] = WW[i * 45 + j];
  for (int i = 0; i < 45; ++i)
    for (int j = 0; j < 45; ++j) out[i * 45 + j] = FINAL[j * 45 + i];
}

// Lower-triangle packed index (i>=j): i*(i+1)/2 + j.
NVMOLKIT_HD inline int packedTril(int i, int j) {
  if (i < j) { const int t = i; i = j; j = t; }
  return i * (i + 1) / 2 + j;
}

}  // namespace drot
}  // namespace semiempirical
}  // namespace nvMolKit

#endif  // NVMOLKIT_SEMIEMPIRICAL_D_ROTATION_DEVICE_H
