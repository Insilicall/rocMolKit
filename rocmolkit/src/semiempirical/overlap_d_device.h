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
// Device-callable d-orbital diatomic STO overlap blocks (PM6_D), ported from the
// PYSEQM diatom_overlap_matrixD and validated bit-exact par-by-par against the
// frozen oracle (tools/semiempirical/validate_doverlap.py + golden_doverlap.json).
// The d-block uses direction cosines (ca,cb,sa,sb of the bond unit vector) — NOT
// a Wigner-D rotation. aintgs/bintgs are the same A/B integrals as the sp case
// (reused from overlap_device.h). d-orbital order: [dz2, dxz, dyz, dx2-y2, dxy].

#ifndef NVMOLKIT_SEMIEMPIRICAL_OVERLAP_D_DEVICE_H
#define NVMOLKIT_SEMIEMPIRICAL_OVERLAP_D_DEVICE_H

#include <cmath>

#include "device_macros.h"
#include "overlap_device.h"  // ovdetail::aintgs / bintgs / kOvAngToBohr

namespace nvMolKit {
namespace semiempirical {

// Direction cosines (ca,cb,sa,sb) of the bond unit vector v = (coordB-coordA)/R.
NVMOLKIT_HD inline void bondAngles(const double v[3], double& ca, double& cb,
                                   double& sa, double& sb) {
  const double xy = std::sqrt(v[0] * v[0] + v[1] * v[1]);
  cb = v[2];
  sb = xy;
  if (xy >= 1e-10) {
    ca = v[0] / xy;
    sa = v[1] / xy;
  } else {
    ca = (v[2] < 0.0) ? -1.0 : 1.0;
    sa = 0.0;
  }
}

// d-s overlap column (5 d-orbitals on A, 1 s on B). dqnA = principal qn of A's d
// shell (3 for P/S/Cl, 4 for Br, 5 for I); qnB = principal qn of B's s shell.
// out[5].
NVMOLKIT_HD inline void dsBlockDev(double zd, double zs, double Rb, int dqnA, int qnB,
                                   double ca, double cb, double sa, double sb, double* out) {
  using namespace ovdetail;
  double A[kOvNMax], B[kOvNMax];
  aintgs(0.5 * Rb * (zd + zs), A);
  bintgs(0.5 * Rb * (zd - zs), B);
  // Reusable radial polynomials (the three appearing across jcall variants).
  const double p431 = (A[2] * (3 * B[0] - B[2]) + A[4] * (3 * B[2] - B[0]) + 4 * A[3] * B[1])
                      - (A[0] * (3 * B[2] - B[4]) + A[2] * (3 * B[4] - B[2]) + 4 * A[1] * B[3]);
  const double p5 = ((A[3] * (3 * B[0] - B[2]) + A[5] * (3 * B[2] - B[0]) + 4 * A[4] * B[1])
                     + (-A[2] * (3 * B[1] - B[3]) - A[4] * (3 * B[3] - B[1]) - 4 * A[3] * B[2]))
                    - ((A[1] * (3 * B[2] - B[4]) + A[3] * (3 * B[4] - B[2]) + 4 * A[2] * B[3])
                       + (-A[0] * (3 * B[3] - B[5]) - A[2] * (3 * B[5] - B[3]) - 4 * A[1] * B[4]));
  const double p6 = (A[4] * (3 * B[0] - B[2]) + A[6] * (3 * B[2] - B[0]) + 4 * A[5] * B[1])
                    + 2.0 * (-A[3] * (3 * B[1] - B[3]) - A[5] * (3 * B[3] - B[1]) - 4 * A[4] * B[2])
                    - 2.0 * (-A[1] * (3 * B[3] - B[5]) - A[3] * (3 * B[5] - B[3]) - 4 * A[2] * B[4])
                    - (A[0] * (3 * B[4] - B[6]) + A[2] * (3 * B[6] - B[4]) + 4 * A[1] * B[5]);
  // jcall 541 (Br + H): same four groups as p5 but signs (+ - - +).
  const double p541 = (A[3] * (3 * B[0] - B[2]) + A[5] * (3 * B[2] - B[0]) + 4 * A[4] * B[1])
                      - (-A[2] * (3 * B[1] - B[3]) - A[4] * (3 * B[3] - B[1]) - 4 * A[3] * B[2])
                      - (A[1] * (3 * B[2] - B[4]) + A[3] * (3 * B[4] - B[2]) + 4 * A[2] * B[3])
                      + (-A[0] * (3 * B[3] - B[5]) - A[2] * (3 * B[5] - B[3]) - 4 * A[1] * B[4]);
  // jcall 642 (Br + 2nd-row): three groups, weights 1 / -2 / 1.
  const double p642 = (A[4] * (3 * B[0] - B[2]) + A[6] * (3 * B[2] - B[0]) + 4 * A[5] * B[1])
                      - 2.0 * (A[2] * (3 * B[2] - B[4]) + A[4] * (3 * B[4] - B[2]) + 4 * A[3] * B[3])
                      + (A[0] * (3 * B[4] - B[6]) + A[2] * (3 * B[6] - B[4]) + 4 * A[1] * B[5]);
  double s311 = 0.0;
  if (dqnA == 3) {
    if (qnB <= 1) s311 = std::pow(zs, 1.5) * std::pow(zd, 3.5) * std::pow(Rb, 5) * p431 / (48.0 * std::sqrt(2.0));
    else if (qnB == 2) s311 = std::pow(zs, 2.5) * std::pow(zd, 3.5) * std::pow(Rb, 6) * p5 / (96.0 * std::sqrt(6.0));
    else s311 = std::pow(zs, 3.5) * std::pow(zd, 3.5) * std::pow(Rb, 7) * p6 / (576.0 * std::sqrt(5.0));
  } else if (dqnA == 4) {
    if (qnB <= 1) s311 = std::pow(zs, 1.5) * std::pow(zd, 4.5) * std::pow(Rb, 6) * p541 / (192.0 * std::sqrt(7.0));
    else if (qnB == 2) s311 = std::pow(zs, 2.5) * std::pow(zd, 4.5) * std::pow(Rb, 7) * p642 / (384.0 * std::sqrt(21.0));
    // qnB 3/4 (jcall 7/8) extend the same pattern when needed.
  }
  (void)p5;
  const double s3 = std::sqrt(3.0), s34 = std::sqrt(0.75);
  out[0] = s311 * s34 * (2 * ca * ca - 1) * sb * sb;
  out[1] = s311 * s3 * ca * sb * cb;
  out[2] = s311 * (cb * cb - 0.5 * sb * sb);
  out[3] = s311 * s3 * sa * sb * cb;
  out[4] = s311 * s3 * sa * ca * sb * sb;
}

// d-p overlap block (5 d on A, 3 p on B). dqnA = principal qn of A's d shell
// (3 for P/S/Cl, 4 for Br); qnB = principal qn of B's p shell. out 5*3 [d][p].
NVMOLKIT_HD inline void dpBlockDev(double zd, double zp, double Rb, int dqnA, int qnB,
                                   double ca, double cb, double sa, double sb, double* out) {
  using namespace ovdetail;
  double A[kOvNMax], B[kOvNMax];
  aintgs(0.5 * Rb * (zd + zp), A);
  bintgs(0.5 * Rb * (zd - zp), B);
  double s321, s322;
  if (dqnA == 3 && qnB <= 2) {  // d(qn3) - p(qn2), jcall 5
    const double pre = std::pow(zp, 2.5) * std::pow(zd, 3.5) * std::pow(Rb, 6);
    s321 = pre
        * ((A[2] * (3 * B[0] - B[2]) + A[3] * (B[1] + B[3]) - A[4] * (B[0] + B[2]) - A[5] * (3 * B[3] - B[1]))
           - (A[0] * (3 * B[2] - B[4]) + A[1] * (B[3] + B[5]) - A[2] * (B[2] + B[4]) - A[3] * (3 * B[5] - B[3])))
        / (96.0 * std::sqrt(2.0));
    s322 = pre
        * (((A[4] - A[2]) * (B[0] - B[2]) + (A[3] - A[5]) * (-B[1] + B[3]))
           - ((A[2] - A[0]) * (B[2] - B[4]) + (A[1] - A[3]) * (-B[3] + B[5])))
        / (32.0 * std::sqrt(6.0));
  } else {
    // jcall 6 (d-qn3 + p-qn3) and jcall 642 (d-qn4 + p-qn2) share the same four
    // groups (G/H), differing only in sign pattern and prefactor/denominator.
    const double G1 = A[3] * (3 * B[0] - B[2]) + A[4] * (B[1] + B[3]) - A[5] * (B[0] + B[2]) - A[6] * (3 * B[3] - B[1]);
    const double G2 = -A[2] * (3 * B[1] - B[3]) - A[3] * (B[2] + B[4]) + A[4] * (B[1] + B[3]) + A[5] * (3 * B[4] - B[2]);
    const double G3 = A[1] * (3 * B[2] - B[4]) + A[2] * (B[3] + B[5]) - A[3] * (B[2] + B[4]) - A[4] * (3 * B[5] - B[3]);
    const double G4 = -A[0] * (3 * B[3] - B[5]) - A[1] * (B[4] + B[6]) + A[2] * (B[3] + B[5]) + A[3] * (3 * B[6] - B[4]);
    const double H1 = (A[5] - A[3]) * (B[0] - B[2]) + (A[4] - A[6]) * -(B[1] - B[3]);
    const double H2 = (A[4] - A[2]) * -(B[1] - B[3]) + (A[3] - A[5]) * (B[2] - B[4]);
    const double H3 = (A[3] - A[1]) * (B[2] - B[4]) + (A[2] - A[4]) * -(B[3] - B[5]);
    const double H4 = (A[2] - A[0]) * -(B[3] - B[5]) + (A[1] - A[3]) * (B[4] - B[6]);
    if (dqnA == 3) {  // jcall 6: signs (+ + - -)
      const double pre = std::pow(zp, 3.5) * std::pow(zd, 3.5) * std::pow(Rb, 7);
      s321 = pre * (G1 + G2 - G3 - G4) / (192.0 * std::sqrt(15.0));
      s322 = pre * (H1 + H2 - H3 - H4) / (192.0 * std::sqrt(5.0));
    } else {  // jcall 642: signs (+ - - +)
      const double pre = std::pow(zp, 2.5) * std::pow(zd, 4.5) * std::pow(Rb, 7);
      s321 = pre * (G1 - G2 - G3 + G4) / (384.0 * std::sqrt(7.0));
      s322 = pre * (H1 - H2 - H3 + H4) / (128.0 * std::sqrt(21.0));
    }
  }
  const double s3 = std::sqrt(3.0), s34 = std::sqrt(0.75), t = 2 * ca * ca - 1;
  out[0]  = -(s321 * s34 * t * sb * sb * ca * sb - s322 * (t * sb * cb * ca * cb + 2 * sa * ca * sb * sa));
  out[1]  = -(s321 * s34 * t * sb * sb * sa * sb - s322 * (t * sb * cb * sa * cb - 2 * sa * ca * sb * ca));
  out[2]  = -(s321 * s34 * t * sb * sb * cb + s322 * (t * sb * cb * sb));
  out[3]  = -(s321 * s3 * ca * sb * cb * ca * sb - s322 * (ca * (2 * cb * cb - 1) * ca * cb + sa * cb * sa));
  out[4]  = -(s321 * s3 * ca * sb * cb * sa * sb - s322 * (ca * (2 * cb * cb - 1) * sa * cb - sa * cb * ca));
  out[5]  = -(s321 * s3 * ca * sb * cb * cb + s322 * (ca * (2 * cb * cb - 1) * sb));
  out[6]  = -(s321 * (cb * cb - 0.5 * sb * sb) * ca * sb + s322 * s3 * sb * cb * ca * cb);
  out[7]  = -(s321 * (cb * cb - 0.5 * sb * sb) * sa * sb + s322 * s3 * sb * cb * sa * cb);
  out[8]  = -(s321 * (cb * cb - 0.5 * sb * sb) * cb - s322 * s3 * sb * cb * sb);
  out[9]  = -(s321 * s3 * sa * sb * cb * ca * sb - s322 * ((sa * (2 * cb * cb - 1)) * ca * cb - ca * cb * sa));
  out[10] = -(s321 * s3 * sa * sb * cb * sa * sb - s322 * ((sa * (2 * cb * cb - 1)) * sa * cb + ca * cb * ca));
  out[11] = -(s321 * s3 * sa * sb * cb * cb + s322 * ((sa * (2 * cb * cb - 1)) * sb));
  out[12] = -(s321 * s3 * sa * ca * sb * sb * ca * sb - s322 * (2 * sa * ca * sb * cb * ca * cb - sb * t * sa));
  out[13] = -(s321 * s3 * sa * ca * sb * sb * sa * sb - s322 * (2 * sa * ca * sb * cb * sa * cb + sb * t * ca));
  out[14] = -(s321 * s3 * sa * ca * sb * sb * cb + s322 * (2 * sa * ca * sb * cb * sb));
}

// d-d overlap 5x5 block (jcall 6). out is 5*5 row-major. Slater-Koster outer
// product, with the dyz-dxy cross term (out[3][4]=out[4][3]) written explicitly.
NVMOLKIT_HD inline void ddBlockDev(double zd1, double zd2, double Rb,
                                   double ca, double cb, double sa, double sb, double* out) {
  using namespace ovdetail;
  double A[kOvNMax], B[kOvNMax];
  aintgs(0.5 * Rb * (zd1 + zd2), A);
  bintgs(0.5 * Rb * (zd1 - zd2), B);
  const double w = std::pow(zd2, 3.5) * std::pow(zd1, 3.5) * std::pow(Rb, 7);
  const double s333 = w * (((A[2] - 2 * A[4] + A[6]) * (B[0] - 2 * B[2] + B[4]))
                           - ((A[0] - 2 * A[2] + A[4]) * (B[2] - 2 * B[4] + B[6]))) / 768.0;
  const double s332 = -w * ((A[2] * (B[2] - B[0]) + A[4] * (B[0] - B[4]) - A[6] * (B[2] - B[4]))
                            - (A[0] * (B[4] - B[2]) + A[2] * (B[2] - B[6]) - A[4] * (B[4] - B[6]))) / 192.0;
  const double s331 = w * ((A[2] * (9 * B[0] - 6 * B[2] + B[4]) - 2 * A[4] * (3 * B[0] - 2 * B[2] + 3 * B[4])
                            + A[6] * (B[0] - 6 * B[2] + 9 * B[4]))
                           - (A[0] * (9 * B[2] - 6 * B[4] + B[6]) - 2 * A[2] * (3 * B[2] - 2 * B[4] + 3 * B[6])
                              + A[4] * (B[2] - 6 * B[4] + 9 * B[6]))) / 1152.0;
  const double s3 = std::sqrt(3.0), s34 = std::sqrt(0.75), t = 2 * ca * ca - 1;
  const double sig[5] = {s34 * t * sb * sb, s3 * ca * sb * cb, cb * cb - 0.5 * sb * sb,
                         s3 * sa * sb * cb, s3 * sa * ca * sb * sb};
  const double p1[5] = {sb * cb * t, ca * (2 * cb * cb - 1), -s3 * sb * cb,
                        sa * (2 * cb * cb - 1), 2 * sa * ca * sb * cb};
  const double p2[5] = {2 * sa * ca * sb, sa * cb, 0.0, -ca * cb, -t * sb};
  const double d1[5] = {t * (cb * cb + 0.5 * sb * sb), -ca * sb * cb, s34 * sb * sb,
                        -sa * sb * cb, 2 * sa * ca * cb * cb + sa * ca * sb * sb};
  const double d2[5] = {2 * sa * ca * cb, -sa * sb, 0.0, ca * sb, -cb * t};
  for (int i = 0; i < 5; ++i)
    for (int j = 0; j < 5; ++j)
      out[i * 5 + j] = s331 * sig[i] * sig[j] + s332 * (p1[i] * p1[j] + p2[i] * p2[j])
                       + s333 * (d1[i] * d1[j] + d2[i] * d2[j]);
  const double e78 = s331 * (s3 * sa * sb * cb) * (s3 * sa * ca * sb * sb)
                     + s332 * ((sa * (2 * cb * cb - 1)) * (2 * sa * ca * sb * cb) + (ca * cb) * (t * sb))
                     + s333 * (-(sa * sb * cb) * (2 * sa * ca * cb * cb + sa * ca * sb * sb)
                               + ca * sb * cb * t);
  out[3 * 5 + 4] = e78;
  out[4 * 5 + 3] = e78;
}

// Full molecular-frame overlap block (nA x nB, orbital order [s,px,py,pz,d...])
// between atoms A and B, including d-orbitals. The sp rows/cols come from the
// validated diatomOverlapSpDev (which already lays them out in the nOrb-strided
// block with the d positions zeroed); the d rows are filled from the validated
// ds/dp/dd blocks. To keep one code path, the heavier (more orbitals) atom is
// always treated as A: if nA < nB the (B,A) block is built and transposed.
// Returns nA*nB. NOTE: only the A-has-d rows are filled (d-s, d-p, d-d); the
// sp-row x B-d-col entries (s-d, p-d when only B has d) are handled by the
// transpose path, so a pair where BOTH atoms carry d (YY) leaves the A-sp x B-d
// block unfilled — that case is gated out until the YY two-center is validated.
NVMOLKIT_HD inline int diatomOverlapDDev(const AtomIntParams& pA, const double coordA[3],
                                         const AtomIntParams& pB, const double coordB[3],
                                         double* out) {
  const int nA = pA.nOrb, nB = pB.nOrb;
  if (nA == 0 || nB == 0) return 0;

  if (nA < nB) {  // ensure the d-bearing / larger atom is A; transpose at the end
    double tmp[81];
    diatomOverlapDDev(pB, coordB, pA, coordA, tmp);
    for (int i = 0; i < nA; ++i)
      for (int j = 0; j < nB; ++j) out[i * nB + j] = tmp[j * nA + i];
    return nA * nB;
  }

  // sp rows/cols (d positions left zero) in the nA x nB layout.
  diatomOverlapSpDev(pA, coordA, pB, coordB, out);
  if (nA < 9) return nA * nB;  // A has no d shell -> done

  const double Rvec[3] = {coordB[0] - coordA[0], coordB[1] - coordA[1], coordB[2] - coordA[2]};
  const double R = std::sqrt(Rvec[0] * Rvec[0] + Rvec[1] * Rvec[1] + Rvec[2] * Rvec[2]);
  const double Rb = R * ovdetail::kOvAngToBohr;
  const double v[3] = {Rvec[0] / R, Rvec[1] / R, Rvec[2] / R};
  double ca, cb, sa, sb;
  bondAngles(v, ca, cb, sa, sb);

  // A's d rows (4..8) against B's s / p / d columns.
  double ds[5];
  dsBlockDev(pA.zetaD, pB.zetaS, Rb, pA.qnD, pB.qn, ca, cb, sa, sb, ds);
  for (int m = 0; m < 5; ++m) out[(4 + m) * nB + 0] = ds[m];
  if (nB >= 4) {
    double dp[15];
    dpBlockDev(pA.zetaD, pB.zetaP, Rb, pA.qnD, pB.qn, ca, cb, sa, sb, dp);
    for (int m = 0; m < 5; ++m)
      for (int q = 0; q < 3; ++q) out[(4 + m) * nB + (1 + q)] = dp[m * 3 + q];
  }
  if (nB == 9) {
    double dd[25];
    ddBlockDev(pA.zetaD, pB.zetaD, Rb, ca, cb, sa, sb, dd);
    for (int m = 0; m < 5; ++m)
      for (int n = 0; n < 5; ++n) out[(4 + m) * nB + (4 + n)] = dd[m * 5 + n];
    // A's sp rows (s,p) against B's d columns: S(s/p_A, d_B) = the d-s/d-p block
    // of the reversed pair (B's d with A's s/p), bond B->A = -v.
    const double vr[3] = {-v[0], -v[1], -v[2]};
    double car, cbr, sar, sbr;
    bondAngles(vr, car, cbr, sar, sbr);
    double dsr[5];
    dsBlockDev(pB.zetaD, pA.zetaS, Rb, pB.qnD, pA.qn, car, cbr, sar, sbr, dsr);
    for (int m = 0; m < 5; ++m) out[0 * nB + (4 + m)] = dsr[m];
    double dpr[15];
    dpBlockDev(pB.zetaD, pA.zetaP, Rb, pB.qnD, pA.qn, car, cbr, sar, sbr, dpr);
    for (int m = 0; m < 5; ++m)
      for (int q = 0; q < 3; ++q) out[(1 + q) * nB + (4 + m)] = dpr[m * 3 + q];
  }
  return nA * nB;
}

}  // namespace semiempirical
}  // namespace nvMolKit

#endif  // NVMOLKIT_SEMIEMPIRICAL_OVERLAP_D_DEVICE_H
