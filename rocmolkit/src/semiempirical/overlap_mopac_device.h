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
// General analytic Slater diatomic overlap, ported from MOPAC 23.2.5
// (openmopac/mopac, Apache-2.0): src/integrals/diat.F90 (ss + diat assembly),
// bfn.F90, coe.F90. Unlike the per-jcall overlap (overlap_d_device.h), this is a
// SINGLE routine valid for ANY principal/angular quantum numbers (incl. d), so it
// handles combinations the per-jcall tables don't tabulate -- notably the
// metal-sp(qn 4/5/6) x ligand-d(qn 3) block (e.g. ZnCl2). Validated bit-exact to
// MOPAC's AUX OVERLAP_MATRIX (tools/semiempirical/validate_mopac_overlap_port.py;
// HCl/H2S/HBr/HI/ZnCl2 to ~1e-15). Orbital order matches the engine + MOPAC:
//   1 s  2 px  3 py  4 pz  5 d(x2-y2)  6 d(xz)  7 d(z2)  8 d(yz)  9 d(xy).

#ifndef NVMOLKIT_SEMIEMPIRICAL_OVERLAP_MOPAC_DEVICE_H
#define NVMOLKIT_SEMIEMPIRICAL_OVERLAP_MOPAC_DEVICE_H

#include <cmath>

#include "device_macros.h"

namespace nvMolKit {
namespace semiempirical {
namespace mopacovl {

constexpr double kA0 = 0.529177210903;  // bohr radius (MOPAC funcon a0)

NVMOLKIT_HD inline double mopFact(int n) {
  double f = 1.0;
  for (int i = 2; i <= n; ++i) f *= static_cast<double>(i);
  return f;
}

// B integrals (MOPAC bfn.F90): bf[0..12] = int_-1^1 eta^k e^{-x eta} d eta.
NVMOLKIT_HD inline void mopBfn(double x, double* bf) {
  const int k = 12;
  const double absx = std::fabs(x);
  if (absx > 3.0) {
    const double expx = std::exp(x), expmx = 1.0 / expx;
    bf[0] = (expx - expmx) / x;
    for (int i = 1; i <= k; ++i)
      bf[i] = (i * bf[i - 1] + ((i & 1) ? -1.0 : 1.0) * expx - expmx) / x;
    return;
  }
  int last;  // truncation of the small-x Taylor series (MOPAC's branches)
  if (absx > 2.0) last = 15;
  else if (absx > 1.0) last = 12;
  else if (absx > 0.5) last = 7;
  else last = 6;
  for (int i = 0; i <= k; ++i) {
    double y = 0.0;
    for (int m = 0; m <= last; ++m) {
      const double xf = (m != 0) ? mopFact(m) : 1.0;
      y += std::pow(-x, m) * (2 * ((m + i + 1) % 2)) / (xf * (m + i + 1));
    }
    bf[i] = y;
  }
}

// General local-frame Slater overlap (MOPAC diat.F90 `ss`). na/nb principal qn;
// la1/lb1/m1 are 1-based angular indices (1=s/sigma, 2=p/pi, 3=d/delta). ua/ub
// Slater exponents; r1 internuclear distance (Angstrom).
NVMOLKIT_HD inline double mopSs(int na, int nb, int la1, int lb1, int m1, double ua, double ub,
                                double r1) {
  const int m = m1 - 1, lb = lb1 - 1, la = la1 - 1;
  const double r = r1 / kA0;
  double aff[3][3][3];
  for (int i = 0; i < 27; ++i) (&aff[0][0][0])[i] = 0.0;
  aff[0][0][0] = 1.0; aff[1][0][0] = 1.0; aff[1][1][0] = std::sqrt(0.5);
  aff[2][0][0] = 1.5; aff[2][1][0] = std::sqrt(1.5); aff[2][2][0] = std::sqrt(0.375);
  aff[2][0][2] = -0.5;
  double bi[13][13];
  for (int i = 0; i < 13; ++i)
    for (int j = 0; j < 13; ++j) bi[i][j] = 0.0;
  for (int i = 0; i < 13; ++i) { bi[i][0] = 1.0; bi[i][i] = 1.0; }
  for (int i = 0; i < 12; ++i)
    for (int j = 1; j <= i; ++j) bi[i + 1][j] = bi[i][j] + bi[i][j - 1];
  const double p = (ua + ub) * r * 0.5, b = (ua - ub) * r * 0.5;
  const double quo = 1.0 / p;
  double af[20];
  af[0] = quo * std::exp(-p);
  for (int n = 1; n < 20; ++n) af[n] = n * quo * af[n - 1] + af[0];
  double bf[13];
  mopBfn(b, bf);
  double s = 0.0;
  const int lam1 = la - m, lbm1 = lb - m;
  for (int i = 0; i <= lam1; i += 2) {
    const int ia = na + i - la, ic = la - i - m;
    for (int j = 0; j <= lbm1; j += 2) {
      const int ib = nb + j - lb, id = lb - j - m;
      double s1 = 0.0;
      const int iab = ia + ib;
      for (int k1 = 0; k1 <= ia; ++k1)
        for (int k2 = 0; k2 <= ib; ++k2)
          for (int k3 = 0; k3 <= ic; ++k3)
            for (int k4 = 0; k4 <= id; ++k4)
              for (int k5 = 0; k5 <= m; ++k5) {
                const int iaf = iab - k1 - k2 + k3 + k4 + 2 * k5;
                for (int k6 = 0; k6 <= m; ++k6) {
                  const int ibf = k1 + k2 + k3 + k4 + 2 * k6;
                  s1 += bi[id][k4] * bi[ic][k3] * bi[ib][k2] * bi[ia][k1] * bi[m][k5] * bi[m][k6]
                        * (1 - 2 * ((m + k2 + k4 + k5 + k6) % 2)) * af[iaf] * bf[ibf];
                }
              }
      s += s1 * aff[la][m][i] * aff[lb][m][j];
    }
  }
  return s * std::pow(r, na + nb + 1) * std::pow(ua, na) * std::pow(ub, nb) / 2.0
         * std::sqrt(ua * ub / (mopFact(na + na) * mopFact(nb + nb))
                     * ((la + la + 1) * (lb + lb + 1)));
}

// Rotation coefficients to the molecular frame (MOPAC coe.F90). c is a flat 1-based
// array of length 76 (c[1..75]); cc(c,i,k,m) = c[i + 3*(k-1) + 15*(m-1)].
NVMOLKIT_HD inline void mopCoe(double x2, double y2, double z2, int nij, double* c) {
  const double rt34 = 0.86602540378444, rt13 = 0.57735026918963;
  double xy = x2 * x2 + y2 * y2;
  const double r = std::sqrt(xy + z2 * z2);
  xy = std::sqrt(xy);
  double ca, cb, sa, sb;
  if (xy >= 1e-10) { ca = x2 / xy; cb = z2 / r; sa = y2 / xy; sb = xy / r; }
  else if (z2 < 0.0) { ca = -1.0; cb = -1.0; sa = 0.0; sb = 0.0; }
  else if (z2 == 0.0) { ca = 0.0; cb = 0.0; sa = 0.0; sb = 0.0; }
  else { ca = 1.0; cb = 1.0; sa = 0.0; sb = 0.0; }
  for (int i = 0; i < 76; ++i) c[i] = 0.0;
  c[37] = 1.0;
  if (nij >= 2) {
    c[56] = ca * cb; c[41] = ca * sb; c[26] = -sa; c[53] = -sb; c[38] = cb; c[23] = 0.0;
    c[50] = sa * cb; c[35] = sa * sb; c[20] = ca;
    if (nij >= 5) {
      const double c2a = 2 * ca * ca - 1.0, c2b = 2 * cb * cb - 1.0, s2a = 2 * sa * ca,
                   s2b = 2 * sb * cb;
      c[75] = c2a * cb * cb + 0.5 * c2a * sb * sb; c[60] = 0.5 * c2a * s2b;
      c[45] = rt34 * c2a * sb * sb; c[30] = -s2a * sb; c[15] = -s2a * cb;
      c[72] = -0.5 * ca * s2b; c[57] = ca * c2b; c[42] = rt34 * ca * s2b; c[27] = -sa * cb;
      c[12] = sa * sb; c[69] = rt13 * sb * sb * 1.5; c[54] = -rt34 * s2b;
      c[39] = cb * cb - 0.5 * sb * sb; c[66] = -0.5 * sa * s2b; c[51] = sa * c2b;
      c[36] = rt34 * sa * s2b; c[21] = ca * cb; c[6] = -ca * sb;
      c[63] = s2a * cb * cb + 0.5 * s2a * sb * sb; c[48] = 0.5 * s2a * s2b;
      c[33] = rt34 * s2a * sb * sb; c[18] = c2a * sb; c[3] = c2a * cb;
    }
  }
}

NVMOLKIT_HD inline double mopCc(const double* c, int i, int k, int m) {
  return c[i + 3 * (k - 1) + 15 * (m - 1)];
}

// ival(i,k) 1-based (MOPAC diat data): di orbital index for (l=i, m-component=k);
// 0 = no orbital. Stored row-major [i-1][k-1].
NVMOLKIT_HD inline int mopIval(int i, int k) {
  static const int kIval[3][5] = {{1, 1, 1, 1, 0}, {0, 3, 4, 2, 0}, {9, 8, 7, 6, 5}};
  return kIval[i - 1][k - 1];
}

// Full diatomic overlap block di[9*9] (row-major, engine/MOPAC orbital order)
// between atom A (natA orbitals, principal qn nA, exponents zs/zp/zd) and atom B,
// with B at displacement xj relative to A. Mirrors MOPAC diat.F90 exactly.
NVMOLKIT_HD inline void mopDiat(int nA, double zsA, double zpA, double zdA, int natA, int nB,
                                double zsB, double zpB, double zdB, int natB, const double xj[3],
                                double* di) {
  for (int i = 0; i < 81; ++i) di[i] = 0.0;
  const double x2 = xj[0], y2 = xj[1], z2 = xj[2];
  const double r = std::sqrt(x2 * x2 + y2 * y2 + z2 * z2);
  if (r < 1e-3) return;
  double c[76];
  mopCoe(x2, y2, z2, (natA > natB ? natA : natB), c);
  const int iaN = natA >= 5 ? 3 : (natA >= 2 ? 2 : 1);
  const int ibN = natB >= 5 ? 3 : (natB >= 2 ? 2 : 1);
  const double ulA[3] = {zsA, zpA, (zdA > 0.3 ? zdA : 0.3)};
  const double ulB[3] = {zsB, zpB, (zdB > 0.3 ? zdB : 0.3)};
  const int nk1 = (iaN - 1 < ibN - 1 ? iaN - 1 : ibN - 1) + 1;
  double s[4][4][4];
  for (int i = 0; i < 64; ++i) (&s[0][0][0])[i] = 0.0;
  for (int i = 1; i <= iaN; ++i)
    for (int j = 1; j <= ibN; ++j)
      for (int k = 1; k <= nk1; ++k) {
        if (k > i || k > j) continue;
        const int pi = (nA > i ? nA : i), pj = (nB > j ? nB : j);
        s[i][j][k] = mopSs(pi, pj, i, j, k, ulA[i - 1], ulB[j - 1], r);
      }
  for (int i = 1; i <= iaN; ++i) {
    const int kmin = 4 - i, kmax = 2 + i;
    for (int j = 1; j <= ibN; ++j) {
      double aa, bb;
      if (j == 2) { aa = -1.0; bb = 1.0; }
      else { aa = 1.0; bb = (j == 3 ? -1.0 : 1.0); }
      const int lmin = 4 - j, lmax = 2 + j;
      for (int k = kmin; k <= kmax; ++k)
        for (int l = lmin; l <= lmax; ++l) {
          const int ii = mopIval(i, k), jj = mopIval(j, l);
          if (ii == 0 || jj == 0) continue;
          di[(ii - 1) * 9 + (jj - 1)] =
              s[i][j][1] * (mopCc(c, i, k, 3) * mopCc(c, j, l, 3)) * aa
              + s[i][j][2] * (mopCc(c, i, k, 4) * mopCc(c, j, l, 4)
                              + mopCc(c, i, k, 2) * mopCc(c, j, l, 2)) * bb
              + s[i][j][3] * (mopCc(c, i, k, 5) * mopCc(c, j, l, 5)
                              + mopCc(c, i, k, 1) * mopCc(c, j, l, 1));
        }
    }
  }
}

}  // namespace mopacovl
}  // namespace semiempirical
}  // namespace nvMolKit

#endif  // NVMOLKIT_SEMIEMPIRICAL_OVERLAP_MOPAC_DEVICE_H
