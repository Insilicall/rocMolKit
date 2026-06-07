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
// Open-shell (UHF) PM6_D Fock build. Mirrors fock_d_device.h branch-for-branch,
// but every two-electron term splits into a pure Coulomb J (from the TOTAL
// density Pt = Pa + Pb) and a pure exchange K (from the same-spin density Ps):
//   F^sigma = H + J(Pt) - K(Ps).
// The closed-shell Fock bakes a factor 1/2 into each term; here the cross-atom
// exchange uses factor -1.0 (vs the closed-shell -0.5), and the one-center d
// block uses the J/K-split W integrals (W_J Coulomb * Pt, 2*W_Kfold exchange
// * Ps -- onecenter_d_data.h). Closed shell (Ps = Pt/2) reproduces fock_d_device
// exactly. Same __host__ __device__ code runs on CPU and (future) GPU.

#ifndef NVMOLKIT_SEMIEMPIRICAL_FOCK_D_UHF_DEVICE_H
#define NVMOLKIT_SEMIEMPIRICAL_FOCK_D_UHF_DEVICE_H

#include "fock_d_device.h"  // spCapped, two-center d machinery, onecenter_d_data

namespace nvMolKit {
namespace semiempirical {

// One-center sp two-electron contribution for spin sigma: J(Pt) - K(Ps). The
// closed-shell fockOneCenterSp folds 1/2 into each term; this splits them.
NVMOLKIT_HD inline void fockOneCenterSpUHF(int nB, const AtomIntParams& p, int s, int norb,
                                           const double* Pt, const double* Ps, double* F) {
  const double Tss = Pt[s * nB + s], Sss = Ps[s * nB + s];
  if (norb == 1) {
    F[s * nB + s] += Tss * p.gss - Sss * p.gss;  // J - K
    return;
  }
  const int p0 = s + 1, p1 = s + 2, p2 = s + 3;
  const double Tpp = Pt[p0 * nB + p0] + Pt[p1 * nB + p1] + Pt[p2 * nB + p2];
  const double Spp = Ps[p0 * nB + p0] + Ps[p1 * nB + p1] + Ps[p2 * nB + p2];
  F[s * nB + s] += (Tss * p.gss + Tpp * p.gsp) - (Sss * p.gss + Spp * p.hsp);
  for (int k = 1; k <= 3; ++k) {
    const int pk = s + k;
    const double Tpk = Pt[pk * nB + pk], Spk = Ps[pk * nB + pk];
    F[pk * nB + pk] += (Tss * p.gsp + Tpk * p.gpp + (Tpp - Tpk) * p.gp2)
                       - (Sss * p.hsp + Spk * p.gpp + (Spp - Spk) * 0.5 * (p.gpp - p.gp2));
    const double off = 2.0 * Pt[s * nB + pk] * p.hsp - Ps[s * nB + pk] * (p.gsp + p.hsp);
    F[s * nB + pk] += off;
    F[pk * nB + s] += off;
  }
  for (int k = 1; k <= 3; ++k)
    for (int l = k + 1; l <= 3; ++l) {
      const int pk = s + k, pl = s + l;
      const double v = Pt[pk * nB + pl] * (p.gpp - p.gp2) - 0.5 * Ps[pk * nB + pl] * (p.gpp + p.gp2);
      F[pk * nB + pl] += v;
      F[pl * nB + pk] += v;
    }
}

// One-center d two-electron contribution for spin sigma, from the J/K-split W:
//   F^sigma += sum W_J * Pp_total  -  2 * sum W_Kfold * Pp_sigma.
// Returns false if z has no baked d split (falls back to RHF-only).
NVMOLKIT_HD inline bool fockOneCenterDUHF(int nB, int z, int s, const double* Pt, const double* Ps,
                                          double* F) {
  using namespace onecenterd;
  const double* WJ = oneCenterDWJ(z);
  const double* WK = oneCenterDWK(z);
  if (WJ == nullptr || WK == nullptr) return false;
  double PtP[kNTril], PsP[kNTril];
  for (int k = 0; k < kNTril; ++k) {
    const int idx = (s + kTrilI[k]) * nB + (s + kTrilJ[k]);
    PtP[k] = Pt[idx] * kWeight45[k];
    PsP[k] = Ps[idx] * kWeight45[k];
  }
  for (int k = 0; k < kNTril; ++k) {
    double fp = 0.0;
    for (int t = kFlOff[k]; t < kFlOff[k + 1]; ++t)
      fp += WJ[kFlWi[t]] * PtP[kFlPi[t]] - 2.0 * WK[kFlWi[t]] * PsP[kFlPi[t]];
    const int i2 = kTrilI[kFlCol[k]], j2 = kTrilJ[kFlCol[k]];
    F[(s + i2) * nB + (s + j2)] += fp;
    if (i2 != j2) F[(s + j2) * nB + (s + i2)] += fp;
  }
  return true;
}

// Generic two-center J(Pt)/K(Ps) for a pair of atoms with a dense w tensor
// w[mu,nu,lam,sig] (na x na x nb x nb), atom blocks at sA/sB. J on both atoms
// from the total density; cross exchange from the same-spin density (factor -1).
NVMOLKIT_HD inline void twoCenterUHFAccum(int nB, int na, int nb, int sA, int sB,
                                          const double* Pt, const double* Ps, double* F,
                                          double (*w)(int, int, int, int, const double*),
                                          const double* wbuf) {
  for (int mu = 0; mu < na; ++mu)
    for (int nu = 0; nu < na; ++nu) {
      double acc = 0.0;
      for (int lam = 0; lam < nb; ++lam)
        for (int sig = 0; sig < nb; ++sig)
          acc += Pt[(sB + lam) * nB + (sB + sig)] * w(mu, nu, lam, sig, wbuf);
      F[(sA + mu) * nB + (sA + nu)] += acc;
    }
  for (int lam = 0; lam < nb; ++lam)
    for (int sig = 0; sig < nb; ++sig) {
      double acc = 0.0;
      for (int mu = 0; mu < na; ++mu)
        for (int nu = 0; nu < na; ++nu)
          acc += Pt[(sA + mu) * nB + (sA + nu)] * w(mu, nu, lam, sig, wbuf);
      F[(sB + lam) * nB + (sB + sig)] += acc;
    }
  for (int mu = 0; mu < na; ++mu)
    for (int lam = 0; lam < nb; ++lam) {
      double acc = 0.0;
      for (int nu = 0; nu < na; ++nu)
        for (int sig = 0; sig < nb; ++sig)
          acc += w(mu, nu, lam, sig, wbuf) * Ps[(sA + nu) * nB + (sB + sig)];
      F[(sA + mu) * nB + (sB + lam)] += -acc;
      F[(sB + lam) * nB + (sA + mu)] += -acc;
    }
}

namespace detail {
NVMOLKIT_HD inline double wYY(int mu, int nu, int lam, int sig, const double* W) {
  return W[((mu * 9 + nu) * 9 + lam) * 9 + sig];
}
NVMOLKIT_HD inline double wYX(int mu, int nu, int lam, int sig, const double* W) {
  return W[((mu * 9 + nu) * 4 + lam) * 4 + sig];
}
NVMOLKIT_HD inline double wSP(int mu, int nu, int lam, int sig, const double* w) {
  return w[detail::wIdxDev(mu, nu, lam, sig)];
}
}  // namespace detail

// Build the open-shell (UHF) PM6_D Fock for spin sigma. Pt = Pa+Pb (total), Ps =
// the same-spin density. Mirrors buildFockDDev exactly, J from Pt / K from Ps.
NVMOLKIT_HD inline void buildFockDUHFDev(int nB, int nAtoms, const AtomIntParams* ap,
                                         const int* start, const int* norb, const double* coords,
                                         const double* H, const double* Pt, const double* Ps,
                                         double* F) {
  for (int t = 0; t < nB * nB; ++t) F[t] = H[t];

  for (int a = 0; a < nAtoms; ++a) {
    const int s = start[a], spn = (norb[a] >= 4) ? 4 : norb[a];
    fockOneCenterSpUHF(nB, ap[a], s, spn, Pt, Ps, F);
    if (norb[a] == 9) fockOneCenterDUHF(nB, ap[a].z, s, Pt, Ps, F);
  }

  for (int i = 0; i < nAtoms; ++i) {
    for (int j = i + 1; j < nAtoms; ++j) {
      int dA = -1, hB = -1;
      if (norb[i] == 9 && norb[j] == 1) { dA = i; hB = j; }
      else if (norb[i] == 1 && norb[j] == 9) { dA = j; hB = i; }
      int yxD = -1, yxS = -1;
      if (norb[i] == 9 && norb[j] == 4) { yxD = i; yxS = j; }
      else if (norb[i] == 4 && norb[j] == 9) { yxD = j; yxS = i; }

      if (norb[i] == 9 && norb[j] == 9) {  // YY: both d
        double W[9 * 9 * 9 * 9];
        yyWMolecular(ap[i], &coords[3 * i], ap[j], &coords[3 * j], W);
        twoCenterUHFAccum(nB, 9, 9, start[i], start[j], Pt, Ps, F, detail::wYY, W);
      } else if (yxD >= 0) {  // YX: d-atom + sp atom (sp block folded into yxW)
        double W[9 * 9 * 4 * 4];
        yxWMolecular(ap[yxD], &coords[3 * yxD], ap[yxS], &coords[3 * yxS], W);
        twoCenterUHFAccum(nB, 9, 4, start[yxD], start[yxS], Pt, Ps, F, detail::wYX, W);
      } else if (dA >= 0) {  // YH: d-atom + H
        double W[81];
        yhWMolecular(ap[dA], &coords[3 * dA], ap[hB], &coords[3 * hB], W);
        const int sA = start[dA], sB = start[hB];
        const double Thh = Pt[sB * nB + sB];
        double sumP = 0.0;
        for (int mu = 0; mu < 9; ++mu)
          for (int nu = 0; nu < 9; ++nu) {
            F[(sA + mu) * nB + (sA + nu)] += Thh * W[mu * 9 + nu];  // J on d-atom
            sumP += Pt[(sA + mu) * nB + (sA + nu)] * W[mu * 9 + nu];
          }
        F[sB * nB + sB] += sumP;  // J on H
        for (int mu = 0; mu < 9; ++mu) {  // cross K (same spin), UHF factor -1
          double ks = 0.0;
          for (int nu = 0; nu < 9; ++nu) ks += W[mu * 9 + nu] * Ps[(sA + nu) * nB + sB];
          F[(sA + mu) * nB + sB] += -ks;
          F[sB * nB + (sA + mu)] += -ks;
        }
      } else {  // sp..sp
        const AtomIntParams pi = spCapped(ap[i]);
        const AtomIntParams pj = spCapped(ap[j]);
        double w[256], e1b[16], e2a[16];
        twoCenterMolecularDev(pi, &coords[3 * i], pj, &coords[3 * j], w, e1b, e2a);
        twoCenterUHFAccum(nB, pi.nOrb, pj.nOrb, start[i], start[j], Pt, Ps, F, detail::wSP, w);
      }
    }
  }
}

}  // namespace semiempirical
}  // namespace nvMolKit

#endif  // NVMOLKIT_SEMIEMPIRICAL_FOCK_D_UHF_DEVICE_H
