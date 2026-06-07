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
// Device-callable NDDO/PM6_D Fock build for the YH scope (one d-atom + H's):
// F = H + G(P) with the one-center sp Slater-Condon block, the one-center d
// two-electron block (baked W integrals + packing maps, onecenter_d_data.h), and
// the two-center d Coulomb/exchange for each d-atom..H pair (the YH 9x9 W: J on
// both atoms + the cross-atom K term). Mirrors tools/semiempirical/
// validate_pm6d.py exactly; same __host__ __device__ code runs on CPU and GPU.

#ifndef NVMOLKIT_SEMIEMPIRICAL_FOCK_D_DEVICE_H
#define NVMOLKIT_SEMIEMPIRICAL_FOCK_D_DEVICE_H

#include "core_hamiltonian_d_device.h"  // spCapped
#include "device_macros.h"
#include "onecenter_d_data.h"
#include "two_center_d_device.h"   // yhWMolecular
#include "two_center_device.h"     // twoCenterMolecularDev, wIdxDev
#include "two_center_yx_device.h"  // yxWMolecular

namespace nvMolKit {
namespace semiempirical {

// One-center sp Slater-Condon contribution for atom a (first min(norb,4) orbs).
NVMOLKIT_HD inline void fockOneCenterSp(int nBasis, const AtomIntParams& p, int s, int norb,
                                        const double* P, double* F) {
  const double Pss = P[s * nBasis + s];
  if (norb == 1) {
    F[s * nBasis + s] += Pss * p.gss * 0.5;
    return;
  }
  const int pk0 = s + 1, pk1 = s + 2, pk2 = s + 3;
  const double Ppp = P[pk0 * nBasis + pk0] + P[pk1 * nBasis + pk1] + P[pk2 * nBasis + pk2];
  const double sp1 = p.gsp - 0.5 * p.hsp;
  const double sp2 = 1.5 * p.hsp - 0.5 * p.gsp;
  const double ppd = 1.25 * p.gp2 - 0.25 * p.gpp;
  const double ppoff = 0.75 * p.gpp - 1.25 * p.gp2;
  F[s * nBasis + s] += Pss * p.gss * 0.5 + Ppp * sp1;
  for (int k = 1; k <= 3; ++k) {
    const int pk = s + k;
    F[pk * nBasis + pk] += Pss * sp1 + P[pk * nBasis + pk] * p.gpp * 0.5
                           + (Ppp - P[pk * nBasis + pk]) * ppd;
    F[s * nBasis + pk] += P[s * nBasis + pk] * sp2;
    F[pk * nBasis + s] += P[pk * nBasis + s] * sp2;
  }
  for (int k = 1; k <= 3; ++k)
    for (int l = k + 1; l <= 3; ++l) {
      const int pk = s + k, pl = s + l;
      F[pk * nBasis + pl] += P[pk * nBasis + pl] * ppoff;
      F[pl * nBasis + pk] += P[pl * nBasis + pk] * ppoff;
    }
}

// One-center d two-electron contribution for a 9-orbital atom a at block start s,
// from the baked W integrals (243) + packing maps. Returns false if z has no
// baked d integrals.
NVMOLKIT_HD inline bool fockOneCenterD(int nBasis, int z, int s, const double* P, double* F) {
  using namespace onecenterd;
  const double* W = oneCenterDW(z);
  if (W == nullptr) return false;
  double Pp[kNTril];
  for (int k = 0; k < kNTril; ++k)
    Pp[k] = P[(s + kTrilI[k]) * nBasis + (s + kTrilJ[k])] * kWeight45[k];
  for (int k = 0; k < kNTril; ++k) {
    double fp = 0.0;
    for (int t = kFlOff[k]; t < kFlOff[k + 1]; ++t) fp += W[kFlWi[t]] * Pp[kFlPi[t]];
    const int i2 = kTrilI[kFlCol[k]], j2 = kTrilJ[kFlCol[k]];
    F[(s + i2) * nBasis + (s + j2)] += fp;
    if (i2 != j2) F[(s + j2) * nBasis + (s + i2)] += fp;
  }
  return true;
}

// Build the PM6_D Fock matrix for one YH-scope molecule.
NVMOLKIT_HD inline void buildFockDDev(int nBasis, int nAtoms, const AtomIntParams* ap,
                                      const int* start, const int* norb, const double* coords,
                                      const double* H, const double* P, double* F) {
  for (int t = 0; t < nBasis * nBasis; ++t) F[t] = H[t];

  // One-center: sp Slater-Condon, plus the d block for 9-orbital atoms.
  for (int a = 0; a < nAtoms; ++a) {
    const int s = start[a], spn = (norb[a] >= 4) ? 4 : norb[a];
    fockOneCenterSp(nBasis, ap[a], s, spn, P, F);
    if (norb[a] == 9) fockOneCenterD(nBasis, ap[a].z, s, P, F);
  }

  // Two-center Coulomb/exchange.
  for (int i = 0; i < nAtoms; ++i) {
    for (int j = i + 1; j < nAtoms; ++j) {
      int dA = -1, hB = -1;
      if (norb[i] == 9 && norb[j] == 1) { dA = i; hB = j; }
      else if (norb[i] == 1 && norb[j] == 9) { dA = j; hB = i; }
      int yxD = -1, yxS = -1;
      if (norb[i] == 9 && norb[j] == 4) { yxD = i; yxS = j; }
      else if (norb[i] == 4 && norb[j] == 9) { yxD = j; yxS = i; }

      if (norb[i] == 9 && norb[j] == 9) {  // YY d two-center (both d): full 9x9 J/K.
        double W[9 * 9 * 9 * 9];
        yyWMolecular(ap[i], &coords[3 * i], ap[j], &coords[3 * j], W);
        const int sA = start[i], sB = start[j];
        auto Wd = [&](int mu, int nu, int lam, int sig) {
          return W[((mu * 9 + nu) * 9 + lam) * 9 + sig];
        };
        for (int mu = 0; mu < 9; ++mu)
          for (int nu = 0; nu < 9; ++nu) {
            double acc = 0.0;
            for (int lam = 0; lam < 9; ++lam)
              for (int sig = 0; sig < 9; ++sig)
                acc += P[(sB + lam) * nBasis + (sB + sig)] * Wd(mu, nu, lam, sig);
            F[(sA + mu) * nBasis + (sA + nu)] += acc;
          }
        for (int lam = 0; lam < 9; ++lam)
          for (int sig = 0; sig < 9; ++sig) {
            double acc = 0.0;
            for (int mu = 0; mu < 9; ++mu)
              for (int nu = 0; nu < 9; ++nu)
                acc += P[(sA + mu) * nBasis + (sA + nu)] * Wd(mu, nu, lam, sig);
            F[(sB + lam) * nBasis + (sB + sig)] += acc;
          }
        for (int mu = 0; mu < 9; ++mu)
          for (int lam = 0; lam < 9; ++lam) {
            double acc = 0.0;
            for (int nu = 0; nu < 9; ++nu)
              for (int sig = 0; sig < 9; ++sig)
                acc += Wd(mu, nu, lam, sig) * P[(sA + nu) * nBasis + (sB + sig)];
            acc *= -0.5;
            F[(sA + mu) * nBasis + (sB + lam)] += acc;
            F[(sB + lam) * nBasis + (sA + mu)] += acc;
          }
      } else if (yxD >= 0) {  // YX d two-center (d-atom + sp atom): full 9x4 J/K.
        // yxW carries the molecular sp block too, so the full contraction is the
        // complete sp + d two-center (no separate sp pass for this pair).
        double W[9 * 9 * 4 * 4];
        yxWMolecular(ap[yxD], &coords[3 * yxD], ap[yxS], &coords[3 * yxS], W);
        const int sA = start[yxD], sB = start[yxS];
        auto Wd = [&](int mu, int nu, int lam, int sig) {
          return W[((mu * 9 + nu) * 4 + lam) * 4 + sig];
        };
        for (int mu = 0; mu < 9; ++mu)
          for (int nu = 0; nu < 9; ++nu) {
            double acc = 0.0;
            for (int lam = 0; lam < 4; ++lam)
              for (int sig = 0; sig < 4; ++sig)
                acc += P[(sB + lam) * nBasis + (sB + sig)] * Wd(mu, nu, lam, sig);
            F[(sA + mu) * nBasis + (sA + nu)] += acc;
          }
        for (int lam = 0; lam < 4; ++lam)
          for (int sig = 0; sig < 4; ++sig) {
            double acc = 0.0;
            for (int mu = 0; mu < 9; ++mu)
              for (int nu = 0; nu < 9; ++nu)
                acc += P[(sA + mu) * nBasis + (sA + nu)] * Wd(mu, nu, lam, sig);
            F[(sB + lam) * nBasis + (sB + sig)] += acc;
          }
        for (int mu = 0; mu < 9; ++mu)
          for (int lam = 0; lam < 4; ++lam) {
            double acc = 0.0;
            for (int nu = 0; nu < 9; ++nu)
              for (int sig = 0; sig < 4; ++sig)
                acc += Wd(mu, nu, lam, sig) * P[(sA + nu) * nBasis + (sB + sig)];
            acc *= -0.5;
            F[(sA + mu) * nBasis + (sB + lam)] += acc;
            F[(sB + lam) * nBasis + (sA + mu)] += acc;
          }
      } else if (dA >= 0) {  // YH d two-center: J on both atoms + cross K
        double W[81];
        // Skip if zA's d charge separations aren't baked: yhWMolecular leaves W
        // uninitialized on false, which would poison the Fock with NaN.
        if (!yhWMolecular(ap[dA], &coords[3 * dA], ap[hB], &coords[3 * hB], W)) continue;
        const int sA = start[dA], sB = start[hB];
        const double Phh = P[sB * nBasis + sB];
        double sumP = 0.0;
        for (int mu = 0; mu < 9; ++mu)
          for (int nu = 0; nu < 9; ++nu) {
            F[(sA + mu) * nBasis + (sA + nu)] += Phh * W[mu * 9 + nu];
            sumP += P[(sA + mu) * nBasis + (sA + nu)] * W[mu * 9 + nu];
          }
        F[sB * nBasis + sB] += sumP;
        for (int mu = 0; mu < 9; ++mu) {
          double ks = 0.0;
          for (int nu = 0; nu < 9; ++nu) ks += W[mu * 9 + nu] * P[(sA + nu) * nBasis + sB];
          ks *= -0.5;
          F[(sA + mu) * nBasis + sB] += ks;
          F[sB * nBasis + (sA + mu)] += ks;
        }
      } else {  // sp..sp two-center (H-H in the YH scope): full sp w tensor
        const AtomIntParams pi = spCapped(ap[i]);
        const AtomIntParams pj = spCapped(ap[j]);
        double w[256], e1b[16], e2a[16];
        twoCenterMolecularDev(pi, &coords[3 * i], pj, &coords[3 * j], w, e1b, e2a);
        const int nA = pi.nOrb, nB = pj.nOrb, sA = start[i], sB = start[j];
        for (int mu = 0; mu < nA; ++mu)
          for (int nu = 0; nu < nA; ++nu) {
            double acc = 0.0;
            for (int lam = 0; lam < nB; ++lam)
              for (int sig = 0; sig < nB; ++sig)
                acc += P[(sB + lam) * nBasis + (sB + sig)] * w[detail::wIdxDev(mu, nu, lam, sig)];
            F[(sA + mu) * nBasis + (sA + nu)] += acc;
          }
        for (int lam = 0; lam < nB; ++lam)
          for (int sig = 0; sig < nB; ++sig) {
            double acc = 0.0;
            for (int mu = 0; mu < nA; ++mu)
              for (int nu = 0; nu < nA; ++nu)
                acc += P[(sA + mu) * nBasis + (sA + nu)] * w[detail::wIdxDev(mu, nu, lam, sig)];
            F[(sB + lam) * nBasis + (sB + sig)] += acc;
          }
        for (int mu = 0; mu < nA; ++mu)
          for (int lam = 0; lam < nB; ++lam) {
            double acc = 0.0;
            for (int nu = 0; nu < nA; ++nu)
              for (int sig = 0; sig < nB; ++sig)
                acc += w[detail::wIdxDev(mu, nu, lam, sig)] * P[(sA + nu) * nBasis + (sB + sig)];
            acc *= -0.5;
            F[(sA + mu) * nBasis + (sB + lam)] += acc;
            F[(sB + lam) * nBasis + (sA + mu)] += acc;
          }
      }
    }
  }
}

// ===========================================================================
// Integral-cache split of buildFockDDev.
//
// The two-center integral tensors (sp-sp w/e1b/e2a, YH 9x9 W, YX 9x9x4x4 W,
// YY 9x9x9x9 W) depend ONLY on geometry + params, NOT on the density P, so they
// are identical across all SCF iterations. precomputeTwoCenterDDev() computes
// them ONCE per molecule into a packed cache; buildFockDDevCached() then reads
// the cache and contracts with the current P. This is a pure arithmetic hoist:
// the contraction below is byte-for-byte the same code as buildFockDDev's
// branches, just sourcing W from the cache instead of recomputing it.
//
// Cache layout (per molecule):
//   - meta[]   : kPairMetaInts ints per unordered pair (i<j), in the same (i,j)
//                double-loop order as buildFockDDev. Fields:
//                  [0] kind  (0 none/skip, 1 YY, 2 YX, 3 YH, 4 spsp)
//                  [1] a-atom index (d-atom for YX/YH; atom i for YY/spsp)
//                  [2] b-atom index (sp/H atom for YX/YH; atom j for YY/spsp)
//                  [3] nA (spsp capped orbs of a) -- spsp only
//                  [4] nB (spsp capped orbs of b) -- spsp only
//                  [5] doff  (offset into the doubles blob for this pair's W)
//   - blob[]   : packed doubles; each pair consumes only its kind's tensor size
//                (spsp 256, YH 81, YX 1296, YY 6561; 0 for skipped pairs).
// ===========================================================================

enum {
  kPmPairNone = 0,
  kPmPairYY = 1,
  kPmPairYX = 2,
  kPmPairYH = 3,
  kPmPairSpsp = 4,
  kPairMetaInts = 6,
};

// Doubles consumed in the cache blob by a pair of the given kind.
NVMOLKIT_HD inline int pm6dPairBlobSize(int kind) {
  switch (kind) {
    case kPmPairYY: return 9 * 9 * 9 * 9;  // 6561
    case kPmPairYX: return 9 * 9 * 4 * 4;  // 1296
    case kPmPairYH: return 81;
    case kPmPairSpsp: return 256;
    default: return 0;
  }
}

// Classify an unordered atom pair (i<j) into its Fock two-center kind, without
// computing any integrals. Mirrors the branch selection in buildFockDDev.
NVMOLKIT_HD inline int pm6dPairKind(int norbi, int norbj) {
  if (norbi == 9 && norbj == 9) return kPmPairYY;
  if ((norbi == 9 && norbj == 4) || (norbi == 4 && norbj == 9)) return kPmPairYX;
  if ((norbi == 9 && norbj == 1) || (norbi == 1 && norbj == 9)) return kPmPairYH;
  return kPmPairSpsp;
}

// Upper bound on the cache blob doubles for one molecule (every pair sized for
// its kind, before the YH-skip is known). Used for scratch sizing; the actual
// fill may consume fewer doubles when a YH pair is skipped.
NVMOLKIT_HD inline long pm6dIntCacheDoubles(int nAtoms, const int* norb) {
  long tot = 0;
  for (int i = 0; i < nAtoms; ++i)
    for (int j = i + 1; j < nAtoms; ++j)
      tot += pm6dPairBlobSize(pm6dPairKind(norb[i], norb[j]));
  return tot;
}

// Fill the integral cache (meta + blob) for one molecule. Walks the (i<j) pairs
// in the SAME order as buildFockDDev, assigns each a blob offset, and computes
// its W tensor once. A YH pair whose d-charge separations aren't baked is marked
// kPmPairNone (skipped in the contraction, exactly as the original `continue`).
// `ywScr` (optional, >=kYWScrDoubles doubles) is reused across pairs to hold the
// large YY/YX 45x45 temporaries off-stack -- only one pair is processed at a time
// here, so a single buffer suffices. Null keeps them on the stack (CPU reference).
NVMOLKIT_HD inline void precomputeTwoCenterDDev(int nAtoms, const AtomIntParams* ap,
                                                const int* start, const int* norb,
                                                const double* coords, int* meta, double* blob,
                                                double* ywScr = nullptr) {
  (void)start;
  int pid = 0;
  long doff = 0;
  for (int i = 0; i < nAtoms; ++i) {
    for (int j = i + 1; j < nAtoms; ++j, ++pid) {
      int* md = &meta[pid * kPairMetaInts];
      const int kind = pm6dPairKind(norb[i], norb[j]);
      md[1] = i; md[2] = j; md[3] = 0; md[4] = 0; md[5] = static_cast<int>(doff);

      if (kind == kPmPairYY) {
        yyWMolecular(ap[i], &coords[3 * i], ap[j], &coords[3 * j], &blob[doff], ywScr);
        md[0] = kPmPairYY;
        doff += 9 * 9 * 9 * 9;
      } else if (kind == kPmPairYX) {
        const int yxD = (norb[i] == 9) ? i : j;
        const int yxS = (norb[i] == 9) ? j : i;
        yxWMolecular(ap[yxD], &coords[3 * yxD], ap[yxS], &coords[3 * yxS], &blob[doff], ywScr);
        md[0] = kPmPairYX;
        md[1] = yxD; md[2] = yxS;
        doff += 9 * 9 * 4 * 4;
      } else if (kind == kPmPairYH) {
        const int dA = (norb[i] == 9) ? i : j;
        const int hB = (norb[i] == 9) ? j : i;
        if (yhWMolecular(ap[dA], &coords[3 * dA], ap[hB], &coords[3 * hB], &blob[doff])) {
          md[0] = kPmPairYH;
          md[1] = dA; md[2] = hB;
          doff += 81;
        } else {
          md[0] = kPmPairNone;  // skipped: consumes no blob bytes
        }
      } else {  // sp..sp
        const AtomIntParams pi = spCapped(ap[i]);
        const AtomIntParams pj = spCapped(ap[j]);
        double e1b[16], e2a[16];
        twoCenterMolecularDev(pi, &coords[3 * i], pj, &coords[3 * j], &blob[doff], e1b, e2a);
        md[0] = kPmPairSpsp;
        md[3] = pi.nOrb; md[4] = pj.nOrb;
        doff += 256;
      }
    }
  }
}

// Build the PM6_D Fock matrix from a precomputed integral cache. The one-center
// blocks (density-dependent) are recomputed here; the two-center contraction
// reads each pair's W from the cache. Byte-identical to buildFockDDev given the
// same cache contents.
NVMOLKIT_HD inline void buildFockDDevCached(int nBasis, int nAtoms, const AtomIntParams* ap,
                                            const int* start, const int* norb,
                                            const int* meta, const double* blob,
                                            const double* H, const double* P, double* F) {
  for (int t = 0; t < nBasis * nBasis; ++t) F[t] = H[t];

  // One-center: sp Slater-Condon, plus the d block for 9-orbital atoms.
  for (int a = 0; a < nAtoms; ++a) {
    const int s = start[a], spn = (norb[a] >= 4) ? 4 : norb[a];
    fockOneCenterSp(nBasis, ap[a], s, spn, P, F);
    if (norb[a] == 9) fockOneCenterD(nBasis, ap[a].z, s, P, F);
  }

  // Two-center Coulomb/exchange, contracting the cached W with P.
  int pid = 0;
  for (int i = 0; i < nAtoms; ++i) {
    for (int j = i + 1; j < nAtoms; ++j, ++pid) {
      const int* md = &meta[pid * kPairMetaInts];
      const int kind = md[0];
      const double* W = &blob[md[5]];

      if (kind == kPmPairYY) {
        const int sA = start[md[1]], sB = start[md[2]];
        auto Wd = [&](int mu, int nu, int lam, int sig) {
          return W[((mu * 9 + nu) * 9 + lam) * 9 + sig];
        };
        for (int mu = 0; mu < 9; ++mu)
          for (int nu = 0; nu < 9; ++nu) {
            double acc = 0.0;
            for (int lam = 0; lam < 9; ++lam)
              for (int sig = 0; sig < 9; ++sig)
                acc += P[(sB + lam) * nBasis + (sB + sig)] * Wd(mu, nu, lam, sig);
            F[(sA + mu) * nBasis + (sA + nu)] += acc;
          }
        for (int lam = 0; lam < 9; ++lam)
          for (int sig = 0; sig < 9; ++sig) {
            double acc = 0.0;
            for (int mu = 0; mu < 9; ++mu)
              for (int nu = 0; nu < 9; ++nu)
                acc += P[(sA + mu) * nBasis + (sA + nu)] * Wd(mu, nu, lam, sig);
            F[(sB + lam) * nBasis + (sB + sig)] += acc;
          }
        for (int mu = 0; mu < 9; ++mu)
          for (int lam = 0; lam < 9; ++lam) {
            double acc = 0.0;
            for (int nu = 0; nu < 9; ++nu)
              for (int sig = 0; sig < 9; ++sig)
                acc += Wd(mu, nu, lam, sig) * P[(sA + nu) * nBasis + (sB + sig)];
            acc *= -0.5;
            F[(sA + mu) * nBasis + (sB + lam)] += acc;
            F[(sB + lam) * nBasis + (sA + mu)] += acc;
          }
      } else if (kind == kPmPairYX) {
        const int sA = start[md[1]], sB = start[md[2]];
        auto Wd = [&](int mu, int nu, int lam, int sig) {
          return W[((mu * 9 + nu) * 4 + lam) * 4 + sig];
        };
        for (int mu = 0; mu < 9; ++mu)
          for (int nu = 0; nu < 9; ++nu) {
            double acc = 0.0;
            for (int lam = 0; lam < 4; ++lam)
              for (int sig = 0; sig < 4; ++sig)
                acc += P[(sB + lam) * nBasis + (sB + sig)] * Wd(mu, nu, lam, sig);
            F[(sA + mu) * nBasis + (sA + nu)] += acc;
          }
        for (int lam = 0; lam < 4; ++lam)
          for (int sig = 0; sig < 4; ++sig) {
            double acc = 0.0;
            for (int mu = 0; mu < 9; ++mu)
              for (int nu = 0; nu < 9; ++nu)
                acc += P[(sA + mu) * nBasis + (sA + nu)] * Wd(mu, nu, lam, sig);
            F[(sB + lam) * nBasis + (sB + sig)] += acc;
          }
        for (int mu = 0; mu < 9; ++mu)
          for (int lam = 0; lam < 4; ++lam) {
            double acc = 0.0;
            for (int nu = 0; nu < 9; ++nu)
              for (int sig = 0; sig < 4; ++sig)
                acc += Wd(mu, nu, lam, sig) * P[(sA + nu) * nBasis + (sB + sig)];
            acc *= -0.5;
            F[(sA + mu) * nBasis + (sB + lam)] += acc;
            F[(sB + lam) * nBasis + (sA + mu)] += acc;
          }
      } else if (kind == kPmPairYH) {
        const int sA = start[md[1]], sB = start[md[2]];
        const double Phh = P[sB * nBasis + sB];
        double sumP = 0.0;
        for (int mu = 0; mu < 9; ++mu)
          for (int nu = 0; nu < 9; ++nu) {
            F[(sA + mu) * nBasis + (sA + nu)] += Phh * W[mu * 9 + nu];
            sumP += P[(sA + mu) * nBasis + (sA + nu)] * W[mu * 9 + nu];
          }
        F[sB * nBasis + sB] += sumP;
        for (int mu = 0; mu < 9; ++mu) {
          double ks = 0.0;
          for (int nu = 0; nu < 9; ++nu) ks += W[mu * 9 + nu] * P[(sA + nu) * nBasis + sB];
          ks *= -0.5;
          F[(sA + mu) * nBasis + sB] += ks;
          F[sB * nBasis + (sA + mu)] += ks;
        }
      } else if (kind == kPmPairSpsp) {
        const int nA = md[3], nB = md[4], sA = start[i], sB = start[j];
        for (int mu = 0; mu < nA; ++mu)
          for (int nu = 0; nu < nA; ++nu) {
            double acc = 0.0;
            for (int lam = 0; lam < nB; ++lam)
              for (int sig = 0; sig < nB; ++sig)
                acc += P[(sB + lam) * nBasis + (sB + sig)] * W[detail::wIdxDev(mu, nu, lam, sig)];
            F[(sA + mu) * nBasis + (sA + nu)] += acc;
          }
        for (int lam = 0; lam < nB; ++lam)
          for (int sig = 0; sig < nB; ++sig) {
            double acc = 0.0;
            for (int mu = 0; mu < nA; ++mu)
              for (int nu = 0; nu < nA; ++nu)
                acc += P[(sA + mu) * nBasis + (sA + nu)] * W[detail::wIdxDev(mu, nu, lam, sig)];
            F[(sB + lam) * nBasis + (sB + sig)] += acc;
          }
        for (int mu = 0; mu < nA; ++mu)
          for (int lam = 0; lam < nB; ++lam) {
            double acc = 0.0;
            for (int nu = 0; nu < nA; ++nu)
              for (int sig = 0; sig < nB; ++sig)
                acc += W[detail::wIdxDev(mu, nu, lam, sig)] * P[(sA + nu) * nBasis + (sB + sig)];
            acc *= -0.5;
            F[(sA + mu) * nBasis + (sB + lam)] += acc;
            F[(sB + lam) * nBasis + (sA + mu)] += acc;
          }
      }
      // kPmPairNone: skipped (matches the original yhWMolecular==false `continue`).
    }
  }
}

}  // namespace semiempirical
}  // namespace nvMolKit

#endif  // NVMOLKIT_SEMIEMPIRICAL_FOCK_D_DEVICE_H
