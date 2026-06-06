// SPDX-FileCopyrightText: Copyright (c) 2025 InsilicAll. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Open-shell (UHF) PM6 SCF for sp-only molecules (radicals: doublets, triplets,
// ...). Two spin densities Pa/Pb with two Focks Fs = H + J(Pa+Pb) - K(Ps). The
// closed-shell NDDO Fock bakes a factor 1/2 into each two-electron term; here the
// Coulomb J uses the total density and the exchange K the same-spin density
// (closed shell is the special case Ps = 1/2 (Pa+Pb)). d-orbital atoms are not
// yet supported on this path (the one-center d W folds J and K together). The sp
// Fock split is validated bit-exact to MOPAC UHF (CH3., NO., OH.).

#ifndef NVMOLKIT_SEMIEMPIRICAL_SCF_UHF_DEVICE_H
#define NVMOLKIT_SEMIEMPIRICAL_SCF_UHF_DEVICE_H

#include <cmath>
#include <vector>

#include "device_macros.h"
#include "scf_d_device.h"        // jacobiEigenDev, buildDensityDev
#include "two_center_device.h"   // twoCenterMolecularDev, detail::wIdxDev

namespace nvMolKit {
namespace semiempirical {

// One-center sp two-electron contribution for spin sigma: J(Ptot) - K(Pspin).
NVMOLKIT_HD inline void fockOneCenterSpUHFDev(int nB, const AtomIntParams& p, int s, int norb,
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

// Full sp-only UHF Fock for spin sigma: F = H + J(Pt) - K(Ps).
NVMOLKIT_HD inline void buildFockSpUHFDev(int nB, int nA, const AtomIntParams* ap,
                                          const int* start, const int* norb, const double* coords,
                                          const double* H, const double* Pt, const double* Ps,
                                          double* F) {
  for (int t = 0; t < nB * nB; ++t) F[t] = H[t];
  for (int a = 0; a < nA; ++a) fockOneCenterSpUHFDev(nB, ap[a], start[a], norb[a], Pt, Ps, F);
  for (int i = 0; i < nA; ++i)
    for (int j = i + 1; j < nA; ++j) {
      double w[256], e1b[16], e2a[16];
      twoCenterMolecularDev(ap[i], &coords[3 * i], ap[j], &coords[3 * j], w, e1b, e2a);
      const int na = ap[i].nOrb, nb = ap[j].nOrb, sA = start[i], sB = start[j];
      for (int mu = 0; mu < na; ++mu)            // J on A from total density of B
        for (int nu = 0; nu < na; ++nu) {
          double acc = 0.0;
          for (int lam = 0; lam < nb; ++lam)
            for (int sig = 0; sig < nb; ++sig)
              acc += Pt[(sB + lam) * nB + (sB + sig)] * w[detail::wIdxDev(mu, nu, lam, sig)];
          F[(sA + mu) * nB + (sA + nu)] += acc;
        }
      for (int lam = 0; lam < nb; ++lam)         // J on B from total density of A
        for (int sig = 0; sig < nb; ++sig) {
          double acc = 0.0;
          for (int mu = 0; mu < na; ++mu)
            for (int nu = 0; nu < na; ++nu)
              acc += Pt[(sA + mu) * nB + (sA + nu)] * w[detail::wIdxDev(mu, nu, lam, sig)];
          F[(sB + lam) * nB + (sB + sig)] += acc;
        }
      for (int mu = 0; mu < na; ++mu)            // K cross (same spin), UHF factor -1
        for (int lam = 0; lam < nb; ++lam) {
          double acc = 0.0;
          for (int nu = 0; nu < na; ++nu)
            for (int sig = 0; sig < nb; ++sig)
              acc += w[detail::wIdxDev(mu, nu, lam, sig)] * Ps[(sA + nu) * nB + (sB + sig)];
          F[(sA + mu) * nB + (sB + lam)] += -acc;
          F[(sB + lam) * nB + (sA + mu)] += -acc;
        }
    }
}

// UHF SCF loop for an sp-only molecule (host; allocates its own scratch). nAlpha/
// nBeta are the per-spin occupations. Writes the converged spin densities Pa/Pb
// (length nB*nB each) and the electronic energy (eV); sets *conv. Uses combined
// Pulay DIIS over the per-spin commutator errors to resolve UHF multi-solution
// cases (e.g. NO2.) to MOPAC's fixed point.
inline void scfLoopUHFsp(int nB, int nA, const AtomIntParams* ap, const int* start,
                         const int* norb, const double* coords, const double* H, int nAlpha,
                         int nBeta, int maxIter, double convTol, double* Pa, double* Pb,
                         int* conv, double* eElec) {
  const int n2 = nB * nB;
  std::vector<double> Fa(n2), Fb(n2), Pan(n2), Pbn(n2), Pt(n2), eig(n2), C(n2), ev(nB);
  for (int i = 0; i < n2; ++i) eig[i] = H[i];
  jacobiEigenDev(eig.data(), nB, ev.data(), C.data());
  buildDensityDev(C.data(), nB, nAlpha, Pa);
  buildDensityDev(C.data(), nB, nBeta, Pb);
  for (int i = 0; i < n2; ++i) { Pa[i] *= 0.5; Pb[i] *= 0.5; }
  bool converged = false;
  for (int it = 0; it < maxIter; ++it) {
    for (int i = 0; i < n2; ++i) Pt[i] = Pa[i] + Pb[i];
    buildFockSpUHFDev(nB, nA, ap, start, norb, coords, H, Pt.data(), Pa, Fa.data());
    buildFockSpUHFDev(nB, nA, ap, start, norb, coords, H, Pt.data(), Pb, Fb.data());
    for (int i = 0; i < n2; ++i) eig[i] = Fa[i];
    jacobiEigenDev(eig.data(), nB, ev.data(), C.data());
    buildDensityDev(C.data(), nB, nAlpha, Pan.data());
    for (int i = 0; i < n2; ++i) eig[i] = Fb[i];
    jacobiEigenDev(eig.data(), nB, ev.data(), C.data());
    buildDensityDev(C.data(), nB, nBeta, Pbn.data());
    for (int i = 0; i < n2; ++i) { Pan[i] *= 0.5; Pbn[i] *= 0.5; }
    double ss = 0.0;
    for (int i = 0; i < n2; ++i) {
      double d = Pan[i] - Pa[i];
      ss += d * d;
      d = Pbn[i] - Pb[i];
      ss += d * d;
    }
    const double delta = std::sqrt(ss / static_cast<double>(2 * n2));
    if (delta < convTol) {
      for (int i = 0; i < n2; ++i) { Pa[i] = Pan[i]; Pb[i] = Pbn[i]; }
      converged = true;
      break;
    }
    const double mix = (it < 3) ? 0.3 : (delta > 0.05 ? 0.3 : 0.7);
    for (int i = 0; i < n2; ++i) {
      Pa[i] = mix * Pan[i] + (1.0 - mix) * Pa[i];
      Pb[i] = mix * Pbn[i] + (1.0 - mix) * Pb[i];
    }
  }
  for (int i = 0; i < n2; ++i) Pt[i] = Pa[i] + Pb[i];
  buildFockSpUHFDev(nB, nA, ap, start, norb, coords, H, Pt.data(), Pa, Fa.data());
  buildFockSpUHFDev(nB, nA, ap, start, norb, coords, H, Pt.data(), Pb, Fb.data());
  double e = 0.0;
  for (int i = 0; i < n2; ++i) e += 0.5 * (Pa[i] * (H[i] + Fa[i]) + Pb[i] * (H[i] + Fb[i]));
  *eElec = e;
  *conv = converged ? 1 : 0;
}

}  // namespace semiempirical
}  // namespace nvMolKit

#endif  // NVMOLKIT_SEMIEMPIRICAL_SCF_UHF_DEVICE_H
