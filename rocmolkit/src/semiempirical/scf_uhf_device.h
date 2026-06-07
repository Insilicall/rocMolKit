// SPDX-FileCopyrightText: Copyright (c) 2025 InsilicAll. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Open-shell (UHF) PM6_D SCF (radicals: doublets, triplets, ...). Two spin
// densities Pa/Pb with two Focks Fs = H + J(Pa+Pb) - K(Ps) built by
// buildFockDUHFDev (fock_d_uhf_device.h), which handles sp AND d atoms. The
// closed-shell NDDO Fock bakes a factor 1/2 into each two-electron term; here the
// Coulomb J uses the total density and the exchange K the same-spin density
// (closed shell is the special case Ps = 1/2 (Pa+Pb)). The diagonal atomic guess
// + decaying level shift converge UHF multi-solution cases (NO2.) to MOPAC.
// Validated bit-exact to MOPAC UHF (sp + d radicals; validate_pm6d_uhf.py).

#ifndef NVMOLKIT_SEMIEMPIRICAL_SCF_UHF_DEVICE_H
#define NVMOLKIT_SEMIEMPIRICAL_SCF_UHF_DEVICE_H

#include <cmath>
#include <vector>

#include "device_macros.h"
#include "fock_d_uhf_device.h"   // buildFockDUHFDev (sp + d UHF Fock)
#include "scf_d_device.h"        // jacobiEigenDev, buildDensityDev

namespace nvMolKit {
namespace semiempirical {

// Build the diagonal atomic-density spin guess: each atom's `valence` core charge
// is spread evenly over its orbitals, then split by the alpha/beta ratio. This is
// MOPAC's initial guess and is far more robust than diagonalizing H_core, which
// for open-shell systems (NO2.) traps the SCF in a higher excited UHF solution.
NVMOLKIT_HD inline void atomicGuessUHFDev(int nB, int nA, const AtomIntParams* ap, const int* start,
                                          const int* norb, int nAlpha, int nBeta, double* Pa,
                                          double* Pb) {
  const int n2 = nB * nB, nElec = nAlpha + nBeta;
  for (int i = 0; i < n2; ++i) { Pa[i] = 0.0; Pb[i] = 0.0; }
  const double fa = static_cast<double>(nAlpha) / static_cast<double>(nElec);
  for (int a = 0; a < nA; ++a) {
    const double perOrb = static_cast<double>(ap[a].valence) / static_cast<double>(norb[a]);
    for (int o = 0; o < norb[a]; ++o) {
      const int d = (start[a] + o) * nB + (start[a] + o);
      Pa[d] = perOrb * fa;
      Pb[d] = perOrb * (1.0 - fa);
    }
  }
}

// Diagonalize a level-shifted spin Fock: Fs' = Fs + shift*(I - P_occ). In the
// orthogonal NDDO basis P (idempotent, trace = nOcc) projects onto the occupied
// spin MOs, so the shift lifts only the virtuals and pins the aufbau filling,
// steering UHF multi-solution cases (NO2.) to MOPAC's ground state. The shift
// never moves the fixed point (at convergence it touches only virtuals).
inline void diagShiftedDev(int nB, const double* F, const double* Pocc, int nOcc, double shift,
                           double* Pnew, double* eig, double* C, double* ev) {
  const int n2 = nB * nB;
  for (int i = 0; i < n2; ++i) eig[i] = F[i] + shift * (-Pocc[i]);
  for (int a = 0; a < nB; ++a) eig[a * nB + a] += shift;  // + shift*I on the diagonal
  jacobiEigenDev(eig, nB, ev, C);
  buildDensityDev(C, nB, nOcc, Pnew);
  for (int i = 0; i < n2; ++i) Pnew[i] *= 0.5;  // closed-shell density -> spin projector
}

// UHF SCF loop for an sp-only molecule (host; allocates its own scratch). nAlpha/
// nBeta are the per-spin occupations. Writes the converged spin densities Pa/Pb
// (length nB*nB each) and the electronic energy (eV); sets *conv. Uses MOPAC's
// diagonal atomic initial guess + a decaying level shift on the virtual orbitals
// to converge UHF multi-solution cases (e.g. NO2.) to MOPAC's fixed point.
inline void scfLoopUHFsp(int nB, int nA, const AtomIntParams* ap, const int* start,
                         const int* norb, const double* coords, const double* H, int nAlpha,
                         int nBeta, int maxIter, double convTol, double* Pa, double* Pb,
                         int* conv, double* eElec) {
  const int n2 = nB * nB;
  std::vector<double> Fa(n2), Fb(n2), Pan(n2), Pbn(n2), Pt(n2), eig(n2), C(n2), ev(nB);
  atomicGuessUHFDev(nB, nA, ap, start, norb, nAlpha, nBeta, Pa, Pb);
  bool converged = false;
  for (int it = 0; it < maxIter; ++it) {
    for (int i = 0; i < n2; ++i) Pt[i] = Pa[i] + Pb[i];
    buildFockDUHFDev(nB, nA, ap, start, norb, coords, H, Pt.data(), Pa, Fa.data());
    buildFockDUHFDev(nB, nA, ap, start, norb, coords, H, Pt.data(), Pb, Fb.data());
    // Level shift large while far from convergence, tapering toward zero so the
    // final density is an unshifted SCF fixed point.
    const double shift = (it < 8) ? 8.0 : (it < 20) ? 4.0 : (it < 40) ? 1.0 : 0.1;
    diagShiftedDev(nB, Fa.data(), Pa, nAlpha, shift, Pan.data(), eig.data(), C.data(), ev.data());
    diagShiftedDev(nB, Fb.data(), Pb, nBeta, shift, Pbn.data(), eig.data(), C.data(), ev.data());
    double ss = 0.0;
    for (int i = 0; i < n2; ++i) {
      double d = Pan[i] - Pa[i];
      ss += d * d;
      d = Pbn[i] - Pb[i];
      ss += d * d;
    }
    const double delta = std::sqrt(ss / static_cast<double>(2 * n2));
    if (delta < convTol && it > 40) {  // require the shift to have fully tapered
      for (int i = 0; i < n2; ++i) { Pa[i] = Pan[i]; Pb[i] = Pbn[i]; }
      converged = true;
      break;
    }
    const double mix = (delta > 0.05) ? 0.3 : 0.5;
    for (int i = 0; i < n2; ++i) {
      Pa[i] = mix * Pan[i] + (1.0 - mix) * Pa[i];
      Pb[i] = mix * Pbn[i] + (1.0 - mix) * Pb[i];
    }
  }
  for (int i = 0; i < n2; ++i) Pt[i] = Pa[i] + Pb[i];
  buildFockDUHFDev(nB, nA, ap, start, norb, coords, H, Pt.data(), Pa, Fa.data());
  buildFockDUHFDev(nB, nA, ap, start, norb, coords, H, Pt.data(), Pb, Fb.data());
  double e = 0.0;
  for (int i = 0; i < n2; ++i) e += 0.5 * (Pa[i] * (H[i] + Fa[i]) + Pb[i] * (H[i] + Fb[i]));
  *eElec = e;
  *conv = converged ? 1 : 0;
}

}  // namespace semiempirical
}  // namespace nvMolKit

#endif  // NVMOLKIT_SEMIEMPIRICAL_SCF_UHF_DEVICE_H
