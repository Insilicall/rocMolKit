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
// Device-callable closed-shell NDDO/PM6_D SCF loop (9-orbital, YH scope). One
// thread runs a whole molecule's SCF: d Fock build (fock_d_device.h) -> Jacobi
// diagonalization -> density -> damped mixing, mirroring the validated
// tools/semiempirical/gen_pm6d_golden.py loop so it converges to the same
// charges. Reuses jacobiEigenDev / buildDensityDev from scf_device.h.

#ifndef NVMOLKIT_SEMIEMPIRICAL_SCF_D_DEVICE_H
#define NVMOLKIT_SEMIEMPIRICAL_SCF_D_DEVICE_H

#include <cmath>

#include "device_macros.h"
#include "fock_d_device.h"
#include "scf_device.h"  // jacobiEigenDev, buildDensityDev

namespace nvMolKit {
namespace semiempirical {

// Full PM6_D SCF for one molecule. H is the (host- or device-built) core
// Hamiltonian. Scratch F/eigA/C/Pnew are each nBasis*nBasis; eval is nBasis;
// density (output) is nBasis*nBasis. On return density holds the converged P,
// eval the final orbital energies, *conv whether it converged, *niter the count.
NVMOLKIT_HD inline void scfLoopDDev(int nBasis, int nAtoms, const AtomIntParams* ap,
                                    const int* start, const int* norb, const double* coords,
                                    const double* H, int nOcc, int maxIter, double convTol,
                                    double* density, double* eval, double* F, double* eigA,
                                    double* C, double* Pnew, double* ecom, double* diisF,
                                    double* diisE, int* conv, int* niter, double* eElec,
                                    int* intMeta, double* intBlob) {
  const int n2 = nBasis * nBasis;
  // Hoist the geometry/param-only two-center integrals out of the SCF loop: they
  // are identical every iteration, so compute them ONCE into the cache here and
  // reuse via buildFockDDevCached below (pure arithmetic hoist, byte-identical).
  precomputeTwoCenterDDev(nAtoms, ap, start, norb, coords, intMeta, intBlob);
  // Initial guess: diagonalize H_core with the d-orbital diagonals shifted far up
  // so d MOs are virtual at iteration 0 (matching the oracle) — this keeps the
  // SCF in the sp-occupied basin instead of collapsing into a d-occupied one.
  for (int i = 0; i < n2; ++i) eigA[i] = H[i];
  for (int a = 0; a < nAtoms; ++a)
    for (int o = 4; o < norb[a]; ++o) {
      const int mu = start[a] + o;
      eigA[mu * nBasis + mu] += 1000.0;
    }
  jacobiEigenDev(eigA, nBasis, eval, C);
  buildDensityDev(C, nBasis, nOcc, density);

  int histN = 0;
  bool converged = false;
  int it = 0;
  bool pseudoMode = false;     // sticky once engaged
  bool pseudoBanned = false;   // permanent real-diag fallback (stall guard)
  int pseudoFallbacks = 0;     // count of blowup fallbacks
  double prevDelta = 1e30;     // previous-iteration density RMS change
  for (it = 0; it < maxIter; ++it) {
    buildFockDDevCached(nBasis, nAtoms, ap, start, norb, intMeta, intBlob, H, density, F);

    // Pulay DIIS (same as the sp scfLoopDev) — extrapolate the Fock from the
    // commutator-error history. This resolves the d-orbital SCF to the oracle's
    // solution where plain damped mixing finds a different fixed point.
    if (it >= 2) {
      for (int i = 0; i < nBasis; ++i)
        for (int j = 0; j < nBasis; ++j) {
          double fp = 0.0, pf = 0.0;
          for (int k = 0; k < nBasis; ++k) {
            fp += F[i * nBasis + k] * density[k * nBasis + j];
            pf += density[i * nBasis + k] * F[k * nBasis + j];
          }
          ecom[i * nBasis + j] = fp - pf;
        }
      if (histN == kScfDiisMax) {
        for (int h = 1; h < kScfDiisMax; ++h)
          for (int i = 0; i < n2; ++i) {
            diisF[(h - 1) * n2 + i] = diisF[h * n2 + i];
            diisE[(h - 1) * n2 + i] = diisE[h * n2 + i];
          }
        histN = kScfDiisMax - 1;
      }
      for (int i = 0; i < n2; ++i) {
        diisF[histN * n2 + i] = F[i];
        diisE[histN * n2 + i] = ecom[i];
      }
      ++histN;

      if (histN >= 2) {
        const int m = histN + 1;
        double B[(kScfDiisMax + 1) * (kScfDiisMax + 1)];
        double rhs[kScfDiisMax + 1];
        double cf[kScfDiisMax + 1];
        for (int i = 0; i < m * m; ++i) B[i] = 0.0;
        for (int i = 0; i < histN; ++i)
          for (int j = 0; j < histN; ++j) {
            double dot = 0.0;
            for (int t = 0; t < n2; ++t) dot += diisE[i * n2 + t] * diisE[j * n2 + t];
            B[i * m + j] = dot;
          }
        for (int i = 0; i < histN; ++i) {
          B[histN * m + i] = -1.0;
          B[i * m + histN] = -1.0;
          rhs[i] = 0.0;
        }
        rhs[histN] = -1.0;
        if (solveLinearDev(B, rhs, m, cf)) {
          for (int i = 0; i < n2; ++i) F[i] = 0.0;
          for (int i = 0; i < histN; ++i)
            for (int t = 0; t < n2; ++t) F[t] += cf[i] * diisF[i * n2 + t];
          for (int i = 0; i < nBasis; ++i)
            for (int j = i + 1; j < nBasis; ++j) {
              const double avg = 0.5 * (F[i * nBasis + j] + F[j * nBasis + i]);
              F[i * nBasis + j] = avg;
              F[j * nBasis + i] = avg;
            }
        }
      }
    }

    // Diagonalize F. After kPseudoNReal real diags, once the SCF is inside the
    // convergence basin (delta < kPseudoTrans), use the cheap Stewart pseudo-diag:
    // it rotates the EXISTING MO columns C toward block-diagonalizing F over the
    // occ-virt block, keeping the eigenvalues `eval` frozen from the last real
    // diag. This is O(nocc*nvirt*nB) vs O(nB^3) for the full Jacobi. fmo (n^2) and
    // ws (n) scratch are carved from eigA (disjoint: fmo uses < nVirt*nOcc entries
    // and ws the trailing nB, both < n2 for nB>=2).
    // Engaged, not banned, past the warm-up, and not on a periodic eigenvalue-
    // refresh iteration (every kPseudoRefresh-th engaged step is a real diag).
    const bool usePseudo = pseudoMode && !pseudoBanned && it >= kPseudoNReal &&
                           ((it - kPseudoNReal) % kPseudoRefresh != 0);
    if (usePseudo) {
      double* fmo = eigA;
      double* ws = eigA + n2 - nBasis;
      pseudoDiagDev(F, C, nBasis, nOcc, eval, ws, fmo);
    } else {
      for (int i = 0; i < n2; ++i) eigA[i] = F[i];
      jacobiEigenDev(eigA, nBasis, eval, C);
    }
    buildDensityDev(C, nBasis, nOcc, Pnew);

    double ss = 0.0;
    for (int i = 0; i < n2; ++i) {
      const double d = Pnew[i] - density[i];
      ss += d * d;
    }
    double delta = std::sqrt(ss / static_cast<double>(n2));

    // Pseudo-diag fallback: if the step blew up (MOPAC's `diff > 1` guard), redo
    // this iteration with a REAL diag from F (re-anchoring C and eval). After a
    // few such fallbacks the molecule is clearly not pseudo-friendly -> ban pseudo
    // permanently and finish on real diags, which provably reach the same fixed
    // point (this is the convergence/correctness safety net).
    if (usePseudo && delta > kPseudoBlowup * prevDelta) {
      for (int i = 0; i < n2; ++i) eigA[i] = F[i];
      jacobiEigenDev(eigA, nBasis, eval, C);
      buildDensityDev(C, nBasis, nOcc, Pnew);
      ss = 0.0;
      for (int i = 0; i < n2; ++i) {
        const double d = Pnew[i] - density[i];
        ss += d * d;
      }
      delta = std::sqrt(ss / static_cast<double>(n2));
      if (++pseudoFallbacks >= kPseudoMaxFallback) pseudoBanned = true;
    }

    // Stall guard: pseudo-diag should converge in roughly the same iteration count
    // as the full diag. If it hasn't converged well into the budget, ban it and
    // finish on real diags so the molecule still reaches the true fixed point.
    if (pseudoMode && !pseudoBanned && it >= kPseudoNReal + maxIter / 4) pseudoBanned = true;
    // Tail lock: ban pseudo once near convergence so the final few iterations are
    // real diags. delta is ~monotone here -> GPU and CPU cross kPseudoLockTol at the
    // same iteration and finish on identical real-diag steps -> converged density is
    // pinned to the true fixed point and bit-exact GPU vs CPU (and vs baseline).
    if (pseudoMode && !pseudoBanned && delta < kPseudoLockTol) pseudoBanned = true;

    if (delta < convTol) {
      // By construction (kPseudoLockTol >> convTol) the converging step is a real
      // diag, so the accepted density is the true fixed point -- bit-identical to the
      // full-diag baseline and identical GPU vs CPU.
      for (int i = 0; i < n2; ++i) density[i] = Pnew[i];
      converged = true;
      break;
    }
    prevDelta = delta;
    // Engage pseudo-diag once inside the basin (sticky thereafter).
    if (!pseudoMode && delta < kPseudoTrans && it + 1 >= kPseudoNReal) pseudoMode = true;
    // Damped mixing (with the oracle's d schedule) feeds the DIIS history.
    double mix;
    if (it < 3) mix = 0.3;
    else if (delta > 0.1) mix = 0.05;
    else if (delta > 0.01) mix = 0.5;
    else mix = 0.8;
    for (int i = 0; i < n2; ++i) density[i] = mix * Pnew[i] + (1.0 - mix) * density[i];
  }
  *conv = converged ? 1 : 0;
  *niter = it + 1;

  // Final Fock + electronic energy E_elec = 0.5 sum(P .* (H + F)).
  buildFockDDevCached(nBasis, nAtoms, ap, start, norb, intMeta, intBlob, H, density, F);
  double e = 0.0;
  for (int i = 0; i < n2; ++i) e += 0.5 * density[i] * (H[i] + F[i]);
  *eElec = e;
}

}  // namespace semiempirical
}  // namespace nvMolKit

#endif  // NVMOLKIT_SEMIEMPIRICAL_SCF_D_DEVICE_H
