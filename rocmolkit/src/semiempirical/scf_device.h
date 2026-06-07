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
// Device-callable closed-shell NDDO/PM6 SCF loop (sp basis). One thread runs a
// whole molecule's SCF: Fock build (fock_device.h) -> block-local Jacobi
// diagonalization -> density -> damped mixing, to convergence. No std:: deps —
// operates on caller-provided scratch so it runs identically on host and device.
// DIIS is omitted (the converged density is the same fixed point; damped mixing
// just takes more iterations). See docs/SEMIEMPIRICAL_DESIGN.md.

#ifndef NVMOLKIT_SEMIEMPIRICAL_SCF_DEVICE_H
#define NVMOLKIT_SEMIEMPIRICAL_SCF_DEVICE_H

#include <cmath>

#include "device_macros.h"
#include "fock_device.h"
#include "two_center_device.h"

namespace nvMolKit {
namespace semiempirical {

// Symmetric eigensolver (cyclic Jacobi) for one n x n matrix. A is overwritten;
// eval (ascending) and evec (columns) are written. All buffers caller-provided.
// Returns false if the off-diagonal norm did not converge within the budget.
NVMOLKIT_HD inline bool jacobiEigenDev(double* A, int n, double* eval, double* evec) {
  for (int i = 0; i < n * n; ++i) evec[i] = 0.0;
  for (int i = 0; i < n; ++i) evec[i * n + i] = 1.0;

  // Converge relative to the (rotation-invariant) Frobenius norm; an absolute
  // threshold is unreachable for larger matrices with large entries.
  double frob2 = 0.0;
  for (int i = 0; i < n * n; ++i) frob2 += A[i] * A[i];
  const double offTol = 1e-26 * (frob2 > 0.0 ? frob2 : 1.0);

  const int maxSweeps = 60 + 4 * n;
  bool converged = false;
  for (int sweep = 0; sweep < maxSweeps; ++sweep) {
    double off = 0.0;
    for (int p = 0; p < n; ++p)
      for (int q = p + 1; q < n; ++q) off += A[p * n + q] * A[p * n + q];
    if (off < offTol) {
      converged = true;
      break;
    }
    for (int p = 0; p < n; ++p) {
      for (int q = p + 1; q < n; ++q) {
        const double apq = A[p * n + q];
        if (std::fabs(apq) < 1e-300) continue;
        const double app = A[p * n + p], aqq = A[q * n + q];
        const double phi = 0.5 * std::atan2(2.0 * apq, aqq - app);
        const double c = std::cos(phi), s = std::sin(phi);
        for (int k = 0; k < n; ++k) {
          const double akp = A[k * n + p], akq = A[k * n + q];
          A[k * n + p] = c * akp - s * akq;
          A[k * n + q] = s * akp + c * akq;
        }
        for (int k = 0; k < n; ++k) {
          const double apk = A[p * n + k], aqk = A[q * n + k];
          A[p * n + k] = c * apk - s * aqk;
          A[q * n + k] = s * apk + c * aqk;
        }
        for (int k = 0; k < n; ++k) {
          const double vkp = evec[k * n + p], vkq = evec[k * n + q];
          evec[k * n + p] = c * vkp - s * vkq;
          evec[k * n + q] = s * vkp + c * vkq;
        }
      }
    }
  }

  // Selection sort of eigenpairs by ascending eigenvalue (n small, no std::sort).
  for (int i = 0; i < n; ++i) eval[i] = A[i * n + i];
  for (int i = 0; i < n; ++i) {
    int mn = i;
    for (int j = i + 1; j < n; ++j)
      if (eval[j] < eval[mn]) mn = j;
    if (mn != i) {
      const double tv = eval[i];
      eval[i] = eval[mn];
      eval[mn] = tv;
      for (int k = 0; k < n; ++k) {
        const double t = evec[k * n + i];
        evec[k * n + i] = evec[k * n + mn];
        evec[k * n + mn] = t;
      }
    }
  }
  return converged;
}

// P = 2 sum_{k<nOcc} C[:,k] C[:,k]^T.
NVMOLKIT_HD inline void buildDensityDev(const double* C, int n, int nOcc, double* P) {
  for (int i = 0; i < n * n; ++i) P[i] = 0.0;
  for (int k = 0; k < nOcc; ++k)
    for (int i = 0; i < n; ++i) {
      const double cik = C[i * n + k];
      for (int j = 0; j < n; ++j) P[i * n + j] += 2.0 * cik * C[j * n + k];
    }
}

// --- Pseudo-diagonalization schedule (MOPAC-style, shared by the CPU scfLoopDDev
// and the cooperative GPU scfLoopDDevCoop so both make identical mode decisions) -
// Do this many REAL Jacobi diagonalizations first (the iteration-0 guess is always
// a real diag; the loop's first kPseudoNReal iterations also use a real diag) so
// the MO basis is close before switching to the cheap Stewart pseudo-diag.
enum { kPseudoNReal = 3 };
// Engage pseudo-diag once the density RMS change drops below this (MOPAC's `trans`
// is 0.2; the SCF is well inside the convergence basin by then). Sticky.
constexpr double kPseudoTrans = 0.1;
// If a pseudo-diag step makes the density change GROW past this factor of the
// previous step (MOPAC turns newdg off when `diff > 1`), redo it with a real diag.
constexpr double kPseudoBlowup = 2.0;
// After this many blowup fallbacks, ban pseudo-diag for the molecule and finish on
// real diagonalizations (guarantees the same converged fixed point as baseline).
enum { kPseudoMaxFallback = 3 };
// Refresh the frozen eigenvalues with a REAL diag every Nth engaged iteration. The
// DIIS-extrapolated Fock keeps moving, so eigenvalues drift; a periodic real diag
// re-anchors eval/C and keeps the cheap pseudo-diag steps tracking the SCF. 1 real
// + (kPseudoRefresh-1) pseudo per cycle.
enum { kPseudoRefresh = 4 };
// The cooperative GPU SCF (one wavefront/molecule) keeps the full tournament Jacobi
// instead of pseudo-diag: pseudo-diag's occ-virt rotations share MO columns and so
// serialize with a __syncthreads per significant element (O(n^2) barriers), which
// is SLOWER than the barrier-light Brent-Luk tournament on a 32-lane wavefront. The
// CPU path (no barriers) takes the pseudo-diag win. The converged FIXED POINT is
// identical either way, so GPU==CPU charges still agree to the SCF floor (~1e-9).
constexpr bool kPseudoCoopEnabled = true;
// Once the density change drops below this (>> convTol), ban pseudo-diag and finish
// the SCF tail on real diagonalizations. delta decreases ~monotonically here, so
// GPU and CPU ban at the SAME iteration and run identical real-diag steps to
// convergence -> the converged density is pinned to the true fixed point and stays
// bit-exact GPU vs CPU (critical for sensitive transition-metal systems). Only the
// few tail iterations (kPseudoLockTol -> convTol) are real diags, so the pseudo
// speedup over the bulk of the SCF is preserved.
constexpr double kPseudoLockTol = 1e-4;

// Stewart pseudo-diagonalization (Stewart, Csaszar, Pulay, J.Comp.Chem 3,227,1982),
// a faithful port of MOPAC's matrix/diag.F90 to a DENSE (not packed) Fock matrix.
//
// Instead of a full O(n^3) Jacobi diag every SCF iteration, it (a) forms only the
// occupied x virtual block of the Fock matrix in the CURRENT MO basis,
//   F_mo[i,j] = C[:,i]^T F C[:,j],  i in virtuals, j in occupied,  (O(nocc*nvirt*n))
// and (b) applies 2x2 Givens rotations that "annihilate" each significant element,
// rotating MO columns i (virtual) and j (occupied) of C. The rotation angle uses
// the FROZEN eigenvalues `eval` from the last real diagonalization (it does NOT
// re-diagonalize the occ-occ / virt-virt blocks). Eigenvalues are unchanged.
//
// Exactly like diag.F90, the F_mo block is computed ONCE (into `fmo`, an n*n
// caller scratch -- only nVirt*nOcc entries used) and the 2x2 rotations then use
// those STORED values (created off-diagonals are ignored -- approximation (C) in
// the paper). At self-consistency F_mo -> 0, so no rotations fire and the density
// rebuilt from the occupied columns is the SAME fixed point a real diag gives. The
// SCF loop still does a real diag on the iteration it declares convergence, so the
// converged density / charges are bit-identical to the full-diag path.
//
// Thresholds match MOPAC: tiny = 0.05*max|F_mo|, bigeps = 1.5e-7. C columns are
// MOs: C[i*n + k] = coeff of basis fn i in MO k. `ws` (n doubles) and `fmo`
// (n*n doubles) are caller scratch.
NVMOLKIT_HD inline void pseudoDiagDev(const double* F, double* C, int n, int nOcc,
                                      const double* eval, double* ws, double* fmo) {
  const double bigeps = 1.5e-7;
  // Pass 1: F_mo(virtual i, occupied j), stored row-major in fmo[(i-nOcc)*nOcc+j].
  double tiny = 0.0;
  for (int i = nOcc; i < n; ++i) {
    for (int r = 0; r < n; ++r) {
      double sum = 0.0;
      for (int k = 0; k < n; ++k) sum += F[r * n + k] * C[k * n + i];
      ws[r] = sum;
    }
    for (int j = 0; j < nOcc; ++j) {
      double sum = 0.0;
      for (int k = 0; k < n; ++k) sum += ws[k] * C[k * n + j];
      fmo[(i - nOcc) * nOcc + j] = sum;
      const double a = std::fabs(sum);
      if (a > tiny) tiny = a;
    }
  }
  tiny = 0.05 * tiny;

  // Pass 2: crude 2x2 rotations to "eliminate" significant occ-virt elements, in
  // the SAME deterministic (virtual outer, occupied inner) order as diag.F90.
  for (int i = nOcc; i < n; ++i) {
    for (int j = 0; j < nOcc; ++j) {
      const double c = fmo[(i - nOcc) * nOcc + j];
      if (std::fabs(c) < tiny) continue;
      const double d = eval[j] - eval[i];
      if (std::fabs(c / d) < bigeps) continue;
      const double e = std::copysign(std::sqrt(4.0 * c * c + d * d), d);
      const double alpha = std::sqrt(0.5 * (1.0 + d / e));
      const double beta = -std::copysign(std::sqrt(1.0 - alpha * alpha), c);
      for (int m = 0; m < n; ++m) {
        const double va = C[m * n + j];
        const double vb = C[m * n + i];
        C[m * n + j] = alpha * va + beta * vb;
        C[m * n + i] = alpha * vb - beta * va;
      }
    }
  }
}

// Max DIIS history depth (matches the CPU reference).
enum { kScfDiisMax = 6 };

// Solve the m x m system B x = rhs by Gaussian elimination with partial pivot
// (m <= kScfDiisMax+1). B/rhs are modified; x is written. Returns false if
// singular.
NVMOLKIT_HD inline bool solveLinearDev(double* B, double* rhs, int m, double* x) {
  for (int col = 0; col < m; ++col) {
    int piv = col;
    for (int r = col + 1; r < m; ++r)
      if (std::fabs(B[r * m + col]) > std::fabs(B[piv * m + col])) piv = r;
    if (std::fabs(B[piv * m + col]) < 1e-14) return false;
    if (piv != col) {
      for (int c = 0; c < m; ++c) {
        const double t = B[piv * m + c];
        B[piv * m + c] = B[col * m + c];
        B[col * m + c] = t;
      }
      const double t = rhs[piv];
      rhs[piv] = rhs[col];
      rhs[col] = t;
    }
    for (int r = 0; r < m; ++r) {
      if (r == col) continue;
      const double f = B[r * m + col] / B[col * m + col];
      for (int c = 0; c < m; ++c) B[r * m + c] -= f * B[col * m + c];
      rhs[r] -= f * rhs[col];
    }
  }
  for (int i = 0; i < m; ++i) x[i] = rhs[i] / B[i * m + i];
  return true;
}

// Full SCF loop for one molecule with Pulay DIIS (matches the CPU reference, so
// it converges to the same solution). H is the (host- or device-built) core
// Hamiltonian. Scratch: F/eigA/C/Pnew/ecom are each n*n; diisF/diisE are
// kScfDiisMax*n*n; eval is n; density (output) is n*n. On return density holds
// the converged P, eval the final orbital energies, *conv whether it converged,
// *niter the iteration count.
NVMOLKIT_HD inline void scfLoopDev(int nBasis, int nAtoms, const AtomIntParams* ap,
                                   const int* start, const int* norb, const double* coords,
                                   const double* H, int nOcc, int maxIter, double convTol,
                                   double* density, double* eval, double* F, double* eigA,
                                   double* C, double* Pnew, double* ecom, double* diisF,
                                   double* diisE, int* conv, int* niter, double* eElec) {
  const int n2 = nBasis * nBasis;

  for (int i = 0; i < n2; ++i) eigA[i] = H[i];
  jacobiEigenDev(eigA, nBasis, eval, C);
  buildDensityDev(C, nBasis, nOcc, density);

  int histN = 0;
  bool converged = false;
  int it = 0;
  for (it = 0; it < maxIter; ++it) {
    buildFockDev(nBasis, nAtoms, ap, start, norb, coords, H, density, F);

    if (it >= 2) {
      // Commutator error e = F*P - P*F.
      for (int i = 0; i < nBasis; ++i)
        for (int j = 0; j < nBasis; ++j) {
          double fp = 0.0, pf = 0.0;
          for (int k = 0; k < nBasis; ++k) {
            fp += F[i * nBasis + k] * density[k * nBasis + j];
            pf += density[i * nBasis + k] * F[k * nBasis + j];
          }
          ecom[i * nBasis + j] = fp - pf;
        }
      // Push (F, e) into the history (drop oldest when full).
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

    for (int i = 0; i < n2; ++i) eigA[i] = F[i];
    jacobiEigenDev(eigA, nBasis, eval, C);
    buildDensityDev(C, nBasis, nOcc, Pnew);

    double ss = 0.0;
    for (int i = 0; i < n2; ++i) {
      const double d = Pnew[i] - density[i];
      ss += d * d;
    }
    const double delta = std::sqrt(ss / static_cast<double>(n2));
    if (delta < convTol) {
      for (int i = 0; i < n2; ++i) density[i] = Pnew[i];
      converged = true;
      break;
    }
    double mix;
    if (it < 3) mix = 0.5;
    else if (delta > 0.1) mix = 0.4;
    else if (delta > 0.01) mix = 0.5;
    else mix = 0.8;
    for (int i = 0; i < n2; ++i) density[i] = mix * Pnew[i] + (1.0 - mix) * density[i];
  }
  *conv = converged ? 1 : 0;
  *niter = it + 1;

  // Final Fock + electronic energy E_elec = 0.5 sum(P .* (H + F)).
  buildFockDev(nBasis, nAtoms, ap, start, norb, coords, H, density, F);
  double e = 0.0;
  for (int i = 0; i < n2; ++i) e += 0.5 * density[i] * (H[i] + F[i]);
  *eElec = e;
}

}  // namespace semiempirical
}  // namespace nvMolKit

#endif  // NVMOLKIT_SEMIEMPIRICAL_SCF_DEVICE_H
