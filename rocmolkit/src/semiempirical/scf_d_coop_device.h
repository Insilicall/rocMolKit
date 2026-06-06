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
// COOPERATIVE (one-warp-per-molecule) PM6_D SCF loop. This is the GPU
// throughput path: a 32-thread block (one RDNA4 wavefront) cooperatively runs
// one molecule's SCF, spreading the O(nB^2)/O(nB^3) work (Jacobi
// diagonalization, density build, DIIS commutator + dot products, mixing)
// across the lanes instead of serializing it on a single lane like the
// one-thread-per-molecule scfLoopDDev.
//
// CORRECTNESS: the per-(p,q) Jacobi rotation angle is recomputed redundantly on
// every lane from the same inputs, and the length-nB inner updates are simply
// SPLIT across lanes -- the same floating-point operations, just distributed.
// The cyclic sweep schedule and pairing order are identical to the serial
// jacobiEigenDev, so the Jacobi output is bit-for-bit identical to the CPU
// reference. The only relaxation comes from cross-lane sum reductions (energy,
// convergence delta, DIIS dot products), which use a fixed lane-0 fold over a
// 32-slot buffer and stay ~1e-12. The Fock build and the (tiny histN<=6) DIIS
// linear solve stay on lane 0 -- O(nAtoms^2) / O(histN^3), dwarfed by the
// O(sweeps*nB^3) diagonalization.
//
// Device-only: these use __syncthreads() and are never compiled into the CPU
// reference (which keeps using scfLoopDDev from scf_d_device.h).

#ifndef NVMOLKIT_SEMIEMPIRICAL_SCF_D_COOP_DEVICE_H
#define NVMOLKIT_SEMIEMPIRICAL_SCF_D_COOP_DEVICE_H

#if defined(__HIPCC__) || defined(__CUDACC__)

#include <cmath>

#include "device_macros.h"
#include "fock_d_device.h"
#include "scf_device.h"  // kScfDiisMax, solveLinearDev

namespace nvMolKit {
namespace semiempirical {

// Warp width for the cooperative path (one RDNA4 wavefront == 32 lanes) and the
// max supported block size (multiple of 32) for the tunable launch.
enum { kCoopWarp = 32, kCoopMaxBlock = 256 };

// Cross-lane sum reduction into a shared slot with a deterministic lane-0 fold
// so the result is reproducible. `sh` is a >=kCoopWarp-double block-local
// scratch; returns the total to every lane.
__device__ inline double coopWarpSum(double partial, int lane, int nLanes, double* sh) {
  sh[lane] = partial;
  __syncthreads();
  double tot = 0.0;
  for (int i = 0; i < nLanes; ++i) tot += sh[i];  // fixed order, all lanes
  __syncthreads();
  return tot;
}

// --- Parallel-ordering (round-robin tournament) cooperative Jacobi --------------
//
// The cyclic jacobiEigenDevCoop barriers TWICE PER (p,q) ROTATION -- O(n^2)
// __syncthreads() per sweep, which dominates for large nB. This variant uses
// the classic Brent-Luk chess-tournament schedule: each parallel STEP rotates
// n/2 DISJOINT index pairs at once (conflict-free, since the pairs partition the
// indices), so a full sweep is (n-1) steps with just 2 barriers each -- O(n)
// barriers per sweep. The schedule is fixed/deterministic, so the result is
// reproducible run-to-run; it merely REORDERS the rotations vs the serial cyclic
// order, so it converges to the same eigenpairs but relaxes GPU-vs-CPU bitwise
// agreement to the SCF floor (~1e-9). Charges-vs-MOPAC stay <=1e-4 (the anchor).
//
// `ring` is a >=(n+1)-int shared scratch for the tournament schedule; `cs` is a
// >=n-double shared scratch for the per-pair (c,s). `sh` is the kCoopWarp
// reduction scratch.
__device__ inline void jacobiEigenDevCoopParallel(double* A, int n, double* eval, double* evec,
                                                  int lane, int nLanes, double* sh,
                                                  int* ring, double* cs) {
  for (int i = lane; i < n * n; i += nLanes) evec[i] = 0.0;
  __syncthreads();
  for (int i = lane; i < n; i += nLanes) evec[i * n + i] = 1.0;
  __syncthreads();

  double frob2p = 0.0;
  for (int i = lane; i < n * n; i += nLanes) frob2p += A[i] * A[i];
  const double frob2 = coopWarpSum(frob2p, lane, nLanes, sh);
  const double offTol = 1e-26 * (frob2 > 0.0 ? frob2 : 1.0);

  // Pad to even m (a dummy index n is "inactive"; its pairs are skipped). The
  // tournament ring holds players 0..m-1; round r pairs (ring[i], ring[m-1-i]).
  const int m = (n + 1) & ~1;     // even
  const int half = m / 2;         // pairs per round
  for (int i = lane; i < m; i += nLanes) ring[i] = i;
  __syncthreads();

  const int maxSweeps = 60 + 4 * n;
  for (int sweep = 0; sweep < maxSweeps; ++sweep) {
    double offp = 0.0;
    for (int idx = lane; idx < n; idx += nLanes)
      for (int q = idx + 1; q < n; ++q) offp += A[idx * n + q] * A[idx * n + q];
    const double off = coopWarpSum(offp, lane, nLanes, sh);
    if (off < offTol) break;

    // (m-1) tournament rounds cover every unordered pair exactly once.
    for (int round = 0; round < m - 1; ++round) {
      // Angles for this round's `half` disjoint pairs, computed in parallel.
      for (int i = lane; i < half; i += nLanes) {
        int p = ring[i], q = ring[m - 1 - i];
        if (p > q) { const int t = p; p = q; q = t; }
        if (q >= n) { cs[2 * i] = 1.0; cs[2 * i + 1] = 0.0; continue; }  // dummy
        const double apq = A[p * n + q];
        if (std::fabs(apq) < 1e-300) { cs[2 * i] = 1.0; cs[2 * i + 1] = 0.0; continue; }
        const double app = A[p * n + p], aqq = A[q * n + q];
        const double phi = 0.5 * std::atan2(2.0 * apq, aqq - app);
        cs[2 * i] = std::cos(phi);
        cs[2 * i + 1] = std::sin(phi);
      }
      __syncthreads();
      // Column phase: pair i owns columns {p,q} (disjoint across pairs).
      for (int i = 0; i < half; ++i) {
        int p = ring[i], q = ring[m - 1 - i];
        if (p > q) { const int t = p; p = q; q = t; }
        if (q >= n) continue;
        const double c = cs[2 * i], sn = cs[2 * i + 1];
        if (sn == 0.0 && c == 1.0) continue;
        for (int k = lane; k < n; k += nLanes) {
          const double akp = A[k * n + p], akq = A[k * n + q];
          A[k * n + p] = c * akp - sn * akq;
          A[k * n + q] = sn * akp + c * akq;
        }
      }
      __syncthreads();
      // Row phase: pair i owns rows {p,q}, plus the eigenvector columns {p,q}.
      for (int i = 0; i < half; ++i) {
        int p = ring[i], q = ring[m - 1 - i];
        if (p > q) { const int t = p; p = q; q = t; }
        if (q >= n) continue;
        const double c = cs[2 * i], sn = cs[2 * i + 1];
        if (sn == 0.0 && c == 1.0) continue;
        for (int k = lane; k < n; k += nLanes) {
          const double apk = A[p * n + k], aqk = A[q * n + k];
          A[p * n + k] = c * apk - sn * aqk;
          A[q * n + k] = sn * apk + c * aqk;
          const double vkp = evec[k * n + p], vkq = evec[k * n + q];
          evec[k * n + p] = c * vkp - sn * vkq;
          evec[k * n + q] = sn * vkp + c * vkq;
        }
      }
      __syncthreads();
      // Advance the tournament: ring[0] fixed; rotate ring[1..m-1] right by one.
      if (lane == 0) {
        const int last = ring[m - 1];
        for (int i = m - 1; i > 1; --i) ring[i] = ring[i - 1];
        ring[1] = last;
      }
      __syncthreads();
    }
  }

  // Eigenvalues + selection sort (serial on lane 0; n small).
  if (lane == 0) {
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
  }
  __syncthreads();
}

// Cooperative symmetric eigensolver (cyclic Jacobi). Same sweep schedule and
// rotation order as jacobiEigenDev; the inner length-n row/col/evec updates are
// split across the warp's lanes. A overwritten; eval (ascending), evec (cols).
__device__ inline void jacobiEigenDevCoop(double* A, int n, double* eval, double* evec,
                                          int lane, int nLanes, double* sh) {
  for (int i = lane; i < n * n; i += nLanes) evec[i] = 0.0;
  __syncthreads();
  for (int i = lane; i < n; i += nLanes) evec[i * n + i] = 1.0;
  __syncthreads();

  double frob2p = 0.0;
  for (int i = lane; i < n * n; i += nLanes) frob2p += A[i] * A[i];
  const double frob2 = coopWarpSum(frob2p, lane, nLanes, sh);
  const double offTol = 1e-26 * (frob2 > 0.0 ? frob2 : 1.0);

  const int maxSweeps = 60 + 4 * n;
  for (int sweep = 0; sweep < maxSweeps; ++sweep) {
    double offp = 0.0;
    for (int p = lane; p < n; p += nLanes)
      for (int q = p + 1; q < n; ++q) offp += A[p * n + q] * A[p * n + q];
    const double off = coopWarpSum(offp, lane, nLanes, sh);
    if (off < offTol) break;

    for (int p = 0; p < n; ++p) {
      for (int q = p + 1; q < n; ++q) {
        const double apq = A[p * n + q];
        if (std::fabs(apq) < 1e-300) continue;
        // Angle recomputed identically on every lane (no shuffle needed).
        const double app = A[p * n + p], aqq = A[q * n + q];
        const double phi = 0.5 * std::atan2(2.0 * apq, aqq - app);
        const double c = std::cos(phi), s = std::sin(phi);
        // Column update A[:,p],A[:,q] -- split over k by lane.
        for (int k = lane; k < n; k += nLanes) {
          const double akp = A[k * n + p], akq = A[k * n + q];
          A[k * n + p] = c * akp - s * akq;
          A[k * n + q] = s * akp + c * akq;
        }
        __syncthreads();
        // Row update A[p,:],A[q,:].
        for (int k = lane; k < n; k += nLanes) {
          const double apk = A[p * n + k], aqk = A[q * n + k];
          A[p * n + k] = c * apk - s * aqk;
          A[q * n + k] = s * apk + c * aqk;
        }
        // Eigenvector accumulation evec[:,p],evec[:,q].
        for (int k = lane; k < n; k += nLanes) {
          const double vkp = evec[k * n + p], vkq = evec[k * n + q];
          evec[k * n + p] = c * vkp - s * vkq;
          evec[k * n + q] = s * vkp + c * vkq;
        }
        __syncthreads();
      }
    }
  }

  // Eigenvalues + selection sort (serial on lane 0; n small).
  if (lane == 0) {
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
  }
  __syncthreads();
}

// P = 2 sum_{k<nOcc} C[:,k] C[:,k]^T -- rows split across lanes.
__device__ inline void buildDensityDevCoop(const double* C, int n, int nOcc, double* P,
                                          int lane, int nLanes) {
  for (int i = lane; i < n; i += nLanes)
    for (int j = 0; j < n; ++j) {
      double acc = 0.0;
      for (int k = 0; k < nOcc; ++k) acc += 2.0 * C[i * n + k] * C[j * n + k];
      P[i * n + j] = acc;
    }
  __syncthreads();
}

// Full cooperative PM6_D SCF for one molecule, run by a 32-thread block.
// Same scratch layout / semantics as scfLoopDDev; `sh` is a >=kCoopWarp-double
// block-local reduction scratch.
__device__ inline void scfLoopDDevCoop(int nBasis, int nAtoms, const AtomIntParams* ap,
                                       const int* start, const int* norb, const double* coords,
                                       const double* H, int nOcc, int maxIter, double convTol,
                                       double* density, double* eval, double* F, double* eigA,
                                       double* C, double* Pnew, double* ecom, double* diisF,
                                       double* diisE, int* conv, int* niter, double* eElec,
                                       int lane, int nLanes, double* sh, int* ring, double* cs) {
  const int n2 = nBasis * nBasis;

  // Initial guess: H_core with d diagonals shifted up so d MOs start virtual.
  for (int i = lane; i < n2; i += nLanes) eigA[i] = H[i];
  __syncthreads();
  if (lane == 0)
    for (int a = 0; a < nAtoms; ++a)
      for (int o = 4; o < norb[a]; ++o) {
        const int mu = start[a] + o;
        eigA[mu * nBasis + mu] += 1000.0;
      }
  __syncthreads();
  jacobiEigenDevCoopParallel(eigA, nBasis, eval, C, lane, nLanes, sh, ring, cs);
  buildDensityDevCoop(C, nBasis, nOcc, density, lane, nLanes);

  int histN = 0;
  bool converged = false;
  int it = 0;
  for (it = 0; it < maxIter; ++it) {
    // Fock build: serial on lane 0 (O(nAtoms^2), cheap vs the diagonalization).
    if (lane == 0)
      buildFockDDev(nBasis, nAtoms, ap, start, norb, coords, H, density, F);
    __syncthreads();

    if (it >= 2) {
      // Commutator FP - PF (O(n^3)) -- rows split across lanes.
      for (int i = lane; i < nBasis; i += nLanes)
        for (int j = 0; j < nBasis; ++j) {
          double fp = 0.0, pf = 0.0;
          for (int k = 0; k < nBasis; ++k) {
            fp += F[i * nBasis + k] * density[k * nBasis + j];
            pf += density[i * nBasis + k] * F[k * nBasis + j];
          }
          ecom[i * nBasis + j] = fp - pf;
        }
      __syncthreads();

      if (lane == 0) {
        if (histN == kScfDiisMax) {
          for (int h = 1; h < kScfDiisMax; ++h)
            for (int i = 0; i < n2; ++i) {
              diisF[(h - 1) * n2 + i] = diisF[h * n2 + i];
              diisE[(h - 1) * n2 + i] = diisE[h * n2 + i];
            }
          histN = kScfDiisMax - 1;
        }
      }
      // Broadcast lane-0's (possibly shifted) histN to the whole block.
      if (lane == 0) sh[0] = (double)histN;
      __syncthreads();
      histN = (int)sh[0];
      __syncthreads();
      for (int i = lane; i < n2; i += nLanes) {
        diisF[histN * n2 + i] = F[i];
        diisE[histN * n2 + i] = ecom[i];
      }
      __syncthreads();
      ++histN;

      if (histN >= 2) {
        const int m = histN + 1;
        __shared__ double B[(kScfDiisMax + 1) * (kScfDiisMax + 1)];
        double rhs[kScfDiisMax + 1];
        double cf[kScfDiisMax + 1];
        if (lane == 0)
          for (int i = 0; i < m * m; ++i) B[i] = 0.0;
        __syncthreads();
        // B[i][j] = <diisE_i, diisE_j> -- fold each dot over the warp.
        for (int i = 0; i < histN; ++i)
          for (int j = 0; j < histN; ++j) {
            double dotp = 0.0;
            for (int t = lane; t < n2; t += nLanes)
              dotp += diisE[i * n2 + t] * diisE[j * n2 + t];
            const double dot = coopWarpSum(dotp, lane, nLanes, sh);
            if (lane == 0) B[i * m + j] = dot;
          }
        __syncthreads();
        if (lane == 0) {
          for (int i = 0; i < histN; ++i) {
            B[histN * m + i] = -1.0;
            B[i * m + histN] = -1.0;
            rhs[i] = 0.0;
          }
          rhs[histN] = -1.0;
          const bool solved = solveLinearDev(B, rhs, m, cf);
          if (solved)
            for (int i = 0; i < histN; ++i) sh[i] = cf[i];
          sh[nLanes - 1] = solved ? 1.0 : 0.0;
        }
        __syncthreads();
        if (sh[nLanes - 1] != 0.0) {
          // F = sum_i cf[i] * diisF[i] -- split over elements.
          for (int t = lane; t < n2; t += nLanes) {
            double acc = 0.0;
            for (int i = 0; i < histN; ++i) acc += sh[i] * diisF[i * n2 + t];
            F[t] = acc;
          }
          __syncthreads();
          for (int i = lane; i < nBasis; i += nLanes)
            for (int j = i + 1; j < nBasis; ++j) {
              const double avg = 0.5 * (F[i * nBasis + j] + F[j * nBasis + i]);
              F[i * nBasis + j] = avg;
              F[j * nBasis + i] = avg;
            }
          __syncthreads();
        }
      }
    }

    for (int i = lane; i < n2; i += nLanes) eigA[i] = F[i];
    __syncthreads();
    jacobiEigenDevCoopParallel(eigA, nBasis, eval, C, lane, nLanes, sh, ring, cs);
    buildDensityDevCoop(C, nBasis, nOcc, Pnew, lane, nLanes);

    double ssp = 0.0;
    for (int i = lane; i < n2; i += nLanes) {
      const double d = Pnew[i] - density[i];
      ssp += d * d;
    }
    const double ss = coopWarpSum(ssp, lane, nLanes, sh);
    const double delta = std::sqrt(ss / static_cast<double>(n2));
    if (delta < convTol) {
      for (int i = lane; i < n2; i += nLanes) density[i] = Pnew[i];
      __syncthreads();
      converged = true;
      break;
    }
    double mix;
    if (it < 3) mix = 0.3;
    else if (delta > 0.1) mix = 0.05;
    else if (delta > 0.01) mix = 0.5;
    else mix = 0.8;
    for (int i = lane; i < n2; i += nLanes)
      density[i] = mix * Pnew[i] + (1.0 - mix) * density[i];
    __syncthreads();
  }
  if (lane == 0) {
    *conv = converged ? 1 : 0;
    *niter = it + 1;
  }

  // Final Fock + electronic energy.
  if (lane == 0) buildFockDDev(nBasis, nAtoms, ap, start, norb, coords, H, density, F);
  __syncthreads();
  double ep = 0.0;
  for (int i = lane; i < n2; i += nLanes) ep += 0.5 * density[i] * (H[i] + F[i]);
  const double e = coopWarpSum(ep, lane, nLanes, sh);
  if (lane == 0) *eElec = e;
  __syncthreads();
}

}  // namespace semiempirical
}  // namespace nvMolKit

#endif  // __HIPCC__ || __CUDACC__
#endif  // NVMOLKIT_SEMIEMPIRICAL_SCF_D_COOP_DEVICE_H
