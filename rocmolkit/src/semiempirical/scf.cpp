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
// Closed-shell NDDO/PM6 SCF (sp basis) host reference, ported from the PYSEQM
// reference nddo_energy / _build_fock. The hot kernels here (Fock build,
// diagonalization, density) are what Stage 7 moves to HIP/rocSOLVER. See
// docs/SEMIEMPIRICAL_DESIGN.md.

#include "scf.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include "core_hamiltonian.h"
#include "pm6_params.h"
#include "two_center.h"

namespace nvMolKit {
namespace semiempirical {

namespace {

int spCount(int z) {
  const int n = pm6NumOrbitals(z);
  return (n >= 4) ? 4 : n;
}

inline int wIdx(int mu, int nu, int lam, int sig) {
  return ((mu * 4 + nu) * 4 + lam) * 4 + sig;
}

// Cyclic Jacobi eigensolver for a symmetric n x n matrix (n small). Writes
// ascending eigenvalues into eval and the corresponding eigenvectors into the
// columns of evec (row-major n x n). A is overwritten.
void jacobiEigen(std::vector<double>& A, int n, std::vector<double>& eval,
                 std::vector<double>& evec) {
  evec.assign(n * n, 0.0);
  for (int i = 0; i < n; ++i) evec[i * n + i] = 1.0;

  for (int sweep = 0; sweep < 100; ++sweep) {
    double off = 0.0;
    for (int p = 0; p < n; ++p)
      for (int q = p + 1; q < n; ++q) off += A[p * n + q] * A[p * n + q];
    if (off < 1e-30) break;

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

  std::vector<int> order(n);
  for (int i = 0; i < n; ++i) order[i] = i;
  std::vector<double> diag(n);
  for (int i = 0; i < n; ++i) diag[i] = A[i * n + i];
  std::sort(order.begin(), order.end(), [&](int a, int b) { return diag[a] < diag[b]; });

  eval.assign(n, 0.0);
  std::vector<double> v(n * n, 0.0);
  for (int j = 0; j < n; ++j) {
    eval[j] = diag[order[j]];
    for (int i = 0; i < n; ++i) v[i * n + j] = evec[i * n + order[j]];
  }
  evec.swap(v);
}

// Density from the lowest nOcc orbitals: P = 2 * sum_k C[:,k] C[:,k]^T.
void buildDensity(const std::vector<double>& C, int n, int nOcc, std::vector<double>& P) {
  P.assign(n * n, 0.0);
  for (int k = 0; k < nOcc; ++k)
    for (int i = 0; i < n; ++i) {
      const double cik = C[i * n + k];
      for (int j = 0; j < n; ++j) P[i * n + j] += 2.0 * cik * C[j * n + k];
    }
}

// F = H + G(P): one-center Slater-Condon + two-center Coulomb/exchange (w).
void buildFock(const std::vector<double>& H, const std::vector<double>& P, int nBasis,
               int nAtoms, const int* atoms, const double* coords,
               const std::vector<int>& start, const std::vector<int>& norb,
               std::vector<double>& F) {
  F = H;

  for (int a = 0; a < nAtoms; ++a) {
    const Pm6ElementParams* p = pm6ParamsForZ(atoms[a]);
    const int s = start[a];
    const double Pss = P[s * nBasis + s];
    if (norb[a] == 1) {
      F[s * nBasis + s] += Pss * p->gss * 0.5;
      continue;
    }
    const int pk0 = s + 1, pk1 = s + 2, pk2 = s + 3;
    const double Ppp = P[pk0 * nBasis + pk0] + P[pk1 * nBasis + pk1] + P[pk2 * nBasis + pk2];
    const double sp1 = p->gsp - 0.5 * p->hsp;
    const double sp2 = 1.5 * p->hsp - 0.5 * p->gsp;
    const double ppd = 1.25 * p->gp2 - 0.25 * p->gpp;
    const double ppoff = 0.75 * p->gpp - 1.25 * p->gp2;

    F[s * nBasis + s] += Pss * p->gss * 0.5 + Ppp * sp1;
    for (int k = 1; k <= 3; ++k) {
      const int pk = s + k;
      F[pk * nBasis + pk] += Pss * sp1 + P[pk * nBasis + pk] * p->gpp * 0.5
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

  for (int i = 0; i < nAtoms; ++i) {
    for (int j = i + 1; j < nAtoms; ++j) {
      double w[256], e1b[16], e2a[16];
      twoCenterMolecular(atoms[i], &coords[3 * i], atoms[j], &coords[3 * j], w, e1b, e2a);
      const int nA = norb[i], nB = norb[j], sA = start[i], sB = start[j];
      // J on A and J on B.
      for (int mu = 0; mu < nA; ++mu)
        for (int nu = 0; nu < nA; ++nu) {
          double acc = 0.0;
          for (int lam = 0; lam < nB; ++lam)
            for (int sig = 0; sig < nB; ++sig)
              acc += P[(sB + lam) * nBasis + (sB + sig)] * w[wIdx(mu, nu, lam, sig)];
          F[(sA + mu) * nBasis + (sA + nu)] += acc;
        }
      for (int lam = 0; lam < nB; ++lam)
        for (int sig = 0; sig < nB; ++sig) {
          double acc = 0.0;
          for (int mu = 0; mu < nA; ++mu)
            for (int nu = 0; nu < nA; ++nu)
              acc += P[(sA + mu) * nBasis + (sA + nu)] * w[wIdx(mu, nu, lam, sig)];
          F[(sB + lam) * nBasis + (sB + sig)] += acc;
        }
      // Exchange: K[mu,lam] = -0.5 sum_{nu,sig} w[mu,nu,lam,sig] P[A nu, B sig].
      for (int mu = 0; mu < nA; ++mu)
        for (int lam = 0; lam < nB; ++lam) {
          double acc = 0.0;
          for (int nu = 0; nu < nA; ++nu)
            for (int sig = 0; sig < nB; ++sig)
              acc += w[wIdx(mu, nu, lam, sig)] * P[(sA + nu) * nBasis + (sB + sig)];
          acc *= -0.5;
          F[(sA + mu) * nBasis + (sB + lam)] += acc;
          F[(sB + lam) * nBasis + (sA + mu)] += acc;
        }
    }
  }
}

// Solve the (nd+1) DIIS system B c = rhs (Gaussian elimination, partial pivot).
// Returns false if singular.
bool solveLinear(std::vector<double> B, std::vector<double> rhs, int m, std::vector<double>& x) {
  for (int col = 0; col < m; ++col) {
    int piv = col;
    for (int r = col + 1; r < m; ++r)
      if (std::fabs(B[r * m + col]) > std::fabs(B[piv * m + col])) piv = r;
    if (std::fabs(B[piv * m + col]) < 1e-14) return false;
    if (piv != col) {
      for (int c = 0; c < m; ++c) std::swap(B[piv * m + c], B[col * m + c]);
      std::swap(rhs[piv], rhs[col]);
    }
    for (int r = 0; r < m; ++r) {
      if (r == col) continue;
      const double f = B[r * m + col] / B[col * m + col];
      for (int c = 0; c < m; ++c) B[r * m + c] -= f * B[col * m + c];
      rhs[r] -= f * rhs[col];
    }
  }
  x.assign(m, 0.0);
  for (int i = 0; i < m; ++i) x[i] = rhs[i] / B[i * m + i];
  return true;
}

}  // namespace

int scfSp(int nAtoms, const int* atoms, const double* coords,
          double* density, double* eigenvalues, ScfResult* out,
          int maxIter, double convTol) {
  const int nBasis = spBasisSize(nAtoms, atoms);
  if (nBasis == 0) return 0;

  int nElec = 0;
  for (int a = 0; a < nAtoms; ++a) nElec += pm6ValenceElectrons(atoms[a]);
  if (nElec % 2 != 0) return 0;  // open shell not handled here
  const int nOcc = nElec / 2;

  std::vector<int> start(nAtoms), norb(nAtoms);
  for (int a = 0, off = 0; a < nAtoms; ++a) {
    start[a] = off;
    norb[a] = spCount(atoms[a]);
    off += norb[a];
  }

  std::vector<double> H(nBasis * nBasis);
  if (buildCoreHamiltonianSp(nAtoms, atoms, coords, H.data()) == 0) return 0;

  // Initial density from the H_core eigenvectors.
  std::vector<double> A = H, eval, C, P;
  jacobiEigen(A, nBasis, eval, C);
  buildDensity(C, nBasis, nOcc, P);

  std::vector<std::vector<double>> diisF, diisE;
  const int kDiisMax = 6;
  std::vector<double> F, Pnew;
  bool converged = false;
  int iteration = 0;
  double delta = 1.0;

  for (iteration = 0; iteration < maxIter; ++iteration) {
    buildFock(H, P, nBasis, nAtoms, atoms, coords, start, norb, F);

    if (iteration >= 2) {
      std::vector<double> e(nBasis * nBasis, 0.0);  // e = F*P - P*F
      for (int i = 0; i < nBasis; ++i)
        for (int j = 0; j < nBasis; ++j) {
          double fp = 0.0, pf = 0.0;
          for (int k = 0; k < nBasis; ++k) {
            fp += F[i * nBasis + k] * P[k * nBasis + j];
            pf += P[i * nBasis + k] * F[k * nBasis + j];
          }
          e[i * nBasis + j] = fp - pf;
        }
      diisF.push_back(F);
      diisE.push_back(e);
      if (static_cast<int>(diisF.size()) > kDiisMax) {
        diisF.erase(diisF.begin());
        diisE.erase(diisE.begin());
      }
      const int nd = static_cast<int>(diisF.size());
      if (nd >= 2) {
        const int m = nd + 1;
        std::vector<double> B(m * m, 0.0), rhs(m, 0.0), c;
        for (int i = 0; i < nd; ++i)
          for (int j = 0; j < nd; ++j) {
            double dot = 0.0;
            for (size_t t = 0; t < diisE[i].size(); ++t) dot += diisE[i][t] * diisE[j][t];
            B[i * m + j] = dot;
          }
        for (int i = 0; i < nd; ++i) {
          B[nd * m + i] = -1.0;
          B[i * m + nd] = -1.0;
        }
        rhs[nd] = -1.0;
        if (solveLinear(B, rhs, m, c)) {
          std::fill(F.begin(), F.end(), 0.0);
          for (int i = 0; i < nd; ++i)
            for (int t = 0; t < nBasis * nBasis; ++t) F[t] += c[i] * diisF[i][t];
        }
      }
    }

    A = F;
    jacobiEigen(A, nBasis, eval, C);
    buildDensity(C, nBasis, nOcc, Pnew);

    double ss = 0.0;
    for (int t = 0; t < nBasis * nBasis; ++t) {
      const double d = Pnew[t] - P[t];
      ss += d * d;
    }
    delta = std::sqrt(ss / (nBasis * nBasis));
    if (delta < convTol) {
      P = Pnew;
      converged = true;
      break;
    }

    double mix;
    if (iteration < 3) mix = 0.5;
    else if (delta > 0.1) mix = 0.4;
    else if (delta > 0.01) mix = 0.5;
    else mix = 0.8;
    for (int t = 0; t < nBasis * nBasis; ++t) P[t] = mix * Pnew[t] + (1.0 - mix) * P[t];
  }

  // Final Fock + electronic energy with the converged density.
  buildFock(H, P, nBasis, nAtoms, atoms, coords, start, norb, F);
  double eElec = 0.0;
  for (int t = 0; t < nBasis * nBasis; ++t) eElec += 0.5 * P[t] * (H[t] + F[t]);

  for (int t = 0; t < nBasis * nBasis; ++t) density[t] = P[t];
  for (int i = 0; i < nBasis; ++i) eigenvalues[i] = eval[i];
  if (out != nullptr) {
    out->nBasis = nBasis;
    out->nIter = iteration + 1;
    out->converged = converged;
    out->electronicEv = eElec;
  }
  return nBasis;
}

bool mullikenCharges(int nAtoms, const int* atoms, const double* coords, double* q) {
  const int nBasis = spBasisSize(nAtoms, atoms);
  if (nBasis == 0) return false;

  std::vector<double> density(nBasis * nBasis), eval(nBasis);
  ScfResult res;
  if (scfSp(nAtoms, atoms, coords, density.data(), eval.data(), &res) == 0) {
    return false;
  }

  for (int a = 0, off = 0; a < nAtoms; ++a) {
    const int c = spCount(atoms[a]);
    double pop = 0.0;
    for (int o = 0; o < c; ++o) {
      const int mu = off + o;
      pop += density[mu * nBasis + mu];
    }
    q[a] = static_cast<double>(pm6ValenceElectrons(atoms[a])) - pop;
    off += c;
  }
  return true;
}

}  // namespace semiempirical
}  // namespace nvMolKit
