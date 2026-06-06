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
// Host CPU driver for the public PM6_D API (scf_d.h). Gathers the per-atom
// parameters, builds H_core, runs the d-orbital SCF (DIIS), and emits Mulliken
// charges + heat of formation — the same device-callable code the GPU kernel
// runs, so the two are bit-exact. See docs/SEMIEMPIRICAL_DESIGN.md.

#include "scf_d.h"

#include <cmath>
#include <vector>

#include "core_hamiltonian.h"  // gatherAtomIntParamsD
#include "core_hamiltonian_d_device.h"
#include "energy_device.h"  // nuclearRepulsionAm1Dev, heatOfFormationKcalDev
#include "pm6_params.h"     // pm6ValenceElectrons
#include "pwcct_device.h"   // heatOfFormationPm6Kcal (canonical/MOPAC core-core)
#include "scf_d_device.h"

namespace nvMolKit {
namespace semiempirical {

bool pm6dCharges(int nAtoms, const int* atoms, const double* coords, double* q,
                 double* hofKcal, int maxIter, double convTol, double* hofPm6Kcal,
                 int charge) {
  if (nAtoms <= 0) return false;

  std::vector<AtomIntParams> ap(nAtoms);
  std::vector<int> start(nAtoms), norb(nAtoms);
  int nBasis = 0, nElec = 0;
  for (int a = 0; a < nAtoms; ++a) {
    if (!gatherAtomIntParamsD(atoms[a], ap[a])) return false;
    start[a] = nBasis;
    norb[a] = ap[a].nOrb;
    nBasis += norb[a];
    nElec += pm6ValenceElectrons(atoms[a]);
  }
  nElec -= charge;  // cation (+) removes electrons; anion (-) adds them
  if (nBasis == 0 || nElec <= 0 || nElec % 2 != 0) return false;  // unsupported / open shell

  const int n2 = nBasis * nBasis;
  std::vector<double> H(n2), density(n2), eval(nBasis), F(n2), eigA(n2), C(n2), Pnew(n2),
      ecom(n2), diisF(kScfDiisMax * n2), diisE(kScfDiisMax * n2);
  buildCoreHamiltonianDDev(nBasis, nAtoms, ap.data(), start.data(), norb.data(), coords, H.data());

  int conv = 0, niter = 0;
  double eElec = 0.0;
  scfLoopDDev(nBasis, nAtoms, ap.data(), start.data(), norb.data(), coords, H.data(), nElec / 2,
              maxIter, convTol, density.data(), eval.data(), F.data(), eigA.data(), C.data(),
              Pnew.data(), ecom.data(), diisF.data(), diisE.data(), &conv, &niter, &eElec);
  if (!conv) return false;

  for (int a = 0; a < nAtoms; ++a) {
    double pop = 0.0;
    for (int o = 0; o < norb[a]; ++o) pop += density[(start[a] + o) * nBasis + (start[a] + o)];
    q[a] = static_cast<double>(pm6ValenceElectrons(atoms[a])) - pop;
  }
  if (hofKcal != nullptr) {
    const double eNuc = nuclearRepulsionAm1Dev(nAtoms, ap.data(), coords);
    *hofKcal = heatOfFormationKcalDev(eElec, eNuc, nAtoms, ap.data());
  }
  if (hofPm6Kcal != nullptr)
    *hofPm6Kcal = heatOfFormationPm6Kcal(eElec, nAtoms, atoms, coords);
  return true;
}

namespace {

// Frozen-density total energy E = 0.5 tr(P (H + F)) + E_nuc (eV) at the given
// geometry: rebuild H_core and Fock from P at `coords` WITHOUT re-solving the SCF.
double frozenDensityEnergy(int nBasis, int nAtoms, const AtomIntParams* ap,
                           const int* start, const int* norb, const double* coords,
                           const double* P, double* H, double* F) {
  buildCoreHamiltonianDDev(nBasis, nAtoms, ap, start, norb, coords, H);
  buildFockDDev(nBasis, nAtoms, ap, start, norb, coords, H, P, F);
  const int n2 = nBasis * nBasis;
  double e = 0.0;
  for (int i = 0; i < n2; ++i) e += 0.5 * P[i] * (H[i] + F[i]);
  return e + nuclearRepulsionAm1Dev(nAtoms, ap, coords);
}

}  // namespace

bool pm6dGradient(int nAtoms, const int* atoms, const double* coords, double* grad,
                  double* energyEv, int maxIter, double convTol, double step) {
  if (nAtoms <= 0) return false;

  std::vector<AtomIntParams> ap(nAtoms);
  std::vector<int> start(nAtoms), norb(nAtoms);
  int nBasis = 0, nElec = 0;
  for (int a = 0; a < nAtoms; ++a) {
    if (!gatherAtomIntParamsD(atoms[a], ap[a])) return false;
    start[a] = nBasis;
    norb[a] = ap[a].nOrb;
    nBasis += norb[a];
    nElec += pm6ValenceElectrons(atoms[a]);
  }
  if (nBasis == 0 || nElec % 2 != 0) return false;  // unsupported / open shell

  const int n2 = nBasis * nBasis;
  std::vector<double> H(n2), density(n2), eval(nBasis), F(n2), eigA(n2), C(n2), Pnew(n2),
      ecom(n2), diisF(kScfDiisMax * n2), diisE(kScfDiisMax * n2);
  buildCoreHamiltonianDDev(nBasis, nAtoms, ap.data(), start.data(), norb.data(), coords, H.data());

  int conv = 0, niter = 0;
  double eElec = 0.0;
  scfLoopDDev(nBasis, nAtoms, ap.data(), start.data(), norb.data(), coords, H.data(), nElec / 2,
              maxIter, convTol, density.data(), eval.data(), F.data(), eigA.data(), C.data(),
              Pnew.data(), ecom.data(), diisF.data(), diisE.data(), &conv, &niter, &eElec);
  if (!conv) return false;

  if (energyEv != nullptr)
    *energyEv = eElec + nuclearRepulsionAm1Dev(nAtoms, ap.data(), coords);

  // Frozen-density central finite difference, per atom and Cartesian direction.
  std::vector<double> disp(3 * nAtoms);
  for (int a = 0; a < nAtoms; ++a) {
    for (int d = 0; d < 3; ++d) {
      for (int k = 0; k < 3 * nAtoms; ++k) disp[k] = coords[k];
      disp[3 * a + d] = coords[3 * a + d] + step;
      const double ep = frozenDensityEnergy(nBasis, nAtoms, ap.data(), start.data(),
                                            norb.data(), disp.data(), density.data(),
                                            H.data(), F.data());
      disp[3 * a + d] = coords[3 * a + d] - step;
      const double em = frozenDensityEnergy(nBasis, nAtoms, ap.data(), start.data(),
                                            norb.data(), disp.data(), density.data(),
                                            H.data(), F.data());
      grad[3 * a + d] = (ep - em) / (2.0 * step);
    }
  }
  return true;
}

bool pm6dOptimize(int nAtoms, const int* atoms, const double* coordsIn, double* coordsOut,
                  double* energyEv, double* gradRms, int* nIter, int maxIter, double gradTol) {
  if (nAtoms <= 0) return false;
  const int nv = 3 * nAtoms;
  std::vector<double> coords(coordsIn, coordsIn + nv), grad(nv), gNew(nv), trial(nv);
  double E = 0.0;
  if (!pm6dGradient(nAtoms, atoms, coords.data(), grad.data(), &E)) return false;

  const int m = 8;
  std::vector<std::vector<double>> sHist, yHist;
  std::vector<double> rhoHist;

  auto dot = [nv](const double* a, const double* b) {
    double s = 0.0;
    for (int i = 0; i < nv; ++i) s += a[i] * b[i];
    return s;
  };

  int it = 0;
  double gRms = std::sqrt(dot(grad.data(), grad.data()) / nv);
  for (; it < maxIter && gRms >= gradTol; ++it) {
    // L-BFGS two-loop recursion for the search direction.
    const int h = static_cast<int>(sHist.size());
    std::vector<double> q(grad), alpha(h), r(nv), dir(nv);
    for (int k = h - 1; k >= 0; --k) {
      alpha[k] = rhoHist[k] * dot(sHist[k].data(), q.data());
      for (int i = 0; i < nv; ++i) q[i] -= alpha[k] * yHist[k][i];
    }
    const double gamma = h > 0 ? dot(sHist[h - 1].data(), yHist[h - 1].data())
                                     / dot(yHist[h - 1].data(), yHist[h - 1].data())
                               : 0.1;
    for (int i = 0; i < nv; ++i) r[i] = gamma * q[i];
    for (int k = 0; k < h; ++k) {
      const double beta = rhoHist[k] * dot(yHist[k].data(), r.data());
      for (int i = 0; i < nv; ++i) r[i] += (alpha[k] - beta) * sHist[k][i];
    }
    for (int i = 0; i < nv; ++i) dir[i] = -r[i];
    double gd = dot(grad.data(), dir.data());
    double step = 1.0;
    if (gd > 0.0) {  // not a descent direction -> reset to steepest descent
      for (int i = 0; i < nv; ++i) dir[i] = -grad[i];
      gd = -dot(grad.data(), grad.data());
      step = 0.05;
    }

    // Backtracking line search (Armijo sufficient-decrease on the re-solved energy).
    double Enew = 0.0;
    bool ok = false;
    for (int ls = 0; ls < 15; ++ls) {
      for (int i = 0; i < nv; ++i) trial[i] = coords[i] + step * dir[i];
      if (pm6dGradient(nAtoms, atoms, trial.data(), gNew.data(), &Enew)
          && Enew <= E + 1e-4 * step * gd) {
        ok = true;
        break;
      }
      step *= 0.5;
    }
    if (!ok) {  // line search failed -> take a tiny step and continue
      step = 1e-4;
      for (int i = 0; i < nv; ++i) trial[i] = coords[i] + step * dir[i];
      if (!pm6dGradient(nAtoms, atoms, trial.data(), gNew.data(), &Enew)) break;
    }

    // Curvature pair; update the limited-memory history when s.y is positive.
    std::vector<double> sK(nv), yK(nv);
    double sy = 0.0;
    for (int i = 0; i < nv; ++i) {
      sK[i] = step * dir[i];
      yK[i] = gNew[i] - grad[i];
      sy += sK[i] * yK[i];
    }
    coords = trial;
    grad = gNew;
    E = Enew;
    if (sy > 1e-10) {
      sHist.push_back(std::move(sK));
      yHist.push_back(std::move(yK));
      rhoHist.push_back(1.0 / sy);
      if (static_cast<int>(sHist.size()) > m) {
        sHist.erase(sHist.begin());
        yHist.erase(yHist.begin());
        rhoHist.erase(rhoHist.begin());
      }
    }
    gRms = std::sqrt(dot(grad.data(), grad.data()) / nv);
  }

  for (int i = 0; i < nv; ++i) coordsOut[i] = coords[i];
  if (energyEv) *energyEv = E;
  if (gradRms) *gradRms = gRms;
  if (nIter) *nIter = it;
  return gRms < gradTol;
}

}  // namespace semiempirical
}  // namespace nvMolKit
