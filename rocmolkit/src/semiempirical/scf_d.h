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
// Public host API for the PM6_D (d-orbital) NDDO semi-empirical engine. The CPU
// path here mirrors scf.h's sp API; the batched GPU path is scf_d_kernels.h
// (scfBatchDGpu). All integral math is the shared __host__ __device__ code, so
// CPU and GPU results are bit-exact. d-bearing atoms (P/S/Cl/Br/I) use the full
// 9-orbital basis; H/He/sp atoms keep their 1/4-orbital basis.

#ifndef NVMOLKIT_SEMIEMPIRICAL_SCF_D_H
#define NVMOLKIT_SEMIEMPIRICAL_SCF_D_H

namespace nvMolKit {
namespace semiempirical {

// Run the closed-shell PM6_D SCF for one molecule on the CPU and return the
// converged Mulliken charges (length nAtoms) and, if hofKcal != nullptr, the heat
// of formation (kcal/mol). atoms: Z array (length nAtoms); coords: nAtoms*3
// row-major, Angstrom. Returns false on an unsupported element, an odd-electron
// (open-shell) molecule, or non-convergence.
bool pm6dCharges(int nAtoms, const int* atoms, const double* coords, double* q,
                 double* hofKcal = nullptr, int maxIter = 800, double convTol = 1e-10);

// PM6_D energy gradient dE/dR for one molecule, via the frozen-density
// (Hellmann-Feynman) method: solve the SCF once for the density P, then take the
// central finite difference of E = 0.5 tr(P (H + F)) + E_nuc rebuilt at each
// displaced geometry WITHOUT re-solving (P is variationally converged, so dE/dP
// vanishes). One SCF + 6*nAtoms integral passes, vs 6N+1 SCF re-solves for a
// plain numerical gradient. grad is nAtoms*3 row-major in eV/Angstrom; if
// energyEv != nullptr it receives the converged electronic+nuclear energy (eV).
// step is the displacement (Angstrom). Returns false on unsupported element,
// open shell, or non-convergence. Mirrors the oracle's analytical_gradient.
bool pm6dGradient(int nAtoms, const int* atoms, const double* coords, double* grad,
                  double* energyEv = nullptr, int maxIter = 800, double convTol = 1e-10,
                  double step = 1e-5);

// PM6_D geometry optimization (L-BFGS with backtracking/Armijo line search) using
// the frozen-density gradient. coordsIn: nAtoms*3 row-major (Angstrom) start;
// coordsOut receives the optimized geometry. Converges when the RMS gradient falls
// below gradTol (eV/Angstrom). If non-null, energyEv / gradRms / nIter receive the
// final energy (eV), RMS gradient, and iteration count. Returns true if the RMS
// gradient reached gradTol within maxIter (else false with the best geometry so
// far in coordsOut). Mirrors the oracle's nddo_optimize.
bool pm6dOptimize(int nAtoms, const int* atoms, const double* coordsIn, double* coordsOut,
                  double* energyEv = nullptr, double* gradRms = nullptr, int* nIter = nullptr,
                  int maxIter = 50, double gradTol = 0.005);

}  // namespace semiempirical
}  // namespace nvMolKit

#endif  // NVMOLKIT_SEMIEMPIRICAL_SCF_D_H
