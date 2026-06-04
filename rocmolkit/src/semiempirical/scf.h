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

#ifndef NVMOLKIT_SEMIEMPIRICAL_SCF_H
#define NVMOLKIT_SEMIEMPIRICAL_SCF_H

namespace nvMolKit {
namespace semiempirical {

struct ScfResult {
  int nBasis = 0;
  int nIter = 0;
  bool converged = false;
  double electronicEv = 0.0;  // 0.5 * sum(P .* (H + F))
};

// Closed-shell NDDO/PM6 self-consistent field for an sp-only molecule.
// Builds H_core (stages 2-4), then iterates F = H + G(P) with DIIS + adaptive
// density mixing until the RMS density change falls below convTol. The basis is
// orthonormal under ZDO, so each cycle is a plain symmetric eigenproblem.
//
// atoms: Z array (length nAtoms); coords: nAtoms*3 row-major (Angstrom).
// Writes the density (nBasis*nBasis row-major) and final orbital eigenvalues
// (nBasis, ascending, eV), fills *out, and returns nBasis. Returns 0 — leaving
// density/eigenvalues/out untouched — if the molecule is unsupported (d-bearing
// or unparameterized atom) or has an odd electron count (open shell).
//
// IMPORTANT: a non-zero return does NOT guarantee convergence. The density is
// only trustworthy when out->converged is true; callers MUST check it (a result
// that hit maxIter, or a failed diagonalization, returns nBasis with
// out->converged == false).
int scfSp(int nAtoms, const int* atoms, const double* coords,
          double* density, double* eigenvalues, ScfResult* out,
          int maxIter = 200, double convTol = 1e-8);

// Mulliken atomic charges q_A = Z_A^core - sum_{mu in A} P[mu,mu], from an SCF
// density. atoms/coords as in scfSp; writes nAtoms charges into q. Returns true
// on success (false if unsupported or the SCF did not converge). Convenience
// wrapper that runs scfSp internally.
bool mullikenCharges(int nAtoms, const int* atoms, const double* coords, double* q);

// PM6_D core-core (nuclear) repulsion energy in eV (AM1-style: (ss|ss) Coulomb
// with the per-element exponential + Gaussian corrections and the N-H/O-H
// special case). atoms/coords as in scfSp.
double nuclearRepulsionEv(int nAtoms, const int* atoms, const double* coords);

// Heat of formation in kcal/mol: (E_elec + E_nuc - sum eisol) * eV->kcal + sum
// eheat. Runs the SCF internally. Writes the result into *hofKcal and returns
// true; returns false if unsupported or the SCF did not converge.
bool heatOfFormationKcal(int nAtoms, const int* atoms, const double* coords, double* hofKcal);

}  // namespace semiempirical
}  // namespace nvMolKit

#endif  // NVMOLKIT_SEMIEMPIRICAL_SCF_H
