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

#ifndef NVMOLKIT_SEMIEMPIRICAL_CORE_HAMILTONIAN_H
#define NVMOLKIT_SEMIEMPIRICAL_CORE_HAMILTONIAN_H

#include "device_macros.h"  // AtomIntParams

namespace nvMolKit {
namespace semiempirical {

// Fill an AtomIntParams from the PM6 table for atomic number z (all fields the
// SCF stages need). Returns false if z is unparameterized. Single source of the
// host-side gather, shared by the CPU wrappers and the GPU batch entries.
bool gatherAtomIntParams(int z, AtomIntParams& out);

// Number of sp basis functions for a molecule (1 per H/He, 4 per other sp
// atom). Returns 0 if any atom is unparameterized or carries d-orbitals (qn>=3,
// not yet supported). Orbital order within an atom is [s, px, py, pz].
int spBasisSize(int nAtoms, const int* atoms);

// Build the NDDO/PM6 core Hamiltonian H_core for an sp-only molecule:
//   - diagonal one-center one-electron terms (Uss on s, Upp on p),
//   - two-center resonance H_uv = 1/2 (beta_u + beta_v) S_uv (validated overlap),
//   - electron-core attraction (e1b) summed over the other atoms.
// atoms: Z array (length nAtoms); coords: nAtoms*3 row-major, Angstrom.
// H is written row-major and must hold at least spBasisSize()^2 doubles.
// Returns nBasis (= spBasisSize), or 0 if the molecule is unsupported.
int buildCoreHamiltonianSp(int nAtoms, const int* atoms, const double* coords, double* H);

}  // namespace semiempirical
}  // namespace nvMolKit

#endif  // NVMOLKIT_SEMIEMPIRICAL_CORE_HAMILTONIAN_H
