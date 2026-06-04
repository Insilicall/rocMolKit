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

#ifndef NVMOLKIT_SEMIEMPIRICAL_SCF_KERNELS_H
#define NVMOLKIT_SEMIEMPIRICAL_SCF_KERNELS_H

namespace nvMolKit {
namespace semiempirical {

// Batched on-device closed-shell NDDO/PM6 SCF. Runs one molecule per GPU thread
// to self-consistency and returns Mulliken charges. Inputs are concatenated:
//   molNAtoms/molNBasis : per-molecule counts (length nMol)
//   atomsAll            : atomic numbers, length sum(nAtoms)
//   coordsAll           : 3*sum(nAtoms) (Angstrom)
//   chargesAll (out)    : per-atom Mulliken charge, length sum(nAtoms)
//   convergedAll (out)  : 1/0 per molecule (length nMol)
// The core Hamiltonian is built host-side and uploaded; the SCF loop (Fock,
// diagonalization, density, mixing) runs entirely on the device. Returns false
// on an unsupported element or odd-electron (open-shell) molecule.
bool scfBatchGpu(int nMol, const int* molNAtoms, const int* molNBasis,
                 const int* atomsAll, const double* coordsAll,
                 double* chargesAll, int* convergedAll,
                 int maxIter = 500, double convTol = 1e-8);

}  // namespace semiempirical
}  // namespace nvMolKit

#endif  // NVMOLKIT_SEMIEMPIRICAL_SCF_KERNELS_H
