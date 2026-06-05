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

}  // namespace semiempirical
}  // namespace nvMolKit

#endif  // NVMOLKIT_SEMIEMPIRICAL_SCF_D_H
