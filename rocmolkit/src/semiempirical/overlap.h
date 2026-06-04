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

#ifndef NVMOLKIT_SEMIEMPIRICAL_OVERLAP_H
#define NVMOLKIT_SEMIEMPIRICAL_OVERLAP_H

namespace nvMolKit {
namespace semiempirical {

// Principal quantum number of the valence shell of element z (periodic-table
// row): H/He=1, Li-Ne=2, Na-Ar=3, K-Kr=4, Rb-Xe=5, else 6. 0 if z<=0.
int principalQn(int z);

// Diatomic STO overlap block between two atoms, in the molecular frame, under
// the NDDO/PM6_SP convention (single-zeta valence Slater orbitals; same-atom
// overlap is the identity by ZDO and is NOT produced here).
//
// Writes an (nA x nB) row-major block into outBlock, where nA/nB are the sp
// orbital counts of zA/zB: 1 for H/He, 4 (s, px, py, pz) otherwise. Orbital
// order is [s, px, py, pz]. coordA/coordB are in Angstrom.
//
// Currently covers valence shells with principal qn in {1, 2} (H, C, N, O, F).
// Heavier atoms (qn >= 3: P, S, Cl, Br, I) write all-zero blocks and are a later
// phase (see docs/SEMIEMPIRICAL_DESIGN.md).
//
// Returns nA*nB on success (the number of elements written, even when that block
// is all zeros for a not-yet-supported qn>=3 shell). Returns 0 only when an
// element is absent from the PM6 parameter table.
int diatomOverlapSp(int zA, const double coordA[3],
                    int zB, const double coordB[3],
                    double* outBlock);

}  // namespace semiempirical
}  // namespace nvMolKit

#endif  // NVMOLKIT_SEMIEMPIRICAL_OVERLAP_H
