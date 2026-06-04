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

#ifndef NVMOLKIT_SEMIEMPIRICAL_TWO_CENTER_H
#define NVMOLKIT_SEMIEMPIRICAL_TWO_CENTER_H

namespace nvMolKit {
namespace semiempirical {

// Pair type for the local-frame two-electron two-center integrals.
enum class PairType { HH, XH, XX };

// Local-frame two-electron two-center integrals (Dewar-Thiel multipole model),
// in eV, for an sp atom pair at distance R_ang (Angstrom). For the XH case the
// heavy atom MUST be zA and the hydrogen zB (the molecular-frame layer handles
// the H-heavy ordering by swapping). Ported bit-exact from the PYSEQM PM6
// two_center_integrals.
//
// Writes the integral array into ri and the core-attraction terms into core:
//   HH: 1 ri, 2 core    XH: 4 ri, 5 core    XX: 22 ri, 8 core
// Returns the number of ri written, or 0 if an element is unparameterized.
// *pairType is set on success.
int twoCenterLocal(int zA, int zB, double R_ang, double* ri, double* core,
                   PairType* pairType);

// Two-electron two-center integrals rotated into the molecular frame for an
// sp atom pair A (zA at coordA) and B (zB at coordB), coordinates in Angstrom.
// Fills:
//   w   : flattened 4x4x4x4 tensor (index ((muA*4+nuA)*4+lamB)*4+sigB), eV.
//   e1b : 4x4, electron (mu,nu on A) attracted to core B (eV).
//   e2a : 4x4, electron (mu,nu on B) attracted to core A (eV).
// nA/nB sp orbital counts are taken from the elements (1 for H, 4 otherwise);
// unused rows/cols are left zero. Returns false if an element is unsupported.
bool twoCenterMolecular(int zA, const double coordA[3],
                        int zB, const double coordB[3],
                        double* w, double* e1b, double* e2a);

}  // namespace semiempirical
}  // namespace nvMolKit

#endif  // NVMOLKIT_SEMIEMPIRICAL_TWO_CENTER_H
