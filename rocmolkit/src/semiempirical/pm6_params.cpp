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

#include "pm6_params.h"

#include "pm6_params_data.h"  // generated kPm6Params[] / kNumPm6Params

namespace nvMolKit {
namespace semiempirical {

const Pm6ElementParams* pm6ParamsForZ(int z) {
  for (int i = 0; i < kNumPm6Params; ++i) {
    if (kPm6Params[i].Z == z) {
      return &kPm6Params[i];
    }
  }
  return nullptr;
}

int pm6ValenceElectrons(int z) {
  // s+p valence (core) electrons for main-group elements. Transition-metal and
  // f-block atoms are not parameterized for the current PM6 phases -> 0.
  if (z <= 0) {
    return 0;
  }
  if (z <= 2) {           // H, He
    return z;
  }
  if (z <= 10) {          // Li..Ne
    return z - 2;
  }
  if (z <= 18) {          // Na..Ar
    return z - 10;
  }
  if (z == 19 || z == 20) {        // K, Ca
    return z - 18;
  }
  if (z >= 31 && z <= 36) {        // Ga..Kr
    return z - 28;
  }
  if (z == 37 || z == 38) {        // Rb, Sr
    return z - 36;
  }
  if (z >= 49 && z <= 54) {        // In..Xe (Br=35, I=53 -> 7)
    return z - 46;
  }
  return 0;  // transition metals / unsupported in current phases
}

int pm6NumOrbitals(int z) {
  const Pm6ElementParams* p = pm6ParamsForZ(z);
  if (p == nullptr || pm6ValenceElectrons(z) == 0) {
    // Not parameterized, or a transition-metal/f-block atom outside the current
    // main-group phases. Stay consistent with pm6ValenceElectrons().
    return 0;
  }
  if (z <= 2) {                 // H, He: s only
    return 1;
  }
  if (p->zeta_d != 0.0) {       // PM6_D d-orbital atoms (P, S, Cl, Br, I, ...)
    return 9;                   // s + 3 p + 5 d
  }
  return 4;                     // s + 3 p
}

}  // namespace semiempirical
}  // namespace nvMolKit
