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

double pm6Eisol(int z) {
  const Pm6ElementParams* p = pm6ParamsForZ(z);
  if (p == nullptr) return 0.0;
  // (ussc, uppc, gssc, gppc, gspc, gp2c, hspc) per element (MOPAC/PYSEQM).
  // Period-3+ analogues share their period-2 occupation pattern.
  double c[7] = {0, 0, 0, 0, 0, 0, 0};
  switch (z) {
    case 1:  c[0] = 1.0; break;                                              // H
    case 6:  c[0] = 2; c[1] = 2; c[2] = 1; c[3] = -0.5; c[4] = 4;  c[5] = 1.5; c[6] = -2; break;  // C
    case 7:                                                                  // N / P
    case 15: c[0] = 2; c[1] = 3; c[2] = 1; c[3] = -1.5; c[4] = 6;  c[5] = 4.5; c[6] = -3; break;
    case 8:                                                                  // O / S
    case 16: c[0] = 2; c[1] = 4; c[2] = 1; c[3] = -0.5; c[4] = 8;  c[5] = 6.5; c[6] = -4; break;
    case 9:                                                                  // F / Cl / Br / I
    case 17:
    case 35:
    case 53: c[0] = 2; c[1] = 5; c[2] = 1; c[3] = 0.5;  c[4] = 10; c[5] = 9.5; c[6] = -5; break;
    default: return 0.0;  // unsupported for HoF
  }
  return c[0] * p->Uss + c[1] * p->Upp + c[2] * p->gss + c[3] * p->gpp
         + c[4] * p->gsp + c[5] * p->gp2 + c[6] * p->hsp;
}

double pm6Eheat(int z) {
  switch (z) {
    case 1:  return 52.102;
    case 6:  return 170.89;
    case 7:  return 113.0;
    case 8:  return 59.559;
    case 9:  return 18.86;
    case 15: return 75.42;
    case 16: return 66.4;
    case 17: return 28.99;
    case 35: return 26.74;
    case 53: return 25.517;
    default: return 0.0;
  }
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
