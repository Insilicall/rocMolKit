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
// Host wrappers over the device-callable two-center integral math in
// two_center_device.h. The math itself lives in the header so the SAME code
// feeds the CPU reference and the HIP kernels; here we only gather the PM6
// parameters for an element pair and forward. See docs/SEMIEMPIRICAL_DESIGN.md.

#include "two_center.h"

#include "core_hamiltonian.h"    // gatherAtomIntParams
#include "two_center_device.h"   // AtomIntParams + *Dev math

namespace nvMolKit {
namespace semiempirical {

namespace {

PairType fromCode(int code) {
  return code == kPairHH ? PairType::HH : (code == kPairXH ? PairType::XH : PairType::XX);
}

}  // namespace

int twoCenterLocal(int zA, int zB, double R_ang, double* ri, double* core, PairType* pairType) {
  AtomIntParams a, b;
  if (!gatherAtomIntParams(zA, a) || !gatherAtomIntParams(zB, b)) return 0;
  int code = 0;
  const int n = twoCenterLocalDev(a, b, R_ang, ri, core, &code);
  if (n > 0) *pairType = fromCode(code);
  return n;
}

bool twoCenterMolecular(int zA, const double coordA[3], int zB, const double coordB[3],
                        double* w, double* e1b, double* e2a) {
  AtomIntParams a, b;
  if (!gatherAtomIntParams(zA, a) || !gatherAtomIntParams(zB, b)) return false;
  return twoCenterMolecularDev(a, coordA, b, coordB, w, e1b, e2a);
}

}  // namespace semiempirical
}  // namespace nvMolKit
