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
// principalQn + a host wrapper over the device-callable diatomic overlap in
// overlap_device.h. The overlap math itself lives in the header so the same code
// feeds the CPU reference and the HIP kernels. See docs/SEMIEMPIRICAL_DESIGN.md.

#include "overlap.h"

#include "overlap_device.h"
#include "pm6_params.h"

namespace nvMolKit {
namespace semiempirical {

int principalQn(int z) {
  if (z <= 0) return 0;
  if (z <= 2) return 1;
  if (z <= 10) return 2;
  if (z <= 18) return 3;
  if (z <= 36) return 4;
  if (z <= 54) return 5;
  return 6;
}

namespace {

int spCount(int z) {
  const int n = pm6NumOrbitals(z);
  return (n >= 4) ? 4 : n;
}

// Gather just the fields the overlap needs (exponents, qn, orbital count).
bool gatherOverlap(int z, AtomIntParams& o) {
  const Pm6ElementParams* p = pm6ParamsForZ(z);
  if (p == nullptr) return false;
  o.zetaS = p->zeta_s;
  o.zetaP = p->zeta_p;
  o.qn = principalQn(z);
  o.nOrb = spCount(z);
  return true;
}

}  // namespace

int diatomOverlapSp(int zA, const double coordA[3], int zB, const double coordB[3],
                    double* outBlock) {
  AtomIntParams a, b;
  if (!gatherOverlap(zA, a) || !gatherOverlap(zB, b)) return 0;
  return diatomOverlapSpDev(a, coordA, b, coordB, outBlock);
}

}  // namespace semiempirical
}  // namespace nvMolKit
