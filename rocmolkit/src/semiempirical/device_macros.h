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

#ifndef NVMOLKIT_SEMIEMPIRICAL_DEVICE_MACROS_H
#define NVMOLKIT_SEMIEMPIRICAL_DEVICE_MACROS_H

// NVMOLKIT_HD marks the integral math so the SAME source feeds both the CPU
// reference (plain C++) and the HIP kernels (__host__ __device__), preserving
// the bit-exact validation across both paths.
#if defined(__HIPCC__) || defined(__CUDACC__)
#define NVMOLKIT_HD __host__ __device__
#else
#define NVMOLKIT_HD
#endif

namespace nvMolKit {
namespace semiempirical {

// Per-atom scalars the integral math needs, gathered once (host-side from the
// PM6 table, or into device arrays for a batch). Carrying explicit values keeps
// the device math free of any global parameter-table lookup.
struct AtomIntParams {
  double zetaS, zetaP;             // valence Slater exponents
  double gss, gsp, gpp, gp2, hsp;  // one-center two-electron integrals
  int qn;        // principal quantum number of the valence shell
  int valence;   // core charge (valence electrons)
  int nOrb;      // sp orbital count: 1 (H/He) or 4
};

}  // namespace semiempirical
}  // namespace nvMolKit

#endif  // NVMOLKIT_SEMIEMPIRICAL_DEVICE_MACROS_H
