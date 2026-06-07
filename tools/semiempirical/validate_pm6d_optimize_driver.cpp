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
// CPU validation driver for the PM6_D geometry optimizer. Reads molecules from
// stdin and prints, per molecule, the convergence flag, iteration count, final
// energy (eV), RMS gradient, and optimized coordinates, for
// tools/semiempirical/validate_pm6d_optimize.py to compare against the oracle
// nddo_optimize. Build with plain g++ (no HIP):
//
//   g++ -std=c++17 -O2 -I../../rocmolkit/src/semiempirical \
//       validate_pm6d_optimize_driver.cpp \
//       ../../rocmolkit/src/semiempirical/{core_hamiltonian,pm6_params,overlap,scf_d}.cpp
//
// stdin format: per molecule "<nAtoms>" then nAtoms lines "<Z> <x> <y> <z>".
// Output per molecule: "<conv> <nIter> <E_eV> <gRms> <x0> <y0> <z0> <x1> ...".

#include <cstdio>
#include <vector>

#include "scf_d.h"  // pm6dOptimize

using namespace nvMolKit::semiempirical;

int main() {
  int nAtoms = 0;
  while (std::scanf("%d", &nAtoms) == 1 && nAtoms > 0) {
    std::vector<int> z(nAtoms);
    std::vector<double> coords(3 * nAtoms);
    for (int a = 0; a < nAtoms; ++a)
      std::scanf("%d %lf %lf %lf", &z[a], &coords[3 * a], &coords[3 * a + 1], &coords[3 * a + 2]);

    std::vector<double> out(3 * nAtoms);
    double E = 0.0, gRms = 0.0;
    int nIter = 0;
    const bool conv = pm6dOptimize(nAtoms, z.data(), coords.data(), out.data(), &E, &gRms, &nIter);
    std::printf("%d %d %.10f %.10f", conv ? 1 : 0, nIter, E, gRms);
    for (int k = 0; k < 3 * nAtoms; ++k) std::printf(" %.10f", out[k]);
    std::printf("\n");
  }
  return 0;
}
