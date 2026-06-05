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
// CPU validation driver for the PM6_D frozen-density gradient. Reads molecules
// from stdin and prints the energy (eV) + per-atom gradient (eV/Angstrom), for
// tools/semiempirical/validate_pm6d_gradient.py to compare against the oracle
// anal_grad.analytical_gradient. SCF settings match the oracle (max_iter=200,
// conv_tol=1e-8, step=1e-5). Build with plain g++ (no HIP):
//
//   g++ -std=c++17 -O2 -I../../rocmolkit/src/semiempirical \
//       validate_pm6d_gradient_driver.cpp \
//       ../../rocmolkit/src/semiempirical/{core_hamiltonian,pm6_params,overlap,scf_d}.cpp
//
// stdin format: per molecule "<nAtoms>" then nAtoms lines "<Z> <x> <y> <z>".
// Output per molecule: "<conv> <E_eV> <g0x> <g0y> <g0z> <g1x> ...".

#include <cstdio>
#include <vector>

#include "scf_d.h"  // pm6dGradient (public host API)

using namespace nvMolKit::semiempirical;

int main() {
  int nAtoms = 0;
  while (std::scanf("%d", &nAtoms) == 1 && nAtoms > 0) {
    std::vector<int> z(nAtoms);
    std::vector<double> coords(3 * nAtoms);
    for (int a = 0; a < nAtoms; ++a) {
      std::scanf("%d %lf %lf %lf", &z[a], &coords[3 * a], &coords[3 * a + 1],
                 &coords[3 * a + 2]);
    }
    std::vector<double> grad(3 * nAtoms);
    double E = 0.0;
    // Match the oracle's analytical_gradient SCF settings (max_iter=200,
    // conv_tol=1e-8) so the converged frozen density coincides bit-exact.
    const bool ok = pm6dGradient(nAtoms, z.data(), coords.data(), grad.data(), &E,
                                 200, 1e-8, 1e-5);
    if (!ok) {
      std::printf("0\n");
    } else {
      std::printf("1 %.10f", E);
      for (int k = 0; k < 3 * nAtoms; ++k) std::printf(" %.10f", grad[k]);
      std::printf("\n");
    }
  }
  return 0;
}
