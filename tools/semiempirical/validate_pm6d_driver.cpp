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
// CPU validation driver for the PM6_D (d-orbital) semi-empirical SCF. Reads
// YH-scope molecules (one d-atom + hydrogens) from stdin and prints the
// converged Mulliken charges, for tools/semiempirical/validate_pm6d_cpp.py to
// compare against data/golden_pm6d_charges.json. Build with plain g++ (no HIP):
//
//   g++ -std=c++17 -O2 -I../../rocmolkit/src/semiempirical \
//       validate_pm6d_driver.cpp \
//       ../../rocmolkit/src/semiempirical/{core_hamiltonian,pm6_params,overlap}.cpp
//
// stdin format: per molecule "<nAtoms>" then nAtoms lines "<Z> <x> <y> <z>".
// Output per molecule: "<conv> <q0> <q1> ...".

#include <cstdio>
#include <vector>

#include "scf_d.h"  // pm6dCharges (public host API)

using namespace nvMolKit::semiempirical;

int main() {
  int nAtoms = 0;
  while (std::scanf("%d", &nAtoms) == 1 && nAtoms > 0) {
    std::vector<int> z(nAtoms);
    std::vector<double> coords(3 * nAtoms);
    for (int a = 0; a < nAtoms; ++a) {
      if (std::scanf("%d %lf %lf %lf", &z[a], &coords[3 * a], &coords[3 * a + 1],
                     &coords[3 * a + 2]) != 4) {
        return 1;
      }
    }

    // Exercise the public host API (scf_d.h) — the same path the library exposes.
    std::vector<double> q(nAtoms);
    double hof = 0.0;
    const bool conv = pm6dCharges(nAtoms, z.data(), coords.data(), q.data(), &hof);
    if (!conv) {
      std::printf("0\n");
      std::fflush(stdout);
      continue;
    }
    std::printf("1 %.6f", hof);
    for (int a = 0; a < nAtoms; ++a) std::printf(" %.6f", q[a]);
    std::printf("\n");
    std::fflush(stdout);
  }
  return 0;
}
