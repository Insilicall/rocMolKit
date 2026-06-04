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
// CPU validation driver for the semi-empirical engine. Reads molecules from
// stdin and prints Mulliken charges + heat of formation, for validate.py to
// compare against the frozen PYSEQM goldens. Build with plain g++ (no HIP):
//
//   g++ -std=c++17 -I../../rocmolkit/src/semiempirical validate_driver.cpp \
//       ../../rocmolkit/src/semiempirical/{scf,core_hamiltonian,two_center,overlap,pm6_params}.cpp
//
// stdin format: per molecule, "<nAtoms>" then nAtoms lines "<Z> <x> <y> <z>".
// Repeat for each molecule; EOF ends. Output per molecule:
//   "<conv> <hof> <q0> <q1> ..."  (conv 1/0; hof kcal/mol; charges per atom)

#include <cstdio>
#include <vector>

#include "core_hamiltonian.h"
#include "scf.h"

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
    std::vector<double> q(nAtoms);
    double hof = 0.0;
    const bool okQ = nvMolKit::semiempirical::mullikenCharges(nAtoms, z.data(), coords.data(), q.data());
    const bool okH = nvMolKit::semiempirical::heatOfFormationKcal(nAtoms, z.data(), coords.data(), &hof);
    std::printf("%d %.6f", (okQ && okH) ? 1 : 0, hof);
    for (int a = 0; a < nAtoms; ++a) std::printf(" %.6f", q[a]);
    std::printf("\n");
    std::fflush(stdout);
  }
  return 0;
}
