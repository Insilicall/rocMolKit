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
// NDDO/PM6 core Hamiltonian assembly (sp basis), ported from the PYSEQM
// reference _build_core_hamiltonian. Combines the validated Stage 2 overlap
// and Stage 4 two-center integrals. See docs/SEMIEMPIRICAL_DESIGN.md.

#include "core_hamiltonian.h"

#include <vector>

#include "overlap.h"
#include "pm6_params.h"
#include "two_center.h"

namespace nvMolKit {
namespace semiempirical {

namespace {

int spCount(int z) {
  const int n = pm6NumOrbitals(z);
  return (n >= 4) ? 4 : n;
}

}  // namespace

int spBasisSize(int nAtoms, const int* atoms) {
  int n = 0;
  for (int a = 0; a < nAtoms; ++a) {
    if (pm6ParamsForZ(atoms[a]) == nullptr) return 0;
    if (principalQn(atoms[a]) >= 3) return 0;  // d-bearing: not yet supported
    const int c = spCount(atoms[a]);
    if (c == 0) return 0;
    n += c;
  }
  return n;
}

int buildCoreHamiltonianSp(int nAtoms, const int* atoms, const double* coords, double* H) {
  const int nBasis = spBasisSize(nAtoms, atoms);
  if (nBasis == 0) return 0;

  std::vector<int> start(nAtoms), norb(nAtoms);
  for (int a = 0, off = 0; a < nAtoms; ++a) {
    start[a] = off;
    norb[a] = spCount(atoms[a]);
    off += norb[a];
  }

  for (int i = 0; i < nBasis * nBasis; ++i) H[i] = 0.0;

  // Diagonal one-center one-electron terms: Uss on s, Upp on the three p.
  for (int a = 0; a < nAtoms; ++a) {
    const Pm6ElementParams* p = pm6ParamsForZ(atoms[a]);
    for (int o = 0; o < norb[a]; ++o) {
      const int mu = start[a] + o;
      H[mu * nBasis + mu] = (o == 0) ? p->Uss : p->Upp;
    }
  }

  // Two-center resonance: H_uv = 1/2 (beta_u + beta_v) S_uv, atom pairs i<j.
  for (int i = 0; i < nAtoms; ++i) {
    const Pm6ElementParams* pi = pm6ParamsForZ(atoms[i]);
    for (int j = i + 1; j < nAtoms; ++j) {
      const Pm6ElementParams* pj = pm6ParamsForZ(atoms[j]);
      double blk[16];
      diatomOverlapSp(atoms[i], &coords[3 * i], atoms[j], &coords[3 * j], blk);
      for (int mo = 0; mo < norb[i]; ++mo) {
        const double bmu = (mo == 0) ? pi->beta_s : pi->beta_p;
        for (int no = 0; no < norb[j]; ++no) {
          const double bnu = (no == 0) ? pj->beta_s : pj->beta_p;
          const double h = 0.5 * (bmu + bnu) * blk[mo * norb[j] + no];
          const int mu = start[i] + mo;
          const int nu = start[j] + no;
          H[mu * nBasis + nu] = h;
          H[nu * nBasis + mu] = h;
        }
      }
    }
  }

  // Electron-core attraction: for each ordered pair (i,j), e1b is the electron
  // on atom i attracted to core j; accumulate into atom i's one-center block.
  for (int i = 0; i < nAtoms; ++i) {
    for (int j = 0; j < nAtoms; ++j) {
      if (i == j) continue;
      double w[256], e1b[16], e2a[16];
      if (!twoCenterMolecular(atoms[i], &coords[3 * i], atoms[j], &coords[3 * j], w, e1b, e2a)) {
        return 0;
      }
      for (int mo = 0; mo < norb[i]; ++mo) {
        for (int no = 0; no < norb[i]; ++no) {
          H[(start[i] + mo) * nBasis + (start[i] + no)] += e1b[mo * 4 + no];
        }
      }
    }
  }
  return nBasis;
}

}  // namespace semiempirical
}  // namespace nvMolKit
