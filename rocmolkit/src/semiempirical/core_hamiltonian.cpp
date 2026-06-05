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
// Host wrappers over the device-callable core Hamiltonian assembly in
// core_hamiltonian_device.h. The math lives in the header so the same code
// builds H_core on the CPU reference and on the GPU. See
// docs/SEMIEMPIRICAL_DESIGN.md.

#include "core_hamiltonian.h"

#include <vector>

#include "core_hamiltonian_device.h"
#include "overlap.h"  // principalQn
#include "pm6_params.h"
#include "two_center_device.h"  // AtomIntParams

namespace nvMolKit {
namespace semiempirical {

namespace {

int spCount(int z) {
  const int n = pm6NumOrbitals(z);
  return (n >= 4) ? 4 : n;
}

}  // namespace

// Fill an AtomIntParams from the PM6 table (all fields the SCF stages may need).
// Shared by the CPU wrappers and the GPU host-side gathers.
bool gatherAtomIntParams(int z, AtomIntParams& o) {
  const Pm6ElementParams* p = pm6ParamsForZ(z);
  if (p == nullptr) return false;
  o.zetaS = p->zeta_s;
  o.zetaP = p->zeta_p;
  o.zetaD = p->zeta_d;
  o.gss = p->gss;
  o.gsp = p->gsp;
  o.gpp = p->gpp;
  o.gp2 = p->gp2;
  o.hsp = p->hsp;
  o.uss = p->Uss;
  o.upp = p->Upp;
  o.udd = p->Udd;
  o.betaS = p->beta_s;
  o.betaP = p->beta_p;
  o.betaD = p->beta_d;
  o.alpha = p->alpha;
  for (int k = 0; k < 4; ++k) {
    o.gaussK[k] = p->gaussianK[k];
    o.gaussL[k] = p->gaussianL[k];
    o.gaussM[k] = p->gaussianM[k];
  }
  o.eisol = pm6Eisol(z);
  o.eheat = pm6Eheat(z);
  o.z = z;
  o.qn = principalQn(z);
  o.valence = pm6ValenceElectrons(z);
  o.nOrb = spCount(z);
  return true;
}

// PM6_D gather: identical to gatherAtomIntParams but nOrb is the full valence
// basis size (9 for d-bearing atoms), so the d-orbital stages see the d shell.
bool gatherAtomIntParamsD(int z, AtomIntParams& o) {
  if (!gatherAtomIntParams(z, o)) return false;
  o.nOrb = pm6NumOrbitals(z);
  return o.nOrb > 0;
}

int spBasisSize(int nAtoms, const int* atoms) {
  int n = 0;
  for (int a = 0; a < nAtoms; ++a) {
    if (pm6ParamsForZ(atoms[a]) == nullptr) return 0;
    // PM6_SP treats P/S/Cl (qn=3) as sp. qn>=4 (Br, I) needs sp overlap formulas
    // not yet ported; d-orbitals (PM6_D) are a later phase.
    if (principalQn(atoms[a]) >= 4) return 0;
    const int c = spCount(atoms[a]);
    if (c == 0) return 0;
    n += c;
  }
  return n;
}

int buildCoreHamiltonianSp(int nAtoms, const int* atoms, const double* coords, double* H) {
  const int nBasis = spBasisSize(nAtoms, atoms);
  if (nBasis == 0) return 0;

  std::vector<AtomIntParams> ap(nAtoms);
  std::vector<int> start(nAtoms), norb(nAtoms);
  for (int a = 0, off = 0; a < nAtoms; ++a) {
    if (!gatherAtomIntParams(atoms[a], ap[a])) return 0;
    start[a] = off;
    norb[a] = ap[a].nOrb;
    off += norb[a];
  }
  buildCoreHamiltonianDev(nBasis, nAtoms, ap.data(), start.data(), norb.data(), coords, H);
  return nBasis;
}

}  // namespace semiempirical
}  // namespace nvMolKit
