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

#ifndef NVMOLKIT_SEMIEMPIRICAL_PM6_PARAMS_H
#define NVMOLKIT_SEMIEMPIRICAL_PM6_PARAMS_H

namespace nvMolKit {
namespace semiempirical {

// PM6 element parameters (Stewart PM6 / MOPAC public set). One row per element,
// indexed by atomic number Z. Energies in eV, exponents in atomic units,
// distances/alpha in their MOPAC conventions. Fields mirror the columns of
// tools/semiempirical/data/pm6_params_mopac.csv; the constant table in
// pm6_params_data.h is generated from that CSV by
// tools/semiempirical/gen_pm6_params_header.py — do not hand-edit the table.
struct Pm6ElementParams {
  int Z;  // atomic number

  double Uss, Upp, Udd;            // one-center one-electron core integrals
  double zeta_s, zeta_p, zeta_d;   // Slater orbital exponents (valence)
  double beta_s, beta_p, beta_d;   // resonance (one-electron two-center) params
  double sOrbExpTail, pOrbExpTail, dOrbExpTail;  // STO inner-shell tail exponents
  double gss, gsp, gpp, gp2, hsp;  // one-center two-electron integrals
  double F0SD, G2SD;               // s-d Slater-Condon (d-orbital methods)
  double rhoCore;                  // additive core multipole radius
  double alpha;                    // core-core exponent
  double EISOL;                    // isolated-atom electronic energy (eV)
  double gaussianK[4], gaussianL[4], gaussianM[4];  // AM1/PM3-style core-core Gaussians
  double zs, zp, zd;               // core-core screened charges
};

// Returns the PM6 parameters for atomic number z, or nullptr if z is not
// parameterized in the table.
const Pm6ElementParams* pm6ParamsForZ(int z);

// Number of valence (core) electrons for z — the core charge used in the
// electron-core attraction and in Mulliken charges. Returns 0 if unknown.
int pm6ValenceElectrons(int z);

// Number of valence basis orbitals for z under PM6_D: 1 (H/He), 4 (sp, no d),
// or 9 (sp+d on P, S, Cl, Br, I and heavier). Returns 0 if unknown.
int pm6NumOrbitals(int z);

}  // namespace semiempirical
}  // namespace nvMolKit

#endif  // NVMOLKIT_SEMIEMPIRICAL_PM6_PARAMS_H
