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

#include "pm6_params.h"

#include "pm6_params_data.h"  // generated kPm6Params[] / kNumPm6Params

namespace nvMolKit {
namespace semiempirical {

const Pm6ElementParams* pm6ParamsForZ(int z) {
  for (int i = 0; i < kNumPm6Params; ++i) {
    if (kPm6Params[i].Z == z) {
      return &kPm6Params[i];
    }
  }
  return nullptr;
}

int pm6ValenceElectrons(int z) {
  // s+p valence (core) electrons for main-group elements. Transition-metal and
  // f-block atoms are not parameterized for the current PM6 phases -> 0.
  if (z <= 0) {
    return 0;
  }
  if (z <= 2) {           // H, He
    return z;
  }
  if (z <= 10) {          // Li..Ne
    return z - 2;
  }
  if (z <= 18) {          // Na..Ar
    return z - 10;
  }
  if (z == 19 || z == 20) {        // K, Ca
    return z - 18;
  }
  if (z == 30 || z == 48 || z == 80) {  // Zn, Cd, Hg (group-12, d10 core -> sp, 2 s valence)
    return 2;
  }
  if (z == 21) {  // Sc -- active-d TM (group 3): tore = 3 (4s + 3d valence electrons).
    // ENABLED: ScF3 (d0, closed-shell) is bit-exact to MOPAC on CPU and GPU
    // (q_Sc = +1.2454 vs MOPAC +1.2455, dq = 1.6e-4 < 1e-3; the residual is the
    // ev/a0 constant truncation 27.21 vs 27.211386, the same floor the 9 main-group
    // HoF elements carry). The two-center 2e integrals (sp via MOPAC reppd's 2*qq,
    // d via riLocalYX) were already bit-exact; the bug was the ELECTRON-CORE
    // attraction: MOPAC's mndod spcore gives the CORE atom a special additive
    // radius po(9) = pocord (AtomIntParams::rhoCore) for the e1b monopole, which the
    // engine had ignored. pocord biases the H_core of every ligand orbital attracted
    // to the Sc core; restoring it (coreAttractionE1bDev in core_hamiltonian_d) drops
    // ||[F,P]|| at MOPAC's density from 1.26 to 3e-3 and fixes the charge. pocord is
    // defined only for Sc/Fe/Ni in PM6 (poc_6), so the fix is a no-op for every
    // main-group d-atom. See validate_pm6d_tm.py.
    return 3;
  }
  if (z >= 22 && z <= 24) {  // Ti, V, Cr -- active-d TM, ENABLED (d0 closed-shell).
    // tore = ios + iop + iod = group number = Z - 18 (MOPAC parameters_C reference
    // config Ti s2d2, V s2d3, Cr s2d4 -> tore 4/5/6). The full PM6 parameter set
    // (Uss/Upp/Udd, betas, F0SD/G2SD, tail exps) is in pm6_params_data.h; the
    // one-center d W is baked by gen_tm_onecenter_w.py and the d charge separations
    // by gen_tm_chargesep.py. The d0 closed-shell oxidation states are BIT-EXACT to
    // MOPAC on the RHF (and GPU batch) path: TiF4 q_Ti=+1.5816 vs MOPAC +1.5817
    // (dq=1.6e-4), TiCl4 dq=3.7e-4, VF5 dq=2.1e-4, CrF6 dq=4.7e-4 (all < 1e-3, the
    // ev/a0 constant-truncation floor). Because Ti(IV)/V(V)/Cr(VI) are d0, the
    // metal carries no d electrons, so the two-center d-block 2e integrals never
    // fire and the SCF reduces to the validated sp + ligand-d path. rhoCore (MOPAC
    // poc_) is undefined for Ti/V/Cr (rhoCore=0 -> regular e1b monopole, which is
    // what MOPAC's mndod spcore does for them). See validate_pm6d_tm.py.
    return z - 18;
  }
  // Mn..Cu (25..29) and the heavier active-d TM (Y-Ag, Hf-Au) stay DISABLED. Their
  // params, one-center d W and d charge separations ARE baked (pm6_params_data.h +
  // gen_tm_*), and the one-center d-block reproduces MOPAC (Cu+ d10 single-ion HoF
  // 272.99 vs MOPAC 273.53, the EISOL-rounding floor). What is NOT yet bit-exact is
  // the TWO-CENTER d-block when the metal d-shell is POPULATED: e.g. CuF (Cu d10)
  // converges to Cu s-pop 0.18 / q=+0.84 instead of MOPAC's s-pop 0.58 / q=+0.50
  // (our state is ~3 eV lower, so it is a genuine metal-d<->ligand two-center 2e /
  // electron-core discrepancy, NOT an SCF local minimum). d0 metals (Ti/V/Cr above)
  // are immune because they have no d electrons; the populated-d metals need the
  // two-center d path resolved before they can be enabled bit-exact. They are left
  // disabled to avoid silent-wrong output.
  if (z >= 31 && z <= 36) {        // Ga..Kr
    return z - 28;
  }
  if (z == 37 || z == 38) {        // Rb, Sr
    return z - 36;
  }
  if (z >= 49 && z <= 54) {        // In..Xe (Br=35, I=53 -> 7)
    return z - 46;
  }
  return 0;  // transition metals / unsupported in current phases
}

double pm6Eisol(int z) {
  const Pm6ElementParams* p = pm6ParamsForZ(z);
  if (p == nullptr) return 0.0;
  // (ussc, uppc, gssc, gppc, gspc, gp2c, hspc) per element (MOPAC/PYSEQM).
  // Period-3+ analogues share their period-2 occupation pattern.
  double c[7] = {0, 0, 0, 0, 0, 0, 0};
  switch (z) {
    case 1:  c[0] = 1.0; break;                                              // H
    case 13: c[0] = 2; c[1] = 1; c[2] = 1; c[3] = 0;    c[4] = 2;  c[5] = 0;   c[6] = 0; break;   // Al (s^2 p^1)
    case 6:                                                                  // C / Si
    case 14: c[0] = 2; c[1] = 2; c[2] = 1; c[3] = -0.5; c[4] = 4;  c[5] = 1.5; c[6] = -2; break;  // (s^2 p^2)
    case 7:                                                                  // N / P
    case 15: c[0] = 2; c[1] = 3; c[2] = 1; c[3] = -1.5; c[4] = 6;  c[5] = 4.5; c[6] = -3; break;
    case 8:                                                                  // O / S
    case 16: c[0] = 2; c[1] = 4; c[2] = 1; c[3] = -0.5; c[4] = 8;  c[5] = 6.5; c[6] = -4; break;
    case 9:                                                                  // F / Cl / Br / I
    case 17:
    case 35:
    case 53: c[0] = 2; c[1] = 5; c[2] = 1; c[3] = 0.5;  c[4] = 10; c[5] = 9.5; c[6] = -5; break;
    default: return 0.0;  // unsupported for HoF
  }
  return c[0] * p->Uss + c[1] * p->Upp + c[2] * p->gss + c[3] * p->gpp
         + c[4] * p->gsp + c[5] * p->gp2 + c[6] * p->hsp;
}

double pm6Eheat(int z) {
  switch (z) {
    case 1:  return 52.102;
    case 13: return 78.800;   // Al (canonical PM6 atomic heat of formation)
    case 14: return 108.390;  // Si
    case 6:  return 170.89;
    case 7:  return 113.0;
    case 8:  return 59.559;
    case 9:  return 18.86;
    case 15: return 75.42;
    case 16: return 66.4;
    case 17: return 28.99;
    case 35: return 26.74;
    case 53: return 25.517;
    default: return 0.0;
  }
}

int pm6NumOrbitals(int z) {
  const Pm6ElementParams* p = pm6ParamsForZ(z);
  if (p == nullptr || pm6ValenceElectrons(z) == 0) {
    // Not parameterized, or a transition-metal/f-block atom outside the current
    // main-group phases. Stay consistent with pm6ValenceElectrons().
    return 0;
  }
  if (z <= 2) {                 // H, He: s only
    return 1;
  }
  if (p->zeta_d != 0.0) {       // PM6_D d-orbital atoms (P, S, Cl, Br, I, ...)
    return 9;                   // s + 3 p + 5 d
  }
  return 4;                     // s + 3 p
}

}  // namespace semiempirical
}  // namespace nvMolKit
