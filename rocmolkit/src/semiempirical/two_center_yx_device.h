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
// Device-callable YX two-center two-electron tensor (d-atom A + sp atom B):
// builds the 9x9x4x4 molecular (mu nu_A | lam sig_B) integral tensor that the
// PM6_D Fock J/K consume. Assembles the 45x45 local-frame block from the
// transpiled d reduced integrals (riLocalYX, d-pairs) and the sp molecular w
// (twoCenterMolecularDev, sp 10x10), then applies the d-orbital rotation. Mirrors
// PYSEQM's two_elec_two_center_int YX path; validated bit-exact (validate_yx_*).

#ifndef NVMOLKIT_SEMIEMPIRICAL_TWO_CENTER_YX_DEVICE_H
#define NVMOLKIT_SEMIEMPIRICAL_TWO_CENTER_YX_DEVICE_H

#include <cmath>

#include "d_localframe_generated.h"
#include "d_rotation_device.h"
#include "device_macros.h"
#include "two_center_d_device.h"  // dChargeSeparations
#include "two_center_device.h"    // twoCenterMolecularDev, computeMultipoleParamsDev, wIdxDev

namespace nvMolKit {
namespace semiempirical {

// Doubles of caller scratch used by yyWMolecular / yxWMolecular when the off-stack
// path is taken (GPU). Layout (offsets into `scr`):
//   ri   [0     .. 2025)   local reduced integrals (YY 2025, YX uses first 450)
//   WW   [2025  .. 4050)   local-frame 45x45 block
//   wmol [4050  .. 6075)   molecular-frame 45x45 block
//   mat  [6075  .. 6750)   15x45 pair-rotation matrix
//   rot  [6750  .. 12825)  3x 45x45 scratch for rotate2Center2ElectronD
enum {
  kYWScrRi = 0,
  kYWScrWW = 2025,
  kYWScrWmol = 4050,
  kYWScrMat = 6075,
  kYWScrRot = 6750,
  kYWScrDoubles = 12825,
};

// Pick a temporary's storage: on the DEVICE always use the provided `scr` (the
// stack array is NOT declared, so it costs zero per-lane private memory -- this is
// what frees up occupancy); on the HOST fall back to a stack array when `scr` is
// null (CPU reference path). `nm` is the pointer name, `off` the scratch offset,
// `sz` the stack-array size.
#ifdef __HIP_DEVICE_COMPILE__
#define NVMOLKIT_YWSCR(nm, off, sz) double* nm = scr + (off)
#else
#define NVMOLKIT_YWSCR(nm, off, sz) \
  double nm##Stk[sz];               \
  double* nm = scr ? scr + (off) : nm##Stk
#endif

// Build the molecular YX tensor W[9*9*4*4] (row-major mu,nu,lam,sig) for the pair
// (pA has d, pB is sp). Returns false if pA has no baked d parameters.
//
// `scr` (optional, >=kYWScrDoubles doubles) holds the large 45x45 temporaries
// off-stack (GPU path); null keeps them on the stack (CPU reference). The
// arithmetic is byte-identical regardless -- only the storage class of the
// temporaries differs, so charges stay bit-exact GPU-vs-CPU and vs MOPAC.
NVMOLKIT_HD inline bool yxWMolecular(const AtomIntParams& pA, const double coordA[3],
                                     const AtomIntParams& pB, const double coordB[3], double* W,
                                     double* scr = nullptr) {
  using namespace detail;
  double dp, ds, dd, rho3, rho4, rho5, rho6;
  if (!dChargeSeparationsTwoCenter(pA.z, dp, ds, dd, rho3, rho4, rho5, rho6)) return false;

  const double Rvec[3] = {coordB[0] - coordA[0], coordB[1] - coordA[1], coordB[2] - coordA[2]};
  const double R = std::sqrt(Rvec[0] * Rvec[0] + Rvec[1] * Rvec[1] + Rvec[2] * Rvec[2]);
  const double r0 = R * kSemiAngToBohr;

  // sp multipole params (da, qa, rho0/1/2) for A and B.
  double daA, qaA, rho0A, rho1A, rho2A, daB, qaB, rho0B, rho1B, rho2B;
  computeMultipoleParamsDev(pA, daA, qaA, rho0A, rho1A, rho2A);
  computeMultipoleParamsDev(pB, daB, qaB, rho0B, rho1B, rho2B);

  // Local-frame d reduced integrals (450), d-pairs only; sp slots are zero.
  NVMOLKIT_YWSCR(ri, kYWScrRi, 450);
  dlocal::riLocalYX(r0, daA, daB, qaA, qaB, dp, ds, dd, rho0A, rho0B, rho1A, rho1B,
                    rho2A, rho2B, rho3, rho4, rho5, rho6, ri);

  // Embed into the 45x45 local WW: rows = B sp-pairs (0..9), cols = A pairs.
  NVMOLKIT_YWSCR(WW, kYWScrWW, 45 * 45);
  for (int i = 0; i < 45 * 45; ++i) WW[i] = 0.0;
  for (int r = 0; r < 10; ++r)
    for (int c = 0; c < 45; ++c) WW[r * 45 + c] = ri[r * 45 + c];

  // sp 10x10 block: the sp molecular two-electron tensor (mu nu_A | lam sig_B),
  // packed lower-triangle [s,px,py,pz]. WW[bpair_B][apair_A].
  AtomIntParams spA = pA, spB = pB;
  if (spA.nOrb > 4) spA.nOrb = 4;
  if (spB.nOrb > 4) spB.nOrb = 4;
  double w[256], e1b[16], e2a[16];
  twoCenterMolecularDev(spA, coordA, spB, coordB, w, e1b, e2a);
  // pair k -> (orbital i,j), i>=j: 0:ss 1:(1,0) 2:(1,1) 3:(2,0) 4:(2,1) 5:(2,2)
  //                                 6:(3,0) 7:(3,1) 8:(3,2) 9:(3,3)
  const int PI[10] = {0, 1, 1, 2, 2, 2, 3, 3, 3, 3};
  const int PJ[10] = {0, 0, 1, 0, 1, 2, 0, 1, 2, 3};
  for (int bp = 0; bp < 10; ++bp)
    for (int ap = 0; ap < 10; ++ap)
      WW[bp * 45 + ap] = w[wIdxDev(PI[bp], PJ[bp], PI[ap], PJ[ap])];

  // Rotate to molecular frame.
  const double v[3] = {Rvec[0] / R, Rvec[1] / R, Rvec[2] / R};
  NVMOLKIT_YWSCR(mat, kYWScrMat, 15 * 45);
  drot::generateRotationMatrixD(v, mat);
  NVMOLKIT_YWSCR(wmol, kYWScrWmol, 45 * 45);
  drot::rotate2Center2ElectronD(WW, mat, wmol, scr ? scr + kYWScrRot : nullptr);

  // Reshape: W[mu,nu,lam,sig] = wmol[packed(lam,sig) , packed(mu,nu)].
  for (int mu = 0; mu < 9; ++mu)
    for (int nu = 0; nu < 9; ++nu)
      for (int lam = 0; lam < 4; ++lam)
        for (int sig = 0; sig < 4; ++sig)
          W[((mu * 9 + nu) * 4 + lam) * 4 + sig] =
              wmol[drot::packedTril(lam, sig) * 45 + drot::packedTril(mu, nu)];
  return true;
}

// Build the molecular YY tensor W[9*9*9*9] (row-major mu,nu,lam,sig) for the pair
// (both atoms carry d). Returns false if either atom has no baked d parameters.
NVMOLKIT_HD inline bool yyWMolecular(const AtomIntParams& pA, const double coordA[3],
                                     const AtomIntParams& pB, const double coordB[3], double* W,
                                     double* scr = nullptr) {
  using namespace detail;
  double dpA, dsA, ddA, r3A, r4A, r5A, r6A, dpB, dsB, ddB, r3B, r4B, r5B, r6B;
  if (!dChargeSeparationsTwoCenter(pA.z, dpA, dsA, ddA, r3A, r4A, r5A, r6A)) return false;
  if (!dChargeSeparationsTwoCenter(pB.z, dpB, dsB, ddB, r3B, r4B, r5B, r6B)) return false;

  const double Rvec[3] = {coordB[0] - coordA[0], coordB[1] - coordA[1], coordB[2] - coordA[2]};
  const double R = std::sqrt(Rvec[0] * Rvec[0] + Rvec[1] * Rvec[1] + Rvec[2] * Rvec[2]);
  const double r0 = R * kSemiAngToBohr;

  double daA, qaA, rho0A, rho1A, rho2A, daB, qaB, rho0B, rho1B, rho2B;
  computeMultipoleParamsDev(pA, daA, qaA, rho0A, rho1A, rho2A);
  computeMultipoleParamsDev(pB, daB, qaB, rho0B, rho1B, rho2B);

  NVMOLKIT_YWSCR(ri, kYWScrRi, 2025);
  dlocal::riLocalYY(r0, daA, daB, qaA, qaB, dpA, dpB, dsA, dsB, ddA, ddB,
                    rho0A, rho0B, rho1A, rho1B, rho2A, rho2B, r3A, r3B, r4A, r4B,
                    r5A, r5B, r6A, r6B, ri);

  // Local WW = riYY (full 45x45); the sp 10x10 is then the sp molecular w.
  NVMOLKIT_YWSCR(WW, kYWScrWW, 45 * 45);
  for (int i = 0; i < 45 * 45; ++i) WW[i] = ri[i];
  AtomIntParams spA = pA, spB = pB;
  if (spA.nOrb > 4) spA.nOrb = 4;
  if (spB.nOrb > 4) spB.nOrb = 4;
  double w[256], e1b[16], e2a[16];
  twoCenterMolecularDev(spA, coordA, spB, coordB, w, e1b, e2a);
  const int PI[10] = {0, 1, 1, 2, 2, 2, 3, 3, 3, 3};
  const int PJ[10] = {0, 0, 1, 0, 1, 2, 0, 1, 2, 3};
  for (int bp = 0; bp < 10; ++bp)
    for (int ap = 0; ap < 10; ++ap)
      WW[bp * 45 + ap] = w[wIdxDev(PI[bp], PJ[bp], PI[ap], PJ[ap])];

  const double v[3] = {Rvec[0] / R, Rvec[1] / R, Rvec[2] / R};
  NVMOLKIT_YWSCR(mat, kYWScrMat, 15 * 45);
  drot::generateRotationMatrixD(v, mat);
  NVMOLKIT_YWSCR(wmol, kYWScrWmol, 45 * 45);
  drot::rotate2Center2ElectronD(WW, mat, wmol, scr ? scr + kYWScrRot : nullptr);

  for (int mu = 0; mu < 9; ++mu)
    for (int nu = 0; nu < 9; ++nu)
      for (int lam = 0; lam < 9; ++lam)
        for (int sig = 0; sig < 9; ++sig)
          W[((mu * 9 + nu) * 9 + lam) * 9 + sig] =
              wmol[drot::packedTril(lam, sig) * 45 + drot::packedTril(mu, nu)];
  return true;
}

}  // namespace semiempirical
}  // namespace nvMolKit

#endif  // NVMOLKIT_SEMIEMPIRICAL_TWO_CENTER_YX_DEVICE_H
