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
// Batched on-device NDDO/PM6_D (d-orbital) SCF for the YH scope. THROUGHPUT
// PATH: one WAVEFRONT (32 lanes) per molecule cooperatively runs the SCF loop
// (H_core -> d Fock -> diagonalize -> density -> mixing) via
// scf_d_coop_device.h, spreading the O(nB^3) Jacobi diagonalization + density /
// DIIS work across the lanes (vs the legacy one-thread-per-molecule serial loop
// in scf_d_device.h, which left 31 of every 32 lanes idle). It then emits
// Mulliken charges + heat of formation -- all on the GPU. The cooperative
// Jacobi uses a deterministic round-robin (Brent-Luk) rotation schedule, so the
// converged charges stay bit-exact to the validated CPU pm6dCharges (the GPU-vs-
// CPU validator measures ~1e-14) and to MOPAC. See docs/SEMIEMPIRICAL_DESIGN.md.

#include <hip/hip_runtime.h>

#include <cstdlib>
#include <vector>

#include "core_hamiltonian.h"           // gatherAtomIntParamsD
#include "core_hamiltonian_d_device.h"  // buildCoreHamiltonianDDev
#include "energy_device.h"              // nuclearRepulsionAm1Dev, heatOfFormationKcalDev
#include "pwcct_device.h"               // heatOfFormationPm6KcalAp (canonical PM6 HoF)
#include "scf_d_coop_device.h"  // cooperative one-warp-per-molecule SCF
#include "scf_d_device.h"
#include "scf_d_kernels.h"

namespace nvMolKit {
namespace semiempirical {

namespace {

// One-warp-per-molecule cooperative SCF: a 32-thread block runs one molecule,
// spreading the O(nB^3) Jacobi diagonalization + density/DIIS work across the
// wavefront's lanes (vs the legacy one-thread-per-molecule serial path). The
// per-(p,q) rotation angle is recomputed identically on every lane, so the
// converged charges match the CPU reference (relaxation only from cross-lane
// sum folds, ~1e-12 -- charges still bit-exact to MOPAC).
__global__ void scfBatchDKernelCoop(int nMol, const AtomIntParams* ap, const int* start,
                                    const int* norb, const double* coords, const int* atomOff,
                                    const int* nAtomsArr, const int* nBasisArr,
                                    const int* nOccArr, const long* scratchOff, double* scratch,
                                    const int* ringOff, int* ringScratch, const int* metaOff,
                                    int* metaScratch, double* chargesOut,
                                    int* convOut, double* hofOut, double* hofPm6Out, int maxIter,
                                    double convTol) {
  const int m = blockIdx.x;  // one block (warp) per molecule
  if (m >= nMol) return;
  const int lane = threadIdx.x;
  const int nLanes = blockDim.x;
  __shared__ double sh[kCoopMaxBlock];  // cross-lane reduction scratch

  const int nB = nBasisArr[m], na = nAtomsArr[m], ao = atomOff[m];
  const int n2 = nB * nB;
  double* base = &scratch[scratchOff[m]];
  double* H = base;
  double* density = H + n2;
  double* eval = density + n2;
  double* F = eval + nB;
  double* eigA = F + n2;
  double* C = eigA + n2;
  double* Pnew = C + n2;
  double* ecom = Pnew + n2;
  double* diisF = ecom + n2;
  double* diisE = diisF + kScfDiisMax * n2;
  double* cs = diisE + kScfDiisMax * n2;  // per-pair (c,s) for the parallel Jacobi
  double* intBlob = cs + (nB + 2);        // two-center integral cache (doubles)
  // Per-molecule scratch for the YY/YX 45x45 temporaries, sitting after the
  // integral-cache blob. Moving these off the per-lane stack is what lifts device
  // occupancy (see kYWScrDoubles in two_center_yx_device.h).
  double* ywScr = intBlob + pm6dIntCacheDoubles(na, &norb[ao]);

  // Round-robin tournament ring (ints) + integral-cache meta (ints) live in
  // separate per-molecule buffers.
  int* ring = &ringScratch[ringOff[m]];
  int* intMeta = &metaScratch[metaOff[m]];

  // Core Hamiltonian: lane 0 builds it (O(nAtoms^2), cheap vs the SCF).
  if (lane == 0)
    buildCoreHamiltonianDDev(nB, na, &ap[ao], &start[ao], &norb[ao], &coords[3 * ao], H);
  __syncthreads();

  int conv = 0, niter = 0;
  double eElec = 0.0;
  scfLoopDDevCoop(nB, na, &ap[ao], &start[ao], &norb[ao], &coords[3 * ao], H, nOccArr[m],
                  maxIter, convTol, density, eval, F, eigA, C, Pnew, ecom, diisF, diisE,
                  &conv, &niter, &eElec, lane, nLanes, sh, ring, cs, intMeta, intBlob, ywScr);

  // Outputs: lane 0 emits charges + heats of formation (all small, O(nAtoms)).
  if (lane == 0) {
    convOut[m] = conv;
    for (int a = 0; a < na; ++a) {
      const int s = start[ao + a];
      double pop = 0.0;
      for (int o = 0; o < norb[ao + a]; ++o) pop += density[(s + o) * nB + (s + o)];
      chargesOut[ao + a] = static_cast<double>(ap[ao + a].valence) - pop;
    }
    const double eNuc = nuclearRepulsionAm1Dev(na, &ap[ao], &coords[3 * ao]);
    hofOut[m] = heatOfFormationKcalDev(eElec, eNuc, na, &ap[ao]);
    if (hofPm6Out != nullptr)
      hofPm6Out[m] = heatOfFormationPm6KcalAp(eElec, na, &ap[ao], &coords[3 * ao]);
  }
}

}  // namespace

bool scfBatchDGpu(int nMol, const int* molNAtoms, const int* molNBasis,
                  const int* atomsAll, const double* coordsAll,
                  double* chargesAll, double* hofAll, int* convergedAll,
                  int maxIter, double convTol, double* hofPm6All, const int* molCharge) {
  if (nMol <= 0) return true;

  std::vector<int> atomOff(nMol), nOcc(nMol), ringOff(nMol), metaOff(nMol);
  std::vector<long> scratchOff(nMol);
  int totAtoms = 0;
  for (int m = 0; m < nMol; ++m) totAtoms += molNAtoms[m];

  // Gather PM6_D params (nOrb up to 9) + molecule-local start/norb FIRST, so the
  // per-molecule integral-cache blob size (which depends on per-atom norb) is
  // known when sizing the device scratch below.
  std::vector<AtomIntParams> ap(totAtoms);
  std::vector<int> start(totAtoms), norb(totAtoms);
  {
    int ao = 0;
    for (int m = 0; m < nMol; ++m) {
      const int na = molNAtoms[m];
      int off = 0, nElec = 0;
      for (int a = 0; a < na; ++a) {
        AtomIntParams& o = ap[ao + a];
        if (!gatherAtomIntParamsD(atomsAll[ao + a], o)) return false;
        start[ao + a] = off;
        norb[ao + a] = o.nOrb;
        off += o.nOrb;
        nElec += o.valence;
      }
      nElec -= molCharge ? molCharge[m] : 0;  // cation (+) removes electrons
      if (nElec <= 0 || nElec % 2 != 0) return false;  // open shell / invalid
      nOcc[m] = nElec / 2;
      ao += na;
    }
  }

  long totScratch = 0, totRing = 0, totMeta = 0;
  {
    int ao = 0;
    for (int m = 0; m < nMol; ++m) {
      atomOff[m] = ao;
      scratchOff[m] = totScratch;
      ringOff[m] = static_cast<int>(totRing);
      metaOff[m] = static_cast<int>(totMeta);
      const long nB = molNBasis[m];
      const int na = molNAtoms[m];
      const long nPairs = static_cast<long>(na) * (na - 1) / 2;
      const long blobDoubles = pm6dIntCacheDoubles(na, &norb[ao]);
      // H,density,F,eigA,C,Pnew,ecom (7 n^2) + diisF,diisE (2*kScfDiisMax n^2)
      // + eval (n) + cs (per-pair cos/sin: nB+2) + intBlob (integral cache)
      // + ywScr (kYWScrDoubles: YY/YX 45x45 temporaries moved off the per-lane
      // stack for occupancy).
      totScratch += (7 + 2 * kScfDiisMax) * nB * nB + nB + (nB + 2) + blobDoubles + kYWScrDoubles;
      // Tournament ring: (nB+1)&~1 ints, padded to nB+2 for safety.
      totRing += nB + 2;
      // Integral-cache meta: kPairMetaInts ints per pair.
      totMeta += nPairs * kPairMetaInts;
      ao += na;
    }
  }

  AtomIntParams* dAp = nullptr;
  int *dStart = nullptr, *dNorb = nullptr, *dAtomOff = nullptr, *dNAtoms = nullptr,
      *dNBasis = nullptr, *dNOcc = nullptr, *dConv = nullptr, *dRingOff = nullptr,
      *dRing = nullptr, *dMetaOff = nullptr, *dMeta = nullptr;
  long* dScratchOff = nullptr;
  double *dCoords = nullptr, *dScratch = nullptr, *dCharges = nullptr, *dHof = nullptr,
         *dHofPm6 = nullptr;
  bool ok = true;
  auto need = [&](hipError_t e) { if (e != hipSuccess) ok = false; };

  need(hipMalloc(&dAp, sizeof(AtomIntParams) * totAtoms));
  need(hipMalloc(&dStart, sizeof(int) * totAtoms));
  need(hipMalloc(&dNorb, sizeof(int) * totAtoms));
  need(hipMalloc(&dAtomOff, sizeof(int) * nMol));
  need(hipMalloc(&dNAtoms, sizeof(int) * nMol));
  need(hipMalloc(&dNBasis, sizeof(int) * nMol));
  need(hipMalloc(&dNOcc, sizeof(int) * nMol));
  need(hipMalloc(&dConv, sizeof(int) * nMol));
  need(hipMalloc(&dScratchOff, sizeof(long) * nMol));
  need(hipMalloc(&dCoords, sizeof(double) * 3 * totAtoms));
  need(hipMalloc(&dScratch, sizeof(double) * totScratch));
  need(hipMalloc(&dCharges, sizeof(double) * totAtoms));
  need(hipMalloc(&dHof, sizeof(double) * nMol));
  need(hipMalloc(&dHofPm6, sizeof(double) * nMol));
  need(hipMalloc(&dRingOff, sizeof(int) * nMol));
  need(hipMalloc(&dRing, sizeof(int) * (totRing > 0 ? totRing : 1)));
  need(hipMalloc(&dMetaOff, sizeof(int) * nMol));
  need(hipMalloc(&dMeta, sizeof(int) * (totMeta > 0 ? totMeta : 1)));

  if (ok) {
    hipMemcpy(dAp, ap.data(), sizeof(AtomIntParams) * totAtoms, hipMemcpyHostToDevice);
    hipMemcpy(dStart, start.data(), sizeof(int) * totAtoms, hipMemcpyHostToDevice);
    hipMemcpy(dNorb, norb.data(), sizeof(int) * totAtoms, hipMemcpyHostToDevice);
    hipMemcpy(dAtomOff, atomOff.data(), sizeof(int) * nMol, hipMemcpyHostToDevice);
    hipMemcpy(dNAtoms, molNAtoms, sizeof(int) * nMol, hipMemcpyHostToDevice);
    hipMemcpy(dNBasis, molNBasis, sizeof(int) * nMol, hipMemcpyHostToDevice);
    hipMemcpy(dNOcc, nOcc.data(), sizeof(int) * nMol, hipMemcpyHostToDevice);
    hipMemcpy(dScratchOff, scratchOff.data(), sizeof(long) * nMol, hipMemcpyHostToDevice);
    hipMemcpy(dRingOff, ringOff.data(), sizeof(int) * nMol, hipMemcpyHostToDevice);
    hipMemcpy(dMetaOff, metaOff.data(), sizeof(int) * nMol, hipMemcpyHostToDevice);
    hipMemcpy(dCoords, coordsAll, sizeof(double) * 3 * totAtoms, hipMemcpyHostToDevice);

    // The YY/YX 45x45 W temporaries now live in per-molecule GLOBAL scratch (see
    // kYWScrDoubles), not the per-lane stack, so the kernel's measured private
    // (scratch) footprint dropped from ~108 KB/lane to ~8 KB/lane. The remaining
    // per-thread stack is just the small recursive twoCenterMolecularDev e1b
    // frames + scalar spills; a 32 KB reservation is ample and -- by no longer
    // reserving 192 KB/lane -- frees device scratch memory for more resident
    // blocks.
    hipDeviceSetLimit(hipLimitStackSize, 32 * 1024);

    // One-WAVEFRONT-per-molecule (32 lanes): the warp cooperatively runs the
    // per-molecule SCF, spreading the O(nB^3) Jacobi diagonalization across the
    // lanes. 32 (a single RDNA4 wavefront) is the safe default; larger blocks
    // (ROCMOLKIT_PM6D_BLOCK, multiple of 32) parallelize the inner length-nB loops
    // further for big basis sizes. Now that the Fock/precompute stack footprint is
    // small, multi-warp blocks no longer risk a scratch page fault.
    int block = 32;
    if (const char* e = std::getenv("ROCMOLKIT_PM6D_BLOCK")) {
      int b = std::atoi(e);
      if (b >= 32 && b <= kCoopMaxBlock && (b % 32) == 0) block = b;
    }
    const int grid = nMol;
    scfBatchDKernelCoop<<<grid, block>>>(nMol, dAp, dStart, dNorb, dCoords, dAtomOff, dNAtoms,
                                         dNBasis, dNOcc, dScratchOff, dScratch, dRingOff, dRing,
                                         dMetaOff, dMeta, dCharges, dConv, dHof, dHofPm6, maxIter,
                                         convTol);
    if (hipDeviceSynchronize() != hipSuccess) ok = false;
  }
  if (ok) {
    hipMemcpy(chargesAll, dCharges, sizeof(double) * totAtoms, hipMemcpyDeviceToHost);
    hipMemcpy(convergedAll, dConv, sizeof(int) * nMol, hipMemcpyDeviceToHost);
    if (hofAll != nullptr) hipMemcpy(hofAll, dHof, sizeof(double) * nMol, hipMemcpyDeviceToHost);
    if (hofPm6All != nullptr)
      hipMemcpy(hofPm6All, dHofPm6, sizeof(double) * nMol, hipMemcpyDeviceToHost);
  }

  hipFree(dAp); hipFree(dStart); hipFree(dNorb); hipFree(dAtomOff); hipFree(dNAtoms);
  hipFree(dNBasis); hipFree(dNOcc); hipFree(dConv); hipFree(dScratchOff);
  hipFree(dCoords); hipFree(dScratch); hipFree(dCharges); hipFree(dHof); hipFree(dHofPm6);
  hipFree(dRingOff); hipFree(dRing); hipFree(dMetaOff); hipFree(dMeta);
  return ok;
}

}  // namespace semiempirical
}  // namespace nvMolKit
