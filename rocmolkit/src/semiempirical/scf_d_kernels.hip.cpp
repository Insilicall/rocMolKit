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
// Batched on-device NDDO/PM6_D (d-orbital) SCF for the YH scope: one thread per
// molecule runs the whole SCF loop (H_core -> d Fock -> diagonalize -> density
// -> mixing) via scf_d_device.h, then emits Mulliken charges + heat of formation
// — all on the GPU. The SAME __host__ __device__ headers feed this and the CPU
// reference, so the GPU result is bit-exact to the validated CPU engine. See
// docs/SEMIEMPIRICAL_DESIGN.md.

#include <hip/hip_runtime.h>

#include <vector>

#include "core_hamiltonian.h"           // gatherAtomIntParamsD
#include "core_hamiltonian_d_device.h"  // buildCoreHamiltonianDDev
#include "energy_device.h"              // nuclearRepulsionAm1Dev, heatOfFormationKcalDev
#include "pwcct_device.h"               // heatOfFormationPm6KcalAp (canonical PM6 HoF)
#include "scf_d_device.h"
#include "scf_d_kernels.h"

namespace nvMolKit {
namespace semiempirical {

namespace {

__global__ void scfBatchDKernel(int nMol, const AtomIntParams* ap, const int* start,
                                const int* norb, const double* coords, const int* atomOff,
                                const int* nAtomsArr, const int* nBasisArr, const int* nOccArr,
                                const long* scratchOff, double* scratch, double* chargesOut,
                                int* convOut, double* hofOut, double* hofPm6Out, int maxIter,
                                double convTol) {
  const int m = blockIdx.x * blockDim.x + threadIdx.x;
  if (m >= nMol) return;
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

  // Build the core Hamiltonian on the device (closing the 100%-GPU PM6_D SCF).
  buildCoreHamiltonianDDev(nB, na, &ap[ao], &start[ao], &norb[ao], &coords[3 * ao], H);

  int conv = 0, niter = 0;
  double eElec = 0.0;
  scfLoopDDev(nB, na, &ap[ao], &start[ao], &norb[ao], &coords[3 * ao], H, nOccArr[m],
              maxIter, convTol, density, eval, F, eigA, C, Pnew, ecom, diisF, diisE,
              &conv, &niter, &eElec);
  convOut[m] = conv;
  for (int a = 0; a < na; ++a) {
    const int s = start[ao + a];
    double pop = 0.0;
    for (int o = 0; o < norb[ao + a]; ++o) pop += density[(s + o) * nB + (s + o)];
    chargesOut[ao + a] = static_cast<double>(ap[ao + a].valence) - pop;
  }
  // Heat of formation on-device (PM6_D uses the AM1-style core-core).
  const double eNuc = nuclearRepulsionAm1Dev(na, &ap[ao], &coords[3 * ao]);
  hofOut[m] = heatOfFormationKcalDev(eElec, eNuc, na, &ap[ao]);
  // Canonical (MOPAC-aligned) PM6 heat of formation (PWCCT core-core).
  if (hofPm6Out != nullptr)
    hofPm6Out[m] = heatOfFormationPm6KcalAp(eElec, na, &ap[ao], &coords[3 * ao]);
}

}  // namespace

bool scfBatchDGpu(int nMol, const int* molNAtoms, const int* molNBasis,
                  const int* atomsAll, const double* coordsAll,
                  double* chargesAll, double* hofAll, int* convergedAll,
                  int maxIter, double convTol, double* hofPm6All, const int* molCharge) {
  if (nMol <= 0) return true;

  std::vector<int> atomOff(nMol), nOcc(nMol);
  std::vector<long> scratchOff(nMol);
  int totAtoms = 0;
  long totScratch = 0;
  for (int m = 0; m < nMol; ++m) {
    atomOff[m] = totAtoms;
    scratchOff[m] = totScratch;
    const long nB = molNBasis[m];
    totAtoms += molNAtoms[m];
    // H,density,F,eigA,C,Pnew,ecom (7 n^2) + diisF,diisE (2*kScfDiisMax n^2) + eval (n)
    totScratch += (7 + 2 * kScfDiisMax) * nB * nB + nB;
  }

  // Gather PM6_D params (nOrb up to 9) + molecule-local start/norb.
  std::vector<AtomIntParams> ap(totAtoms);
  std::vector<int> start(totAtoms), norb(totAtoms);
  for (int m = 0; m < nMol; ++m) {
    const int na = molNAtoms[m], ao = atomOff[m];
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
  }

  AtomIntParams* dAp = nullptr;
  int *dStart = nullptr, *dNorb = nullptr, *dAtomOff = nullptr, *dNAtoms = nullptr,
      *dNBasis = nullptr, *dNOcc = nullptr, *dConv = nullptr;
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

  if (ok) {
    hipMemcpy(dAp, ap.data(), sizeof(AtomIntParams) * totAtoms, hipMemcpyHostToDevice);
    hipMemcpy(dStart, start.data(), sizeof(int) * totAtoms, hipMemcpyHostToDevice);
    hipMemcpy(dNorb, norb.data(), sizeof(int) * totAtoms, hipMemcpyHostToDevice);
    hipMemcpy(dAtomOff, atomOff.data(), sizeof(int) * nMol, hipMemcpyHostToDevice);
    hipMemcpy(dNAtoms, molNAtoms, sizeof(int) * nMol, hipMemcpyHostToDevice);
    hipMemcpy(dNBasis, molNBasis, sizeof(int) * nMol, hipMemcpyHostToDevice);
    hipMemcpy(dNOcc, nOcc.data(), sizeof(int) * nMol, hipMemcpyHostToDevice);
    hipMemcpy(dScratchOff, scratchOff.data(), sizeof(long) * nMol, hipMemcpyHostToDevice);
    hipMemcpy(dCoords, coordsAll, sizeof(double) * 3 * totAtoms, hipMemcpyHostToDevice);

    // The YY Fock branch materializes a 9x9x9x9 (52 KB) tensor per thread, on top
    // of the recursive e1b frames; bump the per-thread stack well above the
    // ~1 KB default to hold it.
    hipDeviceSetLimit(hipLimitStackSize, 192 * 1024);

    const int block = 64;
    const int grid = (nMol + block - 1) / block;
    scfBatchDKernel<<<grid, block>>>(nMol, dAp, dStart, dNorb, dCoords, dAtomOff, dNAtoms,
                                     dNBasis, dNOcc, dScratchOff, dScratch, dCharges,
                                     dConv, dHof, dHofPm6, maxIter, convTol);
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
  return ok;
}

}  // namespace semiempirical
}  // namespace nvMolKit
