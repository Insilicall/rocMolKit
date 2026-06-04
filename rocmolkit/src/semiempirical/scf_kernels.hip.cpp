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
// Batched on-device NDDO/PM6 SCF: one thread per molecule runs the whole SCF
// loop (Fock -> diagonalize -> density -> mixing) via scf_device.h, then emits
// Mulliken charges — all on the GPU, no per-iteration host round-trip. The core
// Hamiltonian is built host-side (validated, one-time) and uploaded. See
// docs/SEMIEMPIRICAL_DESIGN.md (Stage 7c).

#include <hip/hip_runtime.h>

#include <vector>

#include "core_hamiltonian.h"
#include "overlap.h"  // principalQn
#include "pm6_params.h"
#include "scf_device.h"
#include "scf_kernels.h"
#include "two_center_device.h"

namespace nvMolKit {
namespace semiempirical {

namespace {

int spCountHost(int z) {
  const int n = pm6NumOrbitals(z);
  return (n >= 4) ? 4 : n;
}

__global__ void scfBatchKernel(int nMol, const AtomIntParams* ap, const int* start,
                               const int* norb, const double* coords, const double* H,
                               const int* atomOff, const int* nAtomsArr, const int* nBasisArr,
                               const int* basisOff, const int* nOccArr, const long* scratchOff,
                               double* scratch, double* chargesOut, int* convOut,
                               int maxIter, double convTol) {
  const int m = blockIdx.x * blockDim.x + threadIdx.x;
  if (m >= nMol) return;
  const int nB = nBasisArr[m], na = nAtomsArr[m], ao = atomOff[m], bo = basisOff[m];
  const int n2 = nB * nB;
  double* base = &scratch[scratchOff[m]];
  double* density = base;
  double* eval = density + n2;
  double* F = eval + nB;
  double* eigA = F + n2;
  double* C = eigA + n2;
  double* Pnew = C + n2;
  double* ecom = Pnew + n2;
  double* diisF = ecom + n2;
  double* diisE = diisF + kScfDiisMax * n2;

  int conv = 0, niter = 0;
  scfLoopDev(nB, na, &ap[ao], &start[ao], &norb[ao], &coords[3 * ao], &H[bo], nOccArr[m],
             maxIter, convTol, density, eval, F, eigA, C, Pnew, ecom, diisF, diisE,
             &conv, &niter);
  convOut[m] = conv;
  for (int a = 0; a < na; ++a) {
    const int s = start[ao + a];
    double pop = 0.0;
    for (int o = 0; o < norb[ao + a]; ++o) {
      const int mu = s + o;
      pop += density[mu * nB + mu];
    }
    chargesOut[ao + a] = static_cast<double>(ap[ao + a].valence) - pop;
  }
}

}  // namespace

bool scfBatchGpu(int nMol, const int* molNAtoms, const int* molNBasis,
                 const int* atomsAll, const double* coordsAll,
                 double* chargesAll, int* convergedAll, int maxIter, double convTol) {
  if (nMol <= 0) return true;

  std::vector<int> atomOff(nMol), basisOff(nMol), nOcc(nMol);
  std::vector<long> scratchOff(nMol);
  int totAtoms = 0;
  long totBasis2 = 0, totScratch = 0;
  for (int m = 0; m < nMol; ++m) {
    atomOff[m] = totAtoms;
    basisOff[m] = static_cast<int>(totBasis2);
    scratchOff[m] = totScratch;
    const long nB = molNBasis[m];
    totAtoms += molNAtoms[m];
    totBasis2 += nB * nB;
    // density,F,eigA,C,Pnew,ecom (6 n^2) + diisF,diisE (2*kScfDiisMax n^2) + eval (n)
    totScratch += (6 + 2 * kScfDiisMax) * nB * nB + nB;
  }

  // Gather params and build H_core host-side (validated), molecule-local start/norb.
  std::vector<AtomIntParams> ap(totAtoms);
  std::vector<int> start(totAtoms), norb(totAtoms);
  std::vector<double> Hall(totBasis2);
  for (int m = 0; m < nMol; ++m) {
    const int na = molNAtoms[m], ao = atomOff[m], nB = molNBasis[m];
    int off = 0, nElec = 0;
    for (int a = 0; a < na; ++a) {
      const int z = atomsAll[ao + a];
      const Pm6ElementParams* p = pm6ParamsForZ(z);
      if (p == nullptr) return false;
      AtomIntParams& o = ap[ao + a];
      o.zetaS = p->zeta_s; o.zetaP = p->zeta_p;
      o.gss = p->gss; o.gsp = p->gsp; o.gpp = p->gpp; o.gp2 = p->gp2; o.hsp = p->hsp;
      o.qn = principalQn(z);
      o.valence = pm6ValenceElectrons(z);
      o.nOrb = spCountHost(z);
      start[ao + a] = off;
      norb[ao + a] = o.nOrb;
      off += o.nOrb;
      nElec += o.valence;
    }
    if (nElec % 2 != 0) return false;  // open shell not handled
    nOcc[m] = nElec / 2;
    if (buildCoreHamiltonianSp(na, &atomsAll[ao], &coordsAll[3 * ao], &Hall[basisOff[m]]) == 0)
      return false;
  }

  AtomIntParams* dAp = nullptr;
  int *dStart = nullptr, *dNorb = nullptr, *dAtomOff = nullptr, *dNAtoms = nullptr,
      *dNBasis = nullptr, *dBasisOff = nullptr, *dNOcc = nullptr, *dConv = nullptr;
  long* dScratchOff = nullptr;
  double *dCoords = nullptr, *dH = nullptr, *dScratch = nullptr, *dCharges = nullptr;
  bool ok = true;
  auto need = [&](hipError_t e) { if (e != hipSuccess) ok = false; };

  need(hipMalloc(&dAp, sizeof(AtomIntParams) * totAtoms));
  need(hipMalloc(&dStart, sizeof(int) * totAtoms));
  need(hipMalloc(&dNorb, sizeof(int) * totAtoms));
  need(hipMalloc(&dAtomOff, sizeof(int) * nMol));
  need(hipMalloc(&dNAtoms, sizeof(int) * nMol));
  need(hipMalloc(&dNBasis, sizeof(int) * nMol));
  need(hipMalloc(&dBasisOff, sizeof(int) * nMol));
  need(hipMalloc(&dNOcc, sizeof(int) * nMol));
  need(hipMalloc(&dConv, sizeof(int) * nMol));
  need(hipMalloc(&dScratchOff, sizeof(long) * nMol));
  need(hipMalloc(&dCoords, sizeof(double) * 3 * totAtoms));
  need(hipMalloc(&dH, sizeof(double) * totBasis2));
  need(hipMalloc(&dScratch, sizeof(double) * totScratch));
  need(hipMalloc(&dCharges, sizeof(double) * totAtoms));

  if (ok) {
    hipMemcpy(dAp, ap.data(), sizeof(AtomIntParams) * totAtoms, hipMemcpyHostToDevice);
    hipMemcpy(dStart, start.data(), sizeof(int) * totAtoms, hipMemcpyHostToDevice);
    hipMemcpy(dNorb, norb.data(), sizeof(int) * totAtoms, hipMemcpyHostToDevice);
    hipMemcpy(dAtomOff, atomOff.data(), sizeof(int) * nMol, hipMemcpyHostToDevice);
    hipMemcpy(dNAtoms, molNAtoms, sizeof(int) * nMol, hipMemcpyHostToDevice);
    hipMemcpy(dNBasis, molNBasis, sizeof(int) * nMol, hipMemcpyHostToDevice);
    hipMemcpy(dBasisOff, basisOff.data(), sizeof(int) * nMol, hipMemcpyHostToDevice);
    hipMemcpy(dNOcc, nOcc.data(), sizeof(int) * nMol, hipMemcpyHostToDevice);
    hipMemcpy(dScratchOff, scratchOff.data(), sizeof(long) * nMol, hipMemcpyHostToDevice);
    hipMemcpy(dCoords, coordsAll, sizeof(double) * 3 * totAtoms, hipMemcpyHostToDevice);
    hipMemcpy(dH, Hall.data(), sizeof(double) * totBasis2, hipMemcpyHostToDevice);

    const int block = 64;
    const int grid = (nMol + block - 1) / block;
    scfBatchKernel<<<grid, block>>>(nMol, dAp, dStart, dNorb, dCoords, dH, dAtomOff, dNAtoms,
                                    dNBasis, dBasisOff, dNOcc, dScratchOff, dScratch, dCharges,
                                    dConv, maxIter, convTol);
    if (hipDeviceSynchronize() != hipSuccess) ok = false;
  }
  if (ok) {
    hipMemcpy(chargesAll, dCharges, sizeof(double) * totAtoms, hipMemcpyDeviceToHost);
    hipMemcpy(convergedAll, dConv, sizeof(int) * nMol, hipMemcpyDeviceToHost);
  }

  hipFree(dAp); hipFree(dStart); hipFree(dNorb); hipFree(dAtomOff); hipFree(dNAtoms);
  hipFree(dNBasis); hipFree(dBasisOff); hipFree(dNOcc); hipFree(dConv); hipFree(dScratchOff);
  hipFree(dCoords); hipFree(dH); hipFree(dScratch); hipFree(dCharges);
  return ok;
}

}  // namespace semiempirical
}  // namespace nvMolKit
