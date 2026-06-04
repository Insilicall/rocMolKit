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
// Batched NDDO/PM6 Fock-build kernel: one thread per molecule runs the shared
// __host__ __device__ buildFockDev (fock_device.h), so the GPU Fock is identical
// to the CPU reference by construction. See docs/SEMIEMPIRICAL_DESIGN.md.

#include <hip/hip_runtime.h>

#include <vector>

#include "fock_device.h"
#include "overlap.h"  // principalQn
#include "pm6_params.h"
#include "two_center_device.h"

namespace nvMolKit {
namespace semiempirical {

namespace {

int spCountHost(int z) {
  const int n = pm6NumOrbitals(z);
  return (n >= 4) ? 4 : n;
}

__global__ void fockBatchKernel(int nMol, const AtomIntParams* ap, const int* start,
                                const int* norb, const double* coords, const int* atomOff,
                                const int* nAtomsArr, const int* nBasisArr, const int* basisOff,
                                const double* H, const double* P, double* F) {
  const int m = blockIdx.x * blockDim.x + threadIdx.x;
  if (m >= nMol) return;
  const int ao = atomOff[m];
  const int bo = basisOff[m];
  buildFockDev(nBasisArr[m], nAtomsArr[m], &ap[ao], &start[ao], &norb[ao],
               &coords[3 * ao], &H[bo], &P[bo], &F[bo]);
}

}  // namespace

bool buildFockBatchGpu(int nMol, const int* molNAtoms, const int* molNBasis,
                       const int* atomsAll, const double* coordsAll,
                       const double* Hall, const double* Pall, double* Fall) {
  if (nMol <= 0) return true;

  // Host-side prefix sums + per-atom gather (molecule-local start/norb).
  std::vector<int> atomOff(nMol), basisOff(nMol);
  int totAtoms = 0;
  long totBasis2 = 0;
  for (int m = 0; m < nMol; ++m) {
    atomOff[m] = totAtoms;
    basisOff[m] = static_cast<int>(totBasis2);
    totAtoms += molNAtoms[m];
    totBasis2 += static_cast<long>(molNBasis[m]) * molNBasis[m];
  }

  std::vector<AtomIntParams> ap(totAtoms);
  std::vector<int> start(totAtoms), norb(totAtoms);
  for (int m = 0; m < nMol; ++m) {
    int off = 0;
    for (int a = 0; a < molNAtoms[m]; ++a) {
      const int idx = atomOff[m] + a;
      const int z = atomsAll[idx];
      const Pm6ElementParams* p = pm6ParamsForZ(z);
      if (p == nullptr) return false;
      AtomIntParams& o = ap[idx];
      o.zetaS = p->zeta_s; o.zetaP = p->zeta_p;
      o.gss = p->gss; o.gsp = p->gsp; o.gpp = p->gpp; o.gp2 = p->gp2; o.hsp = p->hsp;
      o.qn = principalQn(z);
      o.valence = pm6ValenceElectrons(z);
      o.nOrb = spCountHost(z);
      start[idx] = off;
      norb[idx] = o.nOrb;
      off += o.nOrb;
    }
  }

  AtomIntParams* dAp = nullptr;
  int *dStart = nullptr, *dNorb = nullptr, *dAtomOff = nullptr, *dNAtoms = nullptr,
      *dNBasis = nullptr, *dBasisOff = nullptr;
  double *dCoords = nullptr, *dH = nullptr, *dP = nullptr, *dF = nullptr;
  bool ok = true;
  auto need = [&](hipError_t e) { if (e != hipSuccess) ok = false; };

  need(hipMalloc(&dAp, sizeof(AtomIntParams) * totAtoms));
  need(hipMalloc(&dStart, sizeof(int) * totAtoms));
  need(hipMalloc(&dNorb, sizeof(int) * totAtoms));
  need(hipMalloc(&dAtomOff, sizeof(int) * nMol));
  need(hipMalloc(&dNAtoms, sizeof(int) * nMol));
  need(hipMalloc(&dNBasis, sizeof(int) * nMol));
  need(hipMalloc(&dBasisOff, sizeof(int) * nMol));
  need(hipMalloc(&dCoords, sizeof(double) * 3 * totAtoms));
  need(hipMalloc(&dH, sizeof(double) * totBasis2));
  need(hipMalloc(&dP, sizeof(double) * totBasis2));
  need(hipMalloc(&dF, sizeof(double) * totBasis2));

  if (ok) {
    hipMemcpy(dAp, ap.data(), sizeof(AtomIntParams) * totAtoms, hipMemcpyHostToDevice);
    hipMemcpy(dStart, start.data(), sizeof(int) * totAtoms, hipMemcpyHostToDevice);
    hipMemcpy(dNorb, norb.data(), sizeof(int) * totAtoms, hipMemcpyHostToDevice);
    hipMemcpy(dAtomOff, atomOff.data(), sizeof(int) * nMol, hipMemcpyHostToDevice);
    hipMemcpy(dNAtoms, molNAtoms, sizeof(int) * nMol, hipMemcpyHostToDevice);
    hipMemcpy(dNBasis, molNBasis, sizeof(int) * nMol, hipMemcpyHostToDevice);
    hipMemcpy(dBasisOff, basisOff.data(), sizeof(int) * nMol, hipMemcpyHostToDevice);
    hipMemcpy(dCoords, coordsAll, sizeof(double) * 3 * totAtoms, hipMemcpyHostToDevice);
    hipMemcpy(dH, Hall, sizeof(double) * totBasis2, hipMemcpyHostToDevice);
    hipMemcpy(dP, Pall, sizeof(double) * totBasis2, hipMemcpyHostToDevice);

    const int block = 64;
    const int grid = (nMol + block - 1) / block;
    fockBatchKernel<<<grid, block>>>(nMol, dAp, dStart, dNorb, dCoords, dAtomOff,
                                     dNAtoms, dNBasis, dBasisOff, dH, dP, dF);
    if (hipDeviceSynchronize() != hipSuccess) ok = false;
  }
  if (ok) hipMemcpy(Fall, dF, sizeof(double) * totBasis2, hipMemcpyDeviceToHost);

  hipFree(dAp); hipFree(dStart); hipFree(dNorb); hipFree(dAtomOff); hipFree(dNAtoms);
  hipFree(dNBasis); hipFree(dBasisOff); hipFree(dCoords); hipFree(dH); hipFree(dP); hipFree(dF);
  return ok;
}

}  // namespace semiempirical
}  // namespace nvMolKit
