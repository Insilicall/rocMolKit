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
// First GPU kernel of the semi-empirical engine: batched two-center two-electron
// integrals. Each thread does one atom pair via the shared __host__ __device__
// math (two_center_device.h), so the result is identical to the CPU reference.

#include <hip/hip_runtime.h>

#include <vector>

#include "overlap.h"            // principalQn
#include "pm6_params.h"
#include "two_center.h"
#include "two_center_device.h"  // AtomIntParams + twoCenterMolecularDev

namespace nvMolKit {
namespace semiempirical {

namespace {

int spCountHost(int z) {
  const int n = pm6NumOrbitals(z);
  return (n >= 4) ? 4 : n;
}

bool gatherHost(int z, AtomIntParams& out) {
  const Pm6ElementParams* p = pm6ParamsForZ(z);
  if (p == nullptr) return false;
  out.zetaS = p->zeta_s;
  out.zetaP = p->zeta_p;
  out.gss = p->gss;
  out.gsp = p->gsp;
  out.gpp = p->gpp;
  out.gp2 = p->gp2;
  out.hsp = p->hsp;
  out.qn = principalQn(z);
  out.valence = pm6ValenceElectrons(z);
  out.nOrb = spCountHost(z);
  return true;
}

__global__ void twoCenterBatchKernel(int nPairs, const AtomIntParams* pA, const AtomIntParams* pB,
                                     const double* coordA, const double* coordB,
                                     double* w, double* e1b, double* e2a) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nPairs) return;
  twoCenterMolecularDev(pA[i], &coordA[3 * i], pB[i], &coordB[3 * i],
                        &w[256 * i], &e1b[16 * i], &e2a[16 * i]);
}

}  // namespace

bool twoCenterMolecularBatchGpu(int nPairs, const int* zA, const double* coordA,
                                const int* zB, const double* coordB,
                                double* w, double* e1b, double* e2a) {
  if (nPairs <= 0) return true;

  std::vector<AtomIntParams> hA(nPairs), hB(nPairs);
  for (int i = 0; i < nPairs; ++i) {
    if (!gatherHost(zA[i], hA[i]) || !gatherHost(zB[i], hB[i])) return false;
  }

  AtomIntParams* dA = nullptr;
  AtomIntParams* dB = nullptr;
  double *dCoordA = nullptr, *dCoordB = nullptr, *dW = nullptr, *dE1b = nullptr, *dE2a = nullptr;
  bool ok = true;
  auto fail = [&]() {
    ok = false;
    return false;
  };

  if (hipMalloc(&dA, sizeof(AtomIntParams) * nPairs) != hipSuccess) return fail();
  if (hipMalloc(&dB, sizeof(AtomIntParams) * nPairs) != hipSuccess) return fail();
  if (hipMalloc(&dCoordA, sizeof(double) * 3 * nPairs) != hipSuccess) return fail();
  if (hipMalloc(&dCoordB, sizeof(double) * 3 * nPairs) != hipSuccess) return fail();
  if (hipMalloc(&dW, sizeof(double) * 256 * nPairs) != hipSuccess) return fail();
  if (hipMalloc(&dE1b, sizeof(double) * 16 * nPairs) != hipSuccess) return fail();
  if (hipMalloc(&dE2a, sizeof(double) * 16 * nPairs) != hipSuccess) return fail();

  if (ok) {
    hipMemcpy(dA, hA.data(), sizeof(AtomIntParams) * nPairs, hipMemcpyHostToDevice);
    hipMemcpy(dB, hB.data(), sizeof(AtomIntParams) * nPairs, hipMemcpyHostToDevice);
    hipMemcpy(dCoordA, coordA, sizeof(double) * 3 * nPairs, hipMemcpyHostToDevice);
    hipMemcpy(dCoordB, coordB, sizeof(double) * 3 * nPairs, hipMemcpyHostToDevice);

    const int block = 128;
    const int grid = (nPairs + block - 1) / block;
    twoCenterBatchKernel<<<grid, block>>>(nPairs, dA, dB, dCoordA, dCoordB, dW, dE1b, dE2a);
    if (hipDeviceSynchronize() != hipSuccess) ok = false;
  }

  if (ok) {
    hipMemcpy(w, dW, sizeof(double) * 256 * nPairs, hipMemcpyDeviceToHost);
    hipMemcpy(e1b, dE1b, sizeof(double) * 16 * nPairs, hipMemcpyDeviceToHost);
    hipMemcpy(e2a, dE2a, sizeof(double) * 16 * nPairs, hipMemcpyDeviceToHost);
  }

  hipFree(dA);
  hipFree(dB);
  hipFree(dCoordA);
  hipFree(dCoordB);
  hipFree(dW);
  hipFree(dE1b);
  hipFree(dE2a);
  return ok;
}

}  // namespace semiempirical
}  // namespace nvMolKit
