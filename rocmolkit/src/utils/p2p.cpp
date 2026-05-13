// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "p2p.h"

#include <hip/hip_runtime.h>

#include <optional>
#include <stdexcept>
#include <string>

#include "cuda_error_check.h"
#include "device.h"

namespace nvMolKit {
namespace {

void enablePeerOneWay(int fromGpu, int toGpu) {
  const WithDevice  withDevice(fromGpu);
  const hipError_t err = hipDeviceEnablePeerAccess(toGpu, /*flags=*/0);
  if (err == hipSuccess) {
    return;
  }
  if (err == hipErrorPeerAccessAlreadyEnabled) {
    // Clear only the benign sticky error from the idempotent case; leave any unrelated
    // prior error in place so it surfaces at the next cudaCheckError.
    (void)hipGetLastError();
    return;
  }
  throw std::runtime_error("Failed to enable P2P access from GPU " + std::to_string(fromGpu) + " to GPU " +
                           std::to_string(toGpu) + ": " + hipGetErrorString(err));
}

}  // namespace

void enablePeerAccess(int gpuA, int gpuB) {
  if (gpuA == gpuB) {
    return;
  }
  enablePeerOneWay(gpuA, gpuB);
  enablePeerOneWay(gpuB, gpuA);
}

void copyDeviceToDeviceAsync(void*        dstDevice,
                             const void*  srcDevice,
                             std::size_t  byteCount,
                             int          srcGpu,
                             hipStream_t srcStream,
                             int          dstGpu,
                             hipStream_t dstStream) {
  if (byteCount == 0) {
    return;
  }
  if (srcGpu == dstGpu) {
    const WithDevice withDst(dstGpu);
    cudaCheckError(hipMemcpyAsync(dstDevice, srcDevice, byteCount, hipMemcpyDeviceToDevice, dstStream));
    return;
  }

  // If per-call event create/destroy shows up meaningfully in profiles, promote this helper
  // to a stateful class that owns a reusable event (or a small pool) per (srcGpu, dstGpu) pair.
  // The event must be created with srcGpu current so it binds to the source device.
  std::optional<ScopedCudaEvent> srcReady;
  {
    const WithDevice withSrc(srcGpu);
    srcReady.emplace();
    cudaCheckError(hipEventRecord(srcReady->event(), srcStream));
  }
  {
    const WithDevice withDst(dstGpu);
    cudaCheckError(hipStreamWaitEvent(dstStream, srcReady->event(), 0));
    cudaCheckError(hipMemcpyPeerAsync(dstDevice, dstGpu, srcDevice, srcGpu, byteCount, dstStream));
  }
}

}  // namespace nvMolKit
