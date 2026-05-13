#include "hip/hip_runtime.h"
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "device.h"

#include <hip/hip_runtime.h>

#include "cuda_error_check.h"
#include "nvtx.h"

namespace nvMolKit {

int countCudaDevices() {
  int device_count = 0;
  cudaCheckError(hipGetDeviceCount(&device_count));
  return device_count;
}

WithDevice::WithDevice(int device_id) {
  cudaCheckError(hipGetDevice(&original_device_id_));
  cudaCheckError(hipSetDevice(device_id));
}

WithDevice::~WithDevice() {
  cudaCheckErrorNoThrow(hipSetDevice(original_device_id_));
}

std::optional<hipStream_t> acquireExternalStream(std::uintptr_t streamPtr) {
  auto stream = reinterpret_cast<hipStream_t>(streamPtr);
  if (streamPtr == 0) {
    return stream;
  }
  hipError_t err = hipStreamQuery(stream);
  if (err == hipSuccess || err == hipErrorNotReady) {
    return stream;
  }
  // Clear the sticky error state
  hipGetLastError();
  return std::nullopt;
}

size_t getDeviceFreeMemory() {
  size_t free  = 0;
  size_t total = 0;
  cudaCheckError(hipMemGetInfo(&free, &total));
  return free;
}

ScopedStream::ScopedStream(const char* name) {
  cudaCheckError(hipStreamCreateWithFlags(&original_stream_, hipStreamNonBlocking));
  if (name != nullptr) {
    nvtxNameCudaStreamA(original_stream_, name);
  }
}

ScopedStream::~ScopedStream() noexcept {
  if (original_stream_ == nullptr) {
    return;
  }
  cudaCheckErrorNoThrow(hipStreamSynchronize(original_stream_));
  cudaCheckErrorNoThrow(hipStreamDestroy(original_stream_));
}

ScopedStream::ScopedStream(ScopedStream&& other) noexcept : original_stream_(other.original_stream_) {
  other.original_stream_ = nullptr;
}

ScopedStreamWithPriority::ScopedStreamWithPriority(int priority, const char* name) {
  int leastPriority    = 0;
  int greatestPriority = 0;
  cudaCheckError(hipDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority));

  const int clampedPriority = std::max(greatestPriority, std::min(leastPriority, priority));
  cudaCheckError(hipStreamCreateWithPriority(&stream_, hipStreamNonBlocking, clampedPriority));
  if (name != nullptr) {
    nvtxNameCudaStreamA(stream_, name);
  }
}

ScopedStreamWithPriority::~ScopedStreamWithPriority() noexcept {
  if (stream_ == nullptr) {
    return;
  }
  cudaCheckErrorNoThrow(hipStreamSynchronize(stream_));
  cudaCheckErrorNoThrow(hipStreamDestroy(stream_));
}

ScopedStreamWithPriority::ScopedStreamWithPriority(ScopedStreamWithPriority&& other) noexcept : stream_(other.stream_) {
  other.stream_ = nullptr;
}

ScopedStreamWithPriority& ScopedStreamWithPriority::operator=(ScopedStreamWithPriority&& other) noexcept {
  if (stream_ != nullptr && stream_ != other.stream_) {
    cudaCheckErrorNoThrow(hipStreamSynchronize(stream_));
    cudaCheckErrorNoThrow(hipStreamDestroy(stream_));
  }
  stream_       = other.stream_;
  other.stream_ = nullptr;
  return *this;
}

ScopedCudaEvent::ScopedCudaEvent() {
  cudaCheckError(hipEventCreateWithFlags(&original_event_, hipEventDisableTiming));
}

ScopedCudaEvent::~ScopedCudaEvent() noexcept {
  if (original_event_ == nullptr) {
    return;
  }
  cudaCheckErrorNoThrow(hipEventDestroy(original_event_));
}

ScopedCudaEvent::ScopedCudaEvent(ScopedCudaEvent&& other) noexcept : original_event_(other.original_event_) {
  other.original_event_ = nullptr;
}

ScopedCudaEvent& ScopedCudaEvent::operator=(ScopedCudaEvent&& other) noexcept {
  if (original_event_ != nullptr && original_event_ != other.original_event_) {
    cudaCheckErrorNoThrow(hipEventDestroy(original_event_));
  }
  original_event_       = other.original_event_;
  other.original_event_ = nullptr;
  return *this;
}

ScopedStream& ScopedStream::operator=(ScopedStream&& other) noexcept {
  if (original_stream_ != nullptr && original_stream_ != other.original_stream_) {
    cudaCheckErrorNoThrow(hipStreamDestroy(original_stream_));
  }
  original_stream_       = other.original_stream_;
  other.original_stream_ = nullptr;
  return *this;
}

}  // namespace nvMolKit
