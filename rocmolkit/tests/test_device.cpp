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

#include <hip/hip_runtime.h>
#include <gtest/gtest.h>

#include "cuda_error_check.h"
#include "device.h"

using namespace nvMolKit;

TEST(Device, SetDevice) {
  WithDevice wd(0);
  int        device_id;
  cudaCheckError(hipGetDevice(&device_id)) EXPECT_EQ(device_id, 0);
}

TEST(Device, SetDeviceSwap) {
  int nDevices;
  cudaCheckError(hipGetDeviceCount(&nDevices)) if (nDevices < 2) {
    GTEST_SKIP();
  }
  auto wd = std::make_unique<WithDevice>(1);
  int  device_id;
  cudaCheckError(hipGetDevice(&device_id)) EXPECT_EQ(device_id, 1);
  wd.reset();
  cudaCheckError(hipGetDevice(&device_id))

    EXPECT_EQ(device_id, 0);
}

TEST(Device, GetDeviceFreeMemory) {
  size_t free = getDeviceFreeMemory();
  EXPECT_GT(free, 0);

  int* toAllocate = nullptr;
  // Allocate a megabyte
  cudaCheckError(hipMalloc(&toAllocate, 1000000));
  size_t freeAfter = getDeviceFreeMemory();
  EXPECT_LT(freeAfter, free);
  cudaCheckError(hipFree(toAllocate));
}

TEST(DeviceTest, RoundUpToNearestMultipleOfTwo) {
  EXPECT_EQ(nvMolKit::roundUpToNearestMultipleOfTwo(5), 6);
  EXPECT_EQ(nvMolKit::roundUpToNearestMultipleOfTwo(2), 2);
  EXPECT_EQ(nvMolKit::roundUpToNearestMultipleOfTwo(15), 16);
  EXPECT_EQ(nvMolKit::roundUpToNearestMultipleOfTwo(0), 0);
  EXPECT_EQ(nvMolKit::roundUpToNearestMultipleOfTwo(1), 2);
  EXPECT_EQ(nvMolKit::roundUpToNearestMultipleOfTwo(1023), 1024);
}

TEST(DeviceTest, RoundUpToNearestPowerOfTwo) {
  EXPECT_EQ(nvMolKit::roundUpToNearestPowerOfTwo(5), 8);
  EXPECT_EQ(nvMolKit::roundUpToNearestPowerOfTwo(2), 2);
  EXPECT_EQ(nvMolKit::roundUpToNearestPowerOfTwo(15), 16);
  EXPECT_EQ(nvMolKit::roundUpToNearestPowerOfTwo(0), 1);
  EXPECT_EQ(nvMolKit::roundUpToNearestPowerOfTwo(1), 1);
  EXPECT_EQ(nvMolKit::roundUpToNearestPowerOfTwo(1023), 1024);
}

TEST(DeviceTest, ScopedStream) {
  ScopedStream stream;
  hipStream_t s = stream.stream();
  EXPECT_NE(s, nullptr);
}

TEST(DeviceTest, ScopedStreamMove) {
  ScopedStream stream;
  ScopedStream stream2(std::move(stream));
  EXPECT_EQ(stream.stream(), nullptr);
  EXPECT_NE(stream2.stream(), nullptr);
}

TEST(DeviceTest, AcquireExternalStreamZeroIsDefaultStream) {
  auto result = acquireExternalStream(0);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(*result, nullptr);
}

TEST(DeviceTest, AcquireExternalStreamValidStream) {
  ScopedStream scoped;
  auto         streamPtr = reinterpret_cast<std::uintptr_t>(scoped.stream());
  auto         result    = acquireExternalStream(streamPtr);
  ASSERT_TRUE(result.has_value());
  EXPECT_EQ(*result, scoped.stream());
}

TEST(DeviceTest, AcquireExternalStreamWrongDevice) {
  int nDevices = 0;
  cudaCheckError(hipGetDeviceCount(&nDevices));
  if (nDevices < 2) {
    GTEST_SKIP() << "Need at least 2 GPUs to test cross-device stream rejection";
  }
  // Create a stream on device 1
  hipStream_t foreignStream = nullptr;
  cudaCheckError(hipSetDevice(1));
  cudaCheckError(hipStreamCreate(&foreignStream));
  // Switch back to device 0 and try to acquire
  cudaCheckError(hipSetDevice(0));
  auto result = acquireExternalStream(reinterpret_cast<std::uintptr_t>(foreignStream));
  EXPECT_FALSE(result.has_value());
  // Cleanup
  cudaCheckError(hipSetDevice(1));
  cudaCheckError(hipStreamDestroy(foreignStream));
  cudaCheckError(hipSetDevice(0));
}
