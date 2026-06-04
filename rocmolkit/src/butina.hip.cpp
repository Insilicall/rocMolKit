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

#include <hip/hip_cooperative_groups.h>

#include <hipcub/hipcub.hpp>

#include <algorithm>
#include <numeric>
#include <vector>

#include "butina.h"
#include "cub_helpers.hip.h"
#include "host_vector.h"
#include "nvtx.h"

/**
 * TODO: Future optimizations
 * - Keep a live list of active indices and only dispatch counts for those.
 */
namespace nvMolKit {

namespace {
constexpr int blockSizeCount            = 256;
constexpr int kSubTileSize              = 8;
constexpr int kMinLoopSizeForAssignment = 2;

__device__ __forceinline__ void sumCountsAndStoreClusterSize(const int                  tid,
                                                             const int                  pointIdx,
                                                             const cuda::std::span<int> clusterSizes,
                                                             const int                  localCount) {
  __shared__ hipcub::BlockReduce<int, blockSizeCount>::TempStorage tempStorage;
  const int totalCount = hipcub::BlockReduce<int, blockSizeCount>(tempStorage).Sum(localCount);
  if (tid == 0) {
    clusterSizes[pointIdx] = totalCount;
  }
}

//! Kernel to count the size of each cluster around each point
//! Assigns singleton clusters to a sentinel value for later processing.
//! Looks up and skips finished clusters.
__global__ void butinaKernelCountClusterSize(const cuda::std::span<const uint8_t> hitMatrix,
                                             const cuda::std::span<int>           clusters,
                                             const cuda::std::span<int>           clusterSizes) {
  const auto tid       = static_cast<int>(threadIdx.x);
  const auto pointIdx  = static_cast<int>(blockIdx.x);
  const auto numPoints = static_cast<int>(clusters.size());

  if (clusters[pointIdx] >= 0) {
    clusterSizes[pointIdx] = 0;
    return;
  }

  const cuda::std::span<const uint8_t> hits = hitMatrix.subspan(static_cast<size_t>(pointIdx) * numPoints, numPoints);
  int                                  localCount = 0;
  for (int i = tid; i < numPoints; i += blockSizeCount) {
    if (hits[i]) {
      const int cluster = clusters[i];
      if (cluster < 0) {
        localCount++;
      }
    }
  }

  sumCountsAndStoreClusterSize(tid, pointIdx, clusterSizes, localCount);
}

//! Kernel to count the size of each cluster around each point, assigning a neighborlist for later use.
//! IMPORTANT: This assumes that the maximum cluster size is small enough to fit in the neighborlist, so should only
//! be called when that is known to be true.
template <int NeighborlistMaxSize>
__global__ void butinaKernelCountClusterSizeWithNeighborlist(const cuda::std::span<const uint8_t> hitMatrix,
                                                             const cuda::std::span<int>           clusters,
                                                             const cuda::std::span<int>           clusterSizes,
                                                             const cuda::std::span<int>           neighborList) {
  static_assert(NeighborlistMaxSize % kSubTileSize == 0, "NeighborlistMaxSize must be multiple of kSubTileSize");
  const auto tid       = static_cast<int>(threadIdx.x);
  const auto pointIdx  = static_cast<int>(blockIdx.x);
  const auto numPoints = static_cast<int>(clusters.size());

  __shared__ int neighborlistIndex;
  __shared__ int sharedNeighborlist[NeighborlistMaxSize];

  if (tid == 0) {
    neighborlistIndex = 0;
  }
  if (clusters[pointIdx] >= 0) {
    clusterSizes[pointIdx] = 0;
    return;
  }

  const cuda::std::span<const uint8_t> hits = hitMatrix.subspan(static_cast<size_t>(pointIdx) * numPoints, numPoints);
  int                                  localCount = 0;
  __syncthreads();  // for neighborlistIndex init
  for (int i = tid; i < numPoints; i += blockSizeCount) {
    if (hits[i]) {
      const int cluster = clusters[i];
      if (cluster < 0) {
        localCount++;
        const int index           = atomicAdd(&neighborlistIndex, 1);
        sharedNeighborlist[index] = i;
      }
    }
  }

  // Coalesced write of neighborlist using loop for variable sizes
  __syncthreads();  // for sharedNeighborlist final value
  for (int i = tid; i < NeighborlistMaxSize; i += blockSizeCount) {
    neighborList[pointIdx * NeighborlistMaxSize + i] = (i < neighborlistIndex) ? sharedNeighborlist[i] : -1;
  }

  sumCountsAndStoreClusterSize(tid, pointIdx, clusterSizes, localCount);
}

namespace cg = cooperative_groups;

constexpr int blockSizeAssign      = 128;
constexpr int kTilesPerBlockAssign = blockSizeAssign / kSubTileSize;

template <int NeighborlistMaxSize>
__global__ void attemptAssignClustersFromNeighborlist(const cuda::std::span<int>       clusters,
                                                      const cuda::std::span<const int> clusterSizes,
                                                      const cuda::std::span<const int> neighborList,
                                                      const cuda::std::span<int>       centroids,
                                                      const int*                       designatedMaxIdx,
                                                      int*                             nextClusterIdx) {
  static_assert(NeighborlistMaxSize % kSubTileSize == 0, "NeighborlistMaxSize must be multiple of kSubTileSize");

  const auto     tile8       = cg::tiled_partition<kSubTileSize>(cg::this_thread_block());
  const int      rankInBlock = tile8.meta_group_rank();
  const int      tid         = tile8.thread_rank();
  __shared__ int candidateNeighborsBlock[kTilesPerBlockAssign][NeighborlistMaxSize];
  __shared__ int foundIssueBlock[kTilesPerBlockAssign];

  int* sharedFoundIssue         = &foundIssueBlock[rankInBlock];
  int* sharedCandidateNeighbors = &candidateNeighborsBlock[rankInBlock][0];

  if (tid == 0) {
    foundIssueBlock[rankInBlock] = 0;
  }

  // For global tile index across the grid:
  constexpr int tilesPerBlock = blockSizeAssign / kSubTileSize;
  const int     pointIdx      = blockIdx.x * tilesPerBlock + rankInBlock;
  if (pointIdx >= clusters.size()) {
    return;
  }

  const int clustId = clusters[pointIdx];
  if (clustId >= 0) {
    return;
  }
  const int clusterSize     = clusterSizes[pointIdx];
  const int isDesignatedMax = (pointIdx == *designatedMaxIdx);

  // Load neighborlist into shared memory using loop for variable sizes
  for (int i = tid; i < NeighborlistMaxSize; i += kSubTileSize) {
    sharedCandidateNeighbors[i] = neighborList[pointIdx * NeighborlistMaxSize + i];
  }
  tile8.sync();

  for (int i = 0; i < clusterSize; i++) {
    const int candidateNeighbor            = sharedCandidateNeighbors[i];
    const int candidateNeighborClusterSize = clusterSizes[candidateNeighbor];

    // If neighbor has larger cluster, they should be processed instead
    if (candidateNeighborClusterSize > clusterSize) {
      return;
    }

    // If neighbor has SAME cluster size and lower index, defer to them for consistency
    // Also defer if neighbor is the designated max (guarantees only designated max assigns among ties)
    // Designated max itself skips this check to guarantee forward progress
    if (!isDesignatedMax && candidateNeighborClusterSize == clusterSize &&
        (candidateNeighbor < pointIdx || candidateNeighbor == *designatedMaxIdx)) {
      return;
    }

    // If neighbor has smaller cluster size, we're the better centroid - continue

    // Now we verify that all of these neighbors have the same or fewer neighbors we do. Each thread checks 1 candidate
    // at a time. This will rule out our neighbors being connected to a larger cluster.
    for (int oidx = tid; oidx < candidateNeighborClusterSize; oidx += kSubTileSize) {
      const int otherNeighbor = neighborList[candidateNeighbor * NeighborlistMaxSize + oidx];
      bool      foundMatch    = false;
      // One of the neighbors will be ourselves, by definition.
      if (otherNeighbor == pointIdx) {
        foundMatch = true;
      } else {
        for (int j = 0; j < clusterSize; j++) {
          if (otherNeighbor == sharedCandidateNeighbors[j]) {
            foundMatch = true;
            break;
          }
        }
      }
      if (!foundMatch) {
        // We might still be ok if that neighbor is a smaller cluster.
        // Designated max only bails on strictly larger (which can't happen for the true max).
        if (clusterSizes[otherNeighbor] > clusterSize ||
            (clusterSizes[otherNeighbor] == clusterSize && !isDesignatedMax)) {
          atomicExch(sharedFoundIssue, 1);
        }
      }
    }
    tile8.sync();
    if (*sharedFoundIssue) {
      return;
    }
  }

  // At this point, we have a valid cluster. Assign it.
  int clusterVal;
  if (tid == 0) {
    clusterVal         = atomicAdd(nextClusterIdx, 1);
    clusters[pointIdx] = clusterVal;
    if (!centroids.empty()) {
      centroids[clusterVal] = pointIdx;
    }
  }
  tile8.sync();
  clusterVal = tile8.shfl(clusterVal, 0);
  // Assign neighbors using loop for variable sizes
  for (int i = tid; i < clusterSize; i += kSubTileSize) {
    const int assignIdx = sharedCandidateNeighbors[i];
    if (clusters[assignIdx] < 0) {
      clusters[assignIdx] = clusterVal;
    }
  }
}

//! Kernel to write the cluster assignment for the largest cluster found
__global__ void butinaWriteClusterValue(const cuda::std::span<const uint8_t> hitMatrix,
                                        const cuda::std::span<int>           clusters,
                                        const cuda::std::span<int>           centroids,
                                        const int*                           centralIdx,
                                        const int*                           clusterIdx,
                                        const int*                           maxClusterSize) {
  const size_t numPoints = clusters.size();
  const size_t tid       = threadIdx.x + blockIdx.x * blockDim.x;
  const int    clusterSz = *maxClusterSize;
  if (clusterSz < kMinLoopSizeForAssignment) {
    return;
  }
  const int pointIdx = *centralIdx;
  if (pointIdx < 0) {
    return;
  }
  const int                            clusterVal = *clusterIdx;
  const cuda::std::span<const uint8_t> hits = hitMatrix.subspan(static_cast<size_t>(pointIdx) * numPoints, numPoints);
  if (tid < numPoints) {
    if (hits[tid]) {
      if (clusters[tid] < 0) {
        clusters[tid] = clusterVal;
      }
    }
  }
  if (tid == 0 && !centroids.empty()) {
    centroids[clusterVal] = pointIdx;
  }
}

//! Kernel to increment cluster index after assignment. Must be launched with <<<1, 1>>>.
__global__ void bumpClusterIdxKernel(int* clusterIdx, const int* lastClusterSize) {
  if (*lastClusterSize >= kMinLoopSizeForAssignment) {
    *clusterIdx += 1;
  }
}

constexpr int kSingletonBlockSize = 512;

//! Assign all remaining unassigned points their own singleton cluster IDs.
__global__ void assignSingletonIdsKernel(const cuda::std::span<int> clusters,
                                         const cuda::std::span<int> centroids,
                                         int*                       nextClusterIdx) {
  __shared__ int sharedClusterIdx;
  const int      tid       = threadIdx.x;
  const int      numPoints = static_cast<int>(clusters.size());

  if (tid == 0) {
    sharedClusterIdx = *nextClusterIdx;
  }
  __syncthreads();

  for (int i = tid; i < numPoints; i += kSingletonBlockSize) {
    if (clusters[i] < 0) {
      const int myClusterIdx = atomicAdd(&sharedClusterIdx, 1);
      clusters[i]            = myClusterIdx;
      if (!centroids.empty()) {
        centroids[myClusterIdx] = i;
      }
    }
  }

  __syncthreads();
  if (tid == 0) {
    *nextClusterIdx = sharedClusterIdx;
  }
}

//! Count the size of each cluster and store the result in clusterSizes.
__global__ void countClusterSizesKernel(const cuda::std::span<const int> clusters,
                                        const cuda::std::span<int>       clusterSizes) {
  const int numPoints = static_cast<int>(clusters.size());
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < numPoints; i += blockDim.x * gridDim.x) {
    const int clusterId = clusters[i];
    atomicAdd(&clusterSizes[clusterId], 1);
  }
}

//! Apply the remapping to all cluster assignments.
__global__ void applyNewIndices(const cuda::std::span<int> clusters, const cuda::std::span<const int> remap) {
  const int numPoints = static_cast<int>(clusters.size());
  const int tid       = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (tid < numPoints) {
    clusters[tid] = remap[clusters[tid]];
  }
}

__global__ void remapCentroidsKernel(const cuda::std::span<const int> sortedOriginalIds,
                                     const cuda::std::span<const int> centroids,
                                     const cuda::std::span<int>       remappedCentroids) {
  const int numClusters = static_cast<int>(sortedOriginalIds.size());
  const int idx         = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (idx < numClusters) {
    const int originalId   = sortedOriginalIds[idx];
    remappedCentroids[idx] = centroids[originalId];
  }
}

/**
 * @brief Renumber cluster IDs so larger clusters have smaller IDs
 *
 * 1. Maps clusters by size to cluster ID.
 * 2. Then sorts by size (descending)
 * 3. Creates mapping of old ID -> new ID based on sorted order.
 * 4. Applies new IDs to all points.
 */
void renumberClustersBySize(const cuda::std::span<int> clusters,
                            const cuda::std::span<int> centroids,
                            const int                  numClusters,
                            hipStream_t               stream) {
  if (numClusters <= 1) {
    return;
  }

  const int numPoints = static_cast<int>(clusters.size());

  AsyncDeviceVector<int> clusterSizes(numClusters, stream);
  clusterSizes.zero();

  constexpr int blockSize         = 256;
  const int     numBlocksRenumber = (numClusters + blockSize - 1) / blockSize;

  // Count cluster sizes
  countClusterSizesKernel<<<numBlocksRenumber, blockSize, 0, stream>>>(clusters, toSpan(clusterSizes));
  cudaCheckError(hipGetLastError());

  // Argsort cluster ids by descending size on the host. numClusters is small
  // (<= numPoints) and this runs once, at the very end. A device-wide
  // hipcub::DeviceRadixSort::SortPairs proved unreliable here for these inputs
  // (returned a garbled permutation, corrupting the remap), and a stable host
  // sort is trivially correct. Stable order ties-breaks by original id, matching
  // the previous (negative-size, original-id) key design.
  std::vector<int> sizes(numClusters);
  cudaCheckError(
    hipMemcpyAsync(sizes.data(), clusterSizes.data(), numClusters * sizeof(int), hipMemcpyDefault, stream));
  cudaCheckError(hipStreamSynchronize(stream));

  std::vector<int> sortedOriginalIds(numClusters);
  std::iota(sortedOriginalIds.begin(), sortedOriginalIds.end(), 0);
  std::stable_sort(sortedOriginalIds.begin(), sortedOriginalIds.end(), [&](int a, int b) {
    return sizes[a] > sizes[b];
  });

  std::vector<int> remap(numClusters);
  for (int newId = 0; newId < numClusters; ++newId) {
    remap[sortedOriginalIds[newId]] = newId;
  }

  AsyncDeviceVector<int> remapDev(numClusters, stream);
  cudaCheckError(hipMemcpyAsync(remapDev.data(), remap.data(), numClusters * sizeof(int), hipMemcpyDefault, stream));

  // Apply new indices to all points
  const int numBlocks = (numPoints + blockSize - 1) / blockSize;
  applyNewIndices<<<numBlocks, blockSize, 0, stream>>>(clusters, toSpan(remapDev));
  cudaCheckError(hipGetLastError());

  if (!centroids.empty()) {
    AsyncDeviceVector<int> sortedOriginalIdsDev(numClusters, stream);
    cudaCheckError(hipMemcpyAsync(sortedOriginalIdsDev.data(),
                                  sortedOriginalIds.data(),
                                  numClusters * sizeof(int),
                                  hipMemcpyDefault,
                                  stream));
    AsyncDeviceVector<int> remappedCentroids(numClusters, stream);
    remapCentroidsKernel<<<numBlocksRenumber, blockSize, 0, stream>>>(toSpan(sortedOriginalIdsDev),
                                                                      centroids,
                                                                      toSpan(remappedCentroids));
    cudaCheckError(hipGetLastError());
    cudaCheckError(hipMemcpyAsync(centroids.data(),
                                   remappedCentroids.data(),
                                   numClusters * sizeof(int),
                                   hipMemcpyDeviceToDevice,
                                   stream));
  }

  // Host buffers (remap, sortedOriginalIds) feed async H2D copies above; the
  // caller syncs after we return, but these locals would be gone by then.
  cudaCheckError(hipStreamSynchronize(stream));
}

}  // namespace

#if CUB_VERSION < 200800
constexpr int argMaxBlockSize = 512;

//! Custom ArgMax kernel that returns the largest value and index.
//! Used when CUB's new ArgMax API is not available (CCCL < 2.8.0)
__global__ void lastArgMaxKernel(const int* values, int numItems, int* outVal, int* outIdx) {
  int            maxVal = cuda::std::numeric_limits<int>::min();
  int            maxID  = -1;
  __shared__ int foundMaxVal[argMaxBlockSize];
  __shared__ int foundMaxIds[argMaxBlockSize];
  const auto     tid = static_cast<int>(threadIdx.x);
  for (int i = tid; i < numItems; i += argMaxBlockSize) {
    if (const int val = values[i]; val >= maxVal) {
      maxID  = i;
      maxVal = val;
    }
  }
  foundMaxVal[tid] = maxVal;
  foundMaxIds[tid] = maxID;

  __shared__ hipcub::BlockReduce<int, argMaxBlockSize>::TempStorage storage;
  const int actualMaxVal = hipcub::BlockReduce<int, argMaxBlockSize>(storage).Reduce(maxVal, cubMax());
  __syncthreads();  // For shared memory write of maxVal and maxID
  if (tid == 0) {
    *outVal = actualMaxVal;
    for (int i = argMaxBlockSize - 1; i >= 0; i--) {
      if (foundMaxVal[i] == actualMaxVal) {
        *outIdx = foundMaxIds[i];
        break;
      }
    }
  }
}
#endif  // CUB_VERSION < 200800

//! Helper class to run ArgMax on device data.
//! Uses CUB's DeviceReduce::ArgMax when available (CCCL >= 2.8.0), otherwise falls back to custom kernel.
class ArgMaxRunner {
 public:
  ArgMaxRunner([[maybe_unused]] size_t num_items, hipStream_t stream)
      : stream_(stream)
#if CUB_VERSION >= 200800
        ,
        temp_storage_(getTempStorageSize(num_items, stream), stream)
#endif
  {
  }

  void operator()(int* d_in, int* d_max_value_out, int* d_max_index_out, int num_items) {
#if CUB_VERSION >= 200800
    size_t temp_storage_bytes = temp_storage_.size();
    cudaCheckError(hipcub::DeviceReduce::ArgMax(temp_storage_.data(),
                                             temp_storage_bytes,
                                             d_in,
                                             d_max_value_out,
                                             d_max_index_out,
                                             static_cast<int64_t>(num_items),
                                             stream_));
#else
    lastArgMaxKernel<<<1, argMaxBlockSize, 0, stream_>>>(d_in, num_items, d_max_value_out, d_max_index_out);
    cudaCheckError(hipGetLastError());
#endif
  }

  //! Run ArgMax on a specific stream (used during graph capture)
  void captureOn(hipStream_t captureStream, int* d_in, int* d_max_value_out, int* d_max_index_out, int num_items) {
#if CUB_VERSION >= 200800
    size_t temp_storage_bytes = temp_storage_.size();
    cudaCheckError(hipcub::DeviceReduce::ArgMax(temp_storage_.data(),
                                             temp_storage_bytes,
                                             d_in,
                                             d_max_value_out,
                                             d_max_index_out,
                                             static_cast<int64_t>(num_items),
                                             captureStream));
#else
    lastArgMaxKernel<<<1, argMaxBlockSize, 0, captureStream>>>(d_in, num_items, d_max_value_out, d_max_index_out);
    cudaCheckError(hipGetLastError());
#endif
  }

 private:
#if CUB_VERSION >= 200800
  static size_t getTempStorageSize(size_t num_items, hipStream_t stream) {
    size_t temp_storage_bytes = 0;
    hipcub::DeviceReduce::ArgMax(nullptr,
                              temp_storage_bytes,
                              static_cast<int*>(nullptr),
                              static_cast<int*>(nullptr),
                              static_cast<int*>(nullptr),
                              static_cast<int64_t>(num_items),
                              stream);
    return temp_storage_bytes;
  }
#endif

  hipStream_t stream_;
#if CUB_VERSION >= 200800
  AsyncDeviceVector<uint8_t> temp_storage_;
#endif
};

/**
 * @brief Prune neighborlists by removing assigned neighbors and reordering.
 */
// One 32-thread cooperative-groups tile processes one point. The original used
// hipcub::WarpMergeSort + WarpReduce to compact and count, but those warp-level
// primitives assume the physical wavefront width: on AMD wave64 a default
// hipcub::WarpReduce sums 64 lanes (two of our 32-lane tiles), corrupting the
// neighbor counts and merging unrelated points into one giant cluster. The
// compaction does not need a sort — the valid neighbors only need to occupy the
// first newCount slots, in any order — so we compact in shared memory with a
// per-tile atomic counter. This is wave-size agnostic (a 32-tile is valid on
// both wave32 and wave64) and uses no hipcub warp primitives.
template <int NeighborlistMaxSize>
__global__ void pruneNeighborlistKernel(const cuda::std::span<int> clusters,
                                        const cuda::std::span<int> clusterSizes,
                                        const cuda::std::span<int> neighborList) {
  constexpr int kTileSize      = 32;
  constexpr int kWarpsPerBlock = 4;
  static_assert(NeighborlistMaxSize <= 128, "NeighborlistMaxSize must be <= 128");
  static_assert(NeighborlistMaxSize % 8 == 0, "NeighborlistMaxSize must be multiple of 8");

  __shared__ int compacted[kWarpsPerBlock][NeighborlistMaxSize];
  __shared__ int validCount[kWarpsPerBlock];

  const auto tile     = cg::tiled_partition<kTileSize>(cg::this_thread_block());
  const int  tid      = tile.thread_rank();
  const int  warpId   = tile.meta_group_rank();
  const int  pointIdx = blockIdx.x * kWarpsPerBlock + warpId;

  if (pointIdx >= static_cast<int>(clusters.size())) {
    return;
  }
  if (clusters[pointIdx] >= 0) {
    clusterSizes[pointIdx] = 0;
    return;
  }

  const int currentSize = clusterSizes[pointIdx];
  const int baseOffset   = pointIdx * NeighborlistMaxSize;

  if (tid == 0) {
    validCount[warpId] = 0;
  }
  tile.sync();

  // Compact still-unassigned neighbors to the front (order among them is irrelevant).
  for (int i = tid; i < NeighborlistMaxSize; i += kTileSize) {
    const int neighbor = neighborList[baseOffset + i];
    const bool valid   = (i < currentSize) && (neighbor >= 0) && (clusters[neighbor] < 0);
    if (valid) {
      const int pos             = atomicAdd(&validCount[warpId], 1);
      compacted[warpId][pos]    = neighbor;
    }
  }
  tile.sync();

  const int newCount = validCount[warpId];
  for (int i = tid; i < NeighborlistMaxSize; i += kTileSize) {
    neighborList[baseOffset + i] = (i < newCount) ? compacted[warpId][i] : -1;
  }
  if (tid == 0) {
    clusterSizes[pointIdx] = newCount;
  }
}

// TODO - consolidate this to device vector code.
template <typename T> __global__ void setAllKernel(const size_t numElements, T value, T* dst) {
  const size_t idx = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (idx < numElements) {
    dst[idx] = value;
  }
}
template <typename T> void setAll(const cuda::std::span<T>& vec, const T& value, hipStream_t stream) {
  const size_t numElements = vec.size();
  if (numElements == 0) {
    return;
  }
  constexpr int blockSize = 128;
  const size_t  numBlocks = (numElements + blockSize - 1) / blockSize;
  setAllKernel<<<numBlocks, blockSize, 0, stream>>>(numElements, value, vec.data());
  cudaCheckError(hipGetLastError());
}

//! Inner loop iteration for Butina clustering.
void innerButinaLoop(const int                            numPoints,
                     const cuda::std::span<const uint8_t> hitMatrix,
                     const cuda::std::span<int>           clusters,
                     const cuda::std::span<int>           clusterSizesSpan,
                     const cuda::std::span<int>           centroids,
                     int*                                 maxIndexPtr,
                     int*                                 maxValuePtr,
                     int*                                 clusterIdxPtr,
                     ArgMaxRunner&                        argMaxRunner,
                     hipStream_t                         stream) {
  const int numBlocksFlat = ((static_cast<int>(clusterSizesSpan.size()) - 1) / blockSizeCount) + 1;

  butinaKernelCountClusterSize<<<numPoints, blockSizeCount, 0, stream>>>(hitMatrix, clusters, clusterSizesSpan);
  cudaCheckError(hipGetLastError());

  argMaxRunner.captureOn(stream,
                         clusterSizesSpan.data(),
                         maxValuePtr,
                         maxIndexPtr,
                         static_cast<int>(clusterSizesSpan.size()));

  butinaWriteClusterValue<<<numBlocksFlat, blockSizeCount, 0, stream>>>(hitMatrix,
                                                                        clusters,
                                                                        centroids,
                                                                        maxIndexPtr,
                                                                        clusterIdxPtr,
                                                                        maxValuePtr);
  cudaCheckError(hipGetLastError());
  bumpClusterIdxKernel<<<1, 1, 0, stream>>>(clusterIdxPtr, maxValuePtr);
  cudaCheckError(hipGetLastError());
}

//! Inner loop iteration that attempts assignment then prunes neighborlists.
template <int NeighborlistMaxSize>
void innerButinaLoopWithPruning(const int                  numPoints,
                                const cuda::std::span<int> clusters,
                                const cuda::std::span<int> clusterSizesSpan,
                                const cuda::std::span<int> centroids,
                                int*                       maxIndexPtr,
                                int*                       maxValuePtr,
                                int*                       clusterIdxPtr,
                                const cuda::std::span<int> neighborList,
                                ArgMaxRunner&              argMaxRunner,
                                hipStream_t               stream) {
  const int numBlocksAssign = (numPoints + kTilesPerBlockAssign - 1) / kTilesPerBlockAssign;
  attemptAssignClustersFromNeighborlist<NeighborlistMaxSize>
    <<<numBlocksAssign, blockSizeAssign, 0, stream>>>(clusters,
                                                      clusterSizesSpan,
                                                      neighborList,
                                                      centroids,
                                                      maxIndexPtr,
                                                      clusterIdxPtr);
  cudaCheckError(hipGetLastError());

  // Prune assigned neighbors from all neighborlists and update counts
  constexpr int kWarpsPerBlock  = 4;
  constexpr int kPruneBlockSize = kWarpsPerBlock * 32;
  const int     numBlocksPrune  = (numPoints + kWarpsPerBlock - 1) / kWarpsPerBlock;
  pruneNeighborlistKernel<NeighborlistMaxSize>
    <<<numBlocksPrune, kPruneBlockSize, 0, stream>>>(clusters, clusterSizesSpan, neighborList);
  cudaCheckError(hipGetLastError());

  // Compute argmax for next iteration
  argMaxRunner.captureOn(stream,
                         clusterSizesSpan.data(),
                         maxValuePtr,
                         maxIndexPtr,
                         static_cast<int>(clusterSizesSpan.size()));
}

//! Host-driven replacement for the conditional-WHILE CUDA Graph node, which
//! HIP/ROCm does not support. Runs the inner Butina loop body on the stream,
//! copies the resulting maxValue back to the host, and decides on the CPU
//! whether to iterate again. Do-while semantics: the body always runs at least
//! once (matching the graph's default conditional handle = 1). The per-iteration
//! sync is the documented tradeoff; iteration count equals the number of large
//! clusters, which is small.
void runInnerButinaLoopHost(const int                            numPoints,
                            const cuda::std::span<const uint8_t> hitMatrix,
                            const cuda::std::span<int>           clusters,
                            const cuda::std::span<int>           clusterSizesSpan,
                            const cuda::std::span<int>           centroids,
                            int*                                 maxIndexPtr,
                            int*                                 maxValuePtr,
                            int*                                 clusterIdxPtr,
                            const int                            threshold,
                            int*                                 hostMaxValue,
                            ArgMaxRunner&                        argMaxRunner,
                            hipStream_t                          stream) {
  do {
    innerButinaLoop(numPoints,
                    hitMatrix,
                    clusters,
                    clusterSizesSpan,
                    centroids,
                    maxIndexPtr,
                    maxValuePtr,
                    clusterIdxPtr,
                    argMaxRunner,
                    stream);
    cudaCheckError(hipMemcpyAsync(hostMaxValue, maxValuePtr, sizeof(int), hipMemcpyDefault, stream));
    cudaCheckError(hipStreamSynchronize(stream));
  } while (*hostMaxValue >= threshold);
}

//! Host-driven replacement for the pruning conditional-WHILE CUDA Graph node.
//! Same do-while semantics as runInnerButinaLoopHost; continues while the
//! largest remaining cluster is still big enough to assign (>= kMinLoopSizeForAssignment).
template <int NeighborlistMaxSize>
void runPruningButinaLoopHost(const int                  numPoints,
                              const cuda::std::span<int> clusters,
                              const cuda::std::span<int> clusterSizesSpan,
                              const cuda::std::span<int> neighborListSpan,
                              const cuda::std::span<int> centroids,
                              int*                       maxIndexPtr,
                              int*                       maxValuePtr,
                              int*                       clusterIdxPtr,
                              int*                       hostMaxValue,
                              ArgMaxRunner&              argMaxRunner,
                              hipStream_t                stream) {
  do {
    innerButinaLoopWithPruning<NeighborlistMaxSize>(numPoints,
                                                    clusters,
                                                    clusterSizesSpan,
                                                    centroids,
                                                    maxIndexPtr,
                                                    maxValuePtr,
                                                    clusterIdxPtr,
                                                    neighborListSpan,
                                                    argMaxRunner,
                                                    stream);
    cudaCheckError(hipMemcpyAsync(hostMaxValue, maxValuePtr, sizeof(int), hipMemcpyDefault, stream));
    cudaCheckError(hipStreamSynchronize(stream));
  } while (*hostMaxValue >= kMinLoopSizeForAssignment);
}

/**
 * @brief Build the initial neighborlist and cluster sizes from the hit matrix.
 *
 * This is called once before entering the pruning loop.
 */
template <int NeighborlistMaxSize>
void buildInitialNeighborlist(const int                            numPoints,
                              const cuda::std::span<const uint8_t> hitMatrix,
                              const cuda::std::span<int>           clusters,
                              const cuda::std::span<int>           clusterSizesSpan,
                              const cuda::std::span<int>           neighborList,
                              hipStream_t                         stream) {
  const ScopedNvtxRange range("Build initial neighborlist");
  butinaKernelCountClusterSizeWithNeighborlist<NeighborlistMaxSize>
    <<<numPoints, blockSizeCount, 0, stream>>>(hitMatrix, clusters, clusterSizesSpan, neighborList);
  cudaCheckError(hipGetLastError());
  cudaCheckError(hipStreamSynchronize(stream));
}

template <int NeighborlistMaxSize>
[[maybe_unused]] int butinaGpuImpl(const cuda::std::span<const uint8_t> hitMatrix,
                                   const cuda::std::span<int>           clusters,
                                   const cuda::std::span<int>           centroids,
                                   hipStream_t                         stream) {
  ScopedNvtxRange setupRange("Butina Setup");
  const size_t    numPoints = clusters.size();
  if (!centroids.empty() && centroids.size() != numPoints) {
    throw std::invalid_argument("Centroids size mismatch: " + std::to_string(centroids.size()) +
                                " != " + std::to_string(numPoints));
  }
  setAll(clusters, -1, stream);
  if (const size_t matSize = hitMatrix.size(); numPoints * numPoints != matSize) {
    throw std::runtime_error("Butina size mismatch" + std::to_string(numPoints) +
                             " points^2 != " + std::to_string(matSize) + " neighbor matrix size");
  }
  AsyncDeviceVector<int> clusterSizes(clusters.size(), stream);
  clusterSizes.zero();
  AsyncDeviceVector<int> neighborList(NeighborlistMaxSize * numPoints, stream);
  const auto             neighborListSpan = toSpan(neighborList);

  const AsyncDevicePtr<int> maxIndex(-1, stream);
  const AsyncDevicePtr<int> maxValue(std::numeric_limits<int>::max(), stream);
  const AsyncDevicePtr<int> clusterIdx(0, stream);
  PinnedHostVector<int>     maxCluster(1);
  maxCluster[0] = std::numeric_limits<int>::max();

  ArgMaxRunner argMaxRunner(clusters.size(), stream);

  setupRange.pop();
  const auto clusterSizesSpan = toSpan(clusterSizes);

  // If a neighborlist is up to N, then the cluster is up to N+1 (including the central point).
  constexpr int clusterSizeWithMaxNeighborlist = NeighborlistMaxSize + 1;

  // Host-driven loop control (HIP/ROCm has no conditional-WHILE graph node).
  // The CPU reads maxValue back each iteration and decides when to exit.
  {
    const ScopedNvtxRange loopRange("Large cluster Butina Loop (host-driven)");
    runInnerButinaLoopHost(static_cast<int>(numPoints),
                           hitMatrix,
                           clusters,
                           clusterSizesSpan,
                           centroids,
                           maxIndex.data(),
                           maxValue.data(),
                           clusterIdx.data(),
                           clusterSizeWithMaxNeighborlist,
                           maxCluster.data(),
                           argMaxRunner,
                           stream);
    // maxCluster[0] already holds the final maxValue from the loop's last read.
  }

  // Build neighborlist once, then prune dynamically using a host-driven loop
  if (maxCluster[0] >= kMinLoopSizeForAssignment) {
    buildInitialNeighborlist<NeighborlistMaxSize>(numPoints,
                                                  hitMatrix,
                                                  clusters,
                                                  clusterSizesSpan,
                                                  neighborListSpan,
                                                  stream);

    // Initial argmax to prime the loop (buildInitialNeighborlist already synced)
    argMaxRunner(clusterSizesSpan.data(), maxValue.data(), maxIndex.data(), static_cast<int>(clusterSizesSpan.size()));
    cudaCheckError(hipStreamSynchronize(stream));

    // Host-driven pruning loop (HIP/ROCm has no conditional-WHILE graph node).
    const ScopedNvtxRange loopRange("Small cluster Butina Loop with pruning (host-driven)");
    runPruningButinaLoopHost<NeighborlistMaxSize>(numPoints,
                                                  clusters,
                                                  clusterSizesSpan,
                                                  neighborListSpan,
                                                  centroids,
                                                  maxIndex.data(),
                                                  maxValue.data(),
                                                  clusterIdx.data(),
                                                  maxCluster.data(),
                                                  argMaxRunner,
                                                  stream);
  }

  assignSingletonIdsKernel<<<1, kSingletonBlockSize, 0, stream>>>(clusters, centroids, clusterIdx.data());
  cudaCheckError(hipGetLastError());

  // Renumber clusters to be in descending order.
  cudaCheckError(hipMemcpyAsync(maxCluster.data(), clusterIdx.data(), sizeof(int), hipMemcpyDefault, stream));
  cudaCheckError(hipStreamSynchronize(stream));
  renumberClustersBySize(clusters, centroids, maxCluster[0], stream);
  cudaCheckError(hipStreamSynchronize(stream));
  return maxCluster[0];
}

[[maybe_unused]] int butinaGpu(const cuda::std::span<const uint8_t> hitMatrix,
                               const cuda::std::span<int>           clusters,
                               const int                            neighborlistMaxSize,
                               const cuda::std::span<int>           centroids,
                               hipStream_t                         stream) {
  switch (neighborlistMaxSize) {
    case 8:
      return butinaGpuImpl<8>(hitMatrix, clusters, centroids, stream);
    case 16:
      return butinaGpuImpl<16>(hitMatrix, clusters, centroids, stream);
    case 24:
      return butinaGpuImpl<24>(hitMatrix, clusters, centroids, stream);
    case 32:
      return butinaGpuImpl<32>(hitMatrix, clusters, centroids, stream);
    case 64:
      return butinaGpuImpl<64>(hitMatrix, clusters, centroids, stream);
    case 128:
      return butinaGpuImpl<128>(hitMatrix, clusters, centroids, stream);
    default:
      throw std::invalid_argument("neighborlistMaxSize must be 8, 16, 24, 32, 64, or 128. Got: " +
                                  std::to_string(neighborlistMaxSize));
  }
}

namespace {

__global__ void thresholdDistanceMatrixKernel(const double* __restrict__ matrix,
                                              uint8_t* __restrict__ hits,
                                              const double cutoff,
                                              const size_t numElements) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numElements) {
    hits[idx] = (matrix[idx] <= cutoff);
  }
}

}  // namespace

[[maybe_unused]] int butinaGpu(const cuda::std::span<const double> distanceMatrix,
                               const cuda::std::span<int>          clusters,
                               const double                        cutoff,
                               const int                           neighborlistMaxSize,
                               const cuda::std::span<int>          centroids,
                               hipStream_t                        stream) {
  AsyncDeviceVector<uint8_t> hitMatrix(distanceMatrix.size(), stream);

  constexpr int blockSize = 256;
  const size_t  numBlocks = (distanceMatrix.size() + blockSize - 1) / blockSize;
  thresholdDistanceMatrixKernel<<<numBlocks, blockSize, 0, stream>>>(distanceMatrix.data(),
                                                                     hitMatrix.data(),
                                                                     cutoff,
                                                                     distanceMatrix.size());
  cudaCheckError(hipGetLastError());
  return butinaGpu(toSpan(hitMatrix), clusters, neighborlistMaxSize, centroids, stream);
}

}  // namespace nvMolKit