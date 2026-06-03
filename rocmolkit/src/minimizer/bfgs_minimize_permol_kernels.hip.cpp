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

#include <hipcub/hipcub.hpp>

#include "bfgs_minimize_permol_kernels.h"
#include "cub_helpers.hip.h"
#include "device_vector.h"
#include "dist_geom_kernels_device.hip.h"
#include "mmff_kernels.h"
#include "mmff_kernels_device.hip.h"
#include "versions.h"

namespace nvMolKit {

namespace {
// Precision used for the BFGS minimizer's internal working state and the
// force-field math invoked from this kernel. The shared pipeline context
// positions/energies remain double; conversions happen at the kernel boundary.
using MinReal = float;

constexpr int16_t BLOCK_SIZE           = 32;
constexpr int16_t MAX_LINESEARCH_ITERS = 1000;
constexpr float   FUNCTOL              = 1e-4;
constexpr float   MOVETOL              = 1e-7;
constexpr float   TOLX                 = 4.f * 3e-8f;

__device__ void setMaxStep(const MinReal*                                               pos,
                           const int                                                   numTerms,
                           float*                                                      maxStepOutSquared,
                           typename hipcub::BlockReduce<float, BLOCK_SIZE>::TempStorage& tempStorage) {
  float sumSquaredPos = 0.0;
  for (int i = threadIdx.x; i < numTerms; i += blockDim.x) {
    float dx2 = pos[i] * pos[i];
    sumSquaredPos += dx2;
  }
  using BlockReduce = hipcub::BlockReduce<float, BLOCK_SIZE>;

  const float squaredSum = BlockReduce(tempStorage).Sum(sumSquaredPos);
  if (threadIdx.x == 0) {
    constexpr float maxStepFactorSquared = 100.0 * 100.0;
    *maxStepOutSquared =
      maxStepFactorSquared * max(squaredSum, static_cast<float>(numTerms) * static_cast<float>(numTerms));
  }
}

__device__ void lineSearchSetup(const int                                                   numTerms,
                                const MinReal*                                              posStart,
                                const MinReal*                                              gradStart,
                                const float                                                 maxStepSquared,
                                MinReal*                                                    dirStart,
                                float&                                                      slope,
                                float&                                                      lambdaMin,
                                typename hipcub::BlockReduce<float, BLOCK_SIZE>::TempStorage& tempStorage) {
  const int idxInSys = threadIdx.x;
  using BlockReduce  = hipcub::BlockReduce<float, BLOCK_SIZE>;
  __shared__ float dirSumSquared;

  // ---------------------------------
  //  Scale direction vector if needed
  // ---------------------------------
  float sumSquaredLocal = 0.0;
  for (int i = idxInSys; i < numTerms; i += blockDim.x) {
    float dx2 = dirStart[i] * dirStart[i];
    sumSquaredLocal += dx2;
  }
  float blockSum = BlockReduce(tempStorage).Sum(sumSquaredLocal);
  if (idxInSys == 0) {
    dirSumSquared = blockSum;
  }
  __syncthreads();
  if (dirSumSquared > maxStepSquared) {
    const float inverseScaleSquared = dirSumSquared / maxStepSquared;
    const float scale               = rsqrtf(inverseScaleSquared);
    for (int i = idxInSys; i < numTerms; i += blockDim.x) {
      dirStart[i] *= scale;
    }
  }
  __syncthreads();

  // -------------------------
  // Set slope, check validity
  // -------------------------
  float localSum     = 0.0;
  float localGradSum = 0.0;
  float localDirSum  = 0.0;
  // Each thread computes its partial sum
  for (int i = idxInSys; i < numTerms; i += blockDim.x) {
    localSum += dirStart[i] * gradStart[i];
    localGradSum += gradStart[i] * gradStart[i];
    localDirSum += dirStart[i] * dirStart[i];
  }

  // Perform block-wide reduction to compute the total sum
  blockSum = BlockReduce(tempStorage).Sum(localSum);

  // The first thread in the block writes the result
  if (idxInSys == 0) {
    slope = blockSum;
  }
  __syncthreads();

  // ----------------------
  // Compute initial lambda
  // ----------------------
  float localMax_numerator   = 0.0;
  float localMax_denominator = 1.0;
  // Each thread computes its local maximum
  for (int i = idxInSys; i < numTerms; i += blockDim.x) {
    float temp_numerator   = fabs(dirStart[i]);
    float temp_denominator = fmax(fabs(posStart[i]), 1.0);
    // temp_numerator / temp_denominator > localMax_numerator / localMax_denominator
    // <=>
    // temp_numerator * localMax_denominator > localMax_numerator * temp_denominator
    if (temp_numerator * localMax_denominator > localMax_numerator * temp_denominator) {
      localMax_numerator   = temp_numerator;
      localMax_denominator = temp_denominator;
    }
  }

  float localInvMax = localMax_denominator / (localMax_numerator > 0.0f ? localMax_numerator : 1.0e-20f);
  // Perform block-wide reduction to find the maximum
  float blockInvMax = BlockReduce(tempStorage).Reduce(localInvMax, cubMin());

  // The first thread in the block writes the result
  if (threadIdx.x == 0) {
    lambdaMin = MOVETOL * blockInvMax;
  }
}

__device__ void lineSearchPerturb(const int      numTerms,
                                  const MinReal* refPos,
                                  const MinReal* dirStart,
                                  const float    lambda,
                                  MinReal*       scratchPos) {
  for (int i = threadIdx.x; i < numTerms; i += blockDim.x) {
    scratchPos[i] = refPos[i] + lambda * dirStart[i];
  }
  __syncthreads();
}

__device__ bool lineSearchPostEnergy(const bool  isFirstIter,
                                     const float prevE,
                                     const float newE,
                                     const float slope,
                                     const float lambda,
                                     const float lambdaMin,
                                     float&      lambda2,
                                     float&      eScratch,
                                     float&      lambdaOut) {
  bool converged = false;

  if (threadIdx.x == 0) {
    const float eDiff = newE - prevE;
    if (lambda < lambdaMin || eDiff <= FUNCTOL * lambda * slope) {
      converged = true;
    } else {
      float tmpLambda;
      if (isFirstIter) {
        tmpLambda = -slope / (2.0f * (eDiff - slope));
      } else {
        const float rhs1     = eDiff - lambda * slope;
        const float rhs2     = eScratch - prevE - lambda2 * slope;
        const float rLambda  = 1.0f / static_cast<float>(lambda);
        const float rLambda2 = 1.0f / static_cast<float>(lambda2);
        const float rScale   = 1.0f / (lambda - static_cast<float>(lambda2));
        const float a        = (rhs1 * rLambda * rLambda - rhs2 * rLambda2 * rLambda2) * rScale;
        const float b        = (-lambda2 * rhs1 * rLambda * rLambda + lambda * rhs2 * rLambda2 * rLambda2) * rScale;
        if (a == 0.0f) {
          tmpLambda = -slope / (2.0f * b);
        } else {
          const float disc = b * b - 3.0f * a * slope;
          if (disc < 0.0f) {
            tmpLambda = 0.5f * lambda;
          } else {
            const float sqrtDisc = sqrtf(disc);
            tmpLambda            = (b <= 0.0f) ? (-b + sqrtDisc) / (3.0f * a) : -slope / (b + sqrtDisc);
          }
        }
        tmpLambda = fminf(tmpLambda, 0.5f * lambda);
      }
      lambda2   = lambda;
      eScratch  = newE;
      lambdaOut = fmaxf(tmpLambda, 0.1f * lambda);
    }
  }
  __syncthreads();
  return converged;
}

__device__ void setDirection(const int                                                   numTerms,
                             const MinReal*                                              posFromLineSearch,
                             const MinReal*                                              pos,
                             MinReal*                                                    xi,
                             MinReal*                                                    dGrad,
                             const MinReal*                                              grad,
                             bool&                                                       converged,
                             typename hipcub::BlockReduce<float, BLOCK_SIZE>::TempStorage& tempStorage) {
  float localMax_numerator   = 0.0;
  float localMax_denominator = 1.0;
  for (int i = threadIdx.x; i < numTerms; i += blockDim.x) {
    xi[i]    = posFromLineSearch[i] - pos[i];
    dGrad[i] = grad[i];

    float temp_numerator   = fabs(xi[i]);
    float temp_denominator = fmax(fabs(posFromLineSearch[i]), 1.0);
    // temp_numerator / temp_denominator > localMax_numerator / localMax_denominator
    // <=>
    // temp_numerator * localMax_denominator > localMax_numerator * temp_denominator
    if (temp_numerator * localMax_denominator > localMax_numerator * temp_denominator) {
      localMax_numerator   = temp_numerator;
      localMax_denominator = temp_denominator;
    }
  }

  float localMax = localMax_numerator / localMax_denominator;
  float blockMax = hipcub::BlockReduce<float, BLOCK_SIZE>(tempStorage).Reduce(localMax, cubMax());

  if (threadIdx.x == 0 && blockMax < TOLX) {
    converged = true;
  }
  __syncthreads();
}

template <bool scaleGrads>
__device__ void scaleGrad(const int                                                   numTerms,
                          MinReal*                                                    grad,
                          float&                                                      gradScale,
                          typename hipcub::BlockReduce<float, BLOCK_SIZE>::TempStorage& tempStorage) {
  // See scaleGradKernel in bfgs_minimize.cu for the RDKit 5b1d04d23 (2025.09) rationale.
  constexpr bool kRdkitHasGradScaleFix =
    RDKIT_VERSION_MAJOR > 2025 || (RDKIT_VERSION_MAJOR == 2025 && RDKIT_VERSION_MINOR >= 9);
  gradScale = scaleGrads ? 0.1f : 1.0f;

  float maxGrad = kRdkitHasGradScaleFix ? 0.0f : -1e8f;
  for (int i = threadIdx.x; i < numTerms; i += blockDim.x) {
    if constexpr (scaleGrads) {
      grad[i] *= gradScale;
    }
    const float cmp = kRdkitHasGradScaleFix ? fabsf(grad[i]) : grad[i];
    if (cmp > maxGrad) {
      maxGrad = cmp;
    }
  }

  float blockMax = hipcub::BlockReduce<float, BLOCK_SIZE>(tempStorage).Reduce(maxGrad, cubMax());

  __shared__ float distributedMax[1];
  if (threadIdx.x == 0) {
    distributedMax[0] = blockMax;
  }
  __syncthreads();

  maxGrad = distributedMax[0];

  if (scaleGrads && maxGrad > 10.0f) {
    while (maxGrad * gradScale > 10.0f) {
      gradScale *= 0.5f;
    }
    for (int i = threadIdx.x; i < numTerms; i += blockDim.x) {
      grad[i] *= gradScale;
    }
  }
  __syncthreads();
}

__device__ void updateDGrad(const int                                                   numTerms,
                            const float                                                 gradTol,
                            const float                                                 energy,
                            const float                                                 gradScale,
                            const MinReal*                                              grad,
                            const MinReal*                                              pos,
                            MinReal*                                                    dGrad,
                            bool&                                                       converged,
                            typename hipcub::BlockReduce<float, BLOCK_SIZE>::TempStorage& tempStorage) {
  float localMax = 0.0f;
  for (int i = threadIdx.x; i < numTerms; i += blockDim.x) {
    dGrad[i]   = grad[i] - dGrad[i];
    float temp = fabsf(grad[i]) * fmaxf(fabsf(pos[i]), 1.0f);
    if (temp > localMax) {
      localMax = temp;
    }
  }

  float blockMax = hipcub::BlockReduce<float, BLOCK_SIZE>(tempStorage).Reduce(localMax, cubMax());

  if (threadIdx.x == 0) {
    const float term = fmaxf(energy * gradScale, 1.0f);
    blockMax /= term;
    if (blockMax < gradTol) {
      converged = true;
    }
  }
  __syncthreads();
}

__device__ void updateInverseHessian(const int                                                   numTerms,
                                     MinReal*                                                    invHessian,
                                     MinReal*                                                    dGrad,
                                     MinReal*                                                    xi,
                                     MinReal*                                                    hessDGrad,
                                     MinReal*                                                    grad,
                                     typename hipcub::BlockReduce<float, BLOCK_SIZE>::TempStorage& tempStorage) {
  using BlockReduce = hipcub::BlockReduce<float, BLOCK_SIZE>;

  // Compute hessDGrad = invHessian * dGrad
  for (int row = threadIdx.x; row < numTerms; row += blockDim.x) {
    float dotProduct = 0.0f;
    for (int col = 0; col < numTerms; col++) {
      dotProduct += invHessian[col * numTerms + row] * dGrad[col];
    }
    hessDGrad[row] = dotProduct;
  }
  __syncthreads();

  // Compute BFGS sums
  __shared__ float fac, fae, fad, sumDGrad, sumXi;
  __shared__ bool  needUpdate;

  float sumFac = 0.0f;
  for (int i = threadIdx.x; i < numTerms; i += blockDim.x) {
    sumFac += dGrad[i] * xi[i];
  }
  float facReduced = BlockReduce(tempStorage).Sum(sumFac);
  if (threadIdx.x == 0)
    fac = facReduced;
  __syncthreads();

  float sumFae = 0.0f;
  for (int i = threadIdx.x; i < numTerms; i += blockDim.x) {
    sumFae += dGrad[i] * hessDGrad[i];
  }
  float faeReduced = BlockReduce(tempStorage).Sum(sumFae);
  if (threadIdx.x == 0)
    fae = faeReduced;
  __syncthreads();

  float sumDGradSq = 0.0f;
  for (int i = threadIdx.x; i < numTerms; i += blockDim.x) {
    sumDGradSq += dGrad[i] * dGrad[i];
  }
  float sumDGradReduced = BlockReduce(tempStorage).Sum(sumDGradSq);
  if (threadIdx.x == 0)
    sumDGrad = sumDGradReduced;
  __syncthreads();

  float sumXiSq = 0.0f;
  for (int i = threadIdx.x; i < numTerms; i += blockDim.x) {
    sumXiSq += xi[i] * xi[i];
  }
  float sumXiReduced = BlockReduce(tempStorage).Sum(sumXiSq);
  if (threadIdx.x == 0)
    sumXi = sumXiReduced;
  __syncthreads();

  if (threadIdx.x == 0) {
    constexpr float EPS = 3e-8f;
    needUpdate          = (fac > 0) && ((fac * fac) > (EPS * sumDGrad * sumXi));

    if (needUpdate) {
      fac = 1.0f / fac;
      fad = 1.0f / fae;
    }
  }
  __syncthreads();

  if (needUpdate) {
    // Update dGrad for Hessian update
    for (int i = threadIdx.x; i < numTerms; i += blockDim.x) {
      dGrad[i] = fac * xi[i] - fad * hessDGrad[i];
    }
    __syncthreads();

    // Update inverse Hessian
    for (int row = threadIdx.x; row < numTerms; row += blockDim.x) {
      float pxi  = fac * xi[row];
      float hdgi = fad * hessDGrad[row];
      float dgi  = fae * dGrad[row];

      for (int col = 0; col < numTerms; col++) {
        float pxj    = xi[col];
        float hdgj   = hessDGrad[col];
        float dgj    = dGrad[col];
        float update = pxi * pxj - hdgi * hdgj + dgi * dgj;
        invHessian[col * numTerms + row] += update;
      }
    }
    __syncthreads();
  }

  // Update xi = -invHessian * grad
  for (int row = threadIdx.x; row < numTerms; row += blockDim.x) {
    float dotProduct = 0.0f;
    for (int col = 0; col < numTerms; col++) {
      dotProduct += invHessian[col * numTerms + row] * grad[col];
    }
    xi[row] = -dotProduct;
  }
  __syncthreads();
}

// Helper to get data dimensionality from ForceFieldType at compile time
template <ForceFieldType FFType> struct DataDimTraits;

template <> struct DataDimTraits<ForceFieldType::MMFF> {
  static constexpr int value = 3;
};

template <> struct DataDimTraits<ForceFieldType::ETK> {
  static constexpr int value = 4;
};

template <> struct DataDimTraits<ForceFieldType::DG> {
  static constexpr int value = 4;
};

}  // namespace

template <int            MaxAtoms,
          bool           UseSharedMem,
          ForceFieldType FFType,
          bool           HasConstraints,
          typename TermsType,
          typename IndicesType>
__launch_bounds__(BLOCK_SIZE) __global__ void bfgsMinimizeKernel(const int               numIters,
                                                                 const double            gradTol,
                                                                 const bool              scaleGrads,
                                                                 const TermsType*        terms,
                                                                 const IndicesType*      systemIndices,
                                                                 const int*              molIdList,
                                                                 const int*              atomStarts,
                                                                 const int*              hessianStarts,
                                                                 double*                 positions,
                                                                 MinReal*                grad,
                                                                 MinReal*                inverseHessian,
                                                                 MinReal**               scratchBuffers,
                                                                 double*                 energyOuts,
                                                                 int16_t*                statuses,
                                                                 [[maybe_unused]] double chiralWeight,
                                                                 [[maybe_unused]] double fourthDimWeight) {
  const int     molIdx = molIdList[blockIdx.x];
  const int16_t tid    = threadIdx.x;

  const int     atomStart = atomStarts[molIdx];
  const int     atomEnd   = atomStarts[molIdx + 1];
  const int16_t numAtoms  = atomEnd - atomStart;

  // Use compile-time dimension for correctness
  constexpr int16_t dataDim  = DataDimTraits<FFType>::value;
  constexpr int16_t maxTerms = MaxAtoms * dataDim;
  const int16_t     numTerms = dataDim * numAtoms;

  // Pointers to working memory (either shared or global). The minimizer's
  // internal working state is computed in MinReal (float); the shared context
  // positions stay double and are converted at kernel entry/exit below.
  MinReal* localPos;
  MinReal* localGrad;
  MinReal* localDir;
  MinReal* scratchPos;
  MinReal* dGrad;
  MinReal* oldPos;

  const int termStart = atomStart * dataDim;

  if constexpr (UseSharedMem) {
    // Shared memory for small molecules (≤128 atoms)
    __shared__ MinReal sharedLocalPos[maxTerms];
    __shared__ MinReal sharedLocalGrad[maxTerms];
    __shared__ MinReal sharedLocalDir[maxTerms];
    __shared__ MinReal sharedScratchPos[maxTerms];
    __shared__ MinReal sharedDGrad[maxTerms];

    localPos            = sharedLocalPos;
    localGrad           = sharedLocalGrad;
    localDir            = sharedLocalDir;
    scratchPos          = sharedScratchPos;
    dGrad               = sharedDGrad;
    // For small molecules, grad buffer is unused (using sharedLocalGrad), so reuse it for oldPos
    oldPos              = scratchBuffers[0] + termStart;  // Reuse grad buffer for oldPos
  } else {
    // Global memory for large molecules (>128 atoms) - index into pre-allocated float buffers.
    // scratchBuffers[0] holds the float working copy of positions (the double context
    // positions array can no longer be aliased directly).
    localPos            = scratchBuffers[0] + termStart;  // float working positions
    localGrad           = grad + termStart;                // Use main (float) gradient buffer
    localDir            = scratchBuffers[1] + termStart;    // lineSearchDir
    scratchPos          = scratchBuffers[2] + termStart;    // scratchPositions
    dGrad               = scratchBuffers[3] + termStart;    // hessDGrad
    oldPos              = scratchBuffers[4] + termStart;    // scratchGrad (used as oldPos)
  }

  // Shared scalars
  __shared__ float maxStep;
  __shared__ float prevE;
  __shared__ float currE;
  __shared__ float slope;
  __shared__ float lambda;
  __shared__ float lambdaMin;
  __shared__ float lambda2;
  __shared__ float eScratch;
  __shared__ float gradScale;
  __shared__ bool  converged;
  __shared__ bool  lineSearchConverged;

  // Inverse Hessian in global memory (O(n^2), too large for shared)
  // Indexed by hessianStarts which stores cumulative (numTerms * numTerms) offsets
  MinReal* invHessian = inverseHessian + hessianStarts[molIdx];

  // Initialize float working positions from the double context positions.
  // (float->double convert on read; results are written back at kernel exit.)
  double* globalPos = positions + atomStart * dataDim;
  for (int i = tid; i < numTerms; i += blockDim.x) {
    localPos[i] = static_cast<MinReal>(globalPos[i]);
  }
  __syncthreads();

  // Initialize inverse Hessian to identity
  const int hessianSize = numTerms * numTerms;
  for (int i = tid; i < hessianSize; i += blockDim.x) {
    invHessian[i] = 0.0f;
  }
  __syncthreads();
  for (int i = tid; i < numTerms; i += blockDim.x) {
    invHessian[i * numTerms + i] = 1.0f;
  }

  if (tid == 0) {
    converged = false;
  }
  __syncthreads();

  // Shared temp storage for all BlockReduce operations
  using BlockReduce = hipcub::BlockReduce<float, BLOCK_SIZE>;
  __shared__ typename BlockReduce::TempStorage tempStorage;

  // Compute initial energy
  float threadEnergy;
  if constexpr (FFType == ForceFieldType::MMFF) {
    threadEnergy = MMFF::molEnergy<BLOCK_SIZE, HasConstraints>(*terms, *systemIndices, localPos, molIdx, tid);
  } else if constexpr (FFType == ForceFieldType::ETK) {
    threadEnergy = DistGeom::molEnergyETK(*terms, *systemIndices, localPos, molIdx, tid);
  } else {  // DG
    threadEnergy =
      DistGeom::molEnergyDG<dataDim>(*terms, *systemIndices, localPos, molIdx, chiralWeight, fourthDimWeight, tid);
  }
  const float blockEnergy = BlockReduce(tempStorage).Sum(threadEnergy);

  if (tid == 0) {
    prevE              = blockEnergy;
    energyOuts[molIdx] = blockEnergy;
  }
  __syncthreads();

  for (int i = tid; i < numTerms; i += blockDim.x) {
    localGrad[i] = 0.0f;
  }
  __syncthreads();

  if constexpr (FFType == ForceFieldType::MMFF) {
    MMFF::molGrad<BLOCK_SIZE, HasConstraints>(*terms, *systemIndices, localPos, localGrad, molIdx, tid);
  } else if constexpr (FFType == ForceFieldType::ETK) {
    DistGeom::molGradETK(*terms, *systemIndices, localPos, localGrad, molIdx, tid);
  } else {  // DG
    DistGeom::molGradDG<dataDim>(*terms,
                                 *systemIndices,
                                 localPos,
                                 localGrad,
                                 molIdx,
                                 chiralWeight,
                                 fourthDimWeight,
                                 tid);
  }
  __syncthreads();

  // Scale gradients
  if (scaleGrads) {
    scaleGrad<true>(numTerms, localGrad, gradScale, tempStorage);
  } else {
    scaleGrad<false>(numTerms, localGrad, gradScale, tempStorage);
  }
  // Set initial direction as negative gradient
  for (int i = tid; i < numTerms; i += blockDim.x) {
    localDir[i] = -localGrad[i];
  }
  __syncthreads();

  // Set max step
  setMaxStep(localPos, numTerms, &maxStep, tempStorage);
  __syncthreads();

  // Main BFGS loop
  __shared__ int currIter;
  if (tid == 0) {
    currIter = 0;
  }
  __syncthreads();

  while (!converged && currIter < numIters) {
    // Save current position before line search
    for (int i = tid; i < numTerms; i += blockDim.x) {
      oldPos[i] = localPos[i];
    }
    __syncthreads();

    // Line search setup
    if (tid == 0) {
      lineSearchConverged = false;
      lambda              = 1.0;
    }
    __syncthreads();

    lineSearchSetup(numTerms, localPos, localGrad, maxStep, localDir, slope, lambdaMin, tempStorage);
    __syncthreads();

    // Line search loop
    __shared__ int16_t lineSearchIter;
    if (tid == 0) {
      lineSearchIter = 0;
    }
    __syncthreads();

    while (!lineSearchConverged && lineSearchIter < MAX_LINESEARCH_ITERS) {
      // Perturb positions from saved oldPos (not localPos, which may have been modified)
      lineSearchPerturb(numTerms, oldPos, localDir, lambda, scratchPos);

      // Compute energy at perturbed position (use scratchPos which has the perturbed coordinates)
      float lsThreadEnergy;
      if constexpr (FFType == ForceFieldType::MMFF) {
        lsThreadEnergy = MMFF::molEnergy<BLOCK_SIZE, HasConstraints>(*terms, *systemIndices, scratchPos, molIdx, tid);
      } else if constexpr (FFType == ForceFieldType::ETK) {
        lsThreadEnergy = DistGeom::molEnergyETK(*terms, *systemIndices, scratchPos, molIdx, tid);
      } else {  // DG
        lsThreadEnergy = DistGeom::molEnergyDG<dataDim>(*terms,
                                                        *systemIndices,
                                                        scratchPos,
                                                        molIdx,
                                                        chiralWeight,
                                                        fourthDimWeight,
                                                        tid);
      }
      const float lsBlockEnergy = BlockReduce(tempStorage).Sum(lsThreadEnergy);

      if (tid == 0) {
        currE = lsBlockEnergy;
      }
      __syncthreads();

      // Check convergence and update lambda
      lineSearchConverged =
        lineSearchPostEnergy(lineSearchIter == 0, prevE, currE, slope, lambda, lambdaMin, lambda2, eScratch, lambda);
      __syncthreads();

      if (tid == 0) {
        lineSearchIter++;
      }
      __syncthreads();
    }

    // Update positions with final line search result and compute direction
    for (int i = tid; i < numTerms; i += blockDim.x) {
      localPos[i] = scratchPos[i];
    }
    __syncthreads();

    // Set direction (compute xi = new - old)
    setDirection(numTerms, scratchPos, oldPos, localDir, dGrad, localGrad, converged, tempStorage);
    if (converged) {
      break;
    }

    // Update stored energy for next iteration
    if (tid == 0) {
      prevE = currE;
    }
    __syncthreads();

    // Compute gradients at new position
    for (int i = tid; i < numTerms; i += blockDim.x) {
      localGrad[i] = 0.0f;
    }
    __syncthreads();

    if constexpr (FFType == ForceFieldType::MMFF) {
      MMFF::molGrad<BLOCK_SIZE, HasConstraints>(*terms, *systemIndices, localPos, localGrad, molIdx, tid);
    } else if constexpr (FFType == ForceFieldType::ETK) {
      DistGeom::molGradETK(*terms, *systemIndices, localPos, localGrad, molIdx, tid);
    } else {  // DG
      DistGeom::molGradDG<dataDim>(*terms,
                                   *systemIndices,
                                   localPos,
                                   localGrad,
                                   molIdx,
                                   chiralWeight,
                                   fourthDimWeight,
                                   tid);
    }
    __syncthreads();

    // Scale gradients
    if (scaleGrads) {
      scaleGrad<true>(numTerms, localGrad, gradScale, tempStorage);
    } else {
      scaleGrad<false>(numTerms, localGrad, gradScale, tempStorage);
    }

    // Update dGrad and check convergence
    updateDGrad(numTerms, gradTol, currE, gradScale, localGrad, localPos, dGrad, converged, tempStorage);
    if (converged) {
      break;
    }

    // Update Hessian and compute new direction (reuses scratchPos as hessDGrad)
    updateInverseHessian(numTerms, invHessian, dGrad, localDir, scratchPos, localGrad, tempStorage);

    if (tid == 0) {
      currIter++;
    }
    __syncthreads();
  }

  // The minimizer operated on a float working copy of positions in both the
  // shared-memory and global-memory paths. Write the float result back to the
  // double context positions array (float->double convert).
  for (int i = tid; i < numTerms; i += blockDim.x) {
    globalPos[i] = static_cast<double>(localPos[i]);
  }

  // Write final energy and status
  if (tid == 0) {
    energyOuts[molIdx] = prevE;
    // Write status to match batched kernel behavior (0 = converged, 1 = not converged)
    if (statuses != nullptr) {
      statuses[molIdx] = converged ? 0 : 1;
    }
  }
}

namespace {

template <int            MaxAtoms,
          bool           UseSharedMem,
          ForceFieldType FFType,
          bool           HasConstraints,
          typename TermsType,
          typename IndicesType>
hipError_t launchKernelForSize(int                numMols,
                                const int*         molIdList,
                                int                numIters,
                                double             gradTol,
                                bool               scaleGrads,
                                const TermsType*   devTerms,
                                const IndicesType* devSysIdx,
                                const int*         atomStarts,
                                const int*         hessianStarts,
                                double*            positions,
                                MinReal*           grad,
                                MinReal*           inverseHessian,
                                MinReal**          scratchBuffers,
                                double*            energyOuts,
                                int16_t*           statuses,
                                hipStream_t       stream,
                                double             chiralWeight,
                                double             fourthDimWeight) {
  if (numMols == 0) {
    return hipSuccess;
  }

  bfgsMinimizeKernel<MaxAtoms, UseSharedMem, FFType, HasConstraints, TermsType, IndicesType>
    <<<numMols, BLOCK_SIZE, 0, stream>>>(numIters,
                                         gradTol,
                                         scaleGrads,
                                         devTerms,
                                         devSysIdx,
                                         molIdList,
                                         atomStarts,
                                         hessianStarts,
                                         positions,
                                         grad,
                                         inverseHessian,
                                         scratchBuffers,
                                         energyOuts,
                                         statuses,
                                         chiralWeight,
                                         fourthDimWeight);

  return hipGetLastError();
}

template <ForceFieldType FFType, bool HasConstraints, typename TermsType, typename IndicesType>
hipError_t dispatchByMaxAtoms(int                numMols,
                               const int*         molIdList,
                               int                maxAtoms,
                               int                numIters,
                               double             gradTol,
                               bool               scaleGrads,
                               const TermsType*   devTerms,
                               const IndicesType* devSysIdx,
                               const int*         atomStarts,
                               const int*         hessianStarts,
                               double*            positions,
                               MinReal*           grad,
                               MinReal*           inverseHessian,
                               MinReal**          scratchBuffers,
                               double*            energyOuts,
                               int16_t*           statuses,
                               hipStream_t       stream,
                               double             chiralWeight,
                               double             fourthDimWeight) {
  // Use shared memory for <=128 atoms (in increments of 32), global memory for larger
  if (maxAtoms <= 32) {
    return launchKernelForSize<32, true, FFType, HasConstraints>(numMols,
                                                                 molIdList,
                                                                 numIters,
                                                                 gradTol,
                                                                 scaleGrads,
                                                                 devTerms,
                                                                 devSysIdx,
                                                                 atomStarts,
                                                                 hessianStarts,
                                                                 positions,
                                                                 grad,
                                                                 inverseHessian,
                                                                 scratchBuffers,
                                                                 energyOuts,
                                                                 statuses,
                                                                 stream,
                                                                 chiralWeight,
                                                                 fourthDimWeight);
  } else if (maxAtoms <= 64) {
    return launchKernelForSize<64, true, FFType, HasConstraints>(numMols,
                                                                 molIdList,
                                                                 numIters,
                                                                 gradTol,
                                                                 scaleGrads,
                                                                 devTerms,
                                                                 devSysIdx,
                                                                 atomStarts,
                                                                 hessianStarts,
                                                                 positions,
                                                                 grad,
                                                                 inverseHessian,
                                                                 scratchBuffers,
                                                                 energyOuts,
                                                                 statuses,
                                                                 stream,
                                                                 chiralWeight,
                                                                 fourthDimWeight);
  } else if (maxAtoms <= 96) {
    return launchKernelForSize<96, true, FFType, HasConstraints>(numMols,
                                                                 molIdList,
                                                                 numIters,
                                                                 gradTol,
                                                                 scaleGrads,
                                                                 devTerms,
                                                                 devSysIdx,
                                                                 atomStarts,
                                                                 hessianStarts,
                                                                 positions,
                                                                 grad,
                                                                 inverseHessian,
                                                                 scratchBuffers,
                                                                 energyOuts,
                                                                 statuses,
                                                                 stream,
                                                                 chiralWeight,
                                                                 fourthDimWeight);
  } else if (maxAtoms <= 128) {
    return launchKernelForSize<128, true, FFType, HasConstraints>(numMols,
                                                                  molIdList,
                                                                  numIters,
                                                                  gradTol,
                                                                  scaleGrads,
                                                                  devTerms,
                                                                  devSysIdx,
                                                                  atomStarts,
                                                                  hessianStarts,
                                                                  positions,
                                                                  grad,
                                                                  inverseHessian,
                                                                  scratchBuffers,
                                                                  energyOuts,
                                                                  statuses,
                                                                  stream,
                                                                  chiralWeight,
                                                                  fourthDimWeight);
  } else if (maxAtoms <= 256) {
    return launchKernelForSize<256, false, FFType, HasConstraints>(numMols,
                                                                   molIdList,
                                                                   numIters,
                                                                   gradTol,
                                                                   scaleGrads,
                                                                   devTerms,
                                                                   devSysIdx,
                                                                   atomStarts,
                                                                   hessianStarts,
                                                                   positions,
                                                                   grad,
                                                                   inverseHessian,
                                                                   scratchBuffers,
                                                                   energyOuts,
                                                                   statuses,
                                                                   stream,
                                                                   chiralWeight,
                                                                   fourthDimWeight);
  } else {
    return launchKernelForSize<2048, false, FFType, HasConstraints>(numMols,
                                                                    molIdList,
                                                                    numIters,
                                                                    gradTol,
                                                                    scaleGrads,
                                                                    devTerms,
                                                                    devSysIdx,
                                                                    atomStarts,
                                                                    hessianStarts,
                                                                    positions,
                                                                    grad,
                                                                    inverseHessian,
                                                                    scratchBuffers,
                                                                    energyOuts,
                                                                    statuses,
                                                                    stream,
                                                                    chiralWeight,
                                                                    fourthDimWeight);
  }
}

}  // namespace

hipError_t launchBfgsMinimizePerMolKernel(int                                       numMols,
                                           const int*                                molIds,
                                           int                                       maxAtoms,
                                           const int*                                atomStarts,
                                           const int*                                hessianStarts,
                                           int                                       numIters,
                                           double                                    gradTol,
                                           bool                                      scaleGrads,
                                           const MMFF::EnergyForceContribsDevicePtr& terms,
                                           const MMFF::BatchedIndicesDevicePtr&      systemIndices,
                                           double*                                   positions,
                                           float*                                    grad,
                                           float*                                    inverseHessian,
                                           float**                                   scratchBuffers,
                                           double*                                   energyOuts,
                                           bool                                      hasConstraints,
                                           int16_t*                                  statuses,
                                           hipStream_t                              stream) {
  if (numMols == 0) {
    return hipSuccess;
  }

  const AsyncDevicePtr<MMFF::EnergyForceContribsDevicePtr> devTerms(terms, stream);
  const AsyncDevicePtr<MMFF::BatchedIndicesDevicePtr>      devSysIdx(systemIndices, stream);

  if (hasConstraints) {
    return dispatchByMaxAtoms<ForceFieldType::MMFF, true>(numMols,
                                                          molIds,
                                                          maxAtoms,
                                                          numIters,
                                                          gradTol,
                                                          scaleGrads,
                                                          devTerms.data(),
                                                          devSysIdx.data(),
                                                          atomStarts,
                                                          hessianStarts,
                                                          positions,
                                                          grad,
                                                          inverseHessian,
                                                          scratchBuffers,
                                                          energyOuts,
                                                          statuses,
                                                          stream,
                                                          1.0,
                                                          1.0);
  }
  return dispatchByMaxAtoms<ForceFieldType::MMFF, false>(numMols,
                                                         molIds,
                                                         maxAtoms,
                                                         numIters,
                                                         gradTol,
                                                         scaleGrads,
                                                         devTerms.data(),
                                                         devSysIdx.data(),
                                                         atomStarts,
                                                         hessianStarts,
                                                         positions,
                                                         grad,
                                                         inverseHessian,
                                                         scratchBuffers,
                                                         energyOuts,
                                                         statuses,
                                                         stream,
                                                         1.0,
                                                         1.0);
}
hipError_t launchBfgsMinimizePerMolKernelETK(int                                             numMols,
                                              const int*                                      molIds,
                                              int                                             maxAtoms,
                                              const int*                                      atomStarts,
                                              const int*                                      hessianStarts,
                                              int                                             numIters,
                                              double                                          gradTol,
                                              bool                                            scaleGrads,
                                              const DistGeom::Energy3DForceContribsDevicePtr& terms,
                                              const DistGeom::BatchedIndices3DDevicePtr&      systemIndices,
                                              double*                                         positions,
                                              float*                                          grad,
                                              float*                                          inverseHessian,
                                              float**                                         scratchBuffers,
                                              double*                                         energyOuts,
                                              int16_t*                                        statuses,
                                              hipStream_t                                    stream) {
  if (numMols == 0) {
    return hipSuccess;
  }

  const AsyncDevicePtr<DistGeom::Energy3DForceContribsDevicePtr> devTerms(terms, stream);
  const AsyncDevicePtr<DistGeom::BatchedIndices3DDevicePtr>      devSysIdx(systemIndices, stream);

  return dispatchByMaxAtoms<ForceFieldType::ETK, false>(numMols,
                                                        molIds,
                                                        maxAtoms,
                                                        numIters,
                                                        gradTol,
                                                        scaleGrads,
                                                        devTerms.data(),
                                                        devSysIdx.data(),
                                                        atomStarts,
                                                        hessianStarts,
                                                        positions,
                                                        grad,
                                                        inverseHessian,
                                                        scratchBuffers,
                                                        energyOuts,
                                                        statuses,
                                                        stream,
                                                        1.0,
                                                        1.0);
}

hipError_t launchBfgsMinimizePerMolKernelDG(int                                           numMols,
                                             const int*                                    molIds,
                                             int                                           maxAtoms,
                                             const int*                                    atomStarts,
                                             const int*                                    hessianStarts,
                                             int                                           numIters,
                                             double                                        gradTol,
                                             bool                                          scaleGrads,
                                             const DistGeom::EnergyForceContribsDevicePtr& terms,
                                             const DistGeom::BatchedIndicesDevicePtr&      systemIndices,
                                             double*                                       positions,
                                             float*                                        grad,
                                             float*                                        inverseHessian,
                                             float**                                       scratchBuffers,
                                             double*                                       energyOuts,
                                             double                                        chiralWeight,
                                             double                                        fourthDimWeight,
                                             int16_t*                                      statuses,
                                             hipStream_t                                  stream) {
  if (numMols == 0) {
    return hipSuccess;
  }

  const AsyncDevicePtr<DistGeom::EnergyForceContribsDevicePtr> devTerms(terms, stream);
  const AsyncDevicePtr<DistGeom::BatchedIndicesDevicePtr>      devSysIdx(systemIndices, stream);

  return dispatchByMaxAtoms<ForceFieldType::DG, false>(numMols,
                                                       molIds,
                                                       maxAtoms,
                                                       numIters,
                                                       gradTol,
                                                       scaleGrads,
                                                       devTerms.data(),
                                                       devSysIdx.data(),
                                                       atomStarts,
                                                       hessianStarts,
                                                       positions,
                                                       grad,
                                                       inverseHessian,
                                                       scratchBuffers,
                                                       energyOuts,
                                                       statuses,
                                                       stream,
                                                       chiralWeight,
                                                       fourthDimWeight);
}

}  // namespace nvMolKit