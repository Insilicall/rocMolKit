// Compat shim para macros/símbolos do nvMolKit que não são CUDA padrão e
// que o hipify-perl não consegue traduzir automaticamente.
//
// Force-included in every translation unit of rocmolkit_core via -include flag
// in rocmolkit/CMakeLists.txt.
//
// NOTE: cudaCheckError is NOT defined here. nvMolKit ships its own
// rocmolkit/src/utils/cuda_error_check.h that defines it (and is HIP-clean
// already after hipify-perl). Defining it twice would cause -Wmacro-redefined.

#pragma once

#include <hip/hip_runtime.h>

// CUDA Graphs Conditional Nodes (cudaGraphConditionalHandle, cudaGraphSetConditional)
// não têm equivalente em hipGraph. Usado em src/butina.hip.cpp.
// Stub para não quebrar build da Fase 1 — Butina ganha implementação alternativa na Fase 6.
#ifndef ROCMOLKIT_HAS_GRAPH_CONDITIONAL
using cudaGraphConditionalHandle = void*;
inline void cudaGraphSetConditional(cudaGraphConditionalHandle, int) {
    // No-op stub. Kernel que usa isso precisa ser substituído por loop CPU-side.
}
#endif

// nvMolKit defines NVMOLKIT_CUDA_CC_* macros from CMAKE_CUDA_ARCHITECTURES at
// configure time. They guard NVIDIA-specific SM-version-tuned kernels in
// similarity_kernels.cu. On AMD they have no meaning — define them all as 0
// so the runtime check (isComputeCapabilitySupported) returns false and the
// generic fallback is taken. Phase 5 (similarity) may revisit this with
// AMD-specific WMMA paths.
#ifndef NVMOLKIT_CUDA_CC_80
#define NVMOLKIT_CUDA_CC_80  0
#define NVMOLKIT_CUDA_CC_86  0
#define NVMOLKIT_CUDA_CC_89  0
#define NVMOLKIT_CUDA_CC_90  0
#define NVMOLKIT_CUDA_CC_100 0
#define NVMOLKIT_CUDA_CC_103 0
#define NVMOLKIT_CUDA_CC_120 0
#endif

// Memory carveout symbol — hipify-perl não trata em todos os casos.
// Aliases para o símbolo HIP equivalente.
#ifndef cudaSharedmemCarveoutMaxShared
#define cudaSharedmemCarveoutMaxShared hipSharedMemCarveoutMaxShared
#endif
#ifndef cudaSharedmemCarveoutMaxL1
#define cudaSharedmemCarveoutMaxL1 hipSharedMemCarveoutMaxL1
#endif
#ifndef cudaSharedmemCarveoutDefault
#define cudaSharedmemCarveoutDefault hipSharedMemCarveoutDefault
#endif
