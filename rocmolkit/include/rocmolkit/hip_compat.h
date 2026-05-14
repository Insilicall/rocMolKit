// Compat shim para macros/símbolos do nvMolKit que não são CUDA padrão e
// que o hipify-perl não consegue traduzir automaticamente.
//
// Inclua este header em qualquer .hip.cpp que use cudaCheckError ou aliases.

#pragma once

#include <hip/hip_runtime.h>
#include <stdexcept>
#include <string>

#ifndef cudaCheckError
#define cudaCheckError(call)                                                              \
    do {                                                                                  \
        hipError_t _e = (call);                                                           \
        if (_e != hipSuccess) {                                                           \
            throw std::runtime_error(std::string("HIP error: ") + hipGetErrorString(_e) + \
                                     " at " __FILE__ ":" + std::to_string(__LINE__));     \
        }                                                                                 \
    } while (0)
#endif

// CUDA Graphs Conditional Nodes (cudaGraphConditionalHandle, cudaGraphSetConditional)
// não têm equivalente em hipGraph. Usado em src/butina.hip.cpp.
// Stub para não quebrar build da Fase 1 — Butina ganha implementação alternativa na Fase 6.
#ifndef ROCMOLKIT_HAS_GRAPH_CONDITIONAL
using cudaGraphConditionalHandle = void*;
inline void cudaGraphSetConditional(cudaGraphConditionalHandle, int) {
    // No-op stub. Kernel que usa isso precisa ser substituído por loop CPU-side.
}
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
