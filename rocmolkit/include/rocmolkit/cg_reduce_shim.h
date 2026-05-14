// Shim for NVIDIA's <cooperative_groups/reduce.h>.
// AMD HIP cooperative_groups does not ship cg::reduce / cg::reduce_store_async,
// so we provide a minimal compatible implementation using warp shuffles.
//
// Used by src/minimizer/bfgs_hessian.hip.cpp (BFGS inverse-Hessian update).

#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>

namespace cooperative_groups {

template <typename T>
struct plus {
    __host__ __device__ T operator()(const T& a, const T& b) const { return a + b; }
};

template <typename T>
struct multiplies {
    __host__ __device__ T operator()(const T& a, const T& b) const { return a * b; }
};

// Tree reduction via warp shuffles. Group is expected to be a thread_block_tile
// (compile-time size). On AMD we read group.size() at runtime since
// thread_block_tile<32> may map to a 32-wide subset of a wave64.
template <typename Group, typename T, typename Op>
__device__ T reduce(const Group& g, T val, Op op) {
    const int sz = g.size();
    for (int offset = sz / 2; offset > 0; offset /= 2) {
        T other = __shfl_down(val, offset);
        val = op(val, other);
    }
    return val;
}

// Lane 0 of the group writes the reduced value. "_async" in NVIDIA refers to
// the group's wait semantics — on AMD the wave is implicitly synchronised so
// this is just a plain reduce + conditional store.
template <typename Group, typename T, typename Op>
__device__ void reduce_store_async(const Group& g, T* dst, T val, Op op) {
    T result = reduce(g, val, op);
    if (g.thread_rank() == 0) {
        *dst = result;
    }
}

}  // namespace cooperative_groups
