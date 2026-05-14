// Bisect stub for v0.2.0-alpha segfault investigation.
// Replaces bfgs_hessian.hip.cpp's GPU symbols with no-op host implementations.
// If embedMolecules stops crashing with this in place, the bug is in the
// real bfgs_hessian.hip.cpp (likely the cg_reduce_shim implementation or
// a kernel that depends on it).
//
// Once the diagnosis is done, delete this file and remove the corresponding
// list(FILTER ... bfgs_hessian) line in rocmolkit/CMakeLists.txt.

#include <hip/hip_runtime.h>

namespace nvMolKit {

// Match the exact signature in src/minimizer/bfgs_hessian.h
void updateInverseHessianBFGSBatch(int            /*numActiveSystems*/,
                                   const int16_t* /*statuses*/,
                                   const int*     /*hessianStarts*/,
                                   const int*     /*atomStarts*/,
                                   double*        /*invHessians*/,
                                   double*        /*dGrads*/,
                                   double*        /*xis*/,
                                   double*        /*hessDGrads*/,
                                   const double*  /*grads*/,
                                   int            /*dataDim*/,
                                   bool           /*hasLargeMolecule*/,
                                   const int*     /*activeSystemIndices*/,
                                   hipStream_t    /*stream*/) {
    // No-op stub for bisect.
}

}  // namespace nvMolKit
