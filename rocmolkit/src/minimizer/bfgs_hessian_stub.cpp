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

// updateInverseHessianBFGSBatch — declared in bfgs_minimize.h.
// Real signature lives in src/minimizer/bfgs_hessian.hip.cpp.
// Replaying it here as a no-op host function. The BFGS update is what
// makes the optimiser converge faster; without it, ETKDG should still
// produce coordinates (just less optimised) but should not segfault.
void updateInverseHessianBFGSBatch(int /*n*/,
                                   const short* /*runFinished*/,
                                   const int* /*startIndices*/,
                                   const int* /*sizes*/,
                                   double* /*hessianInv*/,
                                   double* /*facShared*/,
                                   double* /*faeShared*/,
                                   double* /*sumDGradShared*/,
                                   const double* /*xi*/,
                                   int /*ldHess*/,
                                   bool /*hessianFromIdentity*/,
                                   const short* /*newlyFinished*/,
                                   hipStream_t /*stream*/) {
    // No-op stub for bisect.
}

}  // namespace nvMolKit
