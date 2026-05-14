# rocMolKit — Known Issues & Manual Fixes

Tracker of CUDA→HIP translation gaps that `hipify-perl` did not handle and
of features in nvMolKit that have no direct HIP/ROCm equivalent.

## Status legend
- ✅ Fixed in tree
- ⏸ Excluded from current build phase
- 🔧 Workaround applied (no-op shim, etc.)
- ⚠️ Needs manual port

---

## Symbols not translated by hipify-perl

| Symbol / Header | File(s) | Status | Resolution |
|---|---|---|---|
| `#include "*.cuh"` (file renamed but includes not rewritten) | 40 files | ✅ | sed `.cuh"` → `.hip.h"` across src/, tests/, benchmarks/ |
| `#include "nvtx.h"` (NVIDIA Tools Extension) | ~25 files | 🔧 | `rocmolkit/src/utils/nvtx.h` rewritten as no-op shim |
| `nvtxNameCudaStreamA` | `utils/device.cpp:65,89` | 🔧 | inline no-op in nvtx.h shim |
| `<cuda/std/span>`, `<cuda/std/tuple>`, `<cuda/std/cstddef>` (NVIDIA libcudacxx) | 5+ files | 🔧 | `rocmolkit/include/rocmolkit/cuda_std_compat.h` aliases `cuda::std::*` → `std::*` (C++20 required) |
| `<cub/device/device_reduce.cuh>` | `etkdg_kernels.hip.cpp:17` | ✅ | replaced with `<hipcub/device/device_reduce.hpp>` |
| `NVMOLKIT_CUDA_CC_80..120` macros (NVIDIA SM version guards) | `similarity_kernels.hip.cpp` | 🔧 | `hip_compat.h` defines all = 0; AMD takes generic fallback path |
| `cudaSharedmemCarveoutMaxShared` | `substruct/substruct_kernels.hip.cpp:145` | 🔧 | `hip_compat.h` macro alias to `hipSharedMemCarveoutMaxShared` |
| `cudaCheckError(...)` | many `.hip.cpp` | ✅ | upstream `cuda_error_check.h` already HIP-clean after hipify; no shim needed |
| `cudaGraphConditionalHandle` / `cudaGraphSetConditional` | `butina.hip.cpp:654-836` | ⏸ | Excluded from Phase 1; Phase 6 will rewrite as CPU-side dispatch loop |
| `cudaGraphCondTypeWhile` | `butina.hip.cpp:754,836` | ⏸ | Same as above |
| `hipMemcpyDefault` (was `cudaMemcpyDefault`) | `butina.hip.cpp:971,1011` | ⏸ | Excluded with rest of butina; HIP supports `hipMemcpyDefault` though |

---

## Features without direct ROCm equivalent

### CUDA Graphs Conditional Nodes
- nvMolKit uses `cudaGraphConditionalHandleCreate` + `cudaGraphCondTypeWhile`
  to express data-dependent loops inside a captured graph.
- HIP exposes `hipGraph` but **not** conditional/while nodes (as of ROCm 6.2).
- **Plan:** rewrite the Butina clustering loop with a CPU-side `while` that
  dispatches the body kernel and reads the convergence flag back per
  iteration. Slightly higher launch overhead, same functionality.
- Affected: `src/butina.hip.cpp` (~750 lines).

### NVTX (NVIDIA Tools Extension) profiling
- Used purely for profiler annotations (`ScopedNvtxRange`, color tagging).
- AMD has rocTX (`roctracer/roctx.h`) with similar API but different symbol
  names. We ship a no-op shim for now; if profiling becomes important, swap
  the shim body for `roctxRangePushA(name)` / `roctxRangePop()` and link
  `roctx64`.

### CUTLASS / NVIDIA-specific tensor ops
- Not detected in the audit so far. `morgan_fingerprint_kernels.cu` and
  `similarity_kernels.cu` use plain CUDA atomics + warp shuffles which
  hipify handles cleanly.

---

## Architecture-specific concerns

### Wavefront 64 vs Warp 32

nvMolKit assumes warp size 32 throughout:
- `dist_geom_kernels_device.hip.h`: ~30× `constexpr int WARP_SIZE = 32` plus
  `cg::tiled_partition<32>` calls.
- `kernel_utils.hip.h:32`, `load_store.hip.h:78,336`, `similarity_kernels.hip.cpp:148–151`,
  `substruct_algos.hip.h:313`: `__shfl_sync(0xffffffff, ...)` with 32-bit mask.

Behavior on AMD targets:
| Arch family | Wavefront | Effect of `tiled_partition<32>` |
|---|---|---|
| **RDNA3 (gfx1100, gfx1101)** | 32 (configurable) | ✅ native, runs as-is |
| **RDNA2 (gfx1030)** | 32 | ✅ native |
| **CDNA2/CDNA3 (gfx90a, gfx942)** | 64 fixed | ⚠️ Two 32-thread tiles per wavefront — 50% utilization but functionally correct |
| **GCN5 (gfx906)** | 64 fixed | EOL in ROCm 6 — not targeted |

**Decision:** ship as-is for Phase 1–4. CDNA-tuned kernels (rewrite for
wavefront 64) are a Phase 8+ optimization, not correctness blocker.

### Float atomics on RDNA
- `butina.hip.cpp` uses `atomicAdd(int*, int)` and `atomicExch(int*, int)` —
  fully supported on all AMD targets.
- No `atomicAdd(float*, ...)` detected so far. If found, RDNA3 supports them
  natively; older RDNA needs CAS-loop fallback.

### CUDA Graphs (non-conditional)
- `butina.hip.cpp` uses `hipGraphCreate`, `hipGraphInstantiate`, `hipGraphLaunch`
  — these work on AMD. Only the **conditional** node type is missing.

---

## Phase exclusions (current `rocmolkit/CMakeLists.txt`)

Excluded from Phase 1 build but kept in tree (will be ported in their phase):

- `src/butina.{cpp,hip.cpp}` — CUDA Graphs Conditional → Phase 6
- `src/tfd/` (whole subdir) — depends on butina → Phase 7
- `src/substruct/` (whole subdir) — `cudaSharedmemCarveoutMaxShared` and
  others not auto-translated → Phase 7

---

## Open questions

1. ~~**rocBLAS Tensile kernel filter** in `Dockerfile.slim` only saved ~900 MB~~
   ✅ **Resolved**: filter rewrite cut `/opt/rocm/lib/rocblas/` from **3.1 GB → 293 MB**.
2. **comgr** (~141 MB) is required at runtime for HIP JIT but maybe parts of
   it (LLVM bitcode libs for archs we don't target) can be removed.
3. **Wavefront 64 atomics on pre-gfx9 RDNA** if we ever target gfx1030.

## Known bug — non-deterministic segfault (v0.2.0-alpha)

`EmbedMolecules` and `MMFFOptimizeMoleculesConfs` work correctly on small
molecules (CCO, benzene, hexane, p-xylene) most of the time but
intermittently SIGSEGV with no error message. Mid-size molecules (aspirin,
pyridine) crash more reliably. **Heisenbug — disappears under `gdb`**.

### Investigated, did NOT fix

- Adding `if (data_ != nullptr)` guards before every `hipFreeAsync` in
  `AsyncDeviceVector` and `AsyncDevicePtr` (they were already guarded in
  the move-assignment, but not in destructors). No effect.
- Adding `hipStreamSynchronize(stream_)` before `hipFreeAsync` in the
  destructors. **Made it worse** — caused crashes even on small mols
  that previously worked. Reverted.
- `HIP_LAUNCH_BLOCKING=1` does not stop the crash (so it is not pure
  kernel async; some host-side issue is involved).

### Hypothesis

Something between RDKit 2024.09.6 Python wrappers, our boost-python
bindings, and AMD HIP runtime corrupts memory non-deterministically.
Possibilities: (a) rdkit-pypi style ABI drift between RDKit C++ and the
Python wrappers we built; (b) AsyncDeviceVector's stream pointer becoming
dangling when an `etkdg` context tears down; (c) a real device-side OOB
write that only manifests when the heap layout is unfortunate.

### Next investigation needs

- `rocgdb` / `rocprofiler` tooling — install in the devel image.
- Hack: try `HSA_TOOLS_LIB=libhsa-amd-aqlprofile.so` for trace.
- Try compiling with `-fsanitize=address` on the host C++ paths only
  (HIP kernel TUs cannot use ASan but the host code can).
- Reproduce in isolation: minimal C++ test that allocates / frees an
  `AsyncDeviceVector` 1000× and watches for heap corruption.

## Phase 1 fat removed from runtime image (cumulative)

| Category | Before | After | Saving |
|---|---|---|---|
| `rocm/dev-ubuntu` base | 13 GB | — | replaced by ubuntu:22.04 + selective apt |
| `rocm-hip-runtime` meta-package | full | minimal trio (hip-runtime-amd + hsa-rocr + comgr) | several hundred MB |
| `rocsolver` (transitive dep) | 1.7 GB | 0 | 1.7 GB |
| `rocsparse` (transitive dep) | 1.4 GB | 0 | 1.4 GB |
| `/opt/rocm/lib/rocblas/` Tensile kernels | 3.1 GB | 293 MB | 2.8 GB |
| `/opt/rocm/llvm/{bin,libexec,share,include}` | ~500 MB | 0 | ~500 MB |
| `/opt/rocm/share`, `/opt/rocm/include` | ~200 MB | 0 | ~200 MB |
| **Total runtime image** | **~12-13 GB** (oficial) | **~2-3 GB** target | ~10 GB |
