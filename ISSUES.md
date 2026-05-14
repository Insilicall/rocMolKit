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
| `#include "nvtx.h"` (NVIDIA Tools Extension) | ~25 files | 🔧 | `rocmolkit/src/utils/nvtx.h` rewritten as no-op shim |
| `nvtxNameCudaStreamA` | `utils/device.cpp:65,89` | 🔧 | inline no-op in nvtx.h shim |
| `<cub/device/device_reduce.cuh>` | `etkdg_kernels.hip.cpp:17` | ✅ | replaced with `<hipcub/device/device_reduce.hpp>` |
| `cudaSharedmemCarveoutMaxShared` | `substruct/substruct_kernels.hip.cpp:145` | 🔧 | `hip_compat.h` macro alias to `hipSharedMemCarveoutMaxShared` |
| `cudaCheckError(...)` | many `.hip.cpp` | 🔧 | `hip_compat.h` macro using `hipGetErrorString` |
| `cudaGraphConditionalHandle` / `cudaGraphSetConditional` | `butina.hip.cpp:654-836` | ⏸ | Excluded from Phase 1; Phase 6 will rewrite as CPU-side dispatch loop |
| `cudaGraphCondTypeWhile` | `butina.hip.cpp:754,836` | ⏸ | Same as above |

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

1. **rocBLAS Tensile kernel filter** in `Dockerfile.slim` only saved ~900 MB
   instead of the expected 2-3 GB. Need to check if the regex really matched
   the file naming convention used by ROCm 6.2.4.
2. **comgr** (~1.5 GB) is required at runtime for HIP JIT but maybe parts of
   it (LLVM bitcode libs for archs we don't target) can be removed.
3. **Wavefront 64 atomics on `pre-gfx9 RDNA`** if we ever target gfx1030.
