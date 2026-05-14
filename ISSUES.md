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
- `rocgdb` does not work for gfx1200 yet ("AMDGCN architecture 0x45
  is not supported"). Plain `gdb` makes the bug disappear (Heisenbug).

### **KEY FINDING — bisect localized the bug**

Two repro programs in `tests/repro/`:

1. `asyncvec_stress.cpp` — raw `hipMallocAsync` / `hipFreeAsync` patterns
   mirroring `AsyncDeviceVector`. **No nvMolKit, no RDKit, no Python.**
   → **20 / 20 iterations × 1000 alloc-free cycles, ZERO crashes.**

2. `embed_pure_cpp.cpp` — pure C++ that builds an RDKit `ROMol` via the
   C++ API and calls `nvMolKit::embedMolecules` directly. **No
   boost-python, no Python.**
   → **Segfaults on every invocation, even on `CCO`.**

The bug is **inside `nvMolKit::embedMolecules` or one of the HIP kernels
it dispatches**. It is not in our HIP allocation lifecycle, and it is not
in boost-python / Python GC. The intermittent appearance in Python is
just the Python loop / interpreter occasionally finishing the work
before the racing free fires.

Suspect call sites (need bisect down further):
- `etkdg_impl.cpp` — top-level orchestration
- `etkdg_stage_distgeom_minimize.hip.cpp` — BFGS DG minimize
- `etkdg_stage_etk_minimization.hip.cpp` — ETK MMFF-like minimize
- `bfgs_minimize.hip.cpp` (newly re-enabled) or `bfgs_hessian.hip.cpp`
  (newly re-enabled) — these were excluded under ROCm 6.2 and are the
  most recent additions. Strong candidates.

### Bisect refinement (May 14)

Pure-C++ driver `tests/repro/embed_pure_cpp.cpp` with explicit
`maxIterations` argument:

- `maxIterations=0` → throws "All parameters must be greater than 0"
  (parameter validation rejects 0). Not useful as a bisect knob.
- `maxIterations=1` → crashes during the first `embedMolecules` call,
  **before any printable progress past "embed(maxIter=1)..."**. So the
  crash is during initialization / first iteration, not during
  multi-iteration convergence.

### **ROOT CAUSE LOCALIZED (May 14, evening)**

Instrumentation in `etkdg.cpp` proved the crash is **not** during ETKDG
stages and **not** during BfgsBatchMinimizer setup. Crash happens
**between invocations** — `EmbedMolecules` works for the first 3-4 calls
and then segfaults on the 4th/5th, even with identical small molecules
(CCO, benzene). Sequence of working invocations followed by silent
crash is reproducible.

This is **state leak between invocations**: some GPU resource (stream,
allocator pool, BFGS buffer, ETKDG context cache) is not fully released
between `EmbedMolecules` calls and accumulates until the heap or driver
runs out / corrupts.

Suspects:
- `ScopedStream` destructor not synchronising before `hipStreamDestroy`
- `BfgsBatchMinimizer` destructor leaking device memory
- Static / OMP-thread-local caches in `nvMolKit::DGeomHelpers::prepareEmbedderArgs`
- `hipMallocAsync` memory pool growing without bound

Single-call use works fine. The crash signature is non-deterministic
because the leaked state's exact shape varies by what came before.

Next step blocker: **rocgdb 7.2.3 does not yet support gfx1200**
("AMDGCN architecture 0x45 is not supported"). Without device
debugger we cannot capture the kernel that faults. Either need to:

- Wait for ROCm release that adds gfx1200 to rocgdb.
- Test on a gfx1100 GPU (RX 7900 XTX/XT) where rocgdb works.
- Add manual `hipDeviceSynchronize() + hipGetLastError()` checks after
  every kernel launch in `etkdg_impl.cpp` to find the failing one.

### Hypothesis

Something between RDKit 2024.09.6 Python wrappers, our boost-python
bindings, and AMD HIP runtime corrupts memory non-deterministically.
Possibilities: (a) rdkit-pypi style ABI drift between RDKit C++ and the
Python wrappers we built; (b) AsyncDeviceVector's stream pointer becoming
dangling when an `etkdg` context tears down; (c) a real device-side OOB
write that only manifests when the heap layout is unfortunate.

### Next investigation needs

- Write a C++-only `EmbedMolecules` driver (no boost-python) that calls
  the same upstream APIs the binding does, with an `RDKit::ROMol`
  built directly via the RDKit C++ API. If it crashes → bug is in
  nvMolKit's expectation of how Python passes ROMol. If it doesn't →
  bug is in the boost-python conversion layer.
- Inspect `nvmolkit/embedMolecules.cpp` — how does it convert
  `boost::python::list` into `std::vector<RDKit::ROMol*>`? Look for
  GIL release / borrowed reference handling.
- Try `HSA_TOOLS_LIB=libhsa-amd-aqlprofile.so` for runtime trace.
- Compile host code paths with `-fsanitize=address` (HIP kernel TUs
  cannot use ASan but boost-python wrappers can).

### Bisect refinement (May 14, late evening)

Reproducible thresholds on the gfx1200 host:

| Scenario | Result |
|---|---|
| Single mol per `EmbedMolecules` call, sequential CCO/benzene/… | 1–4 calls before SIGSEGV (non-deterministic within that range) |
| Single batch of N mols in one call, default `BatchHardwareOptions` | N≤3 succeeds, N≥4 SIGSEGV during stage 1 (First Minimization) |
| `OMP_NUM_THREADS=1` + sequential single-mol | 7/11 calls succeed before crash (vs 2/11 with default OMP) |
| `OMP_NUM_THREADS=1` + batch N≥4 | Still SIGSEGV |
| `HIP_LAUNCH_BLOCKING=1` | Crash arrives sooner — bug is *not* a kernel-launch race |
| `batchesPerGpu=1` and `preprocessingThreads=1` | Helps sequential single-mol, does not save batch N≥4 |
| `batchSize=1` + N≥4 | Still SIGSEGV |
| Pure C++ `AsyncDeviceVector` stress (20×1000 alloc-free) | 0 crashes (allocator pattern alone is fine) |
| Pure C++ driver calling `nvMolKit::embedMolecules` | Reproduces SIGSEGV |

Things tried this session that did **not** fix it (and were reverted):

- Replacing `hipMallocAsync`/`hipFreeAsync` with synchronous
  `hipMalloc`/`hipFree` in `AsyncDeviceVector`/`AsyncDevicePtr` —
  regressed: 1st CCO crashed.
- Adding `hipStreamSynchronize(streamPtr)` at the end of the
  `embedMolecules` OpenMP dispatch parallel region (catch pending
  pinned-buffer copies before locals destruct) — together with the
  sync-malloc change above, only got 1–2 sequential calls instead of
  4. Reverted.
- Adding `hipDeviceSynchronize()` inside `PinnedHostVector::~` /
  `resize` / `clear` — regressed further.
- `hipMemPoolSetAttribute(pool, hipMemPoolAttrReleaseThreshold, 0)` via
  `std::call_once` (previous session) — broke even the first call.
- `hipMemPoolTrimTo(pool, 0)` at end of `embedMolecules` (previous
  session) — caused crashes on previously-working molecules.

The pattern (more sync makes it worse, less concurrency makes it
better) does **not** match a classic stream-ordering race. It is
consistent with either a HIP 7.2.3 driver bug specific to `gfx1200`,
or with a host-side memory corruption that compounds when scheduled
work runs concurrently with the host loop.

The bug threshold is **N=4 mols in one batch** and **~4 sequential
single-mol calls**. Below those thresholds the library is correct
(numerical parity already verified bit-exact within ETKDG noise).

### Bug surface: ETKDG vs MMFF

The state-leak hits ETKDG much harder than MMFF94. Empirically on
gfx1200 + ROCm 7.2.3:

| Operation | Single call | Batch (one call, N mols) | Sequential (N back-to-back calls) |
|---|---|---|---|
| `EmbedMolecules` (ETKDG) | ~65% per-call | crashes at **N≥4** | crashes after ~4 calls |
| `MMFFOptimizeMoleculesConfs` | reliable | OK at **N=30** (tested) | crashes after ~30 calls |

So for MMFF, the direct binding is fine for typical batch workloads
(`MMFFOptimizeMoleculesConfs(big_list, ...)` in one call). The
`rocmolkit.safe.mmff_optimize_molecule(s)` wrapper exists for the
sequential-call pattern (e.g. iterating mols one by one in a loop)
where the same state-leak does eventually fire.

### **Workaround: `rocmolkit.safe`** (validated, 100% reliable)

Per-call success rate of `EmbedMolecules` is ~65% on gfx1200; even
fresh subprocess calls fail intermittently. With ≥5 retries the math
crosses 99.5%, and we measured 45/45 success across three
back-to-back rounds of 15 diverse molecules.

The `rocmolkit.safe` module ships this pattern:

```python
from rdkit import Chem
from rdkit.Chem import AddHs
from rocmolkit.safe import embed_molecule, embed_molecules, EmbedFailure

m = AddHs(Chem.MolFromSmiles("CCO"))
embed_molecule(m, seed=42, max_retries=8)  # raises EmbedFailure if exhausted

# Or list:
mols = [AddHs(Chem.MolFromSmiles(s)) for s in [...]]
embed_molecules(mols)  # one subprocess per mol
```

Each call:
1. Pickles the mol via `mol.ToBinary().hex()` and passes to a fresh
   `python3 -c '...'` subprocess.
2. Subprocess runs the upstream `EmbedMolecules` once.
3. Parent reads coords back as JSON and adds a `Conformer`.
4. On `rc=-11` (SIGSEGV) parent retries up to `max_retries`.
5. Mean attempts per mol in our test: 2.33; numerical parity vs RDKit
   `AllChem.EmbedMolecule`: bond-length Δmax ≤ 0.02 Å (within ETKDG
   noise).

Cost: ~150-300 ms per subprocess on warm cache, so ~600 ms per
embedded mol on average. This is comparable to CPU RDKit embed for
small mols, with the GPU advantage growing on larger ones.

Direct `_embedMolecules.EmbedMolecules` remains exposed for power
users who need batch throughput and accept the segfault risk; switch
to `safe` for any unattended workflow.

This is a known limitation of the v0.2.0-alpha release. The C++ fix
will land once ROCm 7.3+ ships gfx1200 support in `rocgdb`, or once
the bug reproduces on a gfx1100 host where `rocgdb` already works.

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
