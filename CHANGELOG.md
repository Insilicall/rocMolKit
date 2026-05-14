# Changelog

All notable changes to rocMolKit will be documented here.

## [Unreleased]

### ROCm 7.2.3 + RDNA4 (gfx1200) + 6/6 bindings load (2026-05-14, evening)

Validated end-to-end on a real AMD Radeon RX 9060 XT (Navi 44 / gfx1200) with ROCm 7.2.3:

- **All 6 Python bindings import cleanly** in the local image: `_embedMolecules`, `_mmffOptimization`, `_uffOptimization`, `_batchedForcefield`, `_conformerRmsd`, `_arrayHelpers`.
- API surface exposed: `EmbedMolecules` / `EmbedMoleculesDevice`, `MMFFOptimizeMoleculesConfs(Device)`, `UFFOptimizeMoleculesConfs(Device)`, `MMFFProperties`, `NativeMMFFBatchedForcefield`, `NativeUFFBatchedForcefield`, `buildMMFFPropertiesFromRDKit`, `GetConformerRMSMatrix(Batch)`.
- `rocminfo` inside the container detects gfx1200 + 32 CUs.

#### ROCm 7.2.3 changes vs 6.2

- `--rocm-device-lib-path=/opt/rocm-*/lib/llvm/lib/clang/<N>/lib/amdgcn/bitcode` is required (no auto-discovery).
- `__shfl_sync(mask, ...)` mask must be **64-bit** (`0xffffffffffffffffULL`) — static_assert in `amd_warp_sync_functions.h`.
- `__shfl_sync` / `__syncwarp` / `__ballot_sync` are now real functions in `amd_hip_bf16.h` — our shims gated to `HIP_VERSION_MAJOR < 7`.
- `hipcub::DeviceReduce::TransformReduce` is available (was missing in 6.2) — re-included `bfgs_minimize.hip.cpp`.
- CMake 4.x requires `CMAKE_HIP_ARCHITECTURES` set **before** `enable_language(HIP)`.

#### Newly re-included sources (compile under ROCm 7.x)
- `src/minimizer/bfgs_minimize.hip.cpp` (TransformReduce now available)
- `src/minimizer/bfgs_hessian.hip.cpp` (with new `rocmolkit/cg_reduce_shim.h` for `cooperative_groups::reduce_store_async`)
- `src/symmetric_eigensolver.hip.cpp` (`cuda::std::abs` shim sufficient)

#### Still excluded
- `morgan_fingerprint_kernels` + consumers — `cuda::std::span` deduction guides do not propagate via `using` in templated contexts; needs explicit deduction or std::span at call sites.
- `similarity_kernels` + `similarity.cpp` — PTX inline asm (BMMA, async copy).
- `butina`, `tfd`, `substruct` — CUDA Graphs Conditional Nodes / unported features.

#### Open
- Functional GPU test (`EmbedMolecules(mols, params, n)`) hits a Boost.Python ABI mismatch between the system boost-1.83 used by our bindings and the boost vendored inside `rdkit-pypi`. Resolution path: build RDKit Python wrappers ourselves (`RDK_BUILD_PYTHON_WRAPPERS=ON`) so RDKit Mol types share ABI with the rocmolkit bindings.

### Phase 2 — first Python binding compiled (2026-05-14)

- `rocmolkit/nvmolkit/CMakeLists.txt` builds boost-python MODULE libraries
  for `embedMolecules`, `mmffOptimization`, `uffOptimization`,
  `batchedForcefield`, `conformerRmsd`, `array_helpers`.
- `_embedMolecules.so` links successfully (verified in CI).
- `ROCMOLKIT_BUILD_PYTHON_BINDINGS=ON` default.
- Bindings still disabled (need Phase 4-7 kernels): fingerprints,
  clustering, substructure, tfd.
- `python-bindings-probe` CI job promoted from continue-on-error to gating.

### Phase 1 — mechanical hipify + green build (2026-05-14)

#### Added
- Apache 2.0 LICENSE + NOTICE crediting upstream nvMolKit.
- Multi-stage `Dockerfile.slim` (runtime ~2 GB) and `Dockerfile.devel` for ROCm 6.2.
- `tools/build_rdkit.sh` — minimal RDKit Release_2024_09_6 build from source.
- `tools/hipify_all.sh` — wraps standalone `hipify-perl` over `nvmolkit/`, `src/`, `rdkit_extensions/`, `tests/`, `benchmarks/`.
- `rocmolkit/include/rocmolkit/cuda_std_compat.h` — aliases NVIDIA libcudacxx (`<cuda/std/*>`) to `std::*` (C++20).
- `rocmolkit/include/rocmolkit/hip_compat.h` — shims for `__shfl_sync`, `__syncwarp`, `cuda::std::*`, `NVMOLKIT_CUDA_CC_*`, `cudaSharedmemCarveout*`.
- `rocmolkit/src/utils/nvtx.h` — no-op shim replacing NVIDIA Tools Extension headers.
- GitHub Actions workflows: `ci.yml` (build smoke + Docker slim/devel) with RDKit cache.
- Pytest smoke harness with GPU-marker gate (`tests/conftest.py`).

#### Build results (Phase 1)
- 161 files automatically converted by `hipify-perl`, 0 errors.
- 45/45 object files in `rocmolkit_core` compile on ROCm 6.2.
- Runtime image: **2.04 GB** (target gfx1100; multi-arch ~2.5 GB).
- RDKit C++ kit cached in CI (~30s on cache hit, ~10 min cold build).

#### Excluded from Phase 1 (need Phase 4-7 manual port)
- `src/butina.{cpp,hip.cpp}` — uses CUDA Graphs Conditional Nodes.
- `src/tfd/` — depends on butina.
- `src/substruct/` — `cudaSharedmemCarveoutMaxShared` plus PTX-style helpers.
- `src/minimizer/bfgs_hessian.hip.cpp` — `cooperative_groups/reduce.h`.
- `src/minimizer/bfgs_minimize.hip.cpp` — `hipcub::DeviceReduce::TransformReduce` not in ROCm 6.2.
- `src/morgan_fingerprint_kernels.hip.cpp` — `cooperative_groups::block_tile_memory`.
- `src/similarity_kernels.hip.cpp` — PTX inline asm (BMMA, async copy).
- `src/symmetric_eigensolver.hip.cpp` — `cuda::std::abs` ADL miss in template context.

#### Tooling decisions
- `uv` for Python build deps (fast, modern).
- ROCm Clang directly (`/opt/rocm-*/llvm/bin/clang++`) as both `CMAKE_CXX_COMPILER` and `CMAKE_HIP_COMPILER` — `hipcc` wrapper rejected by CMake 3.22.
- No `conda`/`pixi` — RDKit headers built from source in discarded builder stage.

#### Known limitations
- No GPU validation yet — needs self-hosted ROCm runner (`rocm-runner` job stays `if: false`).
- Python bindings (`nvmolkit/*.cpp`) not yet active — Phase 2.
- Parts of MMFF/UFF batched force-field paths excluded.
