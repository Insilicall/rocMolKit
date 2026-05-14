# Changelog

All notable changes to rocMolKit will be documented here.

## [Unreleased]

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
