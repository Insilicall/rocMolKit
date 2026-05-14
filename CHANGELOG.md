# Changelog

All notable changes to rocMolKit will be documented here.

## [unreleased]

### Fixed
- **Dockerfile.slim runtime imports**: published v0.3.2-alpha-slim
  image cannot `import rocmolkit._embedMolecules`. Two packaging gaps:
  (1) `LD_LIBRARY_PATH` was `/opt/rocm/lib` only — missing
  `/usr/local/lib` (where `librocmolkit_core.so` is COPY'd in) and
  `/opt/rocm/lib/llvm/lib` (where `libomp.so` lives); (2) the runtime
  stage installed neither `libboost1.83-all` nor any other source of
  `libboost_python310.so.1.83.0`, so the boost-python binding failed
  ImportError. Added the boost runtime via the same `ppa:mhier/libboost-latest`
  the builder uses, and a build-time `python3 -c "from rocmolkit._embedMolecules
  import ..."` smoke test that fails the build (and CI) on regression.

### Changed
- **`tools/benchmark.py` rewritten**: drops the `--n`-only mode in favour
  of an `(N, k)` sweep that exercises both GPU parallelism axes. CPU
  baseline now uses RDKit's multi-threaded `EmbedMultipleConfs` —
  honest comparison on a 12-thread Ryzen — with a `numThreads=1` row
  alongside. GPU paths always pass `BatchHardwareOptions(gpuIds=[0])`
  to pin the discrete device. Each phase is SIGALRM-bounded so a stuck
  GPU does not wedge the whole run.

### Added
- **`tools/validate_gpu.sh`**: single-command post-reboot validation.
  Refuses to run if it detects the leaked-VRAM state (>200 MB held with
  no `/dev/kfd` holders), then runs the sweep against the devel image.

## [v0.3.2-alpha] — 2026-05-14

### Fixed
- **v0.3.1-alpha CMake regression**: `ROCMOLKIT_PYTHON_INSTALL_DIR`
  was defined AFTER `add_subdirectory(nvmolkit)`, so nvmolkit's
  `install(TARGETS ...)` rule saw an empty destination and CI
  build-cpu-smoke errored with "install TARGETS given no LIBRARY
  DESTINATION for module target _embedMolecules". Move the variable
  definition before the subdirectory include. Verified `cmake
  configure` clean both with and without `ROCMOLKIT_BUILD_PYTHON_BINDINGS=ON`.

## [v0.3.1-alpha] — 2026-05-14 (broken — superseded by v0.3.2-alpha)

### Fixed
- **Slim image was unusable on v0.3.0-alpha**: `import rocmolkit`
  failed in the published `ghcr.io/insilicall/rocmolkit:slim` because
  the Python package and all `.so` bindings were going to
  `${Python_SITELIB}` (absolute path) in the discarded builder stage
  instead of into `/install` for the COPY into the runtime stage. The
  install destination is now configurable via
  `ROCMOLKIT_PYTHON_INSTALL_DIR`, and `Dockerfile.slim` sets it to a
  path under `CMAKE_INSTALL_PREFIX` so the artefacts actually ride
  the COPY into `/usr/local/lib/python3.10/dist-packages/rocmolkit`.
- `Dockerfile.slim` pins `numpy<2` (rdkit-pypi was compiled against
  numpy 1.x ABI; under 2.x every rdkit import emits a loud
  "_ARRAY_API not found" warning).
- CI `python-bindings-probe` was on `rocm/dev-ubuntu-22.04:6.2`, but
  `bfgs_minimize.hip.cpp` uses `hipcub::DeviceReduce::TransformReduce`
  which only exists in ROCm 7.x. Bumped to 7.2.3 so the job actually
  exercises the same toolchain everything else uses.
- CI `docker-slim-size` budget gate now uses
  `docker buildx build --platform linux/amd64 --load`. The
  `setup-buildx-action` default produces a manifest list (amd64 +
  arm64), tripling the reported image size and breaking the per-arch
  2.5 GB budget gate.
- `docker-publish` slim image now also includes `gfx1200` so RX
  9060/9070 owners can actually run the published artefact.

### Verified
- Slim image at 2018 MB, `import rocmolkit` clean,
  `rocmolkit.safe.{embed,mmff_optimize,uff_optimize}_molecule` all
  importable from a fresh `docker run`.

## [v0.3.0-alpha] — 2026-05-14

### Added
- `rocmolkit.safe.embed_molecule(s)` and `mmff_optimize_molecule(s)` —
  subprocess+retry wrappers that work around the open
  ROCm 7.2.3 + gfx1200 state-leak SIGSEGV. Validated 100% reliability
  across 45 ETKDG embeds and 15 follow-up MMFF optimisations of a
  diverse molecular set; numerical parity vs RDKit ≤ 0.02 Å on bond
  lengths, MMFF energies match qualitatively.
- `tests/test_safe.py` — pytest regression suite gated on the `gpu`
  marker (skipped without `--rocm` or `ROCMOLKIT_HAS_GPU=1`).
- `tests/repro/README.md` — context for the two C++ bisect repros that
  localised the bug to inside `nvMolKit::embedMolecules`.

### Changed
- `rocmolkit.__version__` → `0.3.0`.
- `README.md` quickstart now leads with `rocmolkit.safe`; the direct
  binding is documented as a power-user path with the segfault caveat.
- `ISSUES.md` updated with measured per-call success rates and a
  surface table comparing ETKDG vs MMFF94 thresholds.

### Investigation
- Five additional fixes attempted and reverted this session (sync
  `hipMalloc`, stream sync at end of OpenMP region, `hipDeviceReset()`
  per call, etc.). Each made the bug **worse**, confirming the
  pattern: more aggressive cleanup ↔ earlier crash. Documented so the
  next investigator does not repeat them.
- Bisect refinement: bug threshold is N=4 mols/batch for ETKDG and
  ~30 sequential calls for MMFF94. Pure-C++ `AsyncDeviceVector`
  stress runs clean (the allocator pattern is fine in isolation), so
  the failure is somewhere else in `nvMolKit::embedMolecules`.
- C++ root cause remains blocked on `rocgdb` 7.2.3 not yet supporting
  gfx1200 (`AMDGCN architecture 0x45 is not supported`).

## [Unreleased]

### MMFF94 + ETKDG numerically validated end-to-end on AMD GPU (2026-05-14)

`MMFFOptimizeMoleculesConfs` works on AMD RX 9060 XT (gfx1200) and converges
to the same minimum-energy ethanol conformation that RDKit CPU produces:

```
ethanol, 3 ETKDG conformers:
  BEFORE MMFF: [4.732, 1.981, 5.931] kcal/mol
  AFTER  MMFF: [-1.337, -1.337, -1.337] kcal/mol  ← all converged to global min
  GPU time:    267 ms
```

Energy reduction is dramatic and physically sensible. All three independent
random starts collapse to the same minimum, as expected for a small flexible
molecule. Indicates the BFGS minimizer + MMFF energy/gradient kernels work
correctly on RDNA4.

### Crash triangulation (ETKDG)
- OK: CCO, CCCC, CCCCCC (20 atoms), benzene, toluene (intermittent), p-xylene (18 atoms).
- CRASH: aspirin (21 atoms), pyridine (11 atoms — has aromatic N).
- Crashes appear non-deterministic on the boundary cases (toluene worked the
  second run after first crash). Suggests race condition or uninitialized
  buffer in per-molecule device allocation, not a hard size limit.

### ETKDG runs on AMD GPU - first numerically valid output (2026-05-14, late)

End-to-end GPU execution validated on AMD Radeon RX 9060 XT:

```
=== Ethanol (CCO) - AMD GPU (gfx1200) vs RDKit CPU ===
Bond C-C: GPU=1.506 Å, CPU=1.506 Å  (identical)
Bond C-O: GPU=1.381 Å, CPU=1.387 Å  (diff 0.006 Å)
```

The GPU-generated conformer is chemically valid: bond lengths match RDKit
within sub-angstrom tolerance, all atoms have sensible 3D positions.

**Working configurations** (no crash):
- Ethanol x1 conformer: 792 ms (cold)
- Ethanol x5 conformers: 1280 ms
- Benzene x1 conformer: 766 ms
- Ethanol x2 mols x1 conformer: 633 ms

**Known crash:** `EmbedMolecules` segfaults on aspirin (21 atoms after AddHs)
and other mid-size molecules. CCO + benzene reliably work. Investigation
needed - likely a buffer-size edge case in the per-molecule device data
allocation, not a fundamental incompatibility.

**Stack used:**
- RDKit 2024.09.6 built with `RDK_BUILD_PYTHON_WRAPPERS=ON` so Python rdkit
  shares the same boost-1.83 ABI as our boost-python bindings.
- numpy pinned `<2` (RDKit Python wrappers were compiled against numpy 1.x ABI).
- All else as previous milestone (ROCm 7.2.3, gfx1200, librocmolkit_core.so).

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
