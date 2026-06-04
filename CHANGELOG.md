# Changelog

All notable changes to rocMolKit will be documented here.

## [unreleased]

## [v0.4.3] — 2026-06-04

### Fixed
- **ETKDG silently dropped ~12% of conformers (the 88% "success" was a bug, not a
  limitation).** The BATCHED FP64 BFGS backend converged *nothing* on
  ROCm/gfx1200: `hipcub::DeviceSelect::Flagged` and `DeviceReduce::TransformReduce`
  return 0 regardless of input on this arch (same class as the broken
  `DeviceRadixSort`), so `compactAndCountConverged` reported "all converged" on
  iteration 0 and the minimizer exited before doing any work. HYBRID routing sends
  any batch containing a >64-atom molecule entirely to BATCHED, so one large
  molecule dropped its whole 500-conformer batch — the largest/hardest ~12%, which
  surfaced as an "88% success rate" and inflated throughput. Replaced both broken
  primitives with wavefront-size-agnostic atomic compaction/count kernels. ETKDG
  generation is now **100% success**, energy-validated (generated conformers fall
  in the same MMFF basins as RDKit: median ΔE = 0.00 kcal/mol).

### Performance
- **Substructure: ~6× slower than RDKit → ~9× faster (vs 12-thread).** The scale
  hardening in v0.4.2 chunked *every* search at 1024 targets with a
  sync + `hipMemPoolTrimTo(pool, 0)` between chunks. That safety is needed only
  for recursive SMARTS (GBs of paint scratch); for simple queries it was pure
  overhead. Now chunk size adapts to recursion: recursive → 1024 + trim
  (unchanged, OOM-safe); non-recursive → 65536 (single-shot for typical batches).
  Measured 50k×20 drug-like: 326k → 3,770k pair/s (0.2× → 3.1× vs single-core,
  9× vs 12-thread). Correctness unchanged (840/840 vs RDKit).
- **MMFF optimize: reuse one BFGS minimizer per thread across batches** instead of
  reconstructing (and reallocating all its device buffers) per batch. +22% in the
  high-conformers-per-molecule regime; removes per-batch mem-pool churn.

### Docs
- Performance numbers rewritten to be honest and audit-proof: reported by molecule
  size (small ~12 atoms vs drug-like ~30 atoms), all at 100% success and
  RDKit-validated, with corrected vs-RDKit (5–13×) and vs-mlxmolkit (1.7–2.6×, both
  forks, matched molecule sizes) comparisons. Speedup chart and two-panel
  conformer-scaling figure regenerated with the corrected data.

## [v0.4.2] — 2026-06-04

### Fixed
- **Substructure search wedged the GPU at scale.** A single
  `countSubstructMatches` / `hasSubstructMatch` / `getSubstructMatches` call with
  more than ~3-4k targets — especially with recursive SMARTS (`[$(...)]`) —
  drove the async-allocation pool past 6 GB and locked up the device (recursive
  at 4k hung indefinitely). Root cause: the per-search GPU working set (recursive
  paint scratch + the pool that retains freed blocks) grew with the targets
  handled in one search and was only released when the search ended. Now the
  three entry points process targets in chunks of 1024 — each chunk an
  independent search writing into its own target slice — and between chunks they
  `hipDeviceSynchronize` + `hipMemPoolTrimTo(pool, 0)` to return the memory to
  the device. Peak VRAM is one chunk regardless of total target count, so a
  single call scales to any size and any SMARTS. Results are identical to one big
  call (bit-exact vs RDKit at 20k-50k targets). Recursive SMARTS at 50k targets
  now run in ~3 s with zero mismatches; simple-SMARTS throughput is unchanged.
  The recursive paint also bounds its own block count as a second line of defence.

### Added
- **Building-blocks performance table** in the README (under the conformer
  benchmarks): Morgan fingerprints, Tanimoto similarity, substructure, TFD and
  Butina clustering, GPU vs RDKit single-core and 12-thread on the same host.
  Similarity 645x/658x, TFD 123x/137x, Butina 49x/48x, fingerprints 16x/7x,
  substructure 3.9x/12x. A `RDKit (1 core)` column was also added to the conformer
  table. Scripts: `tools/bench_features.py`, `tools/conformer_cpu_bench.py`.

## [v0.4.1] — 2026-06-04

Packaging hotfix: the v0.4.0 `slim` image never imported (its build-time smoke
test failed), which also blocked the tagged docker publish. Three pre-existing
slim runtime gaps, fixed so the production image imports end-to-end:

### Fixed
- **Dead `roc::hipblas` link** in `rocmolkit_core`. No code calls any hipBLAS
  API (an inherited nvMolKit link), but `libhipblas.so.3` hard-depends on
  `librocsolver.so.0` (~870 MB) and, via hipblaslt, `librocroller.so.1`
  (~84 MB) — so the dead link dragged ~950 MB of unused math libraries into the
  runtime and made the slim image unsatisfiable. Dropped the link (`hiprand`
  stays; `coord_gen` uses it). Shrinks every image.
- **Missing RDKit C++ runtime libs in slim.** The builder compiles RDKit
  without Python wrappers and the runtime installs `rdkit-pypi`, so the bindings
  could not resolve `libRDKitDistGeomHelpers.so.1` et al. Copy `/opt/rdkit/lib`
  (~30 MB) into the runtime and add it to `LD_LIBRARY_PATH`; also copy
  `libomp.so` (a direct core dep) and the small roctx / rocprofiler-register libs.
- **boost-python cross-module import order.** `_mmffOptimization` /
  `_uffOptimization` / `_batchedForcefield` take a `BatchHardwareOptions` default
  argument whose to_python converter is registered by `_embedMolecules`' init;
  importing one of them first raised "No to_python converter for
  BatchHardwareOptions". `rocmolkit/__init__` now eagerly imports
  `_embedMolecules` (best-effort), so any `from rocmolkit._X import ...` works.

Slim image: import smoke test passes; 2177 MB compressed (gfx1200, budget 2500).

## [v0.4.0] — 2026-06-04

First **beta**: every module in the porting plan is now ported to HIP/RDNA4 and
validated against RDKit on a gfx1200 GPU. Joins the already-shipping ETKDG,
MMFF94, UFF, batched forcefield and conformer RMSD.

### Added
- **Morgan fingerprints on GPU (F1)** — bit-exact vs RDKit `MorganGenerator`.
  128-atom tile rewritten to block-level cooperation for AMD wave64;
  `cuda::std::span` CTAD replaced with a `__host__ __device__` helper. Module
  `_Fingerprints`. Validator `tools/fp_validate.py`.
- **Tanimoto/Cosine similarity on GPU (F2)** — bit-exact vs
  `DataStructs.BulkTanimotoSimilarity` (max|Δ|=0 over 10k pairs). NVIDIA binary
  tensor-core (BMMA) PTX guarded out on AMD; the `__popc` popcount fallback
  runs and is correct on wave64. Module `_DataStructs`. Validator
  `tools/sim_validate.py`.
- **Substructure search on GPU (F5)** — 840/840 matches vs RDKit
  `HasSubstructMatch`. Module `_substructure`. Validator `tools/ss_validate.py`.
- **Butina clustering on GPU (F3)** — matches RDKit `Butina.ClusterData`
  exactly at low cutoffs; at high cutoffs differs only by valid parallel
  tie-breaks (every GPU cluster is ball-valid). Module `_clustering`. Validator
  `tools/butina_validate.py`.
- **Torsion Fingerprint Deviation on GPU (F4)** — matches RDKit `GetTFDMatrix`
  to float32 precision (worst max|Δ|=4e-4). Module `_TFD`. Validator
  `tools/tfd_validate.py`.
- **GPU integration test suite** `tests/test_gpu_features.py` — the four feature
  validators wired into pytest as `gpu`-marked tests. Run with
  `pytest tests/ --rocm`; skipped (and `importorskip`-guarded) otherwise.
- **`tools/validate_gpu.sh`**: single-command post-reboot validation.
  Refuses to run if it detects the leaked-VRAM state (>200 MB held with
  no `/dev/kfd` holders), then runs the sweep against the devel image.

### Fixed
- **Butina wave64 correctness** — `pruneNeighborlistKernel` mixed a 32-lane
  cooperative-groups tile with a default-width `hipcub::WarpReduce`, which is
  64-wide on AMD and merged two tiles' neighbor counts, collapsing unrelated
  points into one giant cluster. Rewritten with a wave-agnostic shared-memory
  compaction (no hipcub warp primitives).
- **Butina renumber corruption** — `hipcub::DeviceRadixSort::SortPairs` returned
  a garbled permutation on gfx1200, corrupting the old→new id remap (points
  ended up with negative ids). Replaced with a stable host argsort in
  `renumberClustersBySize`.
- **Conditional-graph clustering loop** — the Butina conditional-WHILE CUDA
  Graph nodes (no HIP equivalent) became host-driven do-while loops.
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
- **Status: alpha → beta.** README banner and roadmap updated; nothing in the
  porting plan is disabled anymore.
- **CI `rocm-runner` job** now builds the devel image and runs
  `pytest tests/ --rocm` (was a non-configuring `ctest` path). Gated on the
  repo variable `ROCM_RUNNER_ONLINE` instead of `if: false`, so it skips cleanly
  until a `[self-hosted, rocm]` runner is registered.
- **`tools/benchmark.py` rewritten**: drops the `--n`-only mode in favour
  of an `(N, k)` sweep that exercises both GPU parallelism axes. CPU
  baseline now uses RDKit's multi-threaded `EmbedMultipleConfs` —
  honest comparison on a 12-thread Ryzen — with a `numThreads=1` row
  alongside. GPU paths always pass `BatchHardwareOptions(gpuIds=[0])`
  to pin the discrete device. Each phase is SIGALRM-bounded so a stuck
  GPU does not wedge the whole run.
- **Version** bumped to 0.4.0; `pyproject.toml` realigned (was a stale 0.1.0 /
  "Pre-Alpha") to match `rocmolkit.__version__` and the beta classifier.

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
