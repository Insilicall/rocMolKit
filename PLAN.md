# rocMolKit — Port Plan (nvMolKit → ROCm/HIP)

> Port of [NVIDIA-Digital-Bio/nvMolKit](https://github.com/NVIDIA-Digital-Bio/nvMolKit) (Apache 2.0) to AMD GPUs via HIP/ROCm.
> Inspired by the [mlxmolkit](https://github.com/guillaume-osmo/mlxmolkit) approach (Apple Silicon), but using **automated CUDA→HIP conversion** instead of a from-scratch reimplementation.

---

## 1. Principles

1. **Minimal Docker image** (explicit requirement). Multi-stage build, runtime stage without SDK, headers, or docs.
2. **Reuse as much nvMolKit code as possible** through `hipify-perl`. Manual rewrite only when unavoidable.
3. **Numerical parity** with RDKit as acceptance gate (same tolerances nvMolKit uses upstream).
4. **ETKDG + MMFF94 first**. The rest lands in isolated phases.
5. **Apache 2.0** preserved; NOTICE clearly attributes the upstream project.

---

## 2. Docker image — "stay lean" strategy

### Problem with the nvMolKit image
The official one uses `rocm/dev-ubuntu-22.04` (~10–15 GB) as runtime. That is overkill in production: it pulls the HIP compiler, headers, docs, profilers, debug symbols.

### Solution: multi-stage with slim runtime

```
┌─────────────────────────────────────────────────┐
│ STAGE 1: builder                                 │
│   FROM ubuntu:22.04 + apt rocm-* selective       │
│   - clang++, cmake, RDKit headers, boost-all     │
│   - builds RDKit from source (~10 min, cached)   │
│   - compiles librocmolkit_core.so + bindings     │
│   - strip --strip-unneeded on every .so          │
└─────────────────────────────────────────────────┘
                        ↓ COPY artifacts
┌─────────────────────────────────────────────────┐
│ STAGE 2: runtime                                 │
│   FROM ubuntu:22.04                              │
│   - apt: libpython3.11, libstdc++6, libgomp1     │
│   - apt: hip-runtime-amd, hsa-rocr, comgr,       │
│         rocblas, rocrand, hiprand                │
│   - rocsolver/rocsparse purged (transitive deps) │
│   - Tensile kernels filtered to GPU_TARGETS only │
│   - rocmolkit .so + rdkit-pypi via uv            │
│   - NO /opt/rocm/bin, NO /opt/rocm/include       │
│   - NO /opt/rocm/share, NO /opt/rocm/llvm        │
└─────────────────────────────────────────────────┘
```

**Achieved size:** **2.04 GB** (single gfx target, gfx1100). Compare:
- `rocm/dev-ubuntu-22.04` ≈ 13 GB
- `rocm/rocm-terminal` ≈ 9 GB
- `rocmolkit:slim` ≈ **2 GB** ✅

### Variants
- `rocmolkit:slim` — minimal runtime (above), published at `ghcr.io/insilicall/rocmolkit:slim`.
- `rocmolkit:devel` — for developers, with hipcc + headers + RDKit dev kit. Published at `ghcr.io/insilicall/rocmolkit:devel`.
- `rocmolkit:cuda-compat` — optional, future, HIP code running on NVIDIA (HIP is portable).

### Additional shrink tricks
- `--squash` on final build.
- Aggressive apt purge in the runtime stage.
- `rdkit-pypi` wheel via `uv` instead of conda (which would pull miniforge).

---

## 3. Conversion strategy

### What is automatic (hipify-perl)
| nvMolKit uses | rocMolKit uses | tool |
|---|---|---|
| `cudaMalloc/cudaMemcpy/cudaStream_t` | `hipMalloc/hipMemcpy/hipStream_t` | `hipify-perl` |
| `__global__`, `__device__`, `__shared__` | same (HIP inherits the syntax) | trivial |
| `cuBLAS` | `hipBLAS` (wrapper) or `rocBLAS` | `hipify-perl` |
| `cuRAND` | `hipRAND` or `rocRAND` | `hipify-perl` |
| `Thrust` | `rocThrust` (~identical API) | recompile |
| `CUB` | `hipCUB` / `rocPRIM` | `hipify-perl` |

### What requires manual work
1. **Wavefront 64 vs warp 32**
   - Kernels using `__shfl_*` assuming 32 threads need parameterisation.
   - Use `warpSize` (HIP intrinsic) instead of hardcoded 32.
   - Intra-warp reductions become intra-wavefront → fewer blocks per CU on CDNA.

2. **cooperative_groups** (nvMolKit uses these in the ETKDG batch optimiser)
   - HIP supports `hip/hip_cooperative_groups.h` but with a smaller subset.
   - Some grid-sync patterns need rework.

3. **Double-precision atomics**
   - AMD CDNA (MI100+) supports them; RDNA (gaming) does not. Detect at runtime.

4. **CMake**
   - `enable_language(CUDA)` → `find_package(hip REQUIRED)` + `set_source_files_properties(... LANGUAGE HIP)`.
   - `CMAKE_CUDA_ARCHITECTURES` → `GPU_TARGETS` (e.g. `gfx90a;gfx942;gfx1100`).
   - **Important:** ROCm Clang directly (not `hipcc`) — CMake 3.22 rejects the wrapper.

5. **boost-python bindings**
   - Work fine; just need `CXX=clang++` (ROCm) when building the modules.
   - RDKit ABI must match (same stdlib).

---

## 4. Phases

| # | Phase | Deliverable | Acceptance criterion | Estimate |
|---|---|---|---|---|
| 0 | **Scaffold** | repo + LICENSE/NOTICE + CI skeleton + multi-stage Dockerfile | `docker build` passes, image under budget | 1–2 days **(✅ done; image 2.04 GB)** |
| 1 | **Mechanical hipify + core build** | all `.cu` → `.hip.cpp`, CMake builds the HIP lib | `cmake --build rocmolkit_core` with no errors | 3–7 days **(✅ DONE — 45/45 obj compiled, 8 files excluded for Phase 4-7 rewrite)** |
| 2 | **First Python binding** | `_embedMolecules.so` links | binding loads in `import rocmolkit.embedMolecules` | done **(✅ Phase 2 first binding green at v0.1.0-alpha)** |
| 3 | **Functional ETKDG** | `EmbedMolecules` returns valid conformers | parity with RDKit on SPICE-100 (RMSD < 0.1 Å vs nvMolKit) | 2–3 weeks (needs AMD GPU runner) |
| 4 | **Functional MMFF94** | `MMFFOptimizeMoleculesConfs` converges | final energy within 1e-3 kcal/mol vs RDKit | 2–3 weeks |
| 5 | **AMD tuning** | benchmark vs RDKit CPU on RX 7900 XTX and/or MI210 | ≥ 5× speedup on batches ≥ 100 mols | continuous |
| 6 | **Fingerprints + Similarity** | Morgan + Tanimoto on GPU | exact match with RDKit (identical bits) | 2 weeks |
| 7 | **Butina clustering** | divide-and-conquer over > 100k mols | results identical to nvMolKit | 1–2 weeks |
| 8 | **UFF, conformerRMSD, TFD** | rest of the API | RDKit parity | 2 weeks each |

---

## 5. Directory layout (current)

```
rocMolKit/
├── LICENSE                    # Apache 2.0 (inherited)
├── NOTICE                     # nvMolKit attribution + rocMolKit authors
├── README.md
├── PLAN.md                    # this file
├── ISSUES.md                  # known gaps + manual fixes
├── CHANGELOG.md               # release notes per phase
├── pyproject.toml             # build via scikit-build-core
├── CMakeLists.txt             # root
├── cmake/
│   └── ROCmTargets.cmake      # gfx target mapping
├── docker/
│   ├── Dockerfile.slim        # minimal runtime (~2 GB)
│   ├── Dockerfile.devel       # with SDK
│   ├── runtime-libs.txt       # legacy reference, no longer used
│   └── .dockerignore
├── tools/
│   ├── build_rdkit.sh         # source build of RDKit Release_2024_09_6
│   ├── hipify_all.sh          # recursive hipify-perl wrapper
│   ├── numerical_diff.py      # parity gate vs RDKit
│   ├── strip_release.sh       # strip + dpkg cleanup
│   └── bin/
│       └── hipify-perl        # standalone vendored copy of the script
├── rocmolkit/                 # mirrors upstream layout
│   ├── include/rocmolkit/
│   │   ├── hip_compat.h       # force-included; warp/macro shims
│   │   └── cuda_std_compat.h  # cuda::std::* → std::* aliases
│   ├── src/                   # HIP kernels + host code (hipified)
│   ├── rdkit_extensions/      # pure C++ RDKit extensions
│   ├── nvmolkit/              # boost-python bindings (Phase 2 active)
│   └── __init__.py
├── tests/
│   ├── test_smoke.py          # CPU-only import + version check
│   ├── conftest.py            # `gpu` marker, --rocm flag
│   └── data/spice_100.smi     # 10 SMILES seed for parity
├── benchmarks/
└── .github/workflows/
    ├── ci.yml                 # build + Docker slim/devel + bindings probe
    └── docker.yml             # publish images to ghcr.io on tag v*
```

---

## 6. Numerical validation (quality gate)

Without this, the port "compiles but is junk". Plan:

1. **Dataset:** same 1000 SMILES from SPICE-2.0.1 that mlxmolkit benchmarks (`bench_conformers.py`).
2. **ETKDG metric:** RMSD between conformers generated by rocMolKit, nvMolKit (GPU reference) and RDKit (CPU reference). Target: 95% of molecules with RMSD < 0.5 Å after alignment.
3. **MMFF metric:** final energy `|E_roc - E_rdkit| / |E_rdkit| < 1e-4`.
4. **CI gate:** fail on regression. Implemented in `tools/numerical_diff.py`; activates when `rocm-runner` job is enabled.

---

## 7. Target hardware

| GPU | Arch | ROCm status | Notes |
|---|---|---|---|
| MI300X / MI300A | gfx942 | official | ideal but expensive |
| MI250 / MI210 | gfx90a | official | datacenter sweet spot |
| RX 7900 XTX/XT | gfx1100 | official (since 6.0) | best consumer; build defaults here |
| RX 6900 XT | gfx1030 | unofficial | works with `HSA_OVERRIDE_GFX_VERSION=10.3.0` |
| Radeon VII | gfx906 | EOL in ROCm 6 | not targeted |

**Default `GPU_TARGETS`:** `gfx1100;gfx90a;gfx942` (RDNA3 + CDNA2 + CDNA3). Override with `-DGPU_TARGETS=...`.

---

## 8. Risks

| Risk | Probability | Mitigation |
|---|---|---|
| **CUDA Graphs Conditional in `butina.cu`** (CONFIRMED Phase 1) | 100% | Defer Butina to Phase 7; rewrite the conditional loop as CPU-side dispatch |
| Custom non-standard `cudaCheckError` macro (CONFIRMED Phase 1) | 100% | Upstream `cuda_error_check.h` is HIP-clean after hipify; no shim needed |
| nvMolKit uses CUTLASS / advanced CUDA-only kernels | medium | rocPRIM as substitute; rewrite where needed |
| ETKDG cooperative-groups don't translate cleanly | low | hipify converted them OK in `dist_geom_kernels_device` and `bfgs_hessian` (verified) |
| Float atomics break on RDNA | high | alternative path with lock-based reduction |
| RDKit ABI incompatible between conda and pip | medium | pin one of them, document |
| CI without an AMD GPU | high | self-hosted runner mandatory, or use HIP-on-CUDA for smoke testing |

---

## 9. Done so far (v0.1.0-alpha, 2026-05-14)

1. ✅ Repo name decided — **rocMolKit**
2. ✅ Default GPU targets set — `gfx1100;gfx90a;gfx942`
3. ✅ Scaffold (Phase 0): repo, LICENSE/NOTICE, multi-stage Dockerfiles, CI skeleton
4. ✅ Hipify mechanical pass (Phase 1): 161 files converted, 45/45 obj compile
5. ✅ First Python binding (Phase 2): `_embedMolecules.so` links cleanly
6. ✅ Docker images published to ghcr.io
7. ✅ GitHub Release v0.1.0-alpha

## 10. Status & remaining-feature plan (updated 2026-06-03)

**Functional + validated on AMD RDNA4 (gfx1200):** ETKDG generation, MMFF94
optimization (both now *faster than mlxmolkit and 12-thread RDKit* — see
[docs/PERFORMANCE_HIP.md](docs/PERFORMANCE_HIP.md)), UFF, batched forcefield,
conformer RMSD, array helpers.

**Remaining modules.** Each is scaffolded (CUDA source + a test suite already
exist) but **disabled in CMake** behind a specific HIP-port blocker (documented
inline in `rocmolkit/CMakeLists.txt`). The plan below ports them in dependency
order; **every phase is "done" only when its existing test suite passes on
gfx1200**, then its `_<Module>.so` binding is added back to
`ROCMOLKIT_PHASE2_BINDINGS` in `rocmolkit/nvmolkit/CMakeLists.txt`.

| Phase | Module | Blocker to remove | Verify with | Depends on |
|---|---|---|---|---|
| F1 | ✅ **Fingerprints (Morgan)** — **done** (bit-exact vs RDKit MorganGenerator, 103/103 incl. 70–110-atom molecules). 128-atom tile rewritten to block-level cooperation (AMD wave64); `cuda::std::span` CTAD replaced with a `__host__ __device__` helper; non-trivial `__shared__` backed by a raw buffer. Caveat: the 64-atom warp-sort still mismaps on wave64, so 32–127-atom molecules route through the (correct) 128-atom kernel — `tools/fp_validate.py`. | `cooperative_groups` tile>wavefront + `cuda::std::span` deduction | `test_morgan_fingerprint` | — |
| F2 | ✅ **Similarity (Tanimoto/Cosine)** — **done** (bit-exact vs RDKit BulkTanimotoSimilarity, 10000/10000 pairs `max\|Δ\|=0`; cosine self-sim diag==1 + symmetric — `tools/sim_validate.py`). NVIDIA BMMA tensor-core PTX (`mma.sync…b1…popc`) guarded out on AMD; `supportsTensorOps()` returns false so the `__popc` fallback runs (correct on wave64). | PTX inline asm (BMMA tensor-core matmul + async copy) in `macros_ptx.hip.h` | `test_similarity` | F1 |
| F3 | ✅ **Butina clustering** — **done** (matches RDKit Butina exactly for cutoffs ≤0.5; at higher cutoffs the partitions differ only by legitimate parallel tie-breaks — every GPU cluster is ball-valid — `tools/butina_validate.py`). Three HIP fixes: conditional-WHILE graph → host-driven loop; `pruneNeighborlistKernel`'s wave64 bug (default `hipcub::WarpReduce` is 64-wide on AMD, merging two 32-tiles) → manual shared-memory compaction; `hipcub::DeviceRadixSort::SortPairs` returned a garbled permutation on gfx1200 → host argsort in `renumberClustersBySize`. | CUDA Graphs Conditional nodes (no hipGraph conditional) | `test_butina` | F1, F2 |
| F4 | ✅ **TFD** — **done** (matches RDKit `GetTFDMatrix` to float32 precision, 25/25 mols, worst max\|Δ\|=4e-4 — `tools/tfd_validate.py`). Pure re-enable: the block-per-molecule kernels in `src/tfd/` use no CUDA-specific features; only needed `src/tfd` on the core include path and a `tfd.cpp` binding special-case. | depends on butina (kernels in `src/tfd/` are already written) | `test_tfd{,_cpu,_gpu,_kernels}` | F3 |
| F5 | ✅ **Substructure — done** (840/840 pairs match RDKit HasSubstructMatch via tools/ss_validate.py) | `cudaSharedmemCarveoutMaxShared` → `hipFuncAttributePreferredSharedMemoryCarveout` (value 100) | `test_substruct_{algos,integration,label_integration,search}` | — (independent) |

Notes / per-phase work:

- **F1 Fingerprints** — replace `block_tile_memory` with an explicit `__shared__`
  scratch buffer; pin the `cuda::std::span` element types so deduction works.
  Re-enable `morgan_fingerprint_kernels.hip.cpp` + `morgan_fingerprint*.cpp`.
- **F2 Similarity** — ✅ done. The bitwise Tanimoto does not need tensor cores:
  the BMMA/async-copy PTX in `macros_ptx.hip.h` is `#if`-guarded to NVIDIA, with
  AMD stubs that only need to compile (`supportsTensorOps()` is false on AMD, so
  the existing `__popc` popcount path runs and matches RDKit exactly).
- **F3 Butina** — ✅ done. The conditional-WHILE graph nodes became host-driven
  do-while loops (CPU reads `maxValue` back each iteration). Two latent kernel
  bugs surfaced on wave64/gfx1200 and were fixed: the neighborlist prune mixed a
  32-lane cooperative-groups tile with a default-width `hipcub::WarpReduce`
  (64-wide on AMD), and `DeviceRadixSort::SortPairs` returned garbage for the
  renumber step (now a host argsort).
- **F4 TFD** — ✅ done. A pure re-enable: the `src/tfd/` kernels are block-per-
  molecule with plain thread loops (no warp primitives, PTX, or `cuda::std`), so
  they compiled unchanged. Added `src/tfd` to the core include path and a
  `tfd.cpp` source special-case for the `_TFD` binding.
- **F5 Substructure** — the blocker is a one-attribute translation; the rest of
  `src/substruct/` (executor, preprocessor, kernels, search; largest binding at
  342 lines) then needs an end-to-end build + the four substruct tests. Can be
  done in parallel with F1–F4.
- **Prereq** — `symmetric_eigensolver` (`cuda::std::abs` shim) only matters if a
  clustering path needs PCA/eigendecomposition; defer until F3 needs it.

Infra, in parallel:

1. ⏳ Self-hosted ROCm runner with a real AMD GPU; flip the `rocm-runner` CI job
   from `if: false` so the test gates above run on every release.
2. ⏳ Rebuild + publish the devel/slim images after each phase so the new
   bindings ship (the current image predates these modules).

---

## 11. Out of scope (declared)

- Simultaneous NVIDIA support via HIP-on-CUDA (possible, but disperses focus). Revisit after Phase 5.
- GUI or integrated notebook.
- Pure-Python rewrite with PyTorch/ROCm (the mlxmolkit path). Slower and loses parity with nvMolKit.
- conda packaging. PyPI wheel only initially.
