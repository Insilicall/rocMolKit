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

## 10. Next steps

1. ⏳ Set up self-hosted ROCm runner with an actual AMD GPU and flip `rocm-runner` job from `if: false`.
2. ⏳ Phase 3: validate ETKDG numerical parity vs RDKit on SPICE-100.
3. ⏳ Phase 4: validate MMFF94 numerical parity.
4. ⏳ Phases 5–8: tune for AMD architectures, port the 8 excluded modules (see [ISSUES.md](ISSUES.md)), enable remaining bindings.

---

## 11. Out of scope (declared)

- Simultaneous NVIDIA support via HIP-on-CUDA (possible, but disperses focus). Revisit after Phase 5.
- GUI or integrated notebook.
- Pure-Python rewrite with PyTorch/ROCm (the mlxmolkit path). Slower and loses parity with nvMolKit.
- conda packaging. PyPI wheel only initially.
