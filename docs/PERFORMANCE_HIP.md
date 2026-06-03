# Why rocMolKit is slow on consumer RDNA — and what makes HIP fast

Measured on an **AMD Radeon RX 9060 XT** (Navi 44, gfx1200, RDNA4, 32 CUs),
ROCm 7.2.3, dataset `tests/data/druglike_100.smi`. TL;DR: **HIP is not the
problem — FP64 on a consumer GPU is.**

## What the nvMolKit benchmarks actually report

From `upstream/CHANGELOG.md`:

- "On an **H200 GPU**, speedups of **400-1000×** can be achieved on datasets up
  to **60k molecules**" — but this is for **Butina clustering**, not conformer
  generation.
- BFGS minimizer: "up to 5× vs the previous version on small molecules".
- Requirement: **NVIDIA datacenter GPU** (V100/A100/H100/H200).
- They ship an **Optuna autotuner** (`docs/autotune.rst`) that picks batch
  size / threading per GPU model. rocMolKit has none.

So the headline numbers are **datacenter hardware + 60k-molecule batches +
a clustering workload + autotuned knobs** — not ETKDG on a consumer card with
a few thousand molecules.

## The real bottleneck: double precision on a consumer GPU

The ETKDG pipeline spends **98.8%** of its time in the BFGS minimization, and
that code is almost entirely `double` (`dist_geom_kernels.hip.cpp`: 188 `double`,
0 `float`; `bfgsMinimizeKernel`: 42 `double`, 1 `float`). nvMolKit uses `double`
on purpose — it is cheap on datacenter GPUs:

| GPU | FP32 | FP64 | FP64 ratio |
|-----|------|------|-----------|
| RX 9060 XT (RDNA4, ours) | 25.6 TFLOPS | ~1.6 TFLOPS | **1/16** |
| NVIDIA H100 (datacenter) | ~67 | ~34 | 1/2 |
| AMD MI300X (datacenter CDNA) | ~163 | ~81 | 1/2 |

On the H200/MI300 FP64 is fast, so the `double` minimizer flies. On the
RX 9060 XT FP64 runs at ~1/16 of FP32, so the same kernels crawl.

## Measured curve (ETKDG, k=4, coordgen on CPU)

| N | GPU conf/s | CPU 12-thread conf/s |
|---|-----------|----------------------|
| 25 | 37 | 3014 |
| 500 | 78 | 875 |
| 1000 | 110 | 871 |
| 2000 | 109 (plateau) | — |

The GPU **saturates at ~110 conf/s** for N≥1000 — it is compute-bound on FP64,
so larger batches do not help. It stays ~8× behind the CPU on this workload.

## What it takes to be fast in HIP

1. **On an AMD datacenter GPU (MI250X/MI300, CDNA)** the current rocMolKit —
   **with no changes** — should be competitive with nvMolKit on H200, because
   FP64 is native and fast there. The HIP port is already datacenter-ready.
2. **On consumer RDNA** the only lever is **FP32**: convert the minimization
   subsystem (`bfgsMinimizeKernel` + the force-field aggregators
   `molEnergyETK/molGradETK/molEnergyDG/molGradDG` + helpers + the inverse
   Hessian buffer) from `double` to `float`. Theoretical ceiling ~16× on the
   FP-bound kernels (25.6 vs 1.6 TFLOPS). High effort, real correctness risk —
   validate bond lengths / energies against RDKit (`test_etkdg_minimize`).
3. A batch-size autotuner is a minor lever here: the GPU saturates on compute,
   not on configuration.

## Robustness notes (this branch)

- Multi-thread hang at N≥2000 was a deadlock in the shared async mem pool —
  fixed by serializing pool calls (`utils/device_vector.h`). N≥4000 still hangs
  (residual contention; candidate: per-thread non-default mem pools).
- The GPU coordinate generator (`ETKDGCoordGenStage`) memory-faults at N≥100;
  the host generator is the safe default (opt in with `ROCMOLKIT_GPU_COORDGEN=1`).
- `fast-math` now matches nvMolKit's `--use_fast_math` (`-ffast-math
  -fno-finite-math-only`, so NaN/Inf checks still work).

Sources: Tom's Hardware / NanoReview RX 9060 XT specs (25.6 TFLOPS FP32);
`upstream/CHANGELOG.md`; `upstream/docs/autotune.rst`.
