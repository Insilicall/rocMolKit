# rocMolKit performance on consumer RDNA4

Measured on an **AMD Radeon RX 9060 XT** (Navi 44, gfx1200, RDNA4, 32 CUs),
ROCm 7.2.3, dataset `tests/data/druglike_100.smi`.

**TL;DR (current):** rocMolKit now **beats the sibling GPU ports on both
workloads**. The long-standing slowness was never FP64 or pipeline overhead —
it was a one-line correctness bug in the in-kernel line search (it ran 1000
inner iterations per BFGS step instead of ~2-3; see *Root cause* below). Fixing
it lifted ETKDG generation ~27x and high-conformer MMFF ~100x.

## Cross-port comparison

Throughput in **conformers/second** (higher is better). rocMolKit numbers are
measured here; mlxmolkit numbers are from its published README. **Hardware
differs** — this is not a same-machine benchmark, so read it as "each port on
the accelerator it targets", not a hardware shoot-out.

Throughput scales strongly with molecule size (MMFF is O(n²) in the non-bonded
terms), so numbers are reported **by molecule size**. All rocMolKit numbers are
at **100% success** and RDKit-validated (generated conformers fall in the same
MMFF energy basins as RDKit: median ΔE = 0.00 kcal/mol, 52/60 drug-like within
1 kcal/mol of RDKit's best conformer).

| Workload | **rocMolKit** small (~12 atoms) | **rocMolKit** drug-like (~30 atoms) | RDKit CPU (12 threads, same host) |
|---|---|---|---|
| **ETKDG generation** (DG + ETK) | **~29,000** | **~4,900** | ~865 |
| **MMFF94 optimize** (fresh conformers) | **~31,500** | **~4,900** | ~907 |

vs RDKit on drug-like (same machine): **~12–13x** single-core, **~5.4–5.7x**
12-thread.

**vs the Apple-Silicon sibling ports** (their published numbers; hardware differs —
Apple M3 Max ~14 TFLOPS vs RX 9060 XT ~25.6 TFLOPS FP32). Both ports validate
against RDKit, so "RDKit-quality conformers/s" is the shared unit:

| same molecule size | rocMolKit | mlxmolkit | rocMolKit lead |
|---|---|---|---|
| ETKDG gen, drug-like, k=10 | ~4,900 | 2,625 (guillaume-osmo) | **~1.9x** |
| full pipeline (+MMFF), k=10 | ~2,450 | 1,473 (guillaume-osmo) | **~1.7x** |
| MMFF optimize, ~12–14 atoms | ~31,500 | ~12,000 (shivampatel10) | **~2.6x** |

- nvMolKit (the original CUDA library) publishes no comparable throughput
  table; it targets datacenter GPUs (H100/A100), so it is omitted rather than
  compared across very different hardware. Note rocMolKit's per-molecule
  minimizer runs in **FP32**, ahead of upstream nvMolKit's FP64 path.

**Caveats (read these):**
- mlxmolkit ran on Apple Silicon (~14 TFLOPS FP32) vs the RX 9060 XT
  (~25.6 TFLOPS FP32) — different hardware. The comparison is port-vs-port on
  each one's target accelerator.
- The two published mlxmolkit forks report on different molecule sets:
  guillaume-osmo benchmarks the full DG→ETK→MMFF pipeline on drug-like
  molecules; shivampatel10 benchmarks MMFF optimization on the ~14-atom MMFF94
  validation set. Compare each at its matched size, as above.
- Measure MMFF on **fresh** conformers optimized **once** (`tools/mmff_fresh.py`),
  not a "warm median" that re-optimizes already-converged geometries — an
  unrealistic workload whose timings are misleading at high k.
- rocMolKit ETKDG now reports **100% generation success** (the earlier ~88–96%
  was a since-fixed bug where a broken hipcub primitive silently dropped the
  largest molecules' batch); mlxmolkit reports ~99.7% convergence — these
  metrics are defined differently and are not a
  like-for-like quality comparison.

## Root cause of the historical slowness — the line-search bug

The per-molecule BFGS kernel (`bfgs_minimize_permol_kernels.hip.cpp`, shared by
the MMFF, ETK and DG force fields) ran its inner line search to the full
`MAX_LINESEARCH_ITERS` (1000) on **every** BFGS step. `lineSearchPostEnergy`
computes the converged flag only on thread 0, but the caller assigned its return
value to the shared `lineSearchConverged` from all 32 lanes, so the other lanes'
`false` raced thread 0's `true` away and the flag never latched. Latching it from
thread 0 only (commit on this branch) dropped the line search to ~2-3 iterations
per step:

This was an early efficiency fix. The current, authoritative throughput (at
**100% success**, after the separate hipcub/BATCHED correctness fix) is in the
[cross-port table](#cross-port-comparison) above: ETKDG ~4,900 conf/s (drug-like)
/ ~29,000 (small); MMFF ~4,900 (drug-like) / ~31,500 (small). MMFF94 energies
match RDKit's basins (median ΔE = 0.00 kcal/mol).
The separate `hipFreeAsync` stream-sync fix (a HIP-7.0 use-after-free) removed an
intermittent GPU memory fault at N≥~900; the two fixes are independent.

## (Historical) Result of an earlier optimization pass — parity with the CPU

> Superseded by the line-search fix above. Kept for the record; the numbers
> below predate it and are no longer current.

This branch (BLOCK_SIZE=32 + FP32 minimization + conf_to_mol) takes the port
from ~20x behind mlxmolkit to **parity with the 12-thread CPU**:

| config | conf/s (median of 6 runs, N=1000 k=4) |
|--------|---------------------------------------|
| original port | ~110 |
| **this branch** | **~865 (~7.9x)** |
| RDKit CPU, 12 threads | ~870 |
| mlxmolkit (Apple Metal) | ~2,300 |

**Measure with ≥6 runs and take the median** — single-shot is misleading here.
The throughput distribution is **bimodal**: FP32 makes the parallel reductions
non-deterministic, so conformer convergence varies run to run; ~1 in 7 runs gets
extra failures (3500/4000 vs 4000/4000) and the ETKDG retries cost ~2x wall time
(9.6 s vs 4.6 s). Reported single-shot numbers like "373" were slow outliers.

## MMFF94 optimize-only — already ahead of mlxmolkit

The numbers above are the **full ETKDG embed** pipeline. The standalone
**MMFF94 optimization** path (`MMFFOptimizeMoleculesConfs`, conformers already
generated) is a different, much faster workload — and the apples-to-apples
comparison against mlxmolkit's MMFF benchmark:

| config (N=1000 k=4, MMFF optimize only) | conf/s |
|-----------------------------------------|--------|
| cold run (first call, includes blit-kernel JIT) | ~14,000 |
| warm, fresh process per run (median) | ~32,000 |
| **warm, in-process median (`tools/mmff_bench.py`)** | **~52,000** |
| mlxmolkit (Apple Metal), reported peak | ~12,000 |

So on this workload rocMolKit is **~3–4x faster than mlxmolkit's peak**, not
behind it. The warm number depends on warmup: a fresh Python process per run
settles around ~32k conf/s, while a long-running process (the realistic service
case) reaches ~52k once the mempool and kernel modules are hot. Optimized
geometries verified correct (MMFF94 energy recomputed via RDKit matches:
26.41 kcal/mol median, stable across runs). At smaller N=250 k=4 throughput is
~10,000 conf/s (host/launch overhead is a larger fraction). Reproduce with
`tools/mmff_bench.py` (writable `/tmp` required — see hazards below).

### Two measurement hazards that produced false numbers

1. **Writable `/tmp` is mandatory.** The HIP runtime JIT-compiles its internal
   blit (copy) kernels via comgr, which writes scratch to `$TMPDIR` (default
   `/tmp`). If the container's `/tmp` is mounted **read-only**, the compile fails
   with `Couldn't create blit kernels! / Could not create BlitManager!` and the
   process **segfaults** — looking exactly like a GPU/driver crash. Mount bench
   scripts elsewhere (`-v ...:/scripts:ro`) and leave `/tmp` writable. A
   standalone AOT-compiled HIP binary can mask this (small copies may use SDMA
   instead of a JIT blit kernel), so it only bites the Python/runtime path.
2. **Pin `gpuIds=[0]`.** This box has both a dGPU (gfx1200) and the Ryzen iGPU
   (gfx1036). gfx1036 is outside `gfx12-generic` and has no code object;
   enumerating it crashes. Pass `BatchHardwareOptions().gpuIds = [0]`.

The earlier "MMFF cache +80% (1259→2263)" claim (commit 7acf15d) did **not**
reproduce under correct measurement: with a writable `/tmp` and warm cache,
baseline and the per-thread-cache build are statistically identical (~32k conf/s
at N=1000 k=4). The persistent per-thread FF-contribs cache is correct and
harmless but neutral for single-call workloads — each `ROMol*` is processed once,
so there is no cross-batch cache reuse to exploit. The original +80% was an
artifact of a read-only `/tmp` (intermittent blit-JIT stalls) plus cold-cache
first runs.

### Rejected in this pass (measured, did not help / broke correctness)
- Fuse check stages; bulk-insert of index arrays — within measurement noise.
- Reuse molecular-system across conformers — impossible: kernels index the
  global position buffer by per-conformer atom offset, so term indices differ.
- Carry per-molecule convergence across relaunches — **broke correctness**
  (75% → 50% success; the energy gate caught it). Reverted.

## Next project: single-kernel rewrite (the path beyond CPU parity → mlxmolkit)

Incremental tweaks are exhausted at CPU parity. Going from ~865 to ~2,300
needs the mlxmolkit architecture — a **fused per-conformer kernel** (DistGeom +
ETK + checks) with **device-side conf_to_mol** (share constraint topology across
a molecule's conformers on-device) and zero host round-trips. This is an
architectural rewrite of the minimization/stage path, not an incremental change
(5 agent attempts confirmed reuse/fusion of the existing stages is high-blast-
radius or noise). Do it as a dedicated effort, and **first get real kernel
profiling working** — `rocprof`/`rocprofv2` fail on this image (missing
`libhsa-amd-aqlprofile64.so.1`); install `rocprofiler-sdk` / `rocprofv3` (ROCm
7.x) so the compute can be targeted with VALU/occupancy/memory counters instead
of trial-and-error. (`Dockerfile.devel` now installs `rocprofiler-sdk` +
`hsa-amd-aqlprofile` + `rocprofiler-register`; `rocprofv3 --kernel-trace` works.)

**First kernel trace (rocprofv3, N=200 k=4) — the target is unambiguous:**

| % GPU time | kernel | launches |
|-----------|--------|----------|
| **86.4%** | **`bfgsMinimizeKernel`** | 6 |
| 6.7% | `fillBufferAligned` (memsets) | 31,416 |
| 3.2% | `copyBuffer` (memcpies) | 22,772 |
| ~2% | DistViolation grad/energy | ~2,200 |

So **86% of GPU compute is the single per-conformer BFGS kernel** — that is the
one to optimize (occupancy, shared-mem pressure, the in-kernel line search /
Hessian update). Next: profile `bfgsMinimizeKernel` with rocprofv3 counters
(VALU utilization, occupancy, LDS) to know whether it is ALU-bound,
occupancy-limited, or memory-bound, then redesign accordingly. The ~10% of tiny
memsets/copies (54k launches) is the secondary target.

### Also worth fixing (consistency, not raw speed)
The FP32 non-determinism causes the slow-retry outlier (~1 in 7). Making the
parallel reductions deterministic (or stabilizing convergence) would remove the
tail and make throughput consistently ~865 rather than averaging ~790.

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

## Reality check: mlxmolkit does ~20x better on weaker hardware

**mlxmolkit** (github.com/guillaume-osmo/mlxmolkit) is *another port of the same
nvMolKit*, to Apple Metal/MLX. Its published numbers:

| Conformers | Time | Throughput |
|-----------|------|-----------|
| 1,000 | 0.43s | **2,342/s** |
| 10,000 | 4.82s | **2,075/s** |

That is **~2,300 conf/s on an M3 Max (~14 TFLOPS FP32)** vs our **110 conf/s on a
RX 9060 XT (25.6 TFLOPS FP32)** — weaker hardware, ~20x faster. So the consumer
GPU is NOT structurally limited; **our port is ~20x inefficient.** A 20x gap is
not FP64 (that is 2-4x) — it is pipeline overhead.

What mlxmolkit does differently:
- **Fused kernel, zero CPU round-trips.** Our pipeline has 12 separate stages
  with a sync between each, a serial CPU coordgen, and chirality/stereo checks
  that round-trip. mlxmolkit fuses DG/ETK/MMFF into in-kernel threadgroup-per-
  conformer passes.
- **FP32 native** (MLX) vs our literal FP64 port.
- **N×k with shared constraints** (`conf_to_mol`, 50% memory savings).
- It uses **BFGS with the dense Hessian** (not L-BFGS) for drug-like sizes and
  states it is *faster* than L-BFGS up to 74 atoms — so the Hessian is fine;
  the overhead is elsewhere.

**Caveat on our own profiling:** the "minimization = 98%" measurement was taken
with `ROCMOLKIT_DEBUG_STAGES` (a sync after every stage), which hides the
launch/sync overhead of the normal async path. Re-profile WITHOUT the debug sync
(hipEvents or rocprofv2) before attributing the gap.

**The real path = rewrite the pipeline fused + FP32, mlxmolkit-style** (move
coordgen onto the GPU, collapse the 12 stages, cut host round-trips), not the
incremental FP64→FP32 tweak tested below.

### Measured: the GPU is idle 56% of the time

Sampling `gpu_busy_percent` during an N=2000 run (216 samples): **median 2%
busy**, mean 45%, **56% of samples below 10% busy**, 44% above 50%. The
distribution is bimodal — 100% (computing a batch) or 2% (idle, waiting on the
CPU). So the 20x gap decomposes as roughly:

- **~2x — host idle gaps** (per-batch setup of 12 stages, chirality/stereo checks
  that round-trip D2H, launch latency between many small kernels). Keeping the
  GPU busy ~doubles throughput.
- **~10x — inefficient compute when it does run**: FP64 (2-4x on RDNA4) +
  occupancy (our 128 threads/molecule vs mlxmolkit's 32 → 4x fewer concurrent
  conformers per CU) + kernel design.

Priority order to close it: (1) cut host round-trips / keep the GPU fed
(GPU coordgen, fewer stages, pipeline batches); (2) drop minimization
threadgroup to ~32 threads/conformer; (3) FP32 across the minimization path.

**DONE — (2) BLOCK_SIZE 128 -> 32: +74%.** Measured N=1000 k=4 on RX 9060 XT:
110 -> 191 conf/s, correctness unchanged (75.0% success, MMFF94 energy median
26.07 -> 26.21, max 103, no outliers). 64 -> 186 c/s; 16 unsafe (breaks tile32).
One-line change (commit e96bdea). Remaining levers: (1) idle gaps and (3) FP32.
This corrects the earlier "98% minimization" reading, which was taken with the
debug per-stage sync and hid the idle gaps.

## FP32 experiment result (measured) — Hessian alone is not the silver bullet

We tested the highest-value FP32 step in isolation: converting the **inverse
Hessian** (the largest `double` structure, O(n^2) global traffic per BFGS
iteration) to `float`, gated by `tools/fp32_gate.py`.

| metric | double baseline | Hessian float |
|--------|-----------------|---------------|
| success (N=200, k=4) | 75.0% | 75.0% |
| MMFF94 energy median | 26.07 | 25.79 |
| MMFF94 energy **max** | 103 | **485** |
| throughput N=1000 | 110 conf/s | **114 conf/s (+3.6%)** |

**~4% speedup and convergence outliers** (max energy 103 -> 485: some conformers
converge poorly in `float`). Reverted.

Why so small: nvMolKit **already** computes the gradients in `float` (the
ALU-heavy part — see `distViolationGrad`). What is left in `double` is the
Hessian (tiny for drug-like molecules, so it fits in cache and the bandwidth
win is small) and the energy / line search (numerically sensitive). The
full FP32 refactor would likely yield ~10-20%, **not** close the 8x gap, and
would degrade correctness. The real ceiling on consumer RDNA is structural
(per-system minimization saturates ~110 conf/s), not precision. A datacenter
CDNA GPU remains the clean answer; for consumer cards, the honest conclusion is
that ETKDG of small molecules belongs on the CPU.

## FP32 conversion plan (TDD — if targeting larger molecules / CDNA tuning)

The minimization subsystem shares one `double` state (positions, gradient,
inverse Hessian) between the force field and BFGS, so the conversion is
all-or-nothing along the minimization path and must be guarded by a correctness
gate. Suggested order, recompiling + re-running the gate after each step:

0. **Gate.** Build with `-DROCMOLKIT_BUILD_TESTS=ON`, run `test_etkdg_minimize`
   (energy decreases; energy/atom below threshold) and record conf success rate
   at N=400/1000. Bond lengths within 0.02 A of RDKit (README claim) is the
   acceptance bar.
1. **Typedef switch.** Add `using MinReal = float;` (vs `double`) in the
   minimization headers so the precision is one compile-time flip for A/B.
2. **Force-field aggregators** (`dist_geom_kernels_device.hip.h`):
   `molEnergyETK/molGradETK/molEnergyDG/molGradDG` + their term sub-functions —
   read `double` positions, compute and accumulate in `MinReal`. This mirrors
   what `distViolationGrad` already does (float interior, double boundary).
3. **`bfgsMinimizeKernel` working set** (`bfgs_minimize_permol_kernels.hip.cpp`):
   `localPos/localGrad/localDir/scratchPos/dGrad` + line-search scalars ->
   `MinReal`. The shared-mem copy `globalPos -> localPos` is the natural
   double<->float boundary.
4. **Inverse Hessian** (host alloc in `BfgsBatchMinimizer` + `updateInverseHessian`
   / `setDirection`) -> `MinReal`. Biggest single win: O(n^2) global traffic and
   FLOPs per iteration, halved + 16x faster ALU.
5. **Reductions**: `hipcub::BlockReduce<double>` -> `<MinReal>`.

Expected sweet spot is **mixed precision**: float forces + float Hessian, but
keep the energy sum (and possibly the Hessian update) in double if convergence
or bond-length accuracy regresses. Measure ETK 3D + DistGeom stage times with
`ROCMOLKIT_DEBUG_STAGES` against the double baseline.

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
