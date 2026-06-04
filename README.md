# rocMolKit

[![ci](https://github.com/Insilicall/rocMolKit/actions/workflows/ci.yml/badge.svg)](https://github.com/Insilicall/rocMolKit/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

**GPU-accelerated conformer generation and force-field optimization for RDKit, on AMD GPUs** — via HIP/ROCm.

HIP/ROCm port of [nvMolKit](https://github.com/NVIDIA-Digital-Bio/nvMolKit) (NVIDIA CUDA, Apache 2.0) — same API surface, AMD backend. On a **consumer** Radeon RX 9060 XT it runs ETKDG conformer generation and MMFF94 optimization **faster than the published Apple-Silicon sibling port ([mlxmolkit](https://github.com/guillaume-osmo/mlxmolkit))** and an order of magnitude faster than multi-threaded RDKit on CPU.

> **Status: beta.** All core modules are ported to HIP and validated against
> RDKit on AMD RDNA4 / gfx1200: **ETKDG** generation and **MMFF94** optimization
> (see Performance), plus **UFF**, **batched forcefield**, **conformer RMSD**,
> **Morgan fingerprints** (bit-exact), **Tanimoto/Cosine similarity** (bit-exact),
> **Butina clustering** (matches RDKit; high-cutoff differences are valid
> parallel tie-breaks), **substructure search** (840/840 vs RDKit) and **TFD**
> (float32 precision). Per-feature porting notes and validators in
> [PLAN.md](PLAN.md); see also [ISSUES.md](ISSUES.md), [CHANGELOG.md](CHANGELOG.md).

## Performance

Conformers per second (higher is better), measured on an **AMD Radeon RX 9060 XT** (Navi 44, gfx1200, RDNA4, 32 CUs) + Ryzen 5 7600, ROCm 7.2.3, dataset `tests/data/druglike_100.smi`.

| Workload | **rocMolKit**<br>(RX 9060 XT) | RDKit CPU<br>(12 threads, same host) | mlxmolkit<br>(Apple Metal, published) |
|---|---|---|---|
| **ETKDG generation** (DG + ETK) | **~6,200** | ~870 | ~2,000–2,600 |
| **MMFF94 optimization** | **~29,000–43,000** | ~1,200 | ~8,700–12,000 |

- **~7× faster than 12-thread RDKit** on ETKDG generation, **~25–35×** on MMFF94 optimization.
- **~2.5–3× faster than mlxmolkit** on both workloads, including the high-conformers-per-molecule regime.

> Hardware differs across ports — mlxmolkit numbers are from its published README on Apple Silicon (~14 TFLOPS FP32) vs the RX 9060 XT (~25.6 TFLOPS FP32). Read it as "each port on the accelerator it targets," not a same-machine shoot-out. Full methodology, caveats, and the root-cause write-up are in [docs/PERFORMANCE_HIP.md](docs/PERFORMANCE_HIP.md).

Reproduce:

```bash
python3 tools/etkdg_bench.py 1000 4     # ETKDG generation, N=1000 molecules × k=4 conformers
python3 tools/mmff_fresh.py  1000 4     # MMFF94 optimization, fresh conformers
```

## Quickstart

Run on a ROCm-capable machine with the prebuilt dev image:

```bash
docker run --rm -it \
    --device=/dev/kfd --device=/dev/dri \
    --group-add video --security-opt seccomp=unconfined \
    ghcr.io/insilicall/rocmolkit:devel \
    python3
```

```python
from rdkit import Chem
from rdkit.Chem import AddHs
from rdkit.Chem.rdDistGeom import ETKDGv3
from rocmolkit._embedMolecules import EmbedMolecules, BatchHardwareOptions
from rocmolkit._mmffOptimization import MMFFOptimizeMoleculesConfs

mols = [AddHs(Chem.MolFromSmiles(s)) for s in
        ("CCO", "c1ccccc1", "CC(=O)O", "CC(C)Cc1ccc(cc1)C(C)C(=O)O")]
params = ETKDGv3()
params.useRandomCoords = True

opts = BatchHardwareOptions()
opts.gpuIds = [0]                 # pin to the discrete GPU — see note below

EmbedMolecules(mols, params, 50, -1, opts)        # 50 conformers per molecule
MMFFOptimizeMoleculesConfs(mols, 200, [], opts)   # MMFF94, maxIters=200
```

> **Pin `gpuIds=[0]`** on Ryzen hosts. Such systems enumerate both an integrated
> GPU (gfx1036) and the discrete GPU; dispatching to the iGPU crashes. Pinning to
> the discrete device avoids it. (The earlier "non-deterministic SIGSEGV" was this
> deterministic iGPU dispatch — see [ISSUES.md](ISSUES.md) "ROOT CAUSE FOUND".)

## Hardware

ROCm 6.2+ on one of:

| GPU | gfx | Status |
|---|---|---|
| RX 9060 XT / RDNA4 | gfx1200 | primary (validated) |
| RX 7900 XTX/XT | gfx1100 | supported |
| MI210 / MI250 | gfx90a | supported |
| MI300 | gfx942 | supported |
| RX 6000 series | gfx1030 | use `HSA_OVERRIDE_GFX_VERSION=10.3.0` |

## Build

```bash
# Minimal production image (< 2 GB)
docker build -f docker/Dockerfile.slim -t rocmolkit:slim .

# Dev image with the ROCm SDK (hipcc, gdb, hipify)
docker build -f docker/Dockerfile.devel -t rocmolkit:devel .

# Local build
cmake -S . -B build -GNinja -DGPU_TARGETS=gfx1200
cmake --build build
```

## Testing

Two layers. The smoke/safe tests are pure Python and need no GPU; the feature
tests are marked `gpu` and validate the GPU result against RDKit on hardware.

```bash
# No GPU: import + package self-consistency (what public CI runs)
pytest tests/ -m "not gpu"

# On a ROCm machine: also run the GPU feature suite (fingerprints, similarity,
# Butina clustering, TFD — each compared bit/precision-exact against RDKit)
pytest tests/ --rocm
```

`tests/test_gpu_features.py` is the integration suite; each test `importorskip`s
its binding, so anything not built is skipped rather than failing. The same
checks also exist as standalone scripts under `tools/` (`fp_validate.py`,
`sim_validate.py`, `butina_validate.py`, `tfd_validate.py`) for ad-hoc runs.

Run them inside the dev image (GPU passthrough + writable `/tmp` for comgr's
JIT blit kernels):

```bash
docker run --rm \
    --device=/dev/kfd --device=/dev/dri \
    --group-add video --group-add render --security-opt seccomp=unconfined \
    -e HIP_VISIBLE_DEVICES=0 -v "$PWD":/work -w /work \
    ghcr.io/insilicall/rocmolkit:devel \
    python3 -m pytest tests/ --rocm -v
```

## Continuous integration

Public CI (`.github/workflows/ci.yml`) runs on PR open / reopen / ready — **not**
on every push to an open PR — plus tag pushes and manual dispatch. It builds
`rocmolkit_core`, the bindings, and the devel/slim images, and runs the non-GPU
tests. The GPU suite runs in the `rocm-runner` job, gated behind a **self-hosted
runner**:

1. Register a runner on a ROCm machine with the labels `self-hosted,rocm`
   (Settings → Actions → Runners → New self-hosted runner). It needs Docker with
   GPU passthrough (`--device=/dev/kfd --device=/dev/dri`).
2. Set the repo variable `ROCM_RUNNER_ONLINE=true` (Settings → Secrets and
   variables → Actions → Variables). The job stays skipped until then, so PRs
   never queue waiting on an absent runner. Flip it back to disable.

Once enabled, every gating CI run builds the devel image from source (so it
ships **all** current bindings) and runs `pytest tests/ --rocm` against RDKit on
the GPU.

### Publishing images

`.github/workflows/docker.yml` builds and pushes `ghcr.io/insilicall/rocmolkit:{devel,slim}`
on `v*` tag pushes. To refresh them with the latest bindings, tag a release
(`git tag vX.Y.Z && git push --tags`). To publish by hand:

```bash
docker build -f docker/Dockerfile.devel -t ghcr.io/insilicall/rocmolkit:devel .
echo "$GHCR_TOKEN" | docker login ghcr.io -u <user> --password-stdin
docker push ghcr.io/insilicall/rocmolkit:devel
```

## How it works

rocMolKit runs the full ETKDG pipeline (DistGeom 4D → ETK 3D → chirality/stereo
checks) and MMFF94 optimization on the GPU, with an in-kernel batched BFGS
minimizer (one warp per conformer, inverse Hessian in global memory). Conformers
are the unit of parallel work, so throughput scales with both the molecule count
(N) and conformers per molecule (k). See [docs/PERFORMANCE_HIP.md](docs/PERFORMANCE_HIP.md)
for the architecture and the optimization history.

## Roadmap

Core modules are ported and validated (ETKDG, MMFF94, UFF, batched forcefield,
conformer RMSD, fingerprints, similarity, Butina clustering, substructure, TFD).
Next: a self-hosted ROCm CI runner to gate the upstream test suites on real
hardware, and refreshed devel/slim images shipping the new bindings. See
[PLAN.md](PLAN.md).

## License

Apache 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE) for attribution to the
upstream nvMolKit.
