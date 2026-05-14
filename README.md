# rocMolKit

[![ci](https://github.com/Insilicall/rocMolKit/actions/workflows/ci.yml/badge.svg)](https://github.com/Insilicall/rocMolKit/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

GPU-accelerated RDKit operations on AMD GPUs via HIP/ROCm.

Port of [nvMolKit](https://github.com/NVIDIA-Digital-Bio/nvMolKit) (NVIDIA CUDA, Apache 2.0). Same API surface, AMD backend.

> Status: **alpha** — ETKDG + MMFF94 functional ✅
> - Stack: ROCm 7.2.3 + Clang 22 + RDKit 2024.09.6 + boost 1.83
> - Validated on **AMD Radeon RX 9060 XT (Navi 44, RDNA4 / gfx1200)**
> - ETKDG bond lengths: GPU within 0.02 Å of RDKit CPU (within ETKDG noise)
> - MMFF94 converges to physically sensible minima
> - 6/6 Python bindings load: embedMolecules, mmffOptimization, uffOptimization,
>   batchedForcefield, conformerRmsd, arrayHelpers
> - **Resolved**: the "non-deterministic SIGSEGV" was deterministic
>   dispatch to the iGPU (gfx1036) on Ryzen hosts that enumerate both an
>   integrated and a discrete GPU. Pin to the discrete GPU with
>   `BatchHardwareOptions(gpuIds=[0])`. The `rocmolkit.safe` subprocess
>   wrapper is still available as a defensive fallback but is no longer
>   the recommended path. See [ISSUES.md](ISSUES.md) "ROOT CAUSE FOUND".
> - Pending: fingerprints, clustering, substructure, tfd (Phases 4-7 kernels)
>
> See [PLAN.md](PLAN.md), [ISSUES.md](ISSUES.md) and [CHANGELOG.md](CHANGELOG.md).

## Quickstart

Install dev image + run on a ROCm-capable machine:

```bash
docker run --rm -it \
    --device=/dev/kfd --device=/dev/dri \
    --group-add video --security-opt seccomp=unconfined \
    ghcr.io/insilicall/rocmolkit:devel \
    python3
```

Recommended path (direct call, GPU pinned to the discrete device):

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
opts.gpuIds = [0]                 # pin to the discrete GPU (see ISSUES.md)

EmbedMolecules(mols, params, 50, -1, opts)        # 50 conformers per mol
MMFFOptimizeMoleculesConfs(mols, maxIters=200)
```

Defensive fallback (`rocmolkit.safe`) — same call wrapped in a fresh
subprocess with retries. Adds ~600 ms/mol of fork+import overhead, so use
it only when you cannot tolerate the (now rare) failures from upstream
ROCm bugs:

```python
from rdkit import Chem
from rdkit.Chem import AddHs
from rocmolkit.safe import embed_molecule, mmff_optimize_molecule

m = AddHs(Chem.MolFromSmiles("CCO"))
embed_molecule(m, seed=42, gpu_id=0)
energies = mmff_optimize_molecule(m)
```

## Performance

Measured on AMD Ryzen 5 7600 (12 threads) + AMD Radeon RX 9060 XT
(gfx1200, 32 CUs, RDNA4) with ROCm 7.2.3 against
`tests/data/druglike_100.smi`. The benchmark sweeps N (molecule count)
× k (conformers per molecule) so the GPU's two parallelism axes both
get exercised.

Reproduce on a clean GPU:

```bash
bash tools/validate_gpu.sh
```

The script aborts if it detects the leaked-VRAM state (HIP runtime
holds memory after a Ctrl+C; recover with `sudo modprobe -r amdgpu &&
sudo modprobe amdgpu` or a reboot — see [ISSUES.md](ISSUES.md)).

### Methodology

- All paths use the same ETKDGv3 parameters and the same SMILES set.
- RDKit CPU baseline uses `EmbedMultipleConfs(numThreads=0)` — RDKit's
  multi-threaded conformer generator. This is the honest CPU baseline
  on a 12-thread host. A `numThreads=1` row is reported alongside for
  the apples-to-apples per-call comparison.
- GPU calls go through the direct binding with
  `BatchHardwareOptions(gpuIds=[0])` — no subprocess overhead.
- Throughput is reported as conformers per second so single-conformer
  and multi-conformer runs compare on the same axis.

Expected shape (filled in by `tools/validate_gpu.sh` on a clean GPU):

| Pipeline | Shape | Wall time | Per-conformer | Success |
|---|---|---|---|---|
| RDKit CPU (1 thread) | N=500, k=50 | _to be filled_ | _to be filled_ | _to be filled_ |
| RDKit CPU (12 threads) | N=500, k=50 | _to be filled_ | _to be filled_ | _to be filled_ |
| rocMolKit GPU (RX 9060 XT) | N=500, k=50 | _to be filled_ | _to be filled_ | _to be filled_ |

### Why k matters

ETKDG's per-molecule work is small for drug-like molecules (~4 ms on
RDKit CPU) so a single-conformer run on a small batch leaves most of
the GPU idle. The GPU advantage scales with k because every conformer
of every molecule becomes an independent unit of parallel work — the
upstream nvMolKit benchmark on H100 shows the gap widening from
roughly even at k=1 to >10× by k=50 on the same drug-like sets. RDNA4
has lower peak FLOPs than H100 so the crossover sits at higher k or N
on this hardware; the sweep above pins down where exactly.

## Hardware

ROCm 6.2+ em uma das GPUs:

| GPU | gfx | Status |
|---|---|---|
| RX 7900 XTX/XT | gfx1100 | oficial |
| MI210/MI250 | gfx90a | oficial |
| MI300 | gfx942 | oficial |
| RX 6000 series | gfx1030 | use `HSA_OVERRIDE_GFX_VERSION=10.3.0` |

## Build

```bash
# Imagem mínima de produção (< 2 GB)
docker build -f docker/Dockerfile.slim -t rocmolkit:slim .

# Imagem dev com SDK
docker build -f docker/Dockerfile.devel -t rocmolkit:devel .

# Build local
cmake -S . -B build -GNinja -DGPU_TARGETS=gfx1100
cmake --build build
```

## Roadmap

Ver [PLAN.md](PLAN.md). Ordem: ETKDG → MMFF94 → fingerprints → similaridade → Butina → resto.

## Licença

Apache 2.0. Veja [LICENSE](LICENSE) e [NOTICE](NOTICE) para atribuição ao nvMolKit upstream.
