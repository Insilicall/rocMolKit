# rocMolKit

[![ci](https://github.com/Insilicall/rocMolKit/actions/workflows/ci.yml/badge.svg)](https://github.com/Insilicall/rocMolKit/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

GPU-accelerated RDKit operations on AMD GPUs via HIP/ROCm.

Port of [nvMolKit](https://github.com/NVIDIA-Digital-Bio/nvMolKit) (NVIDIA CUDA, Apache 2.0). Same API surface, AMD backend.

> Status: **alpha** — ETKDG + MMFF94 functional via `rocmolkit.safe` ✅
> - Stack: ROCm 7.2.3 + Clang 22 + RDKit 2024.09.6 + boost 1.83
> - Validated on **AMD Radeon RX 9060 XT (Navi 44, RDNA4 / gfx1200)**
> - ETKDG bond lengths: GPU within 0.02 Å of RDKit CPU (within ETKDG noise)
> - MMFF94 converges to physically sensible minima
> - 6/6 Python bindings load: embedMolecules, mmffOptimization, uffOptimization,
>   batchedForcefield, conformerRmsd, arrayHelpers
> - **Known bug** (open): direct `_embedMolecules.EmbedMolecules` SIGSEGVs
>   non-deterministically on gfx1200 (~65% per-call success). Use
>   `rocmolkit.safe.embed_molecule(s)` — subprocess+retry wrapper that
>   measures 100% success and adds ~600 ms/mol overhead. See
>   [ISSUES.md](ISSUES.md).
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

Recommended path (deterministic via subprocess+retry):

```python
from rdkit import Chem
from rdkit.Chem import AddHs
from rocmolkit.safe import embed_molecule, mmff_optimize_molecule

m = AddHs(Chem.MolFromSmiles("CCO"))
embed_molecule(m, seed=42)             # GPU ETKDG, retried on segfault
energies = mmff_optimize_molecule(m)   # GPU MMFF94 on the embedded conformer
print(m.GetConformer(0).GetAtomPosition(0))
print("energy:", energies[0])
```

Direct binding (faster batches, but segfault risk on gfx1200 — see ISSUES.md):

```python
from rdkit import Chem
from rdkit.Chem import AddHs
from rdkit.Chem.rdDistGeom import ETKDGv3
from rocmolkit._embedMolecules import EmbedMolecules

mols = [AddHs(Chem.MolFromSmiles(s)) for s in ["CCO", "c1ccccc1", "CC(=O)O"]]  # batch <= 3
params = ETKDGv3(); params.useRandomCoords = True
EmbedMolecules(mols, params, 1)
```

## Performance (preliminary, blocked on the open segfault)

Measured on AMD Ryzen 5 7600 (12 threads) + AMD Radeon RX 9060 XT
(gfx1200, 32 CUs, RDNA4) with ROCm 7.2.3, against
`tests/data/druglike_100.smi` — 50 small drug-like molecules
(median ~15 atoms with hydrogens). Run via
`tools/benchmark.py tests/data/druglike_100.smi --n 50`.

### ETKDG conformer generation

| Pipeline | Wall time | Per-molecule | Success |
|---|---|---|---|
| RDKit CPU (sequential, single thread) | 0.21 s | 4.2 ms/mol (237 mol/s) | 50/50 |
| rocMolKit GPU `safe.embed_molecule` (subprocess+retry) | 235 s | 4798 ms/mol (0.2 mol/s) | 49/50 |
| rocMolKit GPU direct `EmbedMolecules`, batch≥4 | — | — | crashes (open bug) |

### MMFF94 force-field optimisation (ETKDG-embedded mols)

| Pipeline | Wall time | Per-molecule | Success |
|---|---|---|---|
| RDKit CPU ETKDG + MMFF, sequential | 0.38 s | 7.5 ms/mol (133 mol/s) | 50/50 |
| rocMolKit GPU `MMFFOptimizeMoleculesConfs`, batch=30 | 1.01 s | 20.2 ms/mol (49 mol/s) | 50/50 |

### GPU is working

`/sys/class/drm/card1/device/gpu_busy_percent` sampled every 2 s
during the run: **mean 74 %, peak 100 %, 90 of 121 samples above 50 %**.
Confirms kernels are launching and the GPU is doing real work.

### Honest take

For small drug-like molecules **RDKit CPU is hard to beat today**.
RDKit's ETKDG + MMFF94 are highly tuned in single-molecule paths and
on a 12-thread Ryzen the per-molecule overhead is below 10 ms. The
GPU advantage for batched force-field workloads only starts to show at
much larger molecule sizes (proteins, peptides) and batch counts, both
of which we can't run today because of the open ROCm 7.2.3 + gfx1200
state-leak segfault — see [ISSUES.md](ISSUES.md). Once that's resolved
(via a ROCm 7.3+ rocgdb session or a gfx1100 host where the bug
reproduces under a working debugger), benchmarks will be re-run on the
mlxmolkit-style N×k matrix where GPU tools traditionally win.

The two messages worth sending now:

1. **Functional**: ETKDG and MMFF94 produce the right answer (bond
   lengths within 0.02 Å of RDKit, MMFF energies match qualitatively).
2. **GPU-active**: kernels launch, the device runs at >70 % busy on
   batch workloads, the BFGS minimiser converges to physically
   sensible minima.

Performance optimisation comes after the segfault root cause is fixed.

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
