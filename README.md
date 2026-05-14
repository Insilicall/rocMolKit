# rocMolKit

[![ci](https://github.com/Insilicall/rocMolKit/actions/workflows/ci.yml/badge.svg)](https://github.com/Insilicall/rocMolKit/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

GPU-accelerated RDKit operations on AMD GPUs via HIP/ROCm.

Port of [nvMolKit](https://github.com/NVIDIA-Digital-Bio/nvMolKit) (NVIDIA CUDA, Apache 2.0). Same API surface, AMD backend.

> Status: **alpha** — ETKDG + MMFF94 numerically validated on AMD GPU ✅
> - Stack: ROCm 7.2.3 + Clang 22 + RDKit 2024.09.6 + boost 1.83
> - Validated on **AMD Radeon RX 9060 XT (Navi 44, RDNA4 / gfx1200)**
> - ETKDG ethanol bonds: GPU bit-exact vs RDKit CPU (C-C 1.506 Å; C-O within 0.006 Å)
> - MMFF94 converges to global minimum on ethanol in 267 ms
> - 6/6 Python bindings load: embedMolecules, mmffOptimization, uffOptimization,
>   batchedForcefield, conformerRmsd, arrayHelpers
> - **Known bug:** non-deterministic segfault on mid-size molecules (aspirin,
>   pyridine). Small molecules (CCO, benzene, hexane, p-xylene) reliable.
>   Investigation in progress — likely host-side use-after-free.
> - Pending: fingerprints, clustering, substructure, tfd (Phases 4-7 kernels)
>
> See [PLAN.md](PLAN.md), [ISSUES.md](ISSUES.md) and [CHANGELOG.md](CHANGELOG.md).

## Quickstart (quando estiver pronto)

```bash
pip install rocmolkit
```

```python
from rdkit import Chem
from rdkit.Chem import AllChem
from rocmolkit.embedMolecules import EmbedMolecules

mols = [Chem.AddHs(Chem.MolFromSmiles(s)) for s in ["CCO", "c1ccccc1", "CC(=O)O"]]
params = AllChem.ETKDGv3()
params.useRandomCoords = True
EmbedMolecules(molecules=mols, params=params, confsPerMolecule=10)
```

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
