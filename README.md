# rocMolKit

GPU-accelerated RDKit operations on AMD GPUs via HIP/ROCm.

Port of [nvMolKit](https://github.com/NVIDIA-Digital-Bio/nvMolKit) (NVIDIA CUDA, Apache 2.0). Same API surface, AMD backend.

> Status: **pre-alpha**, scaffold only. See [PLAN.md](PLAN.md).

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
