"""Correctness gate for the FP32 experiment.

Generates conformers with rocMolKit (GPU) and scores each generated geometry by
its MMFF94 energy via RDKit (no re-optimization — we measure the geometry the
GPU produced). Reports success rate + energy distribution. Run before (double
baseline) and after each FP32 step; if energy/success regress, the conversion
broke correctness.
"""
import sys
import numpy as np
from rdkit import Chem
from rdkit.Chem import AddHs, AllChem
from rdkit.Chem.rdDistGeom import ETKDGv3
from rocmolkit._embedMolecules import EmbedMolecules, BatchHardwareOptions

N = int(sys.argv[1]) if len(sys.argv) > 1 else 200
K = int(sys.argv[2]) if len(sys.argv) > 2 else 4
SEED = 42

smis = []
with open("/src/tests/data/druglike_100.smi") as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith("#"):
            smis.append(line.split()[0])
base = list(smis)
while len(smis) < N:
    smis.extend(base)
smis = smis[:N]

mols = []
for smi in smis:
    m = Chem.MolFromSmiles(smi)
    if m is None:
        continue
    mols.append(AddHs(m))

p = ETKDGv3()
p.randomSeed = SEED
p.useRandomCoords = True
opts = BatchHardwareOptions()
opts.gpuIds = [0]

EmbedMolecules(mols, p, K, -1, opts)

energies = []
n_conf = 0
n_target = len(mols) * K
for m in mols:
    n_conf += m.GetNumConformers()
    props = AllChem.MMFFGetMoleculeProperties(m)
    if props is None:
        continue
    for conf in m.GetConformers():
        ff = AllChem.MMFFGetMoleculeForceField(m, props, confId=conf.GetId())
        if ff is not None:
            energies.append(ff.CalcEnergy())

e = np.array(energies)
print(f"GATE N={N} k={K} seed={SEED}")
print(f"  conformers: {n_conf}/{n_target}  success={100*n_conf/max(1,n_target):.1f}%")
if len(e):
    print(f"  MMFF94 energy  median={np.median(e):.2f}  mean={np.mean(e):.2f}  "
          f"p90={np.percentile(e,90):.2f}  max={np.max(e):.2f}  (kcal/mol, lower=better)")
    print(f"  scored geometries: {len(e)}")
