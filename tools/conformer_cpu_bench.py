"""RDKit CPU baseline for the conformer table: ETKDG + MMFF94 throughput at
1 core and 12 threads, same dataset/scale as the GPU numbers. Fills the
"RDKit (1 core)" / "RDKit (12 threads)" columns; the GPU and mlx numbers come
from tools/etkdg_bench.py / tools/mmff_fresh.py and the mlxmolkit README.

Run anywhere with RDKit (no GPU needed).
"""

import sys
import time
from concurrent.futures import ThreadPoolExecutor
from rdkit import Chem
from rdkit.Chem import AllChem, AddHs
from rdkit.Chem.rdDistGeom import ETKDGv3, EmbedMultipleConfs

N = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
K = int(sys.argv[2]) if len(sys.argv) > 2 else 4
NTHREADS = 12

smis = [l.split()[0] for l in open("tests/data/druglike_100.smi") if l.strip() and not l.startswith("#")]
base = [s for s in smis if Chem.MolFromSmiles(s)]
smis_n = [base[i % len(base)] for i in range(N)]
TOTAL = N * K


def fresh_mols():
    return [AddHs(Chem.MolFromSmiles(s)) for s in smis_n]


def etkdg(num_threads):
    p = ETKDGv3()
    p.randomSeed = 42
    p.numThreads = num_threads
    mols = fresh_mols()
    t0 = time.perf_counter()
    for m in mols:
        EmbedMultipleConfs(m, numConfs=K, params=p)
    dt = time.perf_counter() - t0
    return TOTAL / dt, mols


def mmff_1core(mols):
    t0 = time.perf_counter()
    for m in mols:
        AllChem.MMFFOptimizeMoleculeConfs(m, maxIters=200)
    return TOTAL / (time.perf_counter() - t0)


def mmff_threads(mols):
    t0 = time.perf_counter()
    with ThreadPoolExecutor(NTHREADS) as ex:
        list(ex.map(lambda m: AllChem.MMFFOptimizeMoleculeConfs(m, maxIters=200), mols))
    return TOTAL / (time.perf_counter() - t0)


print(f"dataset=druglike_100 tiled to N={N}, k={K}  (total {TOTAL} conformers)")

# ETKDG generation
etk1, _ = etkdg(1)
etk12, mols = etkdg(0)  # 0 = RDKit picks min(ncpu,16); the multi-thread baseline
print(f"ETKDG   1 core: {etk1:7.0f} conf/s   |  12 threads: {etk12:7.0f} conf/s")

# MMFF optimization (optimize the freshly embedded conformers)
import copy
mols1 = copy.deepcopy(mols)
mm1 = mmff_1core(mols1)
mm12 = mmff_threads(mols)
print(f"MMFF94  1 core: {mm1:7.0f} conf/s   |  12 threads: {mm12:7.0f} conf/s")
