"""Isolated MMFF94 optimize-only throughput — apples-to-apples vs mlxmolkit's
MMFF benchmark. Conformers are generated with RDKit (NOT timed); only
MMFFOptimizeMoleculesConfs runs on the GPU and is timed.

Usage:  python3 tools/mmff_bench.py [N=1000] [K=4] [smi=tests/data/druglike_100.smi]

Run inside the devel container with a WRITABLE /tmp (the HIP runtime JIT-compiles
its blit kernels via comgr, which writes scratch to $TMPDIR — a read-only /tmp
makes that compile fail and the process segfaults). gpuIds is pinned to [0]
(dGPU gfx1200); the Ryzen iGPU gfx1036 has no compatible code object.

The first run is cold (includes the one-time blit-kernel JIT); take the median
of the warm runs for steady-state throughput.
"""
import math
import sys
import time

from rdkit import Chem
from rdkit.Chem import AddHs, AllChem
from rdkit.Chem.rdDistGeom import ETKDGv3, EmbedMultipleConfs
from rocmolkit._embedMolecules import BatchHardwareOptions  # registers types first
from rocmolkit._mmffOptimization import MMFFOptimizeMoleculesConfs

N = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
K = int(sys.argv[2]) if len(sys.argv) > 2 else 4
SMI = sys.argv[3] if len(sys.argv) > 3 else "tests/data/druglike_100.smi"
RUNS = 6

smis = []
with open(SMI) as f:
    for line in f:
        s = line.strip()
        if s and not s.startswith("#"):
            smis.append(s.split()[0])
base = list(smis)
while len(smis) < N:
    smis.extend(base)
smis = smis[:N]

opts = BatchHardwareOptions()
opts.gpuIds = [0]


def build_mols():
    p = ETKDGv3(); p.randomSeed = 42; p.numThreads = 0
    mols = []
    for s in smis:
        m = AddHs(Chem.MolFromSmiles(s))
        EmbedMultipleConfs(m, numConfs=K, params=p)
        if m.GetNumConformers() > 0:
            mols.append(m)
    return mols


def energy_median(mols):
    es = []
    for m in mols:
        props = AllChem.MMFFGetMoleculeProperties(m)
        if props is None:
            continue
        for c in m.GetConformers():
            ff = AllChem.MMFFGetMoleculeForceField(m, props, confId=c.GetId())
            if ff:
                es.append(ff.CalcEnergy())
    es = [e for e in es if not math.isnan(e)]
    return sorted(es)[len(es) // 2] if es else float("nan")


rates = []
for i in range(RUNS):
    mols = build_mols()  # fresh geometries each run (NOT timed)
    total = sum(m.GetNumConformers() for m in mols)
    t0 = time.perf_counter()
    MMFFOptimizeMoleculesConfs(mols, 200, [], opts)
    dt = time.perf_counter() - t0
    rate = total / dt
    rates.append(rate)
    tag = "cold" if i == 0 else "warm"
    print(f"run {i} [{tag}]: {total} confs in {dt*1000:.0f}ms -> {rate:.0f} conf/s "
          f"| optimized-geom energy median={energy_median(mols):.2f} kcal/mol")

warm = sorted(rates[1:]) if len(rates) > 1 else rates
print(f"\nwarm median = {warm[len(warm)//2]:.0f} conf/s  (N={N} k={K})")
print("(mlxmolkit Metal ~12000 conf/s peak; RDKit CPU ~1100-1300 conf/s at scale)")
