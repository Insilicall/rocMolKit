"""Isolated ETKDG conformer-GENERATION throughput — the heavier workload (full
DistGeom + ETK minimization), apples-to-apples vs mlxmolkit's generation
benchmark (they report ~2300 conf/s on Apple Silicon).

Times ONLY EmbedMolecules on the GPU. SMILES parsing / AddHs is not timed.

Usage:  python3 tools/etkdg_bench.py [N=1000] [K=4] [smi=tests/data/druglike_100.smi]

Run inside the devel container with a WRITABLE /tmp (the HIP runtime JIT-compiles
its blit kernels via comgr, which writes scratch to $TMPDIR — a read-only /tmp
makes that compile fail and the process segfaults). gpuIds is pinned to [0]
(dGPU gfx1200); the Ryzen iGPU gfx1036 has no compatible code object.

The first run is cold (one-time blit-kernel JIT); take the median of the warm
runs for steady-state throughput. Throughput here is bimodal: FP32 makes the
parallel reductions non-deterministic, so convergence (and ETKDG retries) varies
run to run — that is why we report the median of several runs, not single-shot.
"""
import math
import sys
import time

from rdkit import Chem
from rdkit.Chem import AddHs, AllChem
from rdkit.Chem.rdDistGeom import ETKDGv3
from rocmolkit._embedMolecules import EmbedMolecules, BatchHardwareOptions

N = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
K = int(sys.argv[2]) if len(sys.argv) > 2 else 4
SMI = sys.argv[3] if len(sys.argv) > 3 else "tests/data/druglike_100.smi"
RUNS = 7

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


def params():
    p = ETKDGv3()
    p.useRandomCoords = True
    p.randomSeed = 42
    p.numThreads = 0
    return p


opts = BatchHardwareOptions()
opts.gpuIds = [0]
TOTAL = N * K


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
    mols = [AddHs(Chem.MolFromSmiles(s)) for s in smis]  # not timed
    t0 = time.perf_counter()
    EmbedMolecules(mols, params(), K, -1, opts)
    dt = time.perf_counter() - t0
    confs = sum(m.GetNumConformers() for m in mols)
    rate = confs / dt
    rates.append(rate)
    tag = "cold" if i == 0 else "warm"
    print(f"run {i} [{tag}]: {confs}/{TOTAL} confs in {dt*1000:.0f}ms -> {rate:.0f} conf/s "
          f"| success {100*confs/TOTAL:.0f}% | energy median={energy_median(mols):.2f} kcal/mol")

warm = sorted(rates[1:]) if len(rates) > 1 else rates
print(f"\nwarm median = {warm[len(warm)//2]:.0f} conf/s  (N={N} k={K})")
print("(mlxmolkit Metal ~2300 conf/s; RDKit CPU 12-thread ~870 conf/s at scale)")
