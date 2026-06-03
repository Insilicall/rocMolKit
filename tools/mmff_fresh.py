# Honest first-pass MMFF throughput: GPU pre-warmed, then FRESH conformers
# optimized ONCE (what mlxmolkit measures — not re-optimizing converged geoms).
import sys, time
from rdkit import Chem
from rdkit.Chem import AddHs
from rdkit.Chem.rdDistGeom import ETKDGv3, EmbedMultipleConfs
from rocmolkit._embedMolecules import BatchHardwareOptions
from rocmolkit._mmffOptimization import MMFFOptimizeMoleculesConfs
N,K=int(sys.argv[1]),int(sys.argv[2])
smis=[l.split()[0] for l in open("tests/data/druglike_100.smi") if l.strip() and not l.startswith("#")]
b=list(smis)
while len(smis)<N: smis.extend(b)
smis=smis[:N]
p=ETKDGv3();p.randomSeed=42;p.numThreads=0
o=BatchHardwareOptions();o.gpuIds=[0]
def build():
    ms=[]
    for s in smis:
        m=AddHs(Chem.MolFromSmiles(s)); EmbedMultipleConfs(m,numConfs=K,params=p)
        if m.GetNumConformers()>0: ms.append(m)
    return ms
# GPU warmup on a THROWAWAY set (pays JIT/alloc once, leaves nothing optimized)
warm=build(); MMFFOptimizeMoleculesConfs(warm,200,[],o)
# Now time a FRESH set optimized exactly once
fresh=build(); tot=sum(m.GetNumConformers() for m in fresh)
t=time.perf_counter(); MMFFOptimizeMoleculesConfs(fresh,200,[],o); dt=time.perf_counter()-t
print(f"FRESH k={K}: {tot} confs in {dt*1000:.0f}ms -> {tot/dt:.0f} conf/s")
