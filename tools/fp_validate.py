import ctypes, numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator as rfg
import rocmolkit._arrayHelpers
from rocmolkit._Fingerprints import MorganFingerprintGenerator
RADIUS,FPSIZE=3,2048
rdgen=rfg.GetMorganGenerator(radius=RADIUS,fpSize=FPSIZE,countSimulation=False,includeChirality=False,useBondTypes=True,onlyNonzeroInvariants=False)
smis=[l.split()[0] for l in open("tests/data/druglike_100.smi") if l.strip() and not l.startswith("#")]
smis += ["C"*70, "C"*110, "O="+"C"*90]   # large -> 128 path
mols=[m for m in (Chem.MolFromSmiles(s) for s in smis) if m]
gen=MorganFingerprintGenerator(RADIUS,FPSIZE)
arr=gen.GetFingerprintsDevice(mols,0,0)
ai=arr.__cuda_array_interface__; shape=tuple(ai['shape']); ptr=ai['data'][0]
host=np.empty(shape,dtype=np.uint32)
hip=ctypes.CDLL("libamdhip64.so"); hip.hipDeviceSynchronize()
hip.hipMemcpy.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_size_t,ctypes.c_int]
assert hip.hipMemcpy(host.ctypes.data_as(ctypes.c_void_p),ctypes.c_void_p(ptr),int(np.prod(shape))*4,2)==0
def onbits(row):
    s=set()
    for c,w in enumerate(row):
        w=int(w)&0xFFFFFFFF
        for b in range(32):
            if w&(1<<b): s.add(c*32+b)
    return s
npass=nfail=0; fails=[]
for i,m in enumerate(mols):
    rd=set(rdgen.GetFingerprint(m).GetOnBits()); g=onbits(host[i])
    if rd==g: npass+=1
    else: nfail+=1; fails.append((i,m.GetNumAtoms(),len(rd),len(g)))
print(f"PASS {npass}/{len(mols)}  FAIL {nfail}")
for f in fails[:10]: print("  FAIL", f)
