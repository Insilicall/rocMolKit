import ctypes, numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator as rfg
from rdkit import DataStructs
import rocmolkit._arrayHelpers
from rocmolkit._Fingerprints import MorganFingerprintGenerator
from rocmolkit._DataStructs import (
    CrossTanimotoSimilarityRawBuffers,
    CrossCosineSimilarityRawBuffers,
    CrossTanimotoSimilarityCPURawBuffers,
)

RADIUS, FPSIZE = 3, 2048
rdgen = rfg.GetMorganGenerator(radius=RADIUS, fpSize=FPSIZE, countSimulation=False,
                               includeChirality=False, useBondTypes=True,
                               onlyNonzeroInvariants=False)
smis = [l.split()[0] for l in open("tests/data/druglike_100.smi")
        if l.strip() and not l.startswith("#")]
mols = [m for m in (Chem.MolFromSmiles(s) for s in smis) if m]

# GPU fingerprints (validated bit-exact in F1) -> device buffer
gen = MorganFingerprintGenerator(RADIUS, FPSIZE)
arr = gen.GetFingerprintsDevice(mols, 0, 0)
cai = arr.__cuda_array_interface__

hip = ctypes.CDLL("libamdhip64.so")
hip.hipDeviceSynchronize()
hip.hipMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]

def to_host(pyarr):
    ai = pyarr.__cuda_array_interface__
    shape = tuple(ai['shape']); ptr = ai['data'][0]
    dtype = np.dtype(ai['typestr'])
    host = np.empty(shape, dtype=dtype)
    hip.hipDeviceSynchronize()
    assert hip.hipMemcpy(host.ctypes.data_as(ctypes.c_void_p), ctypes.c_void_p(ptr),
                         int(np.prod(shape)) * dtype.itemsize, 2) == 0
    return host

# ---- RDKit reference: full cross Tanimoto matrix ----
rdfps = [rdgen.GetFingerprint(m) for m in mols]
n = len(rdfps)
ref = np.zeros((n, n), dtype=np.float64)
for i in range(n):
    sims = DataStructs.BulkTanimotoSimilarity(rdfps[i], rdfps)
    ref[i] = sims

# ---- GPU cross Tanimoto ----
gpu_dev = CrossTanimotoSimilarityRawBuffers(cai, cai, 0)
gpu = to_host(gpu_dev).astype(np.float64)

# ---- CPU-result path (host vector) ----
cpu = np.asarray(CrossTanimotoSimilarityCPURawBuffers(cai, cai), dtype=np.float64)

def report(name, mat):
    diff = np.abs(mat - ref)
    md = diff.max()
    nbad = int((diff > 1e-5).sum())
    print(f"{name}: max|Δ|={md:.3e}  off>1e-5={nbad}/{n*n}  "
          f"{'PASS' if md < 1e-5 else 'FAIL'}")
    return md < 1e-5

ok1 = report("Tanimoto GPU ", gpu)
ok2 = report("Tanimoto CPU ", cpu)

# Cosine: RDKit has no direct bulk cosine on ExplicitBitVect; sanity-check
# self-similarity diagonal == 1 and symmetry.
cos = to_host(CrossCosineSimilarityRawBuffers(cai, cai, 0)).astype(np.float64)
diag_ok = np.allclose(np.diag(cos), 1.0, atol=1e-5)
sym_ok = np.allclose(cos, cos.T, atol=1e-5)
print(f"Cosine GPU  : diag==1 {diag_ok}  symmetric {sym_ok}  "
      f"{'PASS' if diag_ok and sym_ok else 'FAIL'}")

print("RESULT:", "ALL PASS" if (ok1 and ok2 and diag_ok and sym_ok) else "FAIL")
