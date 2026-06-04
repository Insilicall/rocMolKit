import ctypes, numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdDistGeom import ETKDGv3, EmbedMultipleConfs
from rdkit.Chem.TorsionFingerprints import GetTFDMatrix
import rocmolkit._arrayHelpers
from rocmolkit._TFD import GetTFDMatricesGpuBuffer

NCONF = 8
smis = [l.split()[0] for l in open("tests/data/druglike_100.smi")
        if l.strip() and not l.startswith("#")]

# Build molecules with several conformers each (TFD needs >= 2 confs and at
# least one rotatable/ring torsion). Keep the first 25 that qualify.
mols = []
params = ETKDGv3()
params.randomSeed = 0xF00D
for s in smis:
    m = Chem.MolFromSmiles(s)
    if m is None:
        continue
    mh = Chem.AddHs(m)
    cids = EmbedMultipleConfs(mh, numConfs=NCONF, params=params)
    if len(cids) < 2:
        continue
    try:
        ref = GetTFDMatrix(mh)  # RDKit reference (also validates it has torsions)
    except (RuntimeError, ValueError, IndexError):
        continue  # RDKit GetTFDMatrix is fragile on some topologies; skip those
    if len(ref) == 0:
        continue
    mols.append((mh, ref))
    if len(mols) >= 25:
        break

molObjs = [m for m, _ in mols]

# GPU: flat buffer of TFD values + per-molecule start offsets.
buf, starts = GetTFDMatricesGpuBuffer(molObjs, True, "equal", 2, True)
ai = buf.__cuda_array_interface__
shape = tuple(ai["shape"]); ptr = ai["data"][0]
host = np.empty(shape, dtype=np.dtype(ai["typestr"]))
hip = ctypes.CDLL("libamdhip64.so")
hip.hipDeviceSynchronize()
hip.hipMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
assert hip.hipMemcpy(host.ctypes.data_as(ctypes.c_void_p), ctypes.c_void_p(ptr),
                     int(np.prod(shape)) * host.dtype.itemsize, 2) == 0  # 2=D2H

starts = list(starts)
npass = nfail = 0
worst = 0.0
for m, (mh, ref) in enumerate(mols):
    k = mh.GetNumConformers()
    npairs = k * (k - 1) // 2
    off = starts[m]
    gpu = host[off:off + npairs].astype(np.float64)
    rd = np.asarray(ref, dtype=np.float64)
    if len(gpu) != len(rd):
        nfail += 1
        print(f"  mol {m}: length mismatch gpu={len(gpu)} rdkit={len(rd)}")
        continue
    d = np.abs(gpu - rd).max() if npairs else 0.0
    worst = max(worst, d)
    # GPU stores conformer positions and dihedral angles in float32 while RDKit
    # computes TFD in double; agreement to ~1e-3 is the expected float32 bound.
    # A logic error (wrong pairing/torsion) would show O(0.1-1.0) differences.
    if d < 1e-3:
        npass += 1
    else:
        nfail += 1
        print(f"  mol {m}: max|Δ|={d:.3e} (k={k})")

print(f"PASS {npass}/{len(mols)}  FAIL {nfail}  worst max|Δ|={worst:.3e} (float32 tol=1e-3)")
print("RESULT:", "ALL PASS" if nfail == 0 else "FAIL")
