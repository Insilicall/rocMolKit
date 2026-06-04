"""Feature throughput benchmark: GPU (rocMolKit) vs RDKit CPU (1 core and 12
threads), on the same host. Mirrors the conformer benchmark's columns.

Times the ported building blocks — Morgan fingerprints, Tanimoto similarity,
substructure search, TFD, and Butina clustering — against the RDKit CPU
equivalents, single-threaded and across 12 threads. GPU calls are timed with an
explicit hipDeviceSynchronize so async kernels are fully accounted for. The
12-thread variants use RDKit's native numThreads where available and a thread
pool (RDKit releases the GIL for compute) otherwise; the Butina greedy step is
inherently serial, so its 12-thread number equals 1 core.

Run inside the devel image with GPU passthrough. Molecules are tiled from
tests/data/druglike_100.smi up to the per-op N.
"""

import ctypes
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator as rfg
from rdkit.Chem.rdDistGeom import ETKDGv3, EmbedMultipleConfs
from rdkit.Chem.TorsionFingerprints import GetTFDMatrix
from rdkit.ML.Cluster import Butina

import rocmolkit._arrayHelpers
from rocmolkit._Fingerprints import MorganFingerprintGenerator
from rocmolkit._DataStructs import CrossTanimotoSimilarityRawBuffers
from rocmolkit._substructure import SubstructSearchConfig, countSubstructMatches
from rocmolkit._TFD import GetTFDMatricesGpuBuffer
from rocmolkit._clustering import butina as gpu_butina

RADIUS, FPSIZE, NTHREADS = 3, 2048, 12
_hip = ctypes.CDLL("libamdhip64.so")
_hip.hipMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
_hip.hipMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]


def sync():
    _hip.hipDeviceSynchronize()


def best(fn, n=3):
    fn()  # warm-up
    return min((lambda t0: (fn(), time.perf_counter() - t0)[1])(time.perf_counter()) for _ in range(n))


def gpu_best(fn, n=3):
    sync()
    return best(lambda: (fn(), sync()), n)


def chunks(seq, k):
    step = (len(seq) + k - 1) // k
    return [seq[i:i + step] for i in range(0, len(seq), step)]


_smis = [l.split()[0] for l in open("tests/data/druglike_100.smi") if l.strip() and not l.startswith("#")]
_base = [m for m in (Chem.MolFromSmiles(s) for s in _smis) if m]
rdgen = rfg.GetMorganGenerator(radius=RADIUS, fpSize=FPSIZE)
gen = MorganFingerprintGenerator(RADIUS, FPSIZE)


def tile(n):
    return [_base[i % len(_base)] for i in range(n)]


rows = []


def emit(op, scale, gpu_t, cpu1_t, cpu12_t, unit, work):
    f = lambda t: f"{work / t:,.0f}"
    rows.append((op, scale, f(gpu_t) + " " + unit, f(cpu1_t) + " " + unit, f(cpu12_t) + " " + unit,
                 f"{cpu12_t / gpu_t:.0f}x"))
    print(f"{op:20s} {scale:12s} GPU {f(gpu_t):>14} | 1c {f(cpu1_t):>12} | 12t {f(cpu12_t):>12} {unit}"
          f"  ({cpu12_t/gpu_t:.0f}x vs 12t)", flush=True)


# ---- 1. Morgan fingerprints ----
print('running: fingerprints...', flush=True)
N = 500000
mols = tile(N)
gt = gpu_best(lambda: gen.GetFingerprintsDevice(mols, 0, 0))
c1 = best(lambda: rdgen.GetFingerprints(mols, numThreads=1), n=1)
c12 = best(lambda: rdgen.GetFingerprints(mols, numThreads=NTHREADS), n=1)
emit("Morgan fingerprints", f"N={N}", gt, c1, c12, "mol/s", N)

# ---- 2. Tanimoto similarity (cross NxN) ----
print('running: similarity...', flush=True)
N = 10000
smols = tile(N)
cai = gen.GetFingerprintsDevice(smols, 0, 0).__cuda_array_interface__
gt = gpu_best(lambda: CrossTanimotoSimilarityRawBuffers(cai, cai, 0))
rdfps = [rdgen.GetFingerprint(m) for m in smols]


def _sim_rows(idxs):
    return [DataStructs.BulkTanimotoSimilarity(rdfps[i], rdfps) for i in idxs]


c1 = best(lambda: _sim_rows(range(N)), n=1)
with ThreadPoolExecutor(NTHREADS) as ex:
    c12 = best(lambda: list(ex.map(_sim_rows, chunks(list(range(N)), NTHREADS))), n=1)
emit("Tanimoto similarity", f"{N}x{N}", gt, c1, c12, "pair/s", N * N)

# ---- 3. Substructure search (targets x queries) ----
print("running: substructure...", flush=True)
N = 50000
targets = tile(N)
pats = ["c1ccccc1", "C=O", "C(=O)O", "C(=O)N", "[OH]", "[NH2]", "c1ccncc1", "CCC", "C(=O)[O;H1]", "S", "F",
        "Cl", "N", "c1ccccc1C", "[#7]", "[#8]", "C=C", "C#N", "C(F)(F)F", "[CX3](=O)[OX2H1]"]
queries = [q for q in (Chem.MolFromSmarts(p) for p in pats) if q]
cfg = SubstructSearchConfig()
cfg.gpuIds = [0]
gt = gpu_best(lambda: countSubstructMatches(targets, queries, cfg))


def _ss_rows(ts):
    return [[t.HasSubstructMatch(q) for q in queries] for t in ts]


c1 = best(lambda: _ss_rows(targets), n=1)
with ThreadPoolExecutor(NTHREADS) as ex:
    c12 = best(lambda: list(ex.map(_ss_rows, chunks(targets, NTHREADS))), n=1)
emit("Substructure match", f"{N}x{len(queries)}", gt, c1, c12, "pair/s", N * len(queries))

# ---- 4. TFD (M mols x k conformers) ----
print('running: tfd...', flush=True)
M, K = 1000, 10
params = ETKDGv3()
params.randomSeed = 0xBEEF
emb = []
for s in _smis:
    m = Chem.MolFromSmiles(s)
    if m is None:
        continue
    mh = Chem.AddHs(m)
    if len(EmbedMultipleConfs(mh, numConfs=K, params=params)) < 2:
        continue
    try:
        GetTFDMatrix(mh)
    except (RuntimeError, ValueError, IndexError):
        continue
    emb.append(mh)
    if len(emb) >= 40:
        break
batch = [emb[i % len(emb)] for i in range(M)]
ppm = K * (K - 1) // 2
gt = gpu_best(lambda: GetTFDMatricesGpuBuffer(batch, True, "equal", 2, True))
c1 = best(lambda: [GetTFDMatrix(m) for m in batch], n=1)
with ThreadPoolExecutor(NTHREADS) as ex:
    c12 = best(lambda: list(ex.map(GetTFDMatrix, batch)), n=1)
emit("TFD", f"M={M},k={K}", gt, c1, c12, "pair/s", M * ppm)

# ---- 5. Butina clustering step (precomputed NxN distance matrix) ----
print('running: butina...', flush=True)
N = 8000
cmols = tile(N)
cfps = [rdgen.GetFingerprint(m) for m in cmols]
dist = np.zeros((N, N), dtype=np.float64)
for i in range(N):
    dist[i] = 1.0 - np.asarray(DataStructs.BulkTanimotoSimilarity(cfps[i], cfps))
dptr = ctypes.c_void_p()
assert _hip.hipMalloc(ctypes.byref(dptr), dist.nbytes) == 0
assert _hip.hipMemcpy(dptr, dist.ctypes.data_as(ctypes.c_void_p), dist.nbytes, 1) == 0
dmat = {"shape": (N, N), "data": (dptr.value, False)}
gt = gpu_best(lambda: gpu_butina(dmat, 0.4, 64, False, 0))
cond = dist[np.tril_indices(N, -1)].tolist()
c1 = best(lambda: Butina.ClusterData(cond, N, 0.4, isDistData=True), n=1)
c12 = c1  # greedy assignment is inherently serial — no thread speedup
emit("Butina clustering", f"N={N}", gt, c1, c12, "mol/s", N)

# ---- Markdown table ----
print("\n| Operation | Scale | rocMolKit (GPU) | RDKit (1 core) | RDKit (12 threads) | Speedup vs 12t |")
print("|---|---|---|---|---|---|")
for op, sc, g, c1, c12, sp in rows:
    print(f"| **{op}** | {sc} | **{g}** | {c1} | {c12} | **{sp}** |")
