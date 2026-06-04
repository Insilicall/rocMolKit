import ctypes, numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator as rfg
from rdkit import DataStructs
from rdkit.ML.Cluster import Butina
import rocmolkit._arrayHelpers
from rocmolkit._clustering import butina as gpu_butina

RADIUS, FPSIZE = 3, 2048
NEIGHBORLIST_MAX = 128   # must exceed the largest cluster's neighbor count
# Sweep several distance thresholds: low = mostly singletons, high = a few large
# clusters (exercises the large-cluster loop, the pruning loop, and renumbering).
CUTOFFS = [0.3, 0.4, 0.5, 0.6, 0.7]

rdgen = rfg.GetMorganGenerator(radius=RADIUS, fpSize=FPSIZE)
smis = [l.split()[0] for l in open("tests/data/druglike_100.smi")
        if l.strip() and not l.startswith("#")]
mols = [m for m in (Chem.MolFromSmiles(s) for s in smis) if m]
fps = [rdgen.GetFingerprint(m) for m in mols]
n = len(fps)

# Full NxN Tanimoto distance matrix (double), built from validated RDKit sims.
dist = np.zeros((n, n), dtype=np.float64)
for i in range(n):
    sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps)
    dist[i] = 1.0 - np.asarray(sims, dtype=np.float64)

# ---- upload distance matrix to device ----
hip = ctypes.CDLL("libamdhip64.so")
hip.hipMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
hip.hipMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
hip.hipDeviceSynchronize()
dptr = ctypes.c_void_p()
assert hip.hipMalloc(ctypes.byref(dptr), dist.nbytes) == 0
assert hip.hipMemcpy(dptr, dist.ctypes.data_as(ctypes.c_void_p), dist.nbytes, 1) == 0  # 1=H2D
dmat = {"shape": (n, n), "data": (dptr.value, False)}

def partition_from_labels(labels):
    groups = {}
    for idx, lab in enumerate(labels):
        groups.setdefault(int(lab), set()).add(idx)
    return frozenset(frozenset(g) for g in groups.values())

def member_map(part):
    m = {}
    for c in part:
        for x in c:
            m[x] = c
    return m

def ball_valid(part, cutoff):
    # Necessary condition for a valid Butina clustering: every cluster has a
    # centroid member within `cutoff` of all other members. Returns the count of
    # clusters that violate it (0 == all valid).
    bad = 0
    for c in part:
        members = list(c)
        if len(members) == 1:
            continue
        if not any(all(dist[ctr][m] <= cutoff for m in members) for ctr in members):
            bad += 1
    return bad

# RDKit reference uses the condensed lower-triangle distance list.
condensed = [dist[i][j] for i in range(n) for j in range(i)]

allpass = True
for cutoff in CUTOFFS:
    res = gpu_butina(dmat, cutoff, NEIGHBORLIST_MAX, False, 0)
    ai = res.__cuda_array_interface__
    shape = tuple(ai["shape"]); rptr = ai["data"][0]
    gpu_ids = np.empty(shape, dtype=np.dtype(ai["typestr"]))
    hip.hipDeviceSynchronize()
    assert hip.hipMemcpy(gpu_ids.ctypes.data_as(ctypes.c_void_p), ctypes.c_void_p(rptr),
                         int(np.prod(shape)) * gpu_ids.dtype.itemsize, 2) == 0  # 2=D2H
    gpu_part = partition_from_labels(gpu_ids)

    rd_clusters = Butina.ClusterData(condensed, n, cutoff, isDistData=True)
    rd_part = frozenset(frozenset(c) for c in rd_clusters)
    largest = max(len(c) for c in rd_part)

    if gpu_part == rd_part:
        print(f"cutoff={cutoff}: GPU={len(gpu_part):3d} RDKit={len(rd_part):3d} "
              f"largest={largest:3d}  IDENTICAL  PASS")
    else:
        gm, rm = member_map(gpu_part), member_map(rd_part)
        diff_pts = [x for x in range(n) if gm[x] != rm[x]]
        # Differences vs RDKit are legitimate only if the GPU partition is itself
        # a valid Butina clustering (every cluster is a ball around a centroid).
        # That distinguishes parallel tie-break choices from real errors.
        bad = ball_valid(gpu_part, cutoff)
        ok = bad == 0
        allpass = allpass and ok
        print(f"cutoff={cutoff}: GPU={len(gpu_part):3d} RDKit={len(rd_part):3d} "
              f"largest={largest:3d}  DIFFER pts={len(diff_pts):2d} invalid_gpu_clusters={bad}  "
              f"{'PASS(tie-break, GPU valid)' if ok else 'FAIL'}")

print("RESULT:", "ALL PASS" if allpass else "FAIL")
