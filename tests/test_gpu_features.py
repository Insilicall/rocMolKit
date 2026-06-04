"""GPU integration tests for the ported feature modules (F1-F5).

Each test validates a GPU result against RDKit on real hardware. They are marked
`gpu`, so they only run with `pytest --rocm` (or ROCMOLKIT_HAS_GPU=1) and are
skipped otherwise. The C++ extensions must be built and importable; tests that
can't import their module are skipped via `importorskip`.

Mirrors the standalone validators in tools/{fp,sim,butina,tfd}_validate.py.
"""

from __future__ import annotations

import ctypes
import pathlib

import numpy as np
import pytest

pytestmark = pytest.mark.gpu

DATA = pathlib.Path(__file__).parent / "data" / "druglike_100.smi"


def _smiles(limit: int | None = None) -> list[str]:
    out = []
    for line in DATA.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            out.append(line.split()[0])
            if limit and len(out) >= limit:
                break
    return out


def _hip() -> ctypes.CDLL:
    hip = ctypes.CDLL("libamdhip64.so")
    hip.hipDeviceSynchronize()
    hip.hipMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    hip.hipMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
    return hip


def _to_host(pyarr, hip: ctypes.CDLL) -> np.ndarray:
    """Copy a GPU-resident __cuda_array_interface__ array to a host ndarray."""
    ai = pyarr.__cuda_array_interface__
    shape = tuple(ai["shape"])
    host = np.empty(shape, dtype=np.dtype(ai["typestr"]))
    hip.hipDeviceSynchronize()
    n = int(np.prod(shape)) * host.dtype.itemsize if shape else 0
    assert hip.hipMemcpy(host.ctypes.data_as(ctypes.c_void_p), ctypes.c_void_p(ai["data"][0]), n, 2) == 0
    return host


def test_morgan_fingerprint_bitexact() -> None:
    pytest.importorskip("rocmolkit._arrayHelpers")
    fp_mod = pytest.importorskip("rocmolkit._Fingerprints")
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator as rfg

    radius, fpsize = 3, 2048
    rdgen = rfg.GetMorganGenerator(radius=radius, fpSize=fpsize)
    smis = _smiles() + ["C" * 70, "C" * 110, "O=" + "C" * 90]  # exercise the 128-atom path
    mols = [m for m in (Chem.MolFromSmiles(s) for s in smis) if m]

    gen = fp_mod.MorganFingerprintGenerator(radius, fpsize)
    host = _to_host(gen.GetFingerprintsDevice(mols, 0, 0), _hip())

    def onbits(row: np.ndarray) -> set[int]:
        s = set()
        for col, word in enumerate(row):
            word = int(word) & 0xFFFFFFFF
            for b in range(32):
                if word & (1 << b):
                    s.add(col * 32 + b)
        return s

    fails = [i for i, m in enumerate(mols)
             if set(rdgen.GetFingerprint(m).GetOnBits()) != onbits(host[i])]
    assert not fails, f"{len(fails)} molecules differ from RDKit Morgan bits"


def test_tanimoto_similarity_bitexact() -> None:
    pytest.importorskip("rocmolkit._arrayHelpers")
    fp_mod = pytest.importorskip("rocmolkit._Fingerprints")
    ds = pytest.importorskip("rocmolkit._DataStructs")
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator as rfg

    radius, fpsize = 3, 2048
    rdgen = rfg.GetMorganGenerator(radius=radius, fpSize=fpsize)
    mols = [m for m in (Chem.MolFromSmiles(s) for s in _smiles(50)) if m]
    n = len(mols)

    gen = fp_mod.MorganFingerprintGenerator(radius, fpsize)
    cai = gen.GetFingerprintsDevice(mols, 0, 0).__cuda_array_interface__
    hip = _hip()
    gpu = _to_host(ds.CrossTanimotoSimilarityRawBuffers(cai, cai, 0), hip).astype(np.float64)

    rdfps = [rdgen.GetFingerprint(m) for m in mols]
    ref = np.array([DataStructs.BulkTanimotoSimilarity(rdfps[i], rdfps) for i in range(n)])
    assert np.abs(gpu - ref).max() < 1e-5


def test_butina_matches_rdkit() -> None:
    pytest.importorskip("rocmolkit._arrayHelpers")
    clustering = pytest.importorskip("rocmolkit._clustering")
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator as rfg
    from rdkit.ML.Cluster import Butina

    rdgen = rfg.GetMorganGenerator(radius=3, fpSize=2048)
    mols = [m for m in (Chem.MolFromSmiles(s) for s in _smiles()) if m]
    fps = [rdgen.GetFingerprint(m) for m in mols]
    n = len(fps)
    dist = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        dist[i] = 1.0 - np.asarray(DataStructs.BulkTanimotoSimilarity(fps[i], fps))

    hip = _hip()
    dptr = ctypes.c_void_p()
    assert hip.hipMalloc(ctypes.byref(dptr), dist.nbytes) == 0
    assert hip.hipMemcpy(dptr, dist.ctypes.data_as(ctypes.c_void_p), dist.nbytes, 1) == 0  # H2D
    dmat = {"shape": (n, n), "data": (dptr.value, False)}

    condensed = [dist[i][j] for i in range(n) for j in range(i)]

    def partition(labels):
        groups: dict[int, set[int]] = {}
        for idx, lab in enumerate(labels):
            groups.setdefault(int(lab), set()).add(idx)
        return frozenset(frozenset(g) for g in groups.values())

    def ball_valid(part, cutoff):
        for c in part:
            members = list(c)
            if len(members) > 1 and not any(
                all(dist[ctr][m] <= cutoff for m in members) for ctr in members
            ):
                return False
        return True

    # Low cutoffs: identical to RDKit. High cutoffs: parallel tie-breaks allowed,
    # but the GPU partition must itself be a valid Butina clustering (ball-valid).
    for cutoff in (0.3, 0.4, 0.5):
        gpu = partition(_to_host(clustering.butina(dmat, cutoff, 128, False, 0), hip))
        rd = frozenset(frozenset(c) for c in Butina.ClusterData(condensed, n, cutoff, isDistData=True))
        assert gpu == rd, f"cutoff={cutoff}: GPU partition differs from RDKit"
    for cutoff in (0.6, 0.7):
        gpu = partition(_to_host(clustering.butina(dmat, cutoff, 128, False, 0), hip))
        assert ball_valid(gpu, cutoff), f"cutoff={cutoff}: GPU produced an invalid (non-ball) cluster"


def test_tfd_matches_rdkit() -> None:
    pytest.importorskip("rocmolkit._arrayHelpers")
    tfd_mod = pytest.importorskip("rocmolkit._TFD")
    from rdkit import Chem
    from rdkit.Chem.rdDistGeom import ETKDGv3, EmbedMultipleConfs
    from rdkit.Chem.TorsionFingerprints import GetTFDMatrix

    params = ETKDGv3()
    params.randomSeed = 0xF00D
    mols, refs = [], []
    for s in _smiles():
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        mh = Chem.AddHs(m)
        if len(EmbedMultipleConfs(mh, numConfs=8, params=params)) < 2:
            continue
        try:
            ref = GetTFDMatrix(mh)
        except (RuntimeError, ValueError, IndexError):
            continue  # RDKit GetTFDMatrix is fragile on some topologies
        if ref:
            mols.append(mh)
            refs.append(np.asarray(ref, dtype=np.float64))
        if len(mols) >= 15:
            break

    assert mols, "no molecule yielded a TFD matrix"
    buf, starts = tfd_mod.GetTFDMatricesGpuBuffer(mols, True, "equal", 2, True)
    host = _to_host(buf, _hip())
    starts = list(starts)
    for m, (mh, ref) in enumerate(zip(mols, refs)):
        k = mh.GetNumConformers()
        npairs = k * (k - 1) // 2
        gpu = host[starts[m]:starts[m] + npairs].astype(np.float64)
        assert len(gpu) == len(ref)
        # GPU is float32 vs RDKit double; ~1e-3 is the expected precision bound.
        assert np.abs(gpu - ref).max() < 1e-3, f"mol {m}: TFD exceeds float32 tolerance"
