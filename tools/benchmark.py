"""rocMolKit benchmark — ETKDG + MMFF94 timing vs RDKit CPU.

The GPU wins on three axes: batch size (N), conformers per molecule (k),
and skipping subprocess overhead. The default sweep exercises all three
so the README can show where rocMolKit beats RDKit and where it does not.

Run inside the devel docker image with the GPU mapped:

    docker run --rm \\
        --device=/dev/kfd --device=/dev/dri \\
        --group-add 987 --group-add 983 \\
        --security-opt seccomp=unconfined \\
        -e LD_LIBRARY_PATH=/usr/local/lib:/opt/rdkit/lib:/opt/rocm/lib:/opt/rocm-7.2.3/lib/llvm/lib \\
        -v $PWD:/work -w /work \\
        ghcr.io/insilicall/rocmolkit:devel \\
        python3 tools/benchmark.py tests/data/druglike_100.smi

A --sweep flag re-runs the GPU path across (N, k) pairs so a single
invocation produces the full README performance table.
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AddHs, AllChem
from rdkit.Chem.rdDistGeom import ETKDGv3, EmbedMultipleConfs


def load_smiles(path: Path, n: int | None) -> list[str]:
    smis = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            smi = line.split()[0]
            if Chem.MolFromSmiles(smi) is not None:
                smis.append(smi)
            if n is not None and len(smis) >= n:
                break
    # Cycle the dataset so larger N can be benchmarked without authoring
    # a much bigger SMILES file.
    if n is not None and len(smis) < n and smis:
        base = list(smis)
        while len(smis) < n:
            smis.extend(base)
        smis = smis[:n]
    return smis


def _etkdg_params(seed: int, num_threads: int = 0) -> ETKDGv3:
    p = ETKDGv3()
    p.useRandomCoords = True
    p.randomSeed = seed
    p.numThreads = num_threads
    return p


def rdkit_cpu_etkdg(smis: list[str], k: int, seed: int = 42,
                    num_threads: int = 0) -> tuple[float, int]:
    """RDKit CPU multi-threaded ``EmbedMultipleConfs`` — the honest baseline.

    ``numThreads=0`` lets RDKit pick (``min(ncpu, 16)``); pin to 1 for the
    apples-to-apples single-thread comparison.
    """
    p = _etkdg_params(seed, num_threads)
    mols = [AddHs(Chem.MolFromSmiles(s)) for s in smis]
    t0 = time.perf_counter()
    confs = 0
    for m in mols:
        ids = EmbedMultipleConfs(m, k, p)
        confs += len(ids)
    return time.perf_counter() - t0, confs


def rdkit_cpu_etkdg_mmff(smis: list[str], k: int, seed: int = 42,
                          num_threads: int = 0) -> tuple[float, int]:
    p = _etkdg_params(seed, num_threads)
    mols = [AddHs(Chem.MolFromSmiles(s)) for s in smis]
    t0 = time.perf_counter()
    ok = 0
    for m in mols:
        ids = EmbedMultipleConfs(m, k, p)
        for cid in ids:
            AllChem.MMFFOptimizeMolecule(m, confId=int(cid), maxIters=200)
        ok += len(ids)
    return time.perf_counter() - t0, ok


def rocmolkit_gpu_etkdg(smis: list[str], k: int, seed: int = 42,
                         gpu_id: int = 0) -> tuple[float, int]:
    """Single batched call into the GPU binding with gpuIds pinned.

    Pinning to a discrete-GPU id avoids dispatching to the integrated
    GPU on Ryzen hosts (see ISSUES.md "ROOT CAUSE FOUND"). Without this
    pin the call SIGSEGVs on the first iGPU-dispatched mol.
    """
    from rocmolkit._embedMolecules import EmbedMolecules, BatchHardwareOptions

    p = _etkdg_params(seed)
    opts = BatchHardwareOptions()
    opts.gpuIds = [gpu_id]

    mols = [AddHs(Chem.MolFromSmiles(s)) for s in smis]
    t0 = time.perf_counter()
    EmbedMolecules(mols, p, k, -1, opts)
    dt = time.perf_counter() - t0
    confs = sum(m.GetNumConformers() for m in mols)
    return dt, confs


def rocmolkit_gpu_etkdg_mmff(smis: list[str], k: int, seed: int = 42,
                              gpu_id: int = 0) -> tuple[float, int]:
    from rocmolkit._embedMolecules import EmbedMolecules, BatchHardwareOptions
    from rocmolkit._mmffOptimization import MMFFOptimizeMoleculesConfs

    p = _etkdg_params(seed)
    opts = BatchHardwareOptions()
    opts.gpuIds = [gpu_id]

    mols = [AddHs(Chem.MolFromSmiles(s)) for s in smis]
    t0 = time.perf_counter()
    EmbedMolecules(mols, p, k, -1, opts)
    MMFFOptimizeMoleculesConfs(mols, maxIters=200)
    dt = time.perf_counter() - t0
    confs = sum(m.GetNumConformers() for m in mols)
    return dt, confs


class _Timeout(Exception):
    pass


def _timeout_handler(signum, frame):  # noqa: ARG001
    raise _Timeout()


def run_phase(label: str, fn, *args, timeout_s: int = 600, **kwargs):
    """Run one benchmark phase. SIGALRM-bounded so a stuck GPU does not
    wedge the whole run."""
    print(f"[{label}] starting...", flush=True)
    old = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout_s)
    try:
        t, ok = fn(*args, **kwargs)
        print(f"[{label}] OK: {t:.2f}s, {ok} confs", flush=True)
        return t, ok
    except _Timeout:
        print(f"[{label}] TIMEOUT after {timeout_s}s", flush=True)
        return None, 0
    except Exception as e:
        print(f"[{label}] CRASHED: {type(e).__name__}: {e}", flush=True)
        return None, 0
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def fmt_row(name: str, n: int, k: int, t: float | None, confs: int) -> str:
    total = n * k
    if t is None or confs == 0:
        return f"| {name} | N={n}, k={k} | — | — | 0/{total} |"
    per_conf_ms = 1000 * t / confs
    throughput = confs / t
    return (f"| {name} | N={n}, k={k} | {t:.2f} s | "
            f"{per_conf_ms:.1f} ms/conf ({throughput:.0f} conf/s) | {confs}/{total} |")


SWEEP_DEFAULT = [
    # (N,  k)
    (50,   1),
    (50,  10),
    (50,  50),
    (500,  1),
    (500, 10),
    (500, 50),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("smiles_file", type=Path)
    ap.add_argument("--n", type=int, default=None,
                    help="Cap molecule count (single-shape mode).")
    ap.add_argument("--k", type=int, default=1,
                    help="Conformers per molecule (single-shape mode).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gpu-id", type=int, default=0)
    ap.add_argument("--sweep", action="store_true",
                    help=f"Run the full (N, k) sweep: {SWEEP_DEFAULT}")
    ap.add_argument("--timeout", type=int, default=600,
                    help="Per-phase timeout in seconds (default: 600)")
    ap.add_argument("--skip", action="append", default=[],
                    help="Skip a path by name (cpu1, cpu_mt, gpu, mmff_cpu, mmff_gpu).")
    args = ap.parse_args()

    shapes = SWEEP_DEFAULT if args.sweep else [(args.n or 0, args.k)]
    max_n = max((n for n, _ in shapes if n), default=args.n or 0)
    smis_all = load_smiles(args.smiles_file, max_n if max_n else None)
    if not smis_all:
        print("error: no SMILES loaded", file=sys.stderr)
        sys.exit(1)

    print(f"# rocMolKit benchmark — dataset {args.smiles_file.name}",
          flush=True)
    print(f"# loaded {len(smis_all)} SMILES (cycled to target N if needed)",
          flush=True)

    rows: list[tuple[str, str]] = []

    for (n, k) in shapes:
        n = n or len(smis_all)
        smis = smis_all[:n]
        print(f"\n=== shape N={n} k={k} ===", flush=True)

        if "cpu1" not in args.skip:
            t, c = run_phase(f"RDKit CPU 1-thread N={n} k={k}",
                             rdkit_cpu_etkdg, smis, k, args.seed,
                             num_threads=1, timeout_s=args.timeout)
            rows.append(("ETKDG",
                         fmt_row("RDKit CPU (1 thread)", n, k, t, c)))

        if "cpu_mt" not in args.skip:
            ncpu = os.cpu_count() or 1
            t, c = run_phase(
                f"RDKit CPU {ncpu}-thread N={n} k={k}",
                rdkit_cpu_etkdg, smis, k, args.seed,
                num_threads=0, timeout_s=args.timeout)
            rows.append(("ETKDG",
                         fmt_row(f"RDKit CPU ({ncpu} threads)", n, k, t, c)))

        if "gpu" not in args.skip:
            t, c = run_phase(
                f"rocMolKit GPU direct N={n} k={k}",
                rocmolkit_gpu_etkdg, smis, k, args.seed,
                gpu_id=args.gpu_id, timeout_s=args.timeout)
            rows.append(("ETKDG",
                         fmt_row("rocMolKit GPU (RX 9060 XT)", n, k, t, c)))

        if "mmff_cpu" not in args.skip:
            t, c = run_phase(
                f"RDKit CPU ETKDG+MMFF N={n} k={k}",
                rdkit_cpu_etkdg_mmff, smis, k, args.seed,
                num_threads=0, timeout_s=args.timeout)
            rows.append(("ETKDG + MMFF94",
                         fmt_row(f"RDKit CPU ({os.cpu_count()} threads)",
                                 n, k, t, c)))

        if "mmff_gpu" not in args.skip:
            t, c = run_phase(
                f"rocMolKit GPU ETKDG+MMFF N={n} k={k}",
                rocmolkit_gpu_etkdg_mmff, smis, k, args.seed,
                gpu_id=args.gpu_id, timeout_s=args.timeout)
            rows.append(("ETKDG + MMFF94",
                         fmt_row("rocMolKit GPU (RX 9060 XT)", n, k, t, c)))

    print("\n\n--- RESULTS ---\n", flush=True)
    section = None
    for sec, row in rows:
        if sec != section:
            print(f"\n### {sec}\n")
            print("| Pipeline | Shape | Wall time | Per-conformer | Success |")
            print("|---|---|---|---|---|")
            section = sec
        print(row)

    print("\n_Hardware: AMD Ryzen 5 7600 (12 threads) + "
          "AMD Radeon RX 9060 XT (gfx1200, 32 CUs, RDNA4) on ROCm 7.2.3._")
    print(f"_Dataset: {args.smiles_file.name}._")


if __name__ == "__main__":
    main()
