"""rocMolKit benchmark — ETKDG + MMFF94 timing vs RDKit CPU.

Run inside the devel docker image with the GPU mapped:

    docker run --rm \\
        --device=/dev/kfd --device=/dev/dri \\
        --group-add 987 --group-add 983 \\
        --security-opt seccomp=unconfined \\
        -v $PWD:/work -w /work \\
        rocmolkit:devel-local \\
        python3 tools/benchmark.py tests/data/spice_100.smi --n 100

Outputs a Markdown table that drops straight into README.md.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AddHs, AllChem
from rdkit.Chem.rdDistGeom import ETKDGv3


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
    return smis


def rdkit_cpu_etkdg(smis: list[str], seed: int = 42) -> tuple[float, int]:
    """Sequential RDKit CPU ETKDG via AllChem.EmbedMolecule."""
    p = ETKDGv3()
    p.useRandomCoords = True
    p.randomSeed = seed
    confs = 0
    t0 = time.perf_counter()
    for smi in smis:
        m = AddHs(Chem.MolFromSmiles(smi))
        if AllChem.EmbedMolecule(m, p) >= 0:
            confs += 1
    return time.perf_counter() - t0, confs


def rdkit_cpu_etkdg_mmff(smis: list[str], seed: int = 42) -> tuple[float, int]:
    """Sequential RDKit CPU: ETKDG embed then MMFF94 optimise."""
    p = ETKDGv3()
    p.useRandomCoords = True
    p.randomSeed = seed
    ok = 0
    t0 = time.perf_counter()
    for smi in smis:
        m = AddHs(Chem.MolFromSmiles(smi))
        if AllChem.EmbedMolecule(m, p) < 0:
            continue
        AllChem.MMFFOptimizeMolecule(m, maxIters=200)
        ok += 1
    return time.perf_counter() - t0, ok


def rocmolkit_gpu_safe_etkdg(smis: list[str], seed: int = 42) -> tuple[float, int]:
    """rocmolkit.safe.embed_molecule per-mol (subprocess+retry)."""
    from rocmolkit.safe import embed_molecule

    confs = 0
    t0 = time.perf_counter()
    for smi in smis:
        m = AddHs(Chem.MolFromSmiles(smi))
        try:
            embed_molecule(m, seed=seed, max_retries=10)
            confs += 1
        except Exception as e:
            print(f"  warn: {smi}: {e}", file=sys.stderr)
    return time.perf_counter() - t0, confs


def rocmolkit_gpu_safe_etkdg_mmff(smis: list[str], seed: int = 42) -> tuple[float, int]:
    """rocmolkit.safe full pipeline: embed_molecule + mmff_optimize_molecule."""
    from rocmolkit.safe import embed_molecule, mmff_optimize_molecule

    ok = 0
    t0 = time.perf_counter()
    for smi in smis:
        m = AddHs(Chem.MolFromSmiles(smi))
        try:
            embed_molecule(m, seed=seed, max_retries=10)
            mmff_optimize_molecule(m, max_iters=200)
            ok += 1
        except Exception as e:
            print(f"  warn: {smi}: {e}", file=sys.stderr)
    return time.perf_counter() - t0, ok


def rocmolkit_gpu_direct_etkdg_batch(smis: list[str], batch_size: int, seed: int = 42) -> tuple[float, int]:
    """Direct binding in batches of ``batch_size`` (no subprocess overhead).

    Limited to batch_size <= 3 by the open ROCm 7.2.3 + gfx1200 segfault;
    batches >= 4 are documented to crash. The benchmark uses batch=3 so the
    GPU does see real batched work without hitting the bug.
    """
    from rocmolkit._embedMolecules import EmbedMolecules

    p = ETKDGv3()
    p.useRandomCoords = True
    p.randomSeed = seed

    confs = 0
    t0 = time.perf_counter()
    for i in range(0, len(smis), batch_size):
        chunk = smis[i:i + batch_size]
        mols = [AddHs(Chem.MolFromSmiles(s)) for s in chunk]
        try:
            EmbedMolecules(mols, p, 1)
            confs += sum(1 for m in mols if m.GetNumConformers() > 0)
        except Exception as e:
            print(f"  warn batch {i//batch_size}: {e}", file=sys.stderr)
    return time.perf_counter() - t0, confs


def rocmolkit_gpu_direct_mmff_batch(smis: list[str], batch_size: int = 30, seed: int = 42) -> tuple[float, int]:
    """Direct MMFF batch — works well up to batch_size=30."""
    from rocmolkit import _embedMolecules  # registers BatchHardwareOptions
    from rocmolkit._mmffOptimization import MMFFOptimizeMoleculesConfs

    # Need conformers first; use RDKit CPU to embed (cheap) so we isolate MMFF.
    mols = []
    for smi in smis:
        m = AddHs(Chem.MolFromSmiles(smi))
        if AllChem.EmbedMolecule(m, randomSeed=seed) >= 0:
            mols.append(m)

    ok = 0
    t0 = time.perf_counter()
    for i in range(0, len(mols), batch_size):
        chunk = mols[i:i + batch_size]
        try:
            MMFFOptimizeMoleculesConfs(chunk, maxIters=200)
            ok += len(chunk)
        except Exception as e:
            print(f"  warn batch {i//batch_size}: {e}", file=sys.stderr)
    return time.perf_counter() - t0, ok


def fmt_row(name: str, t: float, n_ok: int, n_total: int) -> str:
    if n_ok == 0:
        return f"| {name} | — | — | 0/{n_total} |"
    per_mol = t / n_ok * 1000
    throughput = n_ok / t
    return f"| {name} | {t:.2f} s | {per_mol:.1f} ms/mol ({throughput:.1f} mol/s) | {n_ok}/{n_total} |"


def run_phase(label: str, fn, *args, **kwargs):
    """Run one benchmark phase with isolation prints + crash recovery."""
    print(f"\n[{label}] starting…", flush=True)
    try:
        t, ok = fn(*args, **kwargs)
        print(f"[{label}] OK: {t:.2f}s, {ok} succeeded", flush=True)
        return t, ok
    except Exception as e:
        print(f"[{label}] CRASHED: {e}", flush=True)
        return None, 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("smiles_file", type=Path)
    ap.add_argument("--n", type=int, default=None, help="Cap molecule count")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip", action="append", default=[],
                    help="Skip a phase by name (cpu_etkdg, safe_etkdg, direct_etkdg_batch, "
                         "cpu_mmff, gpu_mmff_batch). Can repeat.")
    args = ap.parse_args()

    smis = load_smiles(args.smiles_file, args.n)
    n = len(smis)
    print(f"# rocMolKit benchmark — {n} molecules from {args.smiles_file.name}", flush=True)

    rows: list[tuple[str, str]] = []  # (section, table-row)

    if "cpu_etkdg" not in args.skip:
        t, c = run_phase("RDKit CPU ETKDG", rdkit_cpu_etkdg, smis, args.seed)
        if t is not None:
            rows.append(("ETKDG", fmt_row("RDKit CPU (sequential)", t, c, n)))

    if "safe_etkdg" not in args.skip:
        t, c = run_phase("rocMolKit GPU safe ETKDG", rocmolkit_gpu_safe_etkdg, smis, args.seed)
        if t is not None:
            rows.append(("ETKDG", fmt_row("rocMolKit GPU `safe.embed_molecule` (per-mol subprocess)", t, c, n)))

    if "direct_etkdg_batch" not in args.skip:
        t, c = run_phase("rocMolKit GPU direct ETKDG batch=3",
                         rocmolkit_gpu_direct_etkdg_batch, smis, 3, args.seed)
        if t is not None:
            rows.append(("ETKDG", fmt_row("rocMolKit GPU direct `EmbedMolecules`, batch=3", t, c, n)))

    if "cpu_mmff" not in args.skip:
        t, c = run_phase("RDKit CPU ETKDG+MMFF", rdkit_cpu_etkdg_mmff, smis, args.seed)
        if t is not None:
            rows.append(("ETKDG + MMFF94", fmt_row("RDKit CPU (sequential)", t, c, n)))

    if "gpu_mmff_batch" not in args.skip:
        t, c = run_phase("rocMolKit GPU MMFF batch=30",
                         rocmolkit_gpu_direct_mmff_batch, smis, 30, args.seed)
        if t is not None:
            rows.append(("MMFF94", fmt_row("rocMolKit GPU direct `MMFFOptimizeMoleculesConfs`, batch=30", t, c, n)))

    print("\n\n--- RESULTS ---\n", flush=True)
    section = None
    for sec, row in rows:
        if sec != section:
            print(f"\n### {sec}\n")
            print("| Pipeline | Wall time | Per-molecule | Success |")
            print("|---|---|---|---|")
            section = sec
        print(row)

    print(f"\n_Hardware: AMD Ryzen 5 7600 (12 threads) + AMD Radeon RX 9060 XT (gfx1200, 32 CUs, RDNA4) on ROCm 7.2.3._")
    print(f"_Dataset: {n} molecules from {args.smiles_file.name}._")


if __name__ == "__main__":
    main()
