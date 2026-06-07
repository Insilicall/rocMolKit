#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 InsilicAll. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reproducible PM6_D GPU batch-SCF *throughput* benchmark on real molecules.

SECONDARY / perf-only tool — correctness lives in tools/semiempirical/validate_*.
This measures how fast the on-device batched SCF (scfBatchDGpu) chews through a
batch of drug-like molecules vs the CPU single-molecule path (pm6dCharges), on
an AMD gfx1200 via the rocmolkit devel container.

Pipeline:
  1. Parse a curated list of representative drug SMILES, keep only molecules
     whose elements are all PM6_D-supported, add explicit Hs, embed a 3D
     conformer with ETKDG, and replicate (with small coordinate jitter) up to a
     target batch size. Molecules are grouped/reported by atom-count bucket.
  2. Serialize the batch to a flat binary file (see bench_pm6d_driver.cpp).
  3. hipcc-compile + run the driver inside the container (GPU batch SCF + CPU
     loop), parse the throughput JSON, and emit a Markdown + JSON report.

Run (from repo root, host with an AMD GPU + the devel image):

    docker run --rm --device /dev/kfd --device /dev/dri -e HIP_VISIBLE_DEVICES=0 \
      -v "$PWD":/work -w /work rocmolkit:devel-local \
      python3 tools/semiempirical/benchmark/bench_pm6d_gpu.py --target 600

Or let the script invoke docker itself (run on the host, image via --image /
$DOCKER_IMAGE). Use --target to size the batch and --no-cpu to skip the (slow)
CPU baseline.
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent
SRC = ROOT / "rocmolkit" / "src" / "semiempirical"

# PM6_D supported (bit-exact to MOPAC) — keep the generator inside this set.
SUPPORTED = {
    1: "H", 5: "B", 6: "C", 7: "N", 8: "O", 9: "F", 13: "Al", 14: "Si", 15: "P",
    16: "S", 17: "Cl", 21: "Sc", 30: "Zn", 31: "Ga", 32: "Ge", 35: "Br", 48: "Cd",
    50: "Sn", 53: "I", 80: "Hg",
}

# ~40 representative drug-like SMILES spanning H/C/N/O/F/S/Cl/Br + a few P/Si/Zn.
DRUG_SMILES = [
    ("aspirin", "CC(=O)Oc1ccccc1C(=O)O"),
    ("caffeine", "Cn1cnc2c1c(=O)n(C)c(=O)n2C"),
    ("ibuprofen", "CC(C)Cc1ccc(C(C)C(=O)O)cc1"),
    ("paracetamol", "CC(=O)Nc1ccc(O)cc1"),
    ("sulfamethoxazole", "Cc1cc(NS(=O)(=O)c2ccc(N)cc2)no1"),
    ("chlorpromazine", "CN(C)CCCN1c2ccccc2Sc2ccc(Cl)cc21"),
    ("fluoxetine", "CNCCC(Oc1ccc(C(F)(F)F)cc1)c1ccccc1"),
    ("diazepam", "CN1C(=O)CN=C(c2ccccc2)c2cc(Cl)ccc21"),
    ("naproxen", "COc1ccc2cc(C(C)C(=O)O)ccc2c1"),
    ("metformin", "CN(C)C(=N)N=C(N)N"),
    ("warfarin", "CC(=O)CC(c1ccccc1)c1c(O)c2ccccc2oc1=O"),
    ("atorvastatin_core", "CC(C)c1c(C(=O)Nc2ccccc2)c(-c2ccccc2)c(-c2ccc(F)cc2)n1CCC(O)CC(O)CC(=O)O"),
    ("salbutamol", "CC(C)(C)NCC(O)c1ccc(O)c(CO)c1"),
    ("propranolol", "CC(C)NCC(O)COc1cccc2ccccc12"),
    ("amoxicillin", "CC1(C)SC2C(NC(=O)C(N)c3ccc(O)cc3)C(=O)N2C1C(=O)O"),
    ("penicillin_g", "CC1(C)SC2C(NC(=O)Cc3ccccc3)C(=O)N2C1C(=O)O"),
    ("ciprofloxacin", "OC(=O)c1cn(C2CC2)c2cc(N3CCNCC3)c(F)cc2c1=O"),
    ("omeprazole", "COc1ccc2[nH]c(S(=O)Cc3ncc(C)c(OC)c3C)nc2c1"),
    ("ranitidine", "CNC(=C[N+](=O)[O-])NCCSCc1ccc(CN(C)C)o1"),
    ("metronidazole", "Cc1ncc([N+](=O)[O-])n1CCO"),
    ("acetaminophen_dimer", "CC(=O)Nc1ccc(Oc2ccc(NC(C)=O)cc2)cc1"),
    ("nicotine", "CN1CCCC1c1cccnc1"),
    ("serotonin", "NCCc1c[nH]c2ccc(O)cc12"),
    ("dopamine", "NCCc1ccc(O)c(O)c1"),
    ("phenylalanine", "N[C@@H](Cc1ccccc1)C(=O)O"),
    ("tryptophan", "N[C@@H](Cc1c[nH]c2ccccc12)C(=O)O"),
    ("histidine", "N[C@@H](Cc1c[nH]cn1)C(=O)O"),
    ("glucose", "OCC1OC(O)C(O)C(O)C1O"),
    ("uracil", "O=c1cc[nH]c(=O)[nH]1"),
    ("thymine", "Cc1c[nH]c(=O)[nH]c1=O"),
    ("imatinib_frag", "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1"),
    ("celecoxib", "Cc1ccc(-c2cc(C(F)(F)F)nn2-c2ccc(S(N)(=O)=O)cc2)cc1"),
    ("ketoprofen", "CC(C(=O)O)c1cccc(C(=O)c2ccccc2)c1"),
    ("diclofenac", "O=C(O)Cc1ccccc1Nc1c(Cl)cccc1Cl"),
    ("bromhexine", "CN(C1CCCCC1)Cc1cc(Br)cc(Br)c1N"),
    ("4_bromoaniline", "Nc1ccc(Br)cc1"),
    ("triethyl_phosphate", "CCOP(=O)(OCC)OCC"),
    ("dimethyl_silanediol", "C[Si](C)(O)O"),
    ("phenyltrimethoxysilane", "CO[Si](OC)(OC)c1ccccc1"),
    ("zinc_acetate", "CC(=O)[O-].CC(=O)[O-].[Zn+2]"),
    ("benzene", "c1ccccc1"),
    ("toluene", "Cc1ccccc1"),
    ("aniline", "Nc1ccccc1"),
    ("phenol", "Oc1ccccc1"),
    ("methanol", "CO"),
    ("ethanol", "CCO"),
]


def build_molecules(target, seed, max_atoms=0):
    from rdkit import Chem
    from rdkit.Chem import AllChem
    import numpy as np

    base = []  # (name, atoms[list int], coords[np (N,3)], charge)
    skipped = []
    for name, smi in DRUG_SMILES:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            skipped.append((name, "parse"))
            continue
        # Disconnected (salts) — the engine is single-molecule; skip multi-frag.
        if len(Chem.GetMolFrags(mol)) > 1:
            skipped.append((name, "multi-fragment"))
            continue
        molH = Chem.AddHs(mol)
        zs = [a.GetAtomicNum() for a in molH.GetAtoms()]
        bad = sorted({z for z in zs if z not in SUPPORTED})
        if bad:
            skipped.append((name, "unsupported " + ",".join(map(str, bad))))
            continue
        if sum(zs) % 2 != (Chem.GetFormalCharge(molH) % 2):
            # closed-shell electron count = sum(valence Z) - charge must be even;
            # the engine rejects odd-electron systems. Drop radicals/odd species.
            pass
        params = AllChem.ETKDGv3()
        params.randomSeed = seed
        if AllChem.EmbedMolecule(molH, params) != 0:
            skipped.append((name, "embed"))
            continue
        conf = molH.GetConformer()
        coords = np.array([[conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y,
                            conf.GetAtomPosition(i).z] for i in range(molH.GetNumAtoms())])
        charge = Chem.GetFormalCharge(molH)
        # closed-shell requirement: total electrons even
        nelec = sum(zs) - charge
        if nelec % 2 != 0:
            skipped.append((name, "open-shell"))
            continue
        if max_atoms and len(zs) > max_atoms:
            skipped.append((name, f"N={len(zs)}>max_atoms({max_atoms})"))
            continue
        base.append((name, zs, coords.astype(np.float64), charge))

    if not base:
        raise SystemExit("no usable base molecules")

    rng = np.random.default_rng(seed)
    batch = []
    i = 0
    while len(batch) < target:
        name, zs, coords, charge = base[i % len(base)]
        rep = i // len(base)
        c = coords.copy()
        if rep > 0:
            c = c + rng.normal(0.0, 0.01, size=c.shape)  # tiny jitter, distinct geometries
        batch.append((f"{name}#{rep}", zs, c, charge))
        i += 1
    return batch, base, skipped


def write_batch(batch, path):
    import numpy as np

    nMol = len(batch)
    molNAtoms = np.array([len(z) for _, z, _, _ in batch], dtype=np.int32)
    atomsAll = np.array([z for _, zs, _, _ in batch for z in zs], dtype=np.int32)
    coords = np.concatenate([c.reshape(-1) for _, _, c, _ in batch]).astype(np.float64)
    molCharge = np.array([ch for _, _, _, ch in batch], dtype=np.int32)
    totalAtoms = int(molNAtoms.sum())
    with open(path, "wb") as f:
        f.write(struct.pack("<ii", nMol, totalAtoms))
        molNAtoms.tofile(f)
        atomsAll.tofile(f)
        coords.tofile(f)
        molCharge.tofile(f)
    return totalAtoms


def size_buckets(batch):
    buckets = {"<20": [0, 0], "20-40": [0, 0], "40+": [0, 0]}
    for _, zs, _, _ in batch:
        n = len(zs)
        key = "<20" if n < 20 else ("20-40" if n <= 40 else "40+")
        buckets[key][0] += 1
        buckets[key][1] += n
    return buckets


def in_container():
    return subprocess.run(["bash", "-lc", "command -v hipcc"], capture_output=True).returncode == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=600, help="batch size (molecules)")
    ap.add_argument("--seed", type=int, default=0xC0FFEE)
    ap.add_argument("--gpu-repeats", type=int, default=5)
    ap.add_argument("--cpu-repeats", type=int, default=1)
    ap.add_argument("--no-cpu", action="store_true", help="skip CPU baseline (cpu-repeats=0)")
    ap.add_argument("--image", default=os.environ.get("DOCKER_IMAGE", "rocmolkit:devel-local"))
    ap.add_argument("--out", default=str(HERE / "results.json"))
    ap.add_argument("--max-atoms", type=int, default=0,
                    help="drop base molecules with more than this many atoms (0 = no cap). "
                         "The one-thread-per-molecule GPU SCF serializes on the largest "
                         "molecule, so a cap keeps wall time bounded.")
    args = ap.parse_args()
    if args.no_cpu:
        args.cpu_repeats = 0

    batch, base, skipped = build_molecules(args.target, args.seed, args.max_atoms)
    batch_bin = ROOT / "_pm6d_bench_batch.bin"
    totalAtoms = write_batch(batch, batch_bin)
    buckets = size_buckets(batch)

    print(f"[bench] base usable molecules: {len(base)}  (skipped {len(skipped)})")
    for n, why in skipped:
        print(f"        skip {n}: {why}")
    print(f"[bench] batch: {len(batch)} molecules, {totalAtoms} atoms")
    for k, (nm, na) in buckets.items():
        print(f"        bucket {k:>6}: {nm} mol, {na} atoms")

    src_list = "scf_d_kernels.hip.cpp scf_d.cpp core_hamiltonian.cpp pm6_params.cpp overlap.cpp"
    build = (
        f"S=rocmolkit/src/semiempirical; "
        f"hipcc -std=c++17 -O2 --offload-arch=gfx1200 -I$S "
        f"tools/semiempirical/benchmark/bench_pm6d_driver.cpp "
        + " ".join(f"$S/{s}" for s in src_list.split())
        + " -o /tmp/bench_pm6d && "
        f"/tmp/bench_pm6d _pm6d_bench_batch.bin {args.gpu_repeats} {args.cpu_repeats}"
    )

    if in_container():
        r = subprocess.run(["bash", "-lc", build], cwd=ROOT, capture_output=True, text=True)
    else:
        cmd = ["docker", "run", "--rm", "--device", "/dev/kfd", "--device", "/dev/dri",
               "-e", "HIP_VISIBLE_DEVICES=0", "-v", f"{ROOT}:/work", "-w", "/work",
               args.image, "bash", "-lc", build]
        r = subprocess.run(cmd, capture_output=True, text=True)

    sys.stdout.write(r.stdout)
    if r.returncode != 0 and "RESULT_JSON:" not in r.stdout:
        sys.stderr.write(r.stderr[-3000:])
        batch_bin.unlink(missing_ok=True)
        return r.returncode

    result = None
    for line in r.stdout.splitlines():
        if line.startswith("RESULT_JSON:"):
            result = json.loads(line[len("RESULT_JSON:"):])
    batch_bin.unlink(missing_ok=True)
    if result is None:
        sys.stderr.write(r.stderr[-3000:])
        return 1

    report = {
        "device": "gfx1200 (RDNA4)",
        "branch": "wt/bench-gpu",
        "engine": "PM6_D scfBatchDGpu (batched on-device SCF)",
        "n_base_molecules": len(base),
        "skipped": skipped,
        "buckets": {k: {"molecules": v[0], "atoms": v[1]} for k, v in buckets.items()},
        "gpu_repeats": args.gpu_repeats,
        "cpu_repeats": args.cpu_repeats,
        "seed": args.seed,
        "max_atoms": args.max_atoms,
        **result,
    }
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"\n[bench] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
