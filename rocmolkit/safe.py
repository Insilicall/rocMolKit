"""Subprocess-isolated wrappers for rocMolKit operations.

ROCm 7.2.3 + RDNA4 (gfx1200) has a non-deterministic SIGSEGV inside
``EmbedMolecules`` that we cannot pin down without ``rocgdb``
support for gfx1200 (see ``ISSUES.md``). The bug is bounded — a single
molecule call has roughly 65% per-attempt success rate — so a small
retry loop running each call in a fresh subprocess gives effectively
100% reliability.

This module exposes that pattern as a stable API. Use it when you need
deterministic conformer generation today; switch back to the direct
``EmbedMolecules`` binding once ROCm 7.3+ ships and the bug is fixed.

Cost per call: one ``fork`` + ``execve`` + ``import rdkit/rocmolkit``
(~150-300 ms on a warm cache), repeated until success or
``max_retries``. For a typical workload (8 retries, ~2.3 attempts mean)
this is ~600 ms per molecule — comparable to a CPU RDKit embed.
"""

from __future__ import annotations

import json
import subprocess
import sys
from typing import Optional

from rdkit import Chem
from rdkit.Geometry import Point3D


_WORKER = """
import sys, json
from rdkit import Chem
from rdkit.Chem import AddHs
from rdkit.Chem.rdDistGeom import ETKDGv3
from rocmolkit._embedMolecules import EmbedMolecules

mol_pickle_hex = sys.argv[1]
seed = int(sys.argv[2])

m = Chem.Mol(bytes.fromhex(mol_pickle_hex))

p = ETKDGv3()
p.useRandomCoords = True
p.randomSeed = seed

EmbedMolecules([m], p, 1)

if m.GetNumConformers() == 0:
    sys.exit(2)

c = m.GetConformer(0)
coords = [[c.GetAtomPosition(i).x, c.GetAtomPosition(i).y, c.GetAtomPosition(i).z]
          for i in range(m.GetNumAtoms())]
print(json.dumps({"coords": coords}))
"""


class EmbedFailure(RuntimeError):
    """Raised when subprocess retries are exhausted without success."""


def embed_molecule(
    mol: Chem.Mol,
    *,
    seed: int = 42,
    max_retries: int = 8,
    timeout: float = 30.0,
) -> Chem.Mol:
    """Generate one ETKDG conformer in a fresh subprocess, with retry.

    Args:
        mol: RDKit molecule. Hydrogens should already be added (call
            ``Chem.AddHs(mol)`` first if needed).
        seed: Random seed for ETKDGv3 (passed as ``randomSeed``).
        max_retries: Maximum number of subprocess attempts. Each
            subprocess crash counts as one attempt; the next attempt
            starts in a fresh process.
        timeout: Per-attempt timeout in seconds.

    Returns:
        The same ``mol`` with a Conformer added (in-place modification).

    Raises:
        EmbedFailure: When all attempts have been exhausted, either
            because every subprocess crashed or because ETKDG returned
            zero conformers (mol genuinely cannot be embedded).
    """
    pickle_hex = mol.ToBinary().hex()
    last_reason: Optional[str] = None

    for attempt in range(1, max_retries + 1):
        try:
            r = subprocess.run(
                [sys.executable, "-c", _WORKER, pickle_hex, str(seed)],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            last_reason = f"timeout after {timeout}s"
            continue

        if r.returncode == 2:
            # ETKDG explicitly produced zero conformers. Retrying won't help.
            raise EmbedFailure(
                f"ETKDG produced no conformer for molecule (attempt {attempt}); "
                "this is not the segfault bug — molecule may be unembeddable."
            )

        if r.returncode != 0 or not r.stdout.strip():
            last_reason = f"rc={r.returncode}"
            continue

        try:
            payload = json.loads(r.stdout.strip().splitlines()[-1])
        except (json.JSONDecodeError, IndexError):
            last_reason = f"bad output: {r.stdout!r}"
            continue

        coords = payload["coords"]
        conf = Chem.Conformer(mol.GetNumAtoms())
        for i, (x, y, z) in enumerate(coords):
            conf.SetAtomPosition(i, Point3D(x, y, z))
        mol.AddConformer(conf, assignId=True)
        return mol

    raise EmbedFailure(
        f"Exhausted {max_retries} subprocess attempts; last reason: {last_reason}. "
        "This is the known ROCm 7.2.3 + gfx1200 state-leak bug; see ISSUES.md."
    )


def embed_molecules(
    mols: list[Chem.Mol],
    *,
    seed: int = 42,
    max_retries: int = 8,
    timeout: float = 30.0,
) -> list[Chem.Mol]:
    """Embed a list of molecules, one subprocess per molecule.

    The direct ``EmbedMolecules([m1, m2, ...], ...)`` batch path is
    affected by a different threshold of the same bug (any batch with
    N>=4 molecules SIGSEGVs reliably). Calling this helper trades batch
    throughput for reliability — each molecule gets its own subprocess.

    Mutates each mol in-place (adds a Conformer); returns the list for
    chaining.
    """
    for m in mols:
        embed_molecule(m, seed=seed, max_retries=max_retries, timeout=timeout)
    return mols


_MMFF_WORKER = """
import sys, json
from rdkit import Chem
from rocmolkit import _embedMolecules  # registers BatchHardwareOptions converter
from rocmolkit._mmffOptimization import MMFFOptimizeMoleculesConfs

mol_pickle_hex = sys.argv[1]
max_iters = int(sys.argv[2])

m = Chem.Mol(bytes.fromhex(mol_pickle_hex))
if m.GetNumConformers() == 0:
    sys.exit(3)  # caller error: no conformer to optimize

energies = MMFFOptimizeMoleculesConfs([m], maxIters=max_iters)

# Pull back the optimised coordinates of conformer 0.
c = m.GetConformer(0)
coords = [[c.GetAtomPosition(i).x, c.GetAtomPosition(i).y, c.GetAtomPosition(i).z]
          for i in range(m.GetNumAtoms())]
print(json.dumps({"energies": energies[0], "coords": coords}))
"""


def mmff_optimize_molecule(
    mol: Chem.Mol,
    *,
    max_iters: int = 200,
    max_retries: int = 8,
    timeout: float = 30.0,
) -> list[float]:
    """Optimise an existing conformer with MMFF94 in a fresh subprocess.

    The mol must already have at least one Conformer (run ETKDG or
    ``embed_molecule`` first). This helper isolates each call in a
    subprocess for the same reason as ``embed_molecule``: MMFF stress
    runs of 30+ sequential calls reproduce the same state-leak SIGSEGV
    as ETKDG, just with lower per-call probability.

    Returns the list of energies (one per conformer) and updates the
    mol's conformer coordinates in-place.

    Raises ``EmbedFailure`` after ``max_retries`` segfaults.
    """
    if mol.GetNumConformers() == 0:
        raise ValueError(
            "mmff_optimize_molecule requires the mol to already have a "
            "Conformer; call embed_molecule(mol) or RDKit's EmbedMolecule first."
        )

    pickle_hex = mol.ToBinary().hex()
    last_reason: Optional[str] = None

    for attempt in range(1, max_retries + 1):
        try:
            r = subprocess.run(
                [sys.executable, "-c", _MMFF_WORKER, pickle_hex, str(max_iters)],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            last_reason = f"timeout after {timeout}s"
            continue

        if r.returncode == 3:
            raise ValueError("subprocess saw mol with 0 conformers")

        if r.returncode != 0 or not r.stdout.strip():
            last_reason = f"rc={r.returncode}"
            continue

        try:
            payload = json.loads(r.stdout.strip().splitlines()[-1])
        except (json.JSONDecodeError, IndexError):
            last_reason = f"bad output: {r.stdout!r}"
            continue

        # Update mol's coords with optimised ones.
        conf = mol.GetConformer(0)
        for i, (x, y, z) in enumerate(payload["coords"]):
            conf.SetAtomPosition(i, Point3D(x, y, z))
        return payload["energies"]

    raise EmbedFailure(
        f"Exhausted {max_retries} subprocess attempts; last reason: {last_reason}. "
        "This is the known ROCm 7.2.3 + gfx1200 state-leak bug; see ISSUES.md."
    )


def mmff_optimize_molecules(
    mols: list[Chem.Mol],
    *,
    max_iters: int = 200,
    max_retries: int = 8,
    timeout: float = 30.0,
) -> list[list[float]]:
    """Optimise a list of molecules, one subprocess per molecule.

    Returns a list of energy-lists (one per input mol). Each mol is
    updated in-place with its optimised coordinates.
    """
    return [
        mmff_optimize_molecule(
            m, max_iters=max_iters, max_retries=max_retries, timeout=timeout
        )
        for m in mols
    ]
