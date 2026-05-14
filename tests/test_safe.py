"""Regression tests for rocmolkit.safe — the subprocess+retry wrappers.

These exist so that if anyone refactors the workaround they get a loud
failure when the bug returns. Marked ``gpu`` because they exercise the
real GPU pipeline; CI without a self-hosted runner skips them.
"""

from __future__ import annotations

import pytest


@pytest.mark.gpu
def test_safe_module_imports() -> None:
    """The safe module loads without touching HIP."""
    from rocmolkit import safe

    assert hasattr(safe, "embed_molecule")
    assert hasattr(safe, "embed_molecules")
    assert hasattr(safe, "mmff_optimize_molecule")
    assert hasattr(safe, "mmff_optimize_molecules")
    assert issubclass(safe.EmbedFailure, RuntimeError)


@pytest.mark.gpu
def test_embed_single_mol_returns_conformer() -> None:
    """End-to-end: ETKDG embed produces a conformer with sensible coordinates."""
    from rdkit import Chem
    from rdkit.Chem import AddHs

    from rocmolkit.safe import embed_molecule

    m = AddHs(Chem.MolFromSmiles("CCO"))
    embed_molecule(m, seed=42)

    assert m.GetNumConformers() == 1
    conf = m.GetConformer(0)
    # Sanity: all atoms placed at distinct positions.
    positions = {(round(conf.GetAtomPosition(i).x, 3),
                  round(conf.GetAtomPosition(i).y, 3),
                  round(conf.GetAtomPosition(i).z, 3))
                 for i in range(m.GetNumAtoms())}
    assert len(positions) == m.GetNumAtoms(), "atoms collapsed onto each other"


@pytest.mark.gpu
def test_embed_molecules_list() -> None:
    """The list helper handles multiple mols sequentially.

    Retries are bumped above the library default because CI is sensitive
    to the long tail: at ~35% per-call SIGSEGV the joint probability of
    8 retries failing for any one of 3 mols is ~0.07%, which still
    surfaces as flake every few hundred CI runs.
    """
    from rdkit import Chem
    from rdkit.Chem import AddHs

    from rocmolkit.safe import embed_molecules

    mols = [AddHs(Chem.MolFromSmiles(s)) for s in ("CCO", "c1ccccc1", "CC(=O)O")]
    embed_molecules(mols, seed=42, max_retries=15)

    for m in mols:
        assert m.GetNumConformers() == 1


@pytest.mark.gpu
def test_mmff_after_embed_lowers_energy() -> None:
    """MMFF should lower (or at least not raise) the post-embed energy."""
    from rdkit import Chem
    from rdkit.Chem import AddHs

    from rocmolkit.safe import embed_molecule, mmff_optimize_molecule

    m = AddHs(Chem.MolFromSmiles("CCO"))
    embed_molecule(m, seed=42)
    energies = mmff_optimize_molecule(m, max_iters=200)

    assert len(energies) == 1
    # Ethanol global min is around -1.34 kcal/mol; require it converges to a
    # negative energy (any minimisation should beat the random embed).
    assert energies[0] < 0.0, f"MMFF did not lower energy below 0: {energies[0]}"


@pytest.mark.gpu
def test_uff_after_embed_returns_energy() -> None:
    """UFF should return a finite per-conformer energy."""
    import math

    from rdkit import Chem
    from rdkit.Chem import AddHs

    from rocmolkit.safe import embed_molecule, uff_optimize_molecule

    m = AddHs(Chem.MolFromSmiles("CCO"))
    embed_molecule(m, seed=42)
    energies = uff_optimize_molecule(m, max_iters=200)

    assert len(energies) == 1
    assert math.isfinite(energies[0]), f"UFF returned non-finite energy {energies[0]}"


@pytest.mark.gpu
def test_embed_failure_type_is_runtime_error() -> None:
    """``EmbedFailure`` is a RuntimeError so callers can catch it broadly."""
    from rocmolkit.safe import EmbedFailure

    with pytest.raises(RuntimeError):
        raise EmbedFailure("synthetic")
