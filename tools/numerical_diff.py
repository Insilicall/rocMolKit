"""Compara saídas de rocMolKit vs RDKit (e opcionalmente nvMolKit).

Usado pelos testes de paridade (Fase 2 e 3) e pelo CI gate.

ETKDG: RMSD entre conformers gerados, alinhados.
MMFF:  diferença relativa de energia.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def etkdg_rmsd(smiles_file: Path, n_confs: int = 10, threshold: float = 0.5) -> int:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.Chem.rdMolAlign import GetBestRMS

    try:
        import rocmolkit
        from rocmolkit.embedMolecules import EmbedMolecules as rocEmbed
    except ImportError:
        print("rocmolkit não instalado — pulando", file=sys.stderr)
        return 0

    smis = [s.strip() for s in smiles_file.read_text().splitlines() if s.strip()]
    failures = 0

    for smi in smis:
        m_rd = Chem.AddHs(Chem.MolFromSmiles(smi))
        m_roc = Chem.AddHs(Chem.MolFromSmiles(smi))

        params = AllChem.ETKDGv3()
        params.useRandomCoords = True
        params.randomSeed = 42

        AllChem.EmbedMultipleConfs(m_rd, numConfs=n_confs, params=params)
        rocEmbed(molecules=[m_roc], params=params, confsPerMolecule=n_confs)

        if m_roc.GetNumConformers() == 0:
            print(f"FAIL no-confs: {smi}")
            failures += 1
            continue

        # Pega o melhor RMSD entre qualquer par roc/rdkit
        best = min(
            GetBestRMS(m_rd, m_roc, refId=i, prbId=j)
            for i in range(m_rd.GetNumConformers())
            for j in range(m_roc.GetNumConformers())
        )
        if best > threshold:
            print(f"FAIL rmsd={best:.3f} > {threshold}: {smi}")
            failures += 1

    print(f"etkdg_rmsd: {len(smis) - failures}/{len(smis)} ok")
    return failures


def mmff_energy(smiles_file: Path, rel_tol: float = 1e-4) -> int:
    from rdkit import Chem
    from rdkit.Chem import AllChem

    try:
        from rocmolkit.mmffOptimization import MMFFOptimizeMoleculesConfs as rocOpt
    except ImportError:
        print("rocmolkit.mmffOptimization indisponível — pulando", file=sys.stderr)
        return 0

    smis = [s.strip() for s in smiles_file.read_text().splitlines() if s.strip()]
    failures = 0

    for smi in smis:
        m = Chem.AddHs(Chem.MolFromSmiles(smi))
        AllChem.EmbedMolecule(m, randomSeed=42)

        rd_props = AllChem.MMFFGetMoleculeProperties(m)
        rd_ff = AllChem.MMFFGetMoleculeForceField(m, rd_props)
        rd_ff.Minimize()
        e_rd = rd_ff.CalcEnergy()

        m2 = Chem.Mol(m)
        rocOpt([m2])
        roc_ff = AllChem.MMFFGetMoleculeForceField(m2, AllChem.MMFFGetMoleculeProperties(m2))
        e_roc = roc_ff.CalcEnergy()

        rel = abs(e_roc - e_rd) / max(abs(e_rd), 1e-6)
        if rel > rel_tol:
            print(f"FAIL E_rel={rel:.2e} (rd={e_rd:.4f} roc={e_roc:.4f}): {smi}")
            failures += 1

    print(f"mmff_energy: {len(smis) - failures}/{len(smis)} ok")
    return failures


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("kind", choices=["etkdg", "mmff"])
    p.add_argument("smiles_file", type=Path)
    p.add_argument("--threshold", type=float, default=None)
    args = p.parse_args()

    if args.kind == "etkdg":
        return etkdg_rmsd(args.smiles_file, threshold=args.threshold or 0.5)
    return mmff_energy(args.smiles_file, rel_tol=args.threshold or 1e-4)


if __name__ == "__main__":
    sys.exit(0 if main() == 0 else 1)
