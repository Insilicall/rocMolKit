import numpy as np
from rdkit import Chem
from rocmolkit._substructure import SubstructSearchConfig, countSubstructMatches
smis=[l.split()[0] for l in open("tests/data/druglike_100.smi") if l.strip() and not l.startswith("#")]
targets=[m for m in (Chem.MolFromSmiles(s) for s in smis) if m][:60]
patterns=["c1ccccc1","C=O","C(=O)O","C(=O)N","[OH]","[NH2]","c1ccncc1","CCC","C(=O)[O;H1]","S","F","Cl","N","c1ccccc1C"]
queries=[Chem.MolFromSmarts(p) for p in patterns]
cfg=SubstructSearchConfig(); cfg.gpuIds=[0]
counts=np.asarray(countSubstructMatches(targets, queries, cfg))
print("counts shape:", counts.shape)
npair=nmatch_diff=0
for t in range(len(targets)):
    for q in range(len(queries)):
        rd = targets[t].HasSubstructMatch(queries[q])
        gpu = counts[t][q] > 0
        npair+=1
        if rd != gpu: nmatch_diff+=1
print(f"hasMatch pairs: {npair}  mismatches(GPU vs RDKit HasSubstructMatch): {nmatch_diff}")
print("RESULT:", "ALL MATCH ✅" if nmatch_diff==0 else f"❌ {nmatch_diff} differ")
