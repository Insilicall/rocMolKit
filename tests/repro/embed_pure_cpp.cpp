// Pure C++ ETKDG driver - no boost-python, no Python.
// Calls nvMolKit::embedMolecules directly with RDKit C++ ROMol*.
//
// If this crashes intermittently → bug is in nvMolKit's expectation of
// how molecules are passed (lifetime, ownership).
// If this never crashes → bug is in the boost-python conversion layer
// in nvmolkit/embedMolecules.cpp.

#include <cstdio>
#include <vector>
#include <memory>

#include <GraphMol/RDKitBase.h>
#include <GraphMol/SmilesParse/SmilesParse.h>
#include <GraphMol/MolOps.h>
#include <GraphMol/DistGeomHelpers/Embedder.h>

#include "etkdg.h"

int main(int argc, char** argv) {
    const int n_iters = (argc > 1) ? std::atoi(argv[1]) : 20;
    const char* smi = (argc > 2) ? argv[2] : "CC(=O)Oc1ccccc1C(=O)O";  // aspirin

    int ok = 0, fail = 0;
    for (int i = 0; i < n_iters; ++i) {
        try {
            std::unique_ptr<RDKit::ROMol> ro(RDKit::SmilesToMol(smi));
            if (!ro) { std::fprintf(stderr, "iter %d: SMILES parse failed\n", i); ++fail; continue; }
            std::unique_ptr<RDKit::ROMol> roH(RDKit::MolOps::addHs(*ro));

            RDKit::DGeomHelpers::EmbedParameters params = RDKit::DGeomHelpers::ETKDGv3;
            params.useRandomCoords = true;
            params.randomSeed = 42 + i;

            std::vector<RDKit::ROMol*> mols{roH.get()};
            nvMolKit::embedMolecules(mols, params, /*confsPerMolecule=*/1);

            int n_confs = roH->getNumConformers();
            if (n_confs > 0) ++ok;
            else { std::fprintf(stderr, "iter %d: no conformers generated\n", i); ++fail; }
        } catch (const std::exception& e) {
            std::fprintf(stderr, "iter %d: EXCEPTION %s\n", i, e.what());
            ++fail;
        }
    }
    std::printf("=== %d OK / %d FAIL on '%s' ===\n", ok, fail, smi);
    return (fail > 0) ? 1 : 0;
}
