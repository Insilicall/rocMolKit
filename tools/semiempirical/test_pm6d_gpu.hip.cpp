// SPDX-FileCopyrightText: Copyright (c) 2025 InsilicAll. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// GPU validation for the PM6_D (d-orbital) SCF: runs the on-device batched
// engine (scfBatchDGpu) on the YH-scope molecules and checks the charges + heat
// of formation bit-exact against the host CPU engine (already validated against
// the oracle golden) and against the frozen oracle values. Closes the 100%-GPU
// PM6_D path. Build/run in the rocmolkit:devel container (gfx1200):
//
//   hipcc -std=c++17 -O2 -I rocmolkit/src/semiempirical \
//     tools/semiempirical/test_pm6d_gpu.hip.cpp \
//     rocmolkit/src/semiempirical/scf_d_kernels.hip.cpp \
//     rocmolkit/src/semiempirical/{core_hamiltonian,pm6_params,overlap}.cpp -o /tmp/pm6d_gpu

#include <cmath>
#include <cstdio>
#include <vector>

#include "core_hamiltonian.h"
#include "core_hamiltonian_d_device.h"
#include "energy_device.h"
#include "pm6_params.h"
#include "scf_d_device.h"
#include "scf_d_kernels.h"

using namespace nvMolKit::semiempirical;

namespace {

struct Mol {
  const char* name;
  std::vector<int> z;
  std::vector<double> coords;  // 3*nAtoms
  std::vector<double> qGold;   // oracle charges
  double hofGold;              // oracle HoF (kcal/mol)
};

// Host CPU PM6_D SCF (same validated headers) for the GPU==CPU bit-exact check.
void cpuRun(const Mol& mol, std::vector<double>& q, double& hof, int& conv) {
  const int n = static_cast<int>(mol.z.size());
  std::vector<AtomIntParams> ap(n);
  std::vector<int> start(n), norb(n);
  int nBasis = 0, nElec = 0;
  for (int a = 0; a < n; ++a) {
    gatherAtomIntParamsD(mol.z[a], ap[a]);
    start[a] = nBasis;
    norb[a] = ap[a].nOrb;
    nBasis += norb[a];
    nElec += pm6ValenceElectrons(mol.z[a]);
  }
  const int n2 = nBasis * nBasis;
  std::vector<double> H(n2), density(n2), eval(nBasis), F(n2), eigA(n2), C(n2), Pnew(n2),
      ecom(n2), diisF(kScfDiisMax * n2), diisE(kScfDiisMax * n2);
  buildCoreHamiltonianDDev(nBasis, n, ap.data(), start.data(), norb.data(), mol.coords.data(),
                           H.data());
  int niter = 0;
  double eElec = 0.0;
  const int nPairs = n * (n - 1) / 2;
  std::vector<int> intMeta(static_cast<size_t>(nPairs) * kPairMetaInts);
  std::vector<double> intBlob(static_cast<size_t>(pm6dIntCacheDoubles(n, norb.data())));
  scfLoopDDev(nBasis, n, ap.data(), start.data(), norb.data(), mol.coords.data(), H.data(),
              nElec / 2, 800, 1e-10, density.data(), eval.data(), F.data(), eigA.data(),
              C.data(), Pnew.data(), ecom.data(), diisF.data(), diisE.data(), &conv, &niter,
              &eElec, intMeta.data(), intBlob.data());
  const double eNuc = nuclearRepulsionAm1Dev(n, ap.data(), mol.coords.data());
  hof = heatOfFormationKcalDev(eElec, eNuc, n, ap.data());
  q.assign(n, 0.0);
  for (int a = 0; a < n; ++a) {
    double pop = 0.0;
    for (int o = 0; o < norb[a]; ++o) pop += density[(start[a] + o) * nBasis + (start[a] + o)];
    q[a] = static_cast<double>(pm6ValenceElectrons(mol.z[a])) - pop;
  }
}

}  // namespace

int main() {
  std::vector<Mol> mols = {
      {"H2S", {16, 1, 1}, {0, 0, 0, 0.9686, 0, 0.9269, -0.9686, 0, 0.9269},
       {-0.3617, 0.1809, 0.1809}, -61.8634},
      {"PH3", {15, 1, 1, 1},
       {0, 0, 0, 1.1932, 0, 0.77, -0.5966, 1.0333, 0.77, -0.5966, -1.0333, 0.77},
       {-0.0362, 0.0121, 0.0121, 0.0121}, -217.3812},
      {"HCl", {17, 1}, {0, 0, 0, 0, 0, 1.2746}, {-0.2162, 0.2162}, -70.0817},
      {"H2CS", {16, 6, 1, 1},
       {0, 0, 0, 0, 0, 1.61, 0, 0.9281, 2.1944, 0, -0.9281, 2.1944},
       {-0.1241, -0.2028, 0.1634, 0.1634}, -48.8769},  // YX (C-S)
      {"Cl2", {17, 17}, {0, 0, 0, 0, 0, 1.988}, {0.0, 0.0}, -58.9368},  // YY (Cl-Cl)
      {"HBr", {35, 1}, {0, 0, 0, 0, 0, 1.41}, {-0.1868, 0.1868}, -75.4900},  // Br qn4
      {"HI", {53, 1}, {0, 0, 0, 0, 0, 1.609}, {-0.1626, 0.1626}, -34.6896},  // I qn5
      {"CH3I", {53, 6, 1, 1, 1},
       {0, 0, 2.139, 0, 0, 0, 1.028, 0, -0.363, -0.514, 0.890, -0.363, -0.514, -0.890, -0.363},
       {-0.0485, -0.5437, 0.1974, 0.1974, 0.1974}, -140.4681},  // YX (DIIS)
      {"Br2", {35, 35}, {0, 0, 0, 0, 0, 2.28}, {0.0, 0.0}, -105.4548},  // YY (DIIS)
  };

  // Concatenate for the batched GPU call.
  const int nMol = static_cast<int>(mols.size());
  std::vector<int> molNAtoms(nMol), molNBasis(nMol), atomsAll;
  std::vector<double> coordsAll;
  for (int m = 0; m < nMol; ++m) {
    molNAtoms[m] = static_cast<int>(mols[m].z.size());
    int nb = 0;
    for (int z : mols[m].z) nb += pm6NumOrbitals(z);
    molNBasis[m] = nb;
    for (int z : mols[m].z) atomsAll.push_back(z);
    for (double c : mols[m].coords) coordsAll.push_back(c);
  }
  std::vector<double> chargesAll(atomsAll.size()), hofAll(nMol);
  std::vector<int> convAll(nMol);

  if (!scfBatchDGpu(nMol, molNAtoms.data(), molNBasis.data(), atomsAll.data(),
                    coordsAll.data(), chargesAll.data(), hofAll.data(), convAll.data())) {
    std::printf("scfBatchDGpu returned false (unsupported element?)\n");
    return 1;
  }

  double worstGC = 0.0, worstGH = 0.0, worstOC = 0.0, worstOH = 0.0;
  int fails = 0, off = 0;
  for (int m = 0; m < nMol; ++m) {
    std::vector<double> qc;
    double hofc = 0.0;
    int convc = 0;
    cpuRun(mols[m], qc, hofc, convc);

    const int na = molNAtoms[m];
    double gc = 0.0, oc = 0.0;
    std::printf("%-5s conv(gpu/cpu)=%d/%d  HoF gpu=%.4f cpu=%.4f oracle=%.4f  q=[",
                mols[m].name, convAll[m], convc, hofAll[m], hofc, mols[m].hofGold);
    for (int a = 0; a < na; ++a) {
      const double qg = chargesAll[off + a];
      gc = std::fmax(gc, std::fabs(qg - qc[a]));
      oc = std::fmax(oc, std::fabs(qg - mols[m].qGold[a]));
      std::printf("%s%.4f", a ? " " : "", qg);
    }
    const double gh = std::fabs(hofAll[m] - hofc);
    const double oh = std::fabs(hofAll[m] - mols[m].hofGold);
    worstGC = std::fmax(worstGC, gc); worstGH = std::fmax(worstGH, gh);
    worstOC = std::fmax(worstOC, oc); worstOH = std::fmax(worstOH, oh);
    const bool okm = convAll[m] == 1 && gc < 1e-12 && gh < 1e-9 && oc < 1e-4 && oh < 1e-3;
    fails += okm ? 0 : 1;
    std::printf("]  %s\n", okm ? "OK" : "** FAIL");
    off += na;
  }

  std::printf("\nGPU vs CPU: worst |dq|=%.2e  |dHoF|=%.2e   "
              "GPU vs oracle: worst |dq|=%.2e  |dHoF|=%.2e\n",
              worstGC, worstGH, worstOC, worstOH);
  std::printf("%s\n", fails == 0
              ? "OK — 100% GPU PM6_D SCF bit-exact to CPU and matches the oracle golden"
              : "** FAIL");
  return fails == 0 ? 0 : 1;
}
