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
// Python binding for the PM6_D (d-orbital) NDDO semi-empirical engine. Exposes a
// batched entry that runs the whole d-orbital SCF on the GPU for a list of RDKit
// molecules (each with a 3D conformer) and returns the Mulliken charges + heat of
// formation per molecule. Module: _Semiempirical.

#include <GraphMol/Conformer.h>
#include <GraphMol/ROMol.h>

#include <boost/python.hpp>

#include <vector>

#include "semiempirical/scf_d_kernels.h"  // scfBatchDGpu
#include "semiempirical/scf_d.h"          // pm6dGradient, pm6dCharges
#include "semiempirical/pm6_params.h"     // pm6NumOrbitals, pm6ValenceElectrons
#include "semiempirical/h4_device.h"      // pm6dD3H4Correction

namespace {

using namespace boost::python;

// Per-molecule geometry + electronic bookkeeping pulled once from RDKit.
struct MolInfo {
  int na = 0;                  // atom count
  int nBasis = 0;              // PM6_D basis size
  int charge = 0;             // net formal charge
  int mult = 1;               // spin multiplicity (2S+1)
  bool openShell = false;     // route to the CPU UHF path
  bool hasD = false;          // any 9-orbital (d-bearing) atom
  std::vector<int> atoms;     // Z per atom
  std::vector<double> coords;  // 3*na, Angstrom
};

// Read one molecule's atoms/coords and classify it. The spin multiplicity is
// nRadical+1 (high spin) when RDKit carries radical electrons, else a doublet if
// the valence electron count is odd, else closed-shell. Open-shell molecules go
// to the CPU UHF path; the GPU batch handles closed-shell only.
MolInfo readMol(const RDKit::ROMol* mol) {
  MolInfo info;
  info.na = static_cast<int>(mol->getNumAtoms());
  const RDKit::Conformer& conf = mol->getConformer();  // throws if none
  int nElec = 0, nRad = 0;
  for (int a = 0; a < info.na; ++a) {
    const RDKit::Atom* atom = mol->getAtomWithIdx(a);
    const int z = static_cast<int>(atom->getAtomicNum());
    info.atoms.push_back(z);
    const int no = nvMolKit::semiempirical::pm6NumOrbitals(z);
    info.nBasis += no;
    if (no == 9) info.hasD = true;
    info.charge += atom->getFormalCharge();
    nElec += nvMolKit::semiempirical::pm6ValenceElectrons(z);
    nRad += static_cast<int>(atom->getNumRadicalElectrons());
    const RDGeom::Point3D& p = conf.getAtomPos(a);
    info.coords.push_back(p.x);
    info.coords.push_back(p.y);
    info.coords.push_back(p.z);
  }
  nElec -= info.charge;
  info.mult = nRad > 0 ? nRad + 1 : (nElec % 2 != 0 ? 2 : 1);
  info.openShell = info.mult > 1 || nElec % 2 != 0;
  return info;
}

// PM6_D charges + heats of formation for a list of RDKit molecules (each must
// carry a 3D conformer). Closed-shell molecules run on the GPU batch; open-shell
// (radical / odd-electron) molecules fall back to the CPU UHF path so radicals
// are usable through the binding. Returns a list of length len(mols); each
// element is either a tuple (charges, hof_nddo, hof_pm6, hof_d3h4) or None when
// the molecule is unsupported / did not converge.
boost::python::object pm6dChargesBatch(const boost::python::list& mols) {
  const int nMol = static_cast<int>(len(mols));
  std::vector<MolInfo> info(nMol);
  for (int m = 0; m < nMol; ++m)
    info[m] = readMol(extract<const RDKit::ROMol*>(boost::python::object(mols[m])));

  // Gather the closed-shell subset into one GPU batch (open-shell -> CPU below).
  std::vector<int> gpuMolIdx, molNAtoms, molNBasis, molCharge, atomsAll;
  std::vector<double> coordsAll;
  for (int m = 0; m < nMol; ++m) {
    if (info[m].openShell) continue;
    gpuMolIdx.push_back(m);
    molNAtoms.push_back(info[m].na);
    molNBasis.push_back(info[m].nBasis);
    molCharge.push_back(info[m].charge);
    atomsAll.insert(atomsAll.end(), info[m].atoms.begin(), info[m].atoms.end());
    coordsAll.insert(coordsAll.end(), info[m].coords.begin(), info[m].coords.end());
  }

  const int nGpu = static_cast<int>(gpuMolIdx.size());
  std::vector<double> chargesAll(atomsAll.size()), hofAll(nGpu), hofPm6All(nGpu);
  std::vector<int> convAll(nGpu, 0);
  bool gpuOk = true;
  if (nGpu > 0)
    gpuOk = nvMolKit::semiempirical::scfBatchDGpu(
        nGpu, molNAtoms.data(), molNBasis.data(), atomsAll.data(), coordsAll.data(),
        chargesAll.data(), hofAll.data(), convAll.data(), 800, 1e-10, hofPm6All.data(),
        molCharge.data());

  // Scatter the GPU results back into per-molecule slots (indexed by original m).
  std::vector<boost::python::object> result(nMol);
  for (int m = 0; m < nMol; ++m) result[m] = boost::python::object();  // None
  int off = 0;
  for (int g = 0; g < nGpu; ++g) {
    const int m = gpuMolIdx[g];
    const int na = info[m].na;
    if (gpuOk && convAll[g]) {
      boost::python::list q;
      for (int a = 0; a < na; ++a) q.append(chargesAll[off + a]);
      const double corr = nvMolKit::semiempirical::pm6dD3H4Correction(
          na, info[m].atoms.data(), info[m].coords.data());
      result[m] = boost::python::make_tuple(q, hofAll[g], hofPm6All[g], hofAll[g] + corr);
    }
    off += na;
  }

  // Open-shell molecules: solve each on the CPU UHF path with its multiplicity.
  for (int m = 0; m < nMol; ++m) {
    if (!info[m].openShell) continue;
    const int na = info[m].na;
    std::vector<double> q(na);
    double hof = 0.0, hofPm6 = 0.0;
    const bool ok = nvMolKit::semiempirical::pm6dCharges(
        na, info[m].atoms.data(), info[m].coords.data(), q.data(), &hof, 800, 1e-10,
        &hofPm6, info[m].charge, info[m].mult);
    if (!ok) continue;  // None (d-bearing open-shell unsupported, or non-converged)
    boost::python::list qList;
    for (int a = 0; a < na; ++a) qList.append(q[a]);
    const double corr = nvMolKit::semiempirical::pm6dD3H4Correction(
        na, info[m].atoms.data(), info[m].coords.data());
    result[m] = boost::python::make_tuple(qList, hof, hofPm6, hof + corr);
  }

  boost::python::list out;
  for (int m = 0; m < nMol; ++m) out.append(result[m]);
  return out;
}

// PM6_D frozen-density energy gradient (eV/Angstrom) for a list of RDKit
// molecules (each with a 3D conformer). Returns a list of length len(mols); each
// element is either a list of nAtoms (gx, gy, gz) tuples or None when the molecule
// is unsupported / open-shell / did not converge. Runs on the CPU host path.
boost::python::object pm6dGradientBatch(const boost::python::list& mols) {
  const int nMol = static_cast<int>(len(mols));
  boost::python::list out;
  for (int m = 0; m < nMol; ++m) {
    const RDKit::ROMol* mol = extract<const RDKit::ROMol*>(boost::python::object(mols[m]));
    const int na = static_cast<int>(mol->getNumAtoms());
    const RDKit::Conformer& conf = mol->getConformer();  // throws if none
    std::vector<int> atoms(na);
    std::vector<double> coords(3 * na);
    for (int a = 0; a < na; ++a) {
      atoms[a] = static_cast<int>(mol->getAtomWithIdx(a)->getAtomicNum());
      const RDGeom::Point3D& p = conf.getAtomPos(a);
      coords[3 * a] = p.x;
      coords[3 * a + 1] = p.y;
      coords[3 * a + 2] = p.z;
    }
    std::vector<double> grad(3 * na);
    if (!nvMolKit::semiempirical::pm6dGradient(na, atoms.data(), coords.data(), grad.data())) {
      out.append(boost::python::object());  // None
      continue;
    }
    boost::python::list g;
    for (int a = 0; a < na; ++a)
      g.append(boost::python::make_tuple(grad[3 * a], grad[3 * a + 1], grad[3 * a + 2]));
    out.append(g);
  }
  return out;
}

// PM6_D geometry optimization (L-BFGS on the frozen-density gradient) for a list
// of RDKit molecules (each with a 3D conformer used as the starting geometry).
// Returns a list of length len(mols); each element is either a tuple
// (optimized_coords, energy_eV, grad_rms, converged) -- where optimized_coords is
// a list of nAtoms (x, y, z) in Angstrom -- or None when the molecule is
// unsupported / open-shell / the initial SCF did not converge.
boost::python::object pm6dOptimizeBatch(const boost::python::list& mols) {
  const int nMol = static_cast<int>(len(mols));
  boost::python::list out;
  for (int m = 0; m < nMol; ++m) {
    const RDKit::ROMol* mol = extract<const RDKit::ROMol*>(boost::python::object(mols[m]));
    const int na = static_cast<int>(mol->getNumAtoms());
    const RDKit::Conformer& conf = mol->getConformer();  // throws if none
    std::vector<int> atoms(na);
    std::vector<double> coords(3 * na);
    for (int a = 0; a < na; ++a) {
      atoms[a] = static_cast<int>(mol->getAtomWithIdx(a)->getAtomicNum());
      const RDGeom::Point3D& p = conf.getAtomPos(a);
      coords[3 * a] = p.x;
      coords[3 * a + 1] = p.y;
      coords[3 * a + 2] = p.z;
    }
    std::vector<double> opt(3 * na);
    double E = 0.0, gRms = 0.0;
    int nIter = 0;
    const bool conv = nvMolKit::semiempirical::pm6dOptimize(
        na, atoms.data(), coords.data(), opt.data(), &E, &gRms, &nIter);
    if (gRms == 0.0 && nIter == 0 && !conv) {
      out.append(boost::python::object());  // None (initial SCF failed)
      continue;
    }
    boost::python::list xyz;
    for (int a = 0; a < na; ++a)
      xyz.append(boost::python::make_tuple(opt[3 * a], opt[3 * a + 1], opt[3 * a + 2]));
    out.append(boost::python::make_tuple(xyz, E, gRms, conv));
  }
  return out;
}

}  // namespace

BOOST_PYTHON_MODULE(_Semiempirical) {
  def("PM6DCharges", &pm6dChargesBatch, (arg("molecules")),
      "PM6_D (d-orbital NDDO) Mulliken charges + heats of formation for a list of "
      "RDKit molecules with 3D conformers. Returns a list of "
      "(charges, hof_nddo_kcal, hof_pm6_kcal, hof_d3h4_kcal) tuples (or None per "
      "molecule if unsupported / non-converged): hof_nddo is the "
      "PYSEQM-referenced PM6_D heat of formation (AM1-style core-core), hof_pm6 is "
      "the canonical MOPAC-aligned PM6 heat of formation (PWCCT core-core; ~1 "
      "kcal/mol of MOPAC for light + Br, looser for iodine), and hof_d3h4 adds the "
      "post-SCF PM6-D3H4 correction to hof_nddo. Closed-shell molecules run the "
      "whole d-orbital SCF on the GPU; open-shell ones (radicals / odd electron "
      "count, detected via formal charge + radical electrons) fall back to the CPU "
      "UHF path (sp-only — d-bearing open-shell molecules return None).");
  def("PM6DGradient", &pm6dGradientBatch, (arg("molecules")),
      "PM6_D (d-orbital NDDO) frozen-density energy gradient (eV/Angstrom) for a "
      "list of RDKit molecules with 3D conformers. Returns a list of per-molecule "
      "[(gx, gy, gz), ...] (length nAtoms), or None per molecule if unsupported / "
      "open-shell / non-converged. Hellmann-Feynman frozen-density gradient: one "
      "SCF + 6*nAtoms integral passes (no SCF re-solve).");
  def("PM6DOptimize", &pm6dOptimizeBatch, (arg("molecules")),
      "PM6_D (d-orbital NDDO) geometry optimization (L-BFGS on the frozen-density "
      "gradient) for a list of RDKit molecules; each conformer is the starting "
      "geometry. Returns a list of (optimized_coords, energy_eV, grad_rms, "
      "converged) tuples -- optimized_coords is a list of nAtoms (x, y, z) in "
      "Angstrom -- or None per molecule if unsupported / open-shell / the initial "
      "SCF did not converge.");
}
