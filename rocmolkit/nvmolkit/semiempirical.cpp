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
#include "semiempirical/scf_d.h"          // pm6dGradient
#include "semiempirical/pm6_params.h"     // pm6NumOrbitals
#include "semiempirical/h4_device.h"      // pm6dD3H4Correction

namespace {

using namespace boost::python;

// PM6_D charges + heat of formation for a list of RDKit molecules (each must
// carry a 3D conformer). Returns a list of length len(mols); each element is
// either a tuple (list[float] charges, float hof_kcal) or None when the molecule
// is unsupported / open-shell / did not converge.
boost::python::object pm6dChargesBatch(const boost::python::list& mols) {
  const int nMol = static_cast<int>(len(mols));
  std::vector<const RDKit::ROMol*> molsVec(nMol);
  for (int m = 0; m < nMol; ++m)
    molsVec[m] = extract<const RDKit::ROMol*>(boost::python::object(mols[m]));

  std::vector<int> molNAtoms(nMol), molNBasis(nMol), atomsAll;
  std::vector<double> coordsAll;
  for (int m = 0; m < nMol; ++m) {
    const RDKit::ROMol* mol = molsVec[m];
    const int na = static_cast<int>(mol->getNumAtoms());
    const RDKit::Conformer& conf = mol->getConformer();  // throws if none
    molNAtoms[m] = na;
    int nb = 0;
    for (int a = 0; a < na; ++a) {
      const int z = static_cast<int>(mol->getAtomWithIdx(a)->getAtomicNum());
      atomsAll.push_back(z);
      nb += nvMolKit::semiempirical::pm6NumOrbitals(z);
      const RDGeom::Point3D& p = conf.getAtomPos(a);
      coordsAll.push_back(p.x);
      coordsAll.push_back(p.y);
      coordsAll.push_back(p.z);
    }
    molNBasis[m] = nb;
  }

  std::vector<double> chargesAll(atomsAll.size()), hofAll(nMol);
  std::vector<int> convAll(nMol);
  const bool ok = nvMolKit::semiempirical::scfBatchDGpu(
      nMol, molNAtoms.data(), molNBasis.data(), atomsAll.data(), coordsAll.data(),
      chargesAll.data(), hofAll.data(), convAll.data());

  boost::python::list out;
  int off = 0, coff = 0;
  for (int m = 0; m < nMol; ++m) {
    const int na = molNAtoms[m];
    if (!ok || !convAll[m]) {
      out.append(boost::python::object());  // None
    } else {
      boost::python::list q;
      for (int a = 0; a < na; ++a) q.append(chargesAll[off + a]);
      // PM6-D3H4 = NDDO heat of formation + the post-SCF D3 + H4 + H-H correction.
      const double corr = nvMolKit::semiempirical::pm6dD3H4Correction(
          na, &atomsAll[coff], &coordsAll[3 * coff]);
      out.append(boost::python::make_tuple(q, hofAll[m], hofAll[m] + corr));
    }
    off += na;
    coff += na;
  }
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
      "PM6_D (d-orbital NDDO) Mulliken charges + heat of formation for a list of "
      "RDKit molecules with 3D conformers. Returns a list of "
      "(charges, hof_nddo_kcal, hof_d3h4_kcal) tuples (or None per molecule if "
      "unsupported / open-shell / non-converged); hof_d3h4 adds the post-SCF "
      "PM6-D3H4 correction (D3 dispersion + H4 H-bond + H-H repulsion). The whole "
      "d-orbital SCF runs on the GPU.");
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
