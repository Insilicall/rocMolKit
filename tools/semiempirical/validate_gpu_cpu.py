"""Validate the batched GPU PM6_D path bit-exact against the CPU engine.

The whole d-orbital SCF, the heats of formation (PYSEQM-referenced and the
canonical MOPAC-aligned PM6), and the Mulliken charges run on the GPU via
scfBatchDGpu; by design they share the same __host__ __device__ code as the CPU
pm6dCharges, so the two must agree to floating-point rounding. This compiles a
driver with hipcc and runs it on the GPU inside the rocmolkit devel container,
comparing scfBatchDGpu to pm6dCharges over the golden molecule set.

Run on a machine with an AMD GPU + the rocmolkit devel image:

    docker run --rm --device /dev/kfd --device /dev/dri -e HIP_VISIBLE_DEVICES=0 \\
      -v "$PWD":/work -w /work rocmolkit:devel-local \\
      python3 tools/semiempirical/validate_gpu_cpu.py

(or let this script invoke docker itself when DOCKER_IMAGE is set on the host).
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
SRC = ROOT / "rocmolkit" / "src" / "semiempirical"
GOLDEN = ROOT / "tools" / "semiempirical" / "data" / "golden_pm6d_charges.json"

DRIVER = r'''
#include <cstdio>
#include <cmath>
#include <vector>
#include "scf_d_kernels.h"
#include "scf_d.h"
#include "pm6_params.h"
using namespace nvMolKit::semiempirical;
%(MOLS)s
int main(){
  const int nMol = sizeof(ZS)/sizeof(ZS[0]);
  std::vector<int> molNAtoms(nMol), molNBasis(nMol), atomsAll;
  std::vector<double> coordsAll;
  for(int m=0;m<nMol;++m){
    molNAtoms[m]=NA[m]; int nb=0;
    for(int a=0;a<NA[m];++a){ int z=ZS[m][a]; atomsAll.push_back(z); nb+=pm6NumOrbitals(z);
      coordsAll.push_back(CS[m][3*a]); coordsAll.push_back(CS[m][3*a+1]); coordsAll.push_back(CS[m][3*a+2]); }
    molNBasis[m]=nb;
  }
  std::vector<double> charges(atomsAll.size()), hof(nMol), hofPm6(nMol);
  std::vector<int> conv(nMol);
  bool ok=scfBatchDGpu(nMol,molNAtoms.data(),molNBasis.data(),atomsAll.data(),coordsAll.data(),
                       charges.data(),hof.data(),conv.data(),800,1e-10,hofPm6.data());
  double wq=0,wh=0,wp=0; int off=0;
  for(int m=0;m<nMol;++m){
    std::vector<double> q(NA[m]); double ch=0,cp=0;
    pm6dCharges(NA[m],ZS[m],CS[m],q.data(),&ch,800,1e-10,&cp);
    for(int a=0;a<NA[m];++a) wq=std::fmax(wq,std::fabs(charges[off+a]-q[a]));
    wh=std::fmax(wh,std::fabs(hof[m]-ch)); wp=std::fmax(wp,std::fabs(hofPm6[m]-cp));
    off+=NA[m];
  }
  std::printf("GPU vs CPU: worst |dq|=%.2e  |dHoF|=%.2e  |dHoF_pm6|=%.2e kcal  ok=%d  %s\n",
              wq,wh,wp,ok,(ok&&wq<1e-9&&wh<1e-6&&wp<1e-6)?"OK (GPU==CPU)":"** MISMATCH");
  return (ok&&wq<1e-9&&wh<1e-6&&wp<1e-6)?0:1;
}
'''


def main() -> int:
    mols = json.loads(GOLDEN.read_text())
    zs, na, cs = [], [], []
    for m in mols:
        na.append(len(m["atoms"]))
        zs.append("{" + ",".join(str(z) for z in m["atoms"]) + "}")
        cs.append("{" + ",".join(repr(float(x)) for row in m["coords"] for x in row) + "}")
    decl = (f"static const int NA[]={{{','.join(str(x) for x in na)}}};\n"
            f"static const int ZS[][16]={{{','.join(zs)}}};\n"
            f"static const double CS[][48]={{{','.join(cs)}}};\n")
    src = DRIVER.replace("%(MOLS)s", decl)

    img = os.environ.get("DOCKER_IMAGE", "rocmolkit:devel-local")
    # If we are already inside the container (hipcc present), build+run directly.
    inside = subprocess.run(["bash", "-lc", "command -v hipcc"], capture_output=True).returncode == 0
    cf = SRC.parent.parent.parent / "_gpu_cpu_check.cpp"
    cf.write_text(src)
    try:
        build_run = (
            f"S=rocmolkit/src/semiempirical; "
            f"hipcc -std=c++17 -O2 --offload-arch=gfx1200 -I$S _gpu_cpu_check.cpp "
            f"$S/scf_d_kernels.hip.cpp $S/scf_d.cpp $S/core_hamiltonian.cpp $S/pm6_params.cpp "
            f"$S/overlap.cpp -o /tmp/gcc_check && /tmp/gcc_check")
        if inside:
            r = subprocess.run(["bash", "-lc", build_run], cwd=ROOT, capture_output=True, text=True)
        else:
            cmd = ["docker", "run", "--rm", "--device", "/dev/kfd", "--device", "/dev/dri",
                   "-e", "HIP_VISIBLE_DEVICES=0", "-v", f"{ROOT}:/work", "-w", "/work",
                   img, "bash", "-lc", build_run]
            r = subprocess.run(cmd, capture_output=True, text=True)
        print(r.stdout.strip() or r.stderr.strip()[-2000:])
        return r.returncode
    finally:
        cf.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
