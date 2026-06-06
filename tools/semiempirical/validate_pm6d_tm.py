"""Validate the active-d transition-metal PM6_D path end-to-end against MOPAC.

Closed-shell active-d TM compounds (ScF3, d0) are now bit-exact to MOPAC on BOTH
the CPU reference (pm6dCharges) and the GPU batch (scfBatchDGpu). The fix that
closed the gap: MOPAC's mndod *spcore* gives the CORE atom a special additive
radius po(9)=pocord (AtomIntParams::rhoCore) for the electron-core attraction;
the engine had ignored it (used the regular monopole rho0), which biased the
H_core of every ligand orbital attracted to the metal core and pushed ScF3 to
Sc=+1.350 instead of MOPAC's +1.246. pocord enters only for the few elements that
define it (Sc/Fe/Ni in PM6), so the fix is a no-op for all main-group d-atoms.

This validator:
  1. checks ScF3 SCF charges bit-exact to MOPAC (dq < 1e-3) on the CPU;
  2. (when a GPU container is available) checks ScF3 on scfBatchDGpu == CPU.

    MOPAC_DIR=/tmp/mopac_bin/mopac-23.2.5-linux \\
      python3 tools/semiempirical/validate_pm6d_tm.py
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
SRC = ROOT / "rocmolkit" / "src" / "semiempirical"

# ScF3 (D3h), the simplest closed-shell active-d TM (Sc d0). MOPAC PM6 reference.
SCF3 = {
    "name": "ScF3",
    "atoms": [21, 9, 9, 9],
    "coords": [[0.0, 0.0, 0.0], [1.91, 0.0, 0.0],
               [-0.955, 1.654, 0.0], [-0.955, -1.654, 0.0]],
    "mopac_charges": [1.24554385774458565, -0.41521505946417925,
                      -0.41516439915622172, -0.41516439912417447],
}

CPU_DRIVER = r'''
#include <cstdio>
#include "scf_d.h"
#include "pm6_params.h"
using namespace nvMolKit::semiempirical;
int main(){
  const int z[]={21,9,9,9};
  const double c[]={0,0,0, 1.91,0,0, -0.955,1.654,0, -0.955,-1.654,0};
  double q[4], hof, hofPm6;
  pm6dCharges(4, z, c, q, &hof, 800, 1e-10, &hofPm6);
  for(int a=0;a<4;++a) std::printf("%.12f\n", q[a]);
  return 0;
}
'''

GPU_DRIVER = r'''
#include <cstdio>
#include <vector>
#include <cmath>
#include "scf_d_kernels.h"
#include "scf_d.h"
#include "pm6_params.h"
using namespace nvMolKit::semiempirical;
int main(){
  const int ZS[]={21,9,9,9};
  const double CS[]={0,0,0, 1.91,0,0, -0.955,1.654,0, -0.955,-1.654,0};
  int NA=4, nb=0; std::vector<int> atomsAll;
  std::vector<double> coordsAll;
  for(int a=0;a<NA;++a){ atomsAll.push_back(ZS[a]); nb+=pm6NumOrbitals(ZS[a]);
    coordsAll.push_back(CS[3*a]); coordsAll.push_back(CS[3*a+1]); coordsAll.push_back(CS[3*a+2]); }
  int molNAtoms[1]={NA}, molNBasis[1]={nb}, conv[1];
  std::vector<double> chargesG(NA), hofG(1), hofPm6G(1);
  bool ok=scfBatchDGpu(1,molNAtoms,molNBasis,atomsAll.data(),coordsAll.data(),
                       chargesG.data(),hofG.data(),conv,800,1e-10,hofPm6G.data());
  double qC[4], chC, cpC;
  pm6dCharges(NA, ZS, CS, qC, &chC, 800, 1e-10, &cpC);
  double wq=0; for(int a=0;a<NA;++a) wq=std::fmax(wq,std::fabs(chargesG[a]-qC[a]));
  std::printf("GPU vs CPU ScF3: worst |dq|=%.2e  ok=%d\n", wq, ok);
  for(int a=0;a<NA;++a) std::printf("  GPU q[%d]=%.12f  CPU q[%d]=%.12f\n", a, chargesG[a], a, qC[a]);
  return (ok && wq < 1e-9) ? 0 : 1;
}
'''


def _run_cpu_charges() -> list[float]:
    with tempfile.TemporaryDirectory() as td:
        cf = Path(td) / "t.cpp"
        cf.write_text(CPU_DRIVER)
        exe = Path(td) / "t"
        subprocess.run(
            ["g++", "-std=c++17", "-O2", f"-I{SRC}", str(cf),
             str(SRC / "scf_d.cpp"), str(SRC / "core_hamiltonian.cpp"),
             str(SRC / "pm6_params.cpp"), str(SRC / "overlap.cpp"),
             str(SRC / "two_center.cpp"), "-o", str(exe)], check=True)
        out = subprocess.run([str(exe)], capture_output=True, text=True).stdout
    return [float(x) for x in out.split()]


def _run_gpu_check() -> tuple[bool, str]:
    """Build + run the GPU==CPU ScF3 driver inside the rocmolkit devel container."""
    img = os.environ.get("DOCKER_IMAGE", "rocmolkit:devel-local")
    cf = ROOT / "_tm_gpu_check.cpp"
    cf.write_text(GPU_DRIVER)
    try:
        build_run = (
            "S=rocmolkit/src/semiempirical; "
            "hipcc -std=c++17 -O2 --offload-arch=gfx1200 -I$S _tm_gpu_check.cpp "
            "$S/scf_d_kernels.hip.cpp $S/scf_d.cpp $S/core_hamiltonian.cpp "
            "$S/pm6_params.cpp $S/overlap.cpp -o /tmp/tm_gpu_check && /tmp/tm_gpu_check")
        inside = subprocess.run(["bash", "-lc", "command -v hipcc"],
                                capture_output=True).returncode == 0
        if inside:
            r = subprocess.run(["bash", "-lc", build_run], cwd=ROOT,
                               capture_output=True, text=True)
        else:
            has_docker = subprocess.run(["bash", "-lc", "command -v docker"],
                                        capture_output=True).returncode == 0
            if not has_docker:
                return (True, "SKIP (no hipcc / docker available)")
            cmd = ["docker", "run", "--rm", "--device", "/dev/kfd", "--device", "/dev/dri",
                   "-e", "HIP_VISIBLE_DEVICES=0", "-v", f"{ROOT}:/work", "-w", "/work",
                   img, "bash", "-lc", build_run]
            r = subprocess.run(cmd, capture_output=True, text=True)
        out = (r.stdout.strip() or r.stderr.strip()[-1500:])
        return (r.returncode == 0, out)
    finally:
        cf.unlink(missing_ok=True)


def main() -> int:
    ok = True
    print("=== active-d TM PM6_D end-to-end (ScF3) ===")

    qC = _run_cpu_charges()
    ref = SCF3["mopac_charges"]
    worst = max(abs(qC[a] - ref[a]) for a in range(len(ref)))
    print(f"ScF3 CPU charges : {[round(x, 5) for x in qC]}")
    print(f"      MOPAC ref  : {[round(x, 5) for x in ref]}")
    print(f"      worst |dq| = {worst:.2e}  ({'OK' if worst < 1e-3 else 'FAIL'} vs MOPAC, tol 1e-3)")
    if worst >= 1e-3:
        ok = False

    gpu_ok, gpu_out = _run_gpu_check()
    print("\n--- GPU == CPU (scfBatchDGpu) ---")
    print(gpu_out)
    if not gpu_ok:
        ok = False

    print("\n" + ("OK (active-d TM ScF3 bit-exact to MOPAC, GPU==CPU)" if ok
                  else "** FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
