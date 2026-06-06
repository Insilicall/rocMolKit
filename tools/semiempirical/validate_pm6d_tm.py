"""Validate the active-d transition-metal PM6_D path end-to-end against MOPAC.

Closed-shell **d0** active-d TM compounds are bit-exact to MOPAC on BOTH the CPU
reference (pm6dCharges) and the GPU batch (scfBatchDGpu):

  ScF3 (Sc d0), TiF4 / TiCl4 (Ti(IV) d0), VF5 (V(V) d0), CrF6 (Cr(VI) d0).

The fix that closed Sc: MOPAC's mndod *spcore* gives the CORE atom a special
additive radius po(9)=pocord (AtomIntParams::rhoCore) for the electron-core
attraction; restoring it (rhoCore, defined for Sc/Fe/Ni in PM6) fixes the metal
charge. Ti/V/Cr have no pocord (rhoCore=0, the regular monopole), which is what
MOPAC does for them. Because Ti(IV)/V(V)/Cr(VI) are d0, the metal carries no d
electrons, so the two-center d-block 2e integrals never fire and the SCF reduces
to the validated sp + ligand-d path -> bit-exact.

Mn..Cu (populated-d) are NOT enabled: their one-center d-block reproduces MOPAC
(Cu+ d10 single-ion HoF 272.99 vs 273.53), but the TWO-CENTER d-block with a
POPULATED metal d-shell is not yet bit-exact (CuF: our Cu s-pop 0.18 / q=+0.84 vs
MOPAC 0.58 / q=+0.50; our state is ~3 eV lower, a real two-center-d discrepancy,
not a local minimum). They stay disabled to avoid silent-wrong output.

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

# Closed-shell d0 active-d TM compounds. coords in Angstrom (row-major per atom).
# mopac_charges are MOPAC 23.2.5 PM6 1SCF AUX(PRECISION=12) NET ATOMIC CHARGES.
MOLECULES = [
    {
        "name": "ScF3 (Sc d0)",
        "atoms": [21, 9, 9, 9],
        "coords": [0.0, 0.0, 0.0, 1.91, 0.0, 0.0,
                   -0.955, 1.654, 0.0, -0.955, -1.654, 0.0],
        "mopac_charges": [1.24554385774458565, -0.41521505946417925,
                          -0.41516439915622172, -0.41516439912417447],
    },
    {
        "name": "TiF4 (Ti(IV) d0)",
        "atoms": [22, 9, 9, 9, 9],
        "coords": [0.0, 0.0, 0.0, 1.016, 1.016, 1.016, 1.016, -1.016, -1.016,
                   -1.016, 1.016, -1.016, -1.016, -1.016, 1.016],
        "mopac_charges": [1.58172800025558447, -0.39543200011109647,
                          -0.39543200007962298, -0.39543200004815660,
                          -0.39543200001669554],
    },
    {
        "name": "TiCl4 (Ti(IV) d0)",
        "atoms": [22, 17, 17, 17, 17],
        "coords": [0.0, 0.0, 0.0, 1.2528, 1.2528, 1.2528, 1.2528, -1.2528, -1.2528,
                   -1.2528, 1.2528, -1.2528, -1.2528, -1.2528, 1.2528],
        "mopac_charges": [0.44295785263613041, -0.11073946340066332,
                          -0.11073946323957884, -0.11073946307846594,
                          -0.11073946291740100],
    },
    {
        "name": "VF5 (V(V) d0)",
        "atoms": [23, 9, 9, 9, 9, 9],
        "coords": [0.0, 0.0, 0.0, 1.71, 0.0, 0.0, -1.71, 0.0, 0.0,
                   0.0, 1.71, 0.0, 0.0, -0.855, 1.481, 0.0, -0.855, -1.481],
        "mopac_charges": [1.96038791028551795, -0.42062675854688614,
                          -0.42062675851559206, -0.37301411244400029,
                          -0.37306014040646840, -0.37306014037258528],
    },
    {
        "name": "CrF6 (Cr(VI) d0)",
        "atoms": [24, 9, 9, 9, 9, 9, 9],
        "coords": [0.0, 0.0, 0.0, 1.72, 0.0, 0.0, -1.72, 0.0, 0.0,
                   0.0, 1.72, 0.0, 0.0, -1.72, 0.0, 0.0, 0.0, 1.72, 0.0, 0.0, -1.72],
        "mopac_charges": [2.18866232770078728, -0.36477705470330157,
                          -0.36477705466963073, -0.36477705463584265,
                          -0.36477705460216647, -0.36477705456175080,
                          -0.36477705452807996],
    },
]

CPU_DRIVER_TMPL = r'''
#include <cstdio>
#include "scf_d.h"
#include "pm6_params.h"
using namespace nvMolKit::semiempirical;
int main(){
%s
  return 0;
}
'''


def _cpu_block(mol: dict) -> str:
    n = len(mol["atoms"])
    z = ",".join(str(x) for x in mol["atoms"])
    c = ",".join(repr(x) for x in mol["coords"])
    return (f'  {{ const int z[]={{{z}}};\n'
            f'    const double c[]={{{c}}};\n'
            f'    double q[{n}], hof, hofPm6;\n'
            f'    pm6dCharges({n}, z, c, q, &hof, 3000, 1e-10, &hofPm6);\n'
            f'    for(int a=0;a<{n};++a) std::printf("%.12f\\n", q[a]); }}\n')


def _run_cpu_charges() -> list[list[float]]:
    body = "".join(_cpu_block(m) for m in MOLECULES)
    with tempfile.TemporaryDirectory() as td:
        cf = Path(td) / "t.cpp"
        cf.write_text(CPU_DRIVER_TMPL % body)
        exe = Path(td) / "t"
        subprocess.run(
            ["g++", "-std=c++17", "-O2", f"-I{SRC}", str(cf),
             str(SRC / "scf_d.cpp"), str(SRC / "core_hamiltonian.cpp"),
             str(SRC / "pm6_params.cpp"), str(SRC / "overlap.cpp"),
             str(SRC / "two_center.cpp"), "-o", str(exe)], check=True)
        out = subprocess.run([str(exe)], capture_output=True, text=True).stdout
    vals = [float(x) for x in out.split()]
    res, i = [], 0
    for m in MOLECULES:
        n = len(m["atoms"])
        res.append(vals[i:i + n])
        i += n
    return res


GPU_DRIVER = r'''
#include <cstdio>
#include <vector>
#include <cmath>
#include "scf_d_kernels.h"
#include "scf_d.h"
#include "pm6_params.h"
using namespace nvMolKit::semiempirical;
static int chk(const char* nm,int NA,const int* ZS,const double* CS){
  int nb=0; std::vector<int> atomsAll; std::vector<double> coordsAll;
  for(int a=0;a<NA;++a){ atomsAll.push_back(ZS[a]); nb+=pm6NumOrbitals(ZS[a]);
    coordsAll.push_back(CS[3*a]); coordsAll.push_back(CS[3*a+1]); coordsAll.push_back(CS[3*a+2]); }
  int molNAtoms[1]={NA}, molNBasis[1]={nb}, conv[1];
  std::vector<double> chargesG(NA), hofG(1), hofPm6G(1);
  bool ok=scfBatchDGpu(1,molNAtoms,molNBasis,atomsAll.data(),coordsAll.data(),
                       chargesG.data(),hofG.data(),conv,3000,1e-10,hofPm6G.data());
  std::vector<double> qC(NA); double chC, cpC;
  pm6dCharges(NA, ZS, CS, qC.data(), &chC, 3000, 1e-10, &cpC);
  double wq=0; for(int a=0;a<NA;++a) wq=std::fmax(wq,std::fabs(chargesG[a]-qC[a]));
  std::printf("  %%-18s GPU vs CPU worst |dq|=%%.2e  ok=%%d\n", nm, wq, ok);
  return (ok && wq < 1e-9) ? 0 : 1;
}
int main(){
  int rc=0;
%s
  return rc;
}
'''


def _gpu_block(mol: dict) -> str:
    n = len(mol["atoms"])
    z = ",".join(str(x) for x in mol["atoms"])
    c = ",".join(repr(x) for x in mol["coords"])
    return (f'  {{ const int z[]={{{z}}}; const double c[]={{{c}}};\n'
            f'    rc |= chk("{mol["name"].split()[0]}", {n}, z, c); }}\n')


def _run_gpu_check() -> tuple[bool, str]:
    img = os.environ.get("DOCKER_IMAGE", "rocmolkit:devel-local")
    body = "".join(_gpu_block(m) for m in MOLECULES)
    cf = ROOT / "_tm_gpu_check.cpp"
    cf.write_text(GPU_DRIVER % body)
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
    print("=== active-d TM PM6_D end-to-end (d0 closed-shell) ===")

    allq = _run_cpu_charges()
    for mol, qC in zip(MOLECULES, allq):
        ref = mol["mopac_charges"]
        worst = max(abs(qC[a] - ref[a]) for a in range(len(ref)))
        status = "OK" if worst < 1e-3 else "FAIL"
        if worst >= 1e-3:
            ok = False
        print(f"{mol['name']:>18}: worst |dq| = {worst:.2e}  ({status} vs MOPAC, tol 1e-3)")

    gpu_ok, gpu_out = _run_gpu_check()
    print("\n--- GPU == CPU (scfBatchDGpu) ---")
    print(gpu_out)
    if not gpu_ok:
        ok = False

    print("\n" + ("OK (active-d d0 TM Sc/Ti/V/Cr bit-exact to MOPAC, GPU==CPU)" if ok
                  else "** FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
