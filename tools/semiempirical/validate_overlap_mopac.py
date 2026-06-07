"""Sweep every interhalide diatomic overlap (bond along z) against MOPAC's exact
AUX OVERLAP_MATRIX. For each pair we read MOPAC's own AO_ZETA and feed the d-zeta
to the engine dump, so any remaining mismatch is a FORMULA bug, not a parameter
gap. Heavier atom at origin (engine 'heavier-first' order), lighter at +z."""
import os, re, shutil, subprocess, sys, tempfile
from pathlib import Path

MOPAC_DIR = os.environ.get("MOPAC_DIR", "/tmp/mopac_bin/mopac-23.2.5-linux")
MOPAC = f"{MOPAC_DIR}/bin/mopac"
LIB = f"{MOPAC_DIR}/lib"
SYM = {53: "I", 35: "Br", 17: "Cl", 16: "S"}
LAB = ["s", "px", "py", "pz", "x2", "xz", "z2", "yz", "xy"]
HERE = Path(__file__).resolve().parent
SRC = HERE.parent.parent / "rocmolkit" / "src" / "semiempirical"
DUMP = "/tmp/dump_ovl"
WORK = tempfile.mkdtemp(prefix="ovl_mopac_")


def build_dump():
    cc = shutil.which("g++") or shutil.which("c++")
    if cc is None:
        sys.exit("need g++ to build the overlap dump driver")
    subprocess.run([cc, "-std=c++17", "-O2", f"-I{SRC}", str(HERE / "dump_overlap.cpp"),
                    str(SRC / "core_hamiltonian.cpp"), str(SRC / "pm6_params.cpp"),
                    str(SRC / "overlap.cpp"), "-o", DUMP], check=True)

# (heavier Z, lighter Z, probe R in Angstrom)
PAIRS = [(35, 17, 2.18), (35, 16, 2.24), (53, 17, 2.55), (53, 16, 2.40), (53, 35, 2.74)]


def run_mopac(zA, zB, R):
    name = f"{SYM[zA]}{SYM[zB]}z".replace(" ", "")
    mop = f"{WORK}/{name}.mop"
    with open(mop, "w") as f:
        f.write("PM6 1SCF CHARGE=0 PRECISE AUX(PRECISION=12)\nprobe\n\n")
        f.write(f"{SYM[zA]}  0.0 0 0.0 0 0.0 0\n")
        f.write(f"{SYM[zB]}  0.0 0 0.0 0 {R:.5f} 1\n")
    env = dict(os.environ, LD_LIBRARY_PATH=LIB)
    subprocess.run([MOPAC, mop], env=env, capture_output=True)
    aux = open(f"{WORK}/{name}.aux").read()
    # AO_ZETA (first nA+nB entries across however many lines until next key)
    zblock = re.search(r"AO_ZETA\[\d+\]=\s*\n(.*?)\n\s*[A-Z_]+\[", aux, re.S).group(1)
    zetas = [float(x) for x in zblock.split()]
    m = re.search(r"OVERLAP_MATRIX\[\d+\]=\s*#[^\n]*\n(.*?)\n\s*[A-Z_]+(?:\[|=|:)", aux, re.S)
    nums = [float(x) for x in m.group(1).split()]
    nA, nB = 9, 9
    N = nA + nB
    S = [[0.0] * N for _ in range(N)]
    k = 0
    for i in range(N):
        for j in range(i + 1):
            S[i][j] = S[j][i] = nums[k]; k += 1
    mat = [[S[i][nA + j] for j in range(nB)] for i in range(nA)]  # A rows, B cols
    zdA = zetas[4]       # atom A d zeta (5th AO is first d)
    zdB = zetas[nA + 4]  # atom B d zeta
    return mat, zdA, zdB


def run_engine(zA, zB, R, zdA, zdB):
    out = subprocess.run([DUMP, str(zA), str(zB), f"{R:.5f}", f"{zdA:.6f}", f"{zdB:.6f}"],
                         capture_output=True, text=True).stdout
    rows = []
    for ln in out.splitlines():
        p = ln.split()
        if len(p) == 10 and p[0] in LAB and p[1].lstrip("-")[0].isdigit():
            rows.append([float(x) for x in p[1:]])
    return rows


def main():
    build_dump()
    overall = 0.0
    for zA, zB, R in PAIRS:
        mop, zdA, zdB = run_mopac(zA, zB, R)
        eng = run_engine(zA, zB, R, zdA, zdB)
        big = []
        for i in range(9):
            for j in range(9):
                d = min(abs(eng[i][j] - mop[i][j]), abs(eng[i][j] + mop[i][j]))
                if d > 1e-4:
                    big.append((d, f"{LAB[i]}-{LAB[j]} eng={eng[i][j]:+.5f} mop={mop[i][j]:+.5f}"))
        big.sort(reverse=True)
        w = big[0][0] if big else 0.0
        overall = max(overall, w)
        tag = "OK" if w < 1e-3 else "** MISMATCH"
        print(f"{SYM[zA]}-{SYM[zB]:<3} (zdA={zdA:.4f} zdB={zdB:.4f}) worst|d|={w:.5f}  {tag}")
        for d, s in big[:6]:
            print(f"     {d:.5f}  {s}")
    print(f"\nOVERALL worst |d| = {overall:.5f}  {'OK (<1e-3)' if overall < 1e-3 else '** FORMULA BUGS REMAIN'}")
    return 0 if overall < 1e-3 else 1


if __name__ == "__main__":
    sys.exit(main())
