"""Validate the ACTIVE-d transition-metal diatomic Slater overlap bit-exact
against MOPAC 23.2.5 AUX OVERLAP_MATRIX.

The active-d transition metals (Sc-Cu, Y-Ag, Hf-Au) carry a valence d shell whose
principal quantum number is ONE BELOW the 4s/4p (5s/5p, 6s/6p) sp shell -- e.g.
Sc uses 4s/4p but 3d. MOPAC reads the principal qn per shell via npq(Z,3); our
mopDiat now takes a separate d-shell qn (qnD). This test reproduces MOPAC's
OVERLAP_MATRIX for TM compounds using the per-orbital ATOM_PQN MOPAC dumps,
proving the qnD = qn-1 rule for the overlap is exact.

Reuses the validated ss/coe/cc/parse machinery from validate_mopac_overlap_port.py;
only diat() is made qnD-aware (npqA[2] = d-shell qn instead of the sp qn).
"""
import os, sys, math
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import validate_mopac_overlap_port as base  # ss, coe, cc, IVAL, parse_aux, tri_to_full, run_mopac


def diat_tm(nA, nDA, zsA, zpA, zdA, natA, nB, nDB, zsB, zpB, zdB, natB, xj):
    """MOPAC diat with a per-shell principal qn: s/p use nsp, d uses nD."""
    di = np.zeros((9, 9))
    x2, y2, z2 = xj
    r = math.sqrt(x2 * x2 + y2 * y2 + z2 * z2)
    c, _ = base.coe(x2, y2, z2, natA, natB)
    iaN = 3 if natA >= 5 else (2 if natA >= 2 else 1)
    ibN = 3 if natB >= 5 else (2 if natB >= 2 else 1)
    ulA = [zsA, zpA, max(zdA, 0.3)]
    ulB = [zsB, zpB, max(zdB, 0.3)]
    npqA = [nA, nA, nDA]
    npqB = [nB, nB, nDB]
    a = iaN - 1
    b = ibN - 1
    nk1 = min(a, b) + 1
    s = np.zeros((4, 4, 4))
    for i in range(1, iaN + 1):
        for j in range(1, ibN + 1):
            for k in range(1, nk1 + 1):
                if k > i or k > j:
                    continue
                pi = max(npqA[i - 1], i)
                pj = max(npqB[j - 1], j)
                s[i][j][k] = base.ss(pi, pj, i, j, k, ulA[i - 1], ulB[j - 1], r)
    for i in range(1, iaN + 1):
        kmin = 4 - i
        kmax = 2 + i
        for j in range(1, ibN + 1):
            if j == 2:
                aa, bbv = -1.0, 1.0
            else:
                aa = 1.0
                bbv = -1.0 if j == 3 else 1.0
            lmin = 4 - j
            lmax = 2 + j
            for k in range(kmin, kmax + 1):
                for l in range(lmin, lmax + 1):
                    ii = base.IVAL[(i, k)]
                    jj = base.IVAL[(j, l)]
                    if ii == 0 or jj == 0:
                        continue
                    di[ii - 1, jj - 1] = (
                        s[i][j][1] * (base.cc(c, i, k, 3) * base.cc(c, j, l, 3)) * aa
                        + s[i][j][2] * (base.cc(c, i, k, 4) * base.cc(c, j, l, 4)
                                        + base.cc(c, i, k, 2) * base.cc(c, j, l, 2)) * bbv
                        + s[i][j][3] * (base.cc(c, i, k, 5) * base.cc(c, j, l, 5)
                                        + base.cc(c, i, k, 1) * base.cc(c, j, l, 1))
                    )
    return di


def atoms_from_aux(zeta, pqn, natorb_list):
    """Carve per-atom (nsp, nD, zs, zp, zd, natorb) from MOPAC AO_ZETA/ATOM_PQN.

    ATOM_PQN is per-orbital: orbital 0 (s) gives nsp, orbital 4 (first d) gives
    the d-shell principal qn -- this is exactly qnD (= nsp-1 for active-d TM)."""
    res = []
    k = 0
    for nat in natorb_list:
        nsp = pqn[k]
        nD = pqn[k + 4] if nat >= 9 else nsp
        zs = zeta[k]
        zp = zeta[k + 1] if nat >= 4 else 0.0
        zd = zeta[k + 4] if nat >= 9 else 0.0
        res.append((nsp, nD, zs, zp, zd, nat))
        k += nat
    return res


def build(atoms_aos, coords):
    offs = []
    o = 0
    for a in atoms_aos:
        offs.append(o)
        o += a[5]
    N = o
    S = np.eye(N)
    for A in range(len(atoms_aos)):
        for B in range(len(atoms_aos)):
            if A == B:
                continue
            nA, nDA, zsA, zpA, zdA, natA = atoms_aos[A]
            nB, nDB, zsB, zpB, zdB, natB = atoms_aos[B]
            xj = [coords[B][d] - coords[A][d] for d in range(3)]
            di = diat_tm(nA, nDA, zsA, zpA, zdA, natA, nB, nDB, zsB, zpB, zdB, natB, xj)
            for i in range(natA):
                for j in range(natB):
                    S[offs[A] + i, offs[B] + j] = di[i, j]
    return S


# Active-d transition-metal compounds MOPAC computes (mix of d0 / d10 / open shell;
# the OVERLAP is geometry+param only, independent of the SCF occupation). natorb
# per atom is derived from MOPAC's AO_ATOMINDEX at run time.
CASES = [
    ("ScF3", "Sc 0 0 0\nF 1.91 0 0\nF -0.955 1.654 0\nF -0.955 -1.654 0",
     [[0, 0, 0], [1.91, 0, 0], [-0.955, 1.654, 0], [-0.955, -1.654, 0]]),
    ("TiCl4", "Ti 0 0 0\nCl 1.27 1.27 1.27\nCl -1.27 -1.27 1.27\n"
              "Cl -1.27 1.27 -1.27\nCl 1.27 -1.27 -1.27",
     [[0, 0, 0], [1.27, 1.27, 1.27], [-1.27, -1.27, 1.27],
      [-1.27, 1.27, -1.27], [1.27, -1.27, -1.27]]),
    ("VCl4", "V 0 0 0\nCl 1.30 1.30 1.30\nCl -1.30 -1.30 1.30\n"
             "Cl -1.30 1.30 -1.30\nCl 1.30 -1.30 -1.30",
     [[0, 0, 0], [1.30, 1.30, 1.30], [-1.30, -1.30, 1.30],
      [-1.30, 1.30, -1.30], [1.30, -1.30, -1.30]]),
    ("CuF", "Cu 0 0 0\nF 0 0 1.75",
     [[0, 0, 0], [0, 0, 1.75]]),
    ("CuCl", "Cu 0 0 0\nCl 0 0 2.05",
     [[0, 0, 0], [0, 0, 2.05]]),
]


def run_mopac_xyz(name, geom):
    """Cartesian (0-opt) MOPAC run -> AO_ZETA/ATOM_PQN/OVERLAP_MATRIX + per-atom
    natorb derived from AO_ATOMINDEX (robust to elements MOPAC treats sp-only,
    e.g. PM6 Zn which carries no d in the basis)."""
    import re
    base_path = f"{base.WORK}/{name}"
    body = "\n".join(
        f"{ln.split()[0]} {ln.split()[1]} 0 {ln.split()[2]} 0 {ln.split()[3]} 0"
        for ln in geom.strip().split("\n")
    )
    open(base_path + ".mop", "w").write(
        f"PM6 1SCF AUX(PRECISION=12) GEO-OK\n{name}\n\n{body}\n"
    )
    import subprocess
    subprocess.run([base.MOP, base_path + ".mop"],
                   env=dict(os.environ, LD_LIBRARY_PATH=base.LIB),
                   capture_output=True)
    zeta, pqn, ovv = base.parse_aux(base_path + ".aux")
    txt = open(base_path + ".aux").read()
    idx = [int(x) for x in
           re.search(r"AO_ATOMINDEX\[\d+\]=\s*\n(.*?)\n\s*[A-Z]", txt, re.S).group(1).split()]
    nat = [idx.count(a) for a in sorted(set(idx))]
    return zeta, pqn, ovv, nat


if __name__ == "__main__":
    if not os.path.exists(base.MOP):
        sys.exit(f"set MOPAC_DIR (no mopac at {base.MOP})")
    worst = 0.0
    for name, geom, coords in CASES:
        zeta, pqn, ovv, nat = run_mopac_xyz(name, geom)
        n = sum(nat)
        S = build(atoms_from_aux(zeta, pqn, nat), coords)
        Sm = base.tri_to_full(ovv, n)
        d = np.max(np.abs(S - Sm))
        worst = max(worst, d)
        dqn = pqn[4]  # metal d-shell principal qn
        print(f"{name:7} nAO={n:3} metal-d qn={dqn} worst |dS| = {d:.2e}")
    ok = worst < 1e-12
    print(f"\nworst |dS| = {worst:.2e}  "
          f"{'OK (active-d TM overlap reproduced bit-exact)' if ok else '** MISMATCH'}")

    # The active-d TM OVERLAP (above) is bit-exact to MOPAC and is the validated
    # deliverable. The SCF CHARGES are NOT yet bit-exact, so the active-d metals
    # stay DISABLED in pm6ValenceElectrons (tore=0 -> the SCF refuses them) to avoid
    # silent-wrong output. Root cause, localized for Sc/ScF3:
    #   * overlap, d charge separations (dp/ds/dd), additive radii (rho3..rho6),
    #     sp multipoles (da/qa/rho0..2, qn_sp=4) and the one-center d W (qn_d=3) all
    #     match MOPAC 23.2.5 bit-exact; H_core matches MOPAC's dumped one-electron
    #     matrix to the EV-truncation floor;
    #   * but the d two-center two-ELECTRON Fock is wrong: ||[F,P]|| at MOPAC's
    #     converged density is ~1.26 for ScF3 vs ~0.0017 for the validated main-group
    #     H2S, so the SCF converges to Sc=+1.350 (MOPAC +1.246). The error is in the
    #     integrals coupling the 3d orbitals to a ligand's p-multipoles (these never
    #     enter e1b/H_core); riLocalYX (PYSEQM-derived) diverges from MOPAC's MNDO-d
    #     reppd2/rijkl/charg only when qn_sp != qn_d (active-d), being bit-exact for
    #     main-group d-atoms (P/S/Cl/Br/I, qn_sp=qn_d).
    # The driver call below reports the gap honestly; with Sc disabled it returns
    # ok=0 (the SCF refuses the metal). It does not gate the exit code.
    drv = os.environ.get("DRV_TM", "/tmp/drv_tm")
    if os.path.exists(drv):
        import subprocess
        print("\n--- ScF3 SCF charge diagnostic (active-d DISABLED: ok=0 expected) ---")
        print("reference  Sc=+1.24554 (MOPAC); engine (when forced on) gave +1.350")
        stdin = ("4 0 1\n21 0.0 0.0 0.0\n9 1.91 0.0 0.0\n"
                 "9 -0.955 1.654 0.0\n9 -0.955 -1.654 0.0\n")
        r = subprocess.run([drv], input=stdin, capture_output=True, text=True)
        print("rocMolKit:", r.stdout.strip() or r.stderr.strip())

    sys.exit(0 if ok else 1)
