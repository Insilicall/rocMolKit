"""PM6_SP Mulliken-charge anchor for P/S/Cl (treated sp, no d-orbitals).

Phase 2 extends element coverage to P, S, Cl (principal qn = 3) treated as
sp-only — the PM6_SP method. These are the bit-exactness targets for the
sp-electronic structure, frozen from the oracle (guillaume-osmo/mlxmolkit native
PM6_SP, MIT -> PYSEQM numpy port, BSD-3). Verified bit-exact (worst |dq| ~ 4e-7)
by the C++ and HIP paths.

Note on energy: the SCF density / charges are independent of the core-core
repulsion, so they match PM6_SP exactly. Heat of formation is NOT frozen here:
PM6/PM6_SP use pairwise core-core terms (PWCCT), which the engine does not yet
port (it uses the AM1-style core-core of PM6_D), so the HoF convention differs
for these elements. See docs/SEMIEMPIRICAL_DESIGN.md.

Tuple layout: (name, Z, coords_Angstrom, charges)  — charges over all atoms.
"""

GOLDEN_PM6SP = [
    ("H2S", [16, 1, 1],
     [[0, 0, 0], [0.9686, 0, 0.9269], [-0.9686, 0, 0.9269]],
     [-0.315550, 0.157775, 0.157775]),
    ("PH3", [15, 1, 1, 1],
     [[0, 0, 0], [1.196, 0, 0.823], [-0.598, 1.036, 0.823], [-0.598, -1.036, 0.823]],
     [-0.114339, 0.038098, 0.038121, 0.038121]),
    ("HCl", [17, 1], [[0, 0, 0], [1.275, 0, 0]], [-0.213540, 0.213540]),
    ("CSC", [6, 16, 6, 1, 1, 1, 1, 1, 1],
     [[-1.5, 0, 0.35], [0, 0, -0.7], [1.5, 0, 0.35], [-2.38, 0, -0.3],
      [-1.56, 0.886, 0.988], [-1.56, -0.886, 0.988], [2.38, 0, -0.3],
      [1.56, 0.886, 0.988], [1.56, -0.886, 0.988]],
     [-0.485017, -0.058524, -0.485017, 0.181039, 0.166620, 0.166620, 0.181039, 0.166620, 0.166620]),
    ("CH3SH", [6, 16, 1, 1, 1, 1],
     [[-1.064, 0, 0.227], [0.722, 0, -0.35], [-1.064, 0.886, 0.88],
      [-1.064, -0.886, 0.88], [-1.908, 0, -0.45], [1.245, 0.886, 0.3]],
     [-0.445331, -0.203956, 0.152831, 0.175941, 0.183870, 0.136644]),
    ("CH3Cl", [6, 17, 1, 1, 1],
     [[0, 0, 0], [1.785, 0, 0], [-0.366, 0.515, 0.892],
      [-0.366, 0.515, -0.892], [-0.366, -1.029, 0]],
     [-0.368738, -0.136876, 0.168588, 0.168588, 0.168437]),
    ("CCl4", [6, 17, 17, 17, 17],
     [[0, 0, 0], [1.02, 1.02, 1.02], [-1.02, -1.02, 1.02],
      [-1.02, 1.02, -1.02], [1.02, -1.02, -1.02]],
     [0.138346, -0.034586, -0.034586, -0.034586, -0.034586]),
]
