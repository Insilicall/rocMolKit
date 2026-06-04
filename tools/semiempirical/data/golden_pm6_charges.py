"""PM6_D Mulliken-charge validation anchor, frozen from PYSEQM/MOPAC.

This is the bit-exactness target for the rocMolKit semi-empirical SCF engine
(see docs/SEMIEMPIRICAL_DESIGN.md). We do not trust our own numbers — a result
counts only when the SCF reproduces these charges to the stated tolerance.

Coverage:
  - sp-only (H/C/N/O/F): match PYSEQM to machine precision.
  - YH (one d-orbital heavy atom + H): match to machine precision.
  - YX / YY (heavy-heavy with d-orbitals): match to ~0.01 e; the residual is the
    d-d two-center Fock term, the last piece of the d-orbital port.

Geometries are in Angstrom. `q_heavy` lists the expected Mulliken charge on each
heavy (non-H) atom, in atom order. Provenance: guillaume-osmo/mlxmolkit (MIT),
tests/test_pm6_d_native.py, themselves checked against PYSEQM.

Tuple layout: (name, Z, coords, q_heavy, tol, tier)
  tier ∈ {"sp", "yh", "yx", "yy"} — which phase of the port must pass it.
"""

GOLDEN_PM6 = [
    # ---- sp-only: Phase 1 must reproduce these to machine precision ----
    ("H2O", [8, 1, 1],
     [[0, 0, 0], [0.7572, 0, 0.5868], [-0.7572, 0, 0.5868]],
     [-0.6093], 1e-3, "sp"),
    ("CH4", [6, 1, 1, 1, 1],
     [[0, 0, 0], [0.629, 0.629, 0.629], [-0.629, -0.629, 0.629],
      [-0.629, 0.629, -0.629], [0.629, -0.629, -0.629]],
     [-0.6545], 1e-3, "sp"),
    ("HF", [9, 1], [[0, 0, 0], [0.917, 0, 0]], [-0.2638], 1e-3, "sp"),
    ("NH3", [7, 1, 1, 1],
     [[0, 0, 0.122], [0, 0.939, -0.286],
      [0.813, -0.470, -0.286], [-0.813, -0.470, -0.286]],
     [-0.5786], 1e-3, "sp"),
    ("COC", [6, 8, 6, 1, 1, 1, 1, 1, 1],  # dimethyl ether
     [[-1.165, 0.0, 0.183], [0.0, 0.0, -0.616], [1.165, 0.0, 0.183],
      [-2.020, 0.0, -0.498], [-1.225, 0.886, 0.821], [-1.225, -0.886, 0.821],
      [2.020, 0.0, -0.498], [1.225, 0.886, 0.821], [1.225, -0.886, 0.821]],
     [-0.195, -0.393, -0.195], 1e-3, "sp"),
    ("CNC", [6, 7, 6, 1, 1, 1, 1, 1, 1, 1],  # dimethylamine
     [[-1.215, 0.085, 0.0], [0.0, -0.612, 0.0], [1.215, 0.085, 0.0],
      [0.0, -1.236, 0.812], [-2.080, -0.587, 0.0],
      [-1.275, 0.726, 0.882], [-1.275, 0.726, -0.882],
      [2.080, -0.587, 0.0], [1.275, 0.726, 0.882], [1.275, 0.726, -0.882]],
     [-0.329, -0.374, -0.329], 1e-3, "sp"),

    # ---- YH: one d-orbital heavy atom + hydrogens (Phase 3) ----
    ("H2S", [16, 1, 1],
     [[0, 0, 0], [0.9686, 0, 0.9269], [-0.9686, 0, 0.9269]],
     [-0.3617], 1e-3, "yh"),
    ("PH3", [15, 1, 1, 1],
     [[0, 0, 0], [1.196, 0, 0.823],
      [-0.598, 1.036, 0.823], [-0.598, -1.036, 0.823]],
     [-0.0366], 1e-3, "yh"),
    ("HCl", [17, 1], [[0, 0, 0], [1.275, 0, 0]], [-0.2164], 1e-3, "yh"),
    ("HBr", [35, 1], [[0, 0, 0], [1.414, 0, 0]], [-0.1878], 1e-3, "yh"),

    # ---- YX: heavy-heavy with d-orbitals (Phase 3, ~0.01 e) ----
    ("CSC", [6, 16, 6, 1, 1, 1, 1, 1, 1],
     [[-1.500, 0.0, 0.350], [0.0, 0.0, -0.700], [1.500, 0.0, 0.350],
      [-2.380, 0.0, -0.300], [-1.560, 0.886, 0.988], [-1.560, -0.886, 0.988],
      [2.380, 0.0, -0.300], [1.560, 0.886, 0.988], [1.560, -0.886, 0.988]],
     [-0.488, -0.046, -0.488], 1e-2, "yx"),
    ("CH3SH", [6, 16, 1, 1, 1, 1],
     [[-1.064, 0.0, 0.227], [0.722, 0.0, -0.350],
      [-1.064, 0.886, 0.880], [-1.064, -0.886, 0.880],
      [-1.908, 0.0, -0.450], [1.245, 0.886, 0.300]],
     [-0.467, -0.205], 1e-2, "yx"),
    ("CH3Cl", [6, 17, 1, 1, 1],
     [[0, 0, 0], [1.785, 0, 0],
      [-0.366, 0.515, 0.892], [-0.366, 0.515, -0.892], [-0.366, -1.029, 0]],
     [-0.369, -0.135], 1e-2, "yx"),
    ("CH3Br", [6, 35, 1, 1, 1],
     [[0, 0, 0], [1.939, 0, 0],
      [-0.366, 0.515, 0.892], [-0.366, 0.515, -0.892], [-0.366, -1.029, 0]],
     [-0.423, -0.114], 1e-2, "yx"),

    # ---- YY: both atoms have d-orbitals (Phase 3, ~0.01 e) ----
    ("CCl4", [6, 17, 17, 17, 17],
     [[0, 0, 0], [1.020, 1.020, 1.020], [-1.020, -1.020, 1.020],
      [-1.020, 1.020, -1.020], [1.020, -1.020, -1.020]],
     [0.177, -0.044, -0.044, -0.044, -0.044], 1e-2, "yy"),
]
