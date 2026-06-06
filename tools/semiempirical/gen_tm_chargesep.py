"""Compute the active-d transition-metal d charge separations (dp, ds, dd,
rho3..rho6) bit-exact to PYSEQM's pyseqm_d_params, for baking into
two_center_d_device.h dChargeSeparations.

These are the d-orbital analogues of the sp dipole/quadrupole charge separations.
For an active-d metal (category "A"), PYSEQM (seqm/seqm_functions/two_elec_two_center_int.py
_pm6_d_param_from_key + cal_par.POIJ) builds them from Slater-Condon radial
integrals of the sp (qn0) and d (qn0-1) shells using the tail exponents
(s/p/d_orb_exp_tail == zsn/zpn/zdn) for the Slater-Condon parts and the orbital
exponents (zeta_s/p/d) for the AIJL multipole separations, with the G2SD override.
rho3=DD0, rho4=DP, rho5=DS, rho6=DD (the POIJ additive-term radii); dp/ds/dd are
the dipole / s-d / d-d separations.

Validated bit-exact (to ~1e-9, the published-param rounding floor) against the
existing Cl golden in two_center_d_device.h. Requires torch (for cal_par.POIJ)
and the PYSEQM checkout at /tmp/PYSEQM.

    python3 tools/semiempirical/gen_tm_chargesep.py
"""
import sys
import math
import importlib.util
import types

PYSEQM = "/tmp/PYSEQM"
sys.path.insert(0, PYSEQM)
import torch  # noqa: E402

# Load the two PYSEQM modules we need WITHOUT importing the seqm package __init__
# (which pulls in torch-geometric / h5py / psutil etc.). We register stub package
# objects so the relative imports inside the loaded files resolve.
for p in ["seqm", "seqm.seqm_functions"]:
    if p not in sys.modules:
        mod = types.ModuleType(p)
        mod.__path__ = [f"{PYSEQM}/{p.replace('.', '/')}"]
        sys.modules[p] = mod

BASE = f"{PYSEQM}/seqm/seqm_functions/"


def _load(modname, path):
    spec = importlib.util.spec_from_file_location(modname, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[modname] = m
    spec.loader.exec_module(m)
    return m


_load("seqm.seqm_functions.constants", BASE + "constants.py")
cal = _load("seqm.seqm_functions.cal_par", BASE + "cal_par.py")

# two_elec_two_center_int.py imports many heavy siblings at module top; exec it in
# a namespace, skipping the relative imports except cal_par / constants.
import numpy as _np  # noqa: E402
_ns = {"torch": torch, "math": math, "sys": sys, "numpy": _np, "time": __import__("time")}
_good = []
for _line in open(BASE + "two_elec_two_center_int.py").read().split("\n"):
    if _line.startswith("from .cal_par"):
        _good.append("from seqm.seqm_functions.cal_par import *")
    elif _line.startswith("from .constants"):
        _good.append("from seqm.seqm_functions.constants import ev")
    elif _line.startswith("from ."):
        continue
    else:
        _good.append(_line)
exec("\n".join(_good), _ns)
_pm6_d_param_from_key = _ns["_pm6_d_param_from_key"]
POIJ = cal.POIJ


def charge_separations(category, z, qn0, zetas, zetap, zetad, zs, zp, zd, g2sd):
    """Return (dp, ds, dd, rho3, rho4, rho5, rho6) for the given element.

    category: "A" (active-d TM, d shell = qn0-1) or "B" (main-group d, qn0).
    zetas/zetap/zetad: ORBITAL exponents (zeta_s/p/d).
    zs/zp/zd: TAIL exponents (s/p/d_orb_exp_tail == zsn/zpn/zdn).
    """
    key = ("PM6", category, z, qn0, zetas, zetap, zetad, zs, zp, zd, g2sd)
    ds_add, dp_add, dd_add, dd0_add, dd4, dp3, aij52, aij43, aij63 = _pm6_d_param_from_key(key)
    dp = aij52 / math.sqrt(5)
    ds = math.sqrt(aij43 * math.sqrt(1.0 / 15.0)) * math.sqrt(2.0)
    dorbdorb = math.sqrt(2.0 * aij63 / 7.0)
    def t(x):
        return torch.tensor([x], dtype=torch.double)
    DS = POIJ(2, t(ds), t(ds_add)).item()
    FG = dd0_add + dd_add + 4 / 49 * dd4
    FG1 = dd0_add + 0.5 * dd_add - 24 / 441 * dd4
    FG2 = dd0_add - dd_add + 6 / 441 * dd4
    DD0 = POIJ(0, t(1.0), t(0.2 * (FG + 2 * FG1 + 2 * FG2))).item()
    Dp = aij52 / math.sqrt(5.0)
    FGp = dp_add + dp3
    FG1p = 3 / 49 * 245 / 27 * dp3
    DP = POIJ(1, t(Dp), t(FGp - 1.8 * FG1p)).item()
    Dd = math.sqrt(2.0 * aij63 / 7.0)
    FGd = 3 / 4 * dd_add + 20 / 441 * dd4
    FG1d = 35 / 441 * dd4
    DD = POIJ(2, t(Dd), t(FGd - (20.0 / 35.0) * FG1d)).item()
    return dp, ds, dorbdorb, DD0, DP, DS, DD


_NAMES = ["dp", "ds", "dd", "rho3", "rho4", "rho5", "rho6"]


def _report(label, vals, golden=None):
    print(f"\n{label}:")
    for i, (n, v) in enumerate(zip(_NAMES, vals)):
        if golden is not None:
            print(f"  {n}: {v:.8f}   golden {golden[i]:.8f}   d={abs(v - golden[i]):.2e}")
        else:
            print(f"  {n} = {v:.8f};")


if __name__ == "__main__":
    # Self-test against the existing Cl golden (category B, qn0=3) in
    # two_center_d_device.h -- confirms the method reproduces a known element.
    cl = charge_separations("B", 17, 3, 2.63705, 2.11815, 1.32403,
                            0.9563, 2.46407, 6.41033, 0.0)
    cl_gold = [0.75100923, 1.10740952, 1.51053979,
               0.30216047, 1.03731998, 2.34288535, 0.72430196]
    _report("Cl self-test (vs two_center_d_device.h golden)", cl, cl_gold)

    # Active-d transition metals Sc-Cu (Z=21-29): category "A" (sp qn0=4, d qn0-1=3).
    # (Z, name, zeta_s, zeta_p, zeta_d, zsn, zpn, zdn, G2SD) -- full-precision MOPAC
    # orbital exponents (zs6/zp6/zd6) + tail exps (zsn6/zpn6/zdn6) + G2SD (g2sd6) from
    # parameters_for_PM6_C.F90. These -> two_center_d_device.h dChargeSeparations.
    ACTIVE_D = [
        (21, "Sc", 1.402469, 1.345196, 1.859012, 0.848418, 2.451729, 0.789372, 5.380136),
        (22, "Ti", 5.324777, 1.164068, 1.418280, 1.045904, 1.076844, 0.717945, 3.396235),
        (23, "V",  1.974330, 1.063106, 1.394806, 1.094426, 0.755378, 1.099367, 1.831407),
        (24, "Cr", 3.283460, 1.029394, 1.623119, 1.619853, 0.848266, 1.405015, 2.000300),
        (25, "Mn", 2.131680, 1.525880, 2.607800, 1.132450, 1.390740, 0.962550, 1.105330),
        (26, "Fe", 1.479150, 6.002246, 1.080747, 1.459152, 1.392614, 2.161909, 1.601345),
        (27, "Co", 1.166613, 3.000000, 1.860218, 0.519518, 1.000000, 0.352115, 1.680225),
        (28, "Ni", 1.591828, 2.304739, 2.514761, 0.746470, 0.753327, 1.461345, 1.880502),
        (29, "Cu", 1.669096, 3.000000, 2.734990, 1.899598, 3.000000, 1.484317, 9.847577),
    ]
    for z, nm, zs, zp, zd, zsn, zpn, zdn, g2sd in ACTIVE_D:
        v = charge_separations("A", z, 4, zs, zp, zd, zsn, zpn, zdn, g2sd)
        _report(f"{nm} (Z={z}) -> two_center_d_device.h dChargeSeparations", v)
