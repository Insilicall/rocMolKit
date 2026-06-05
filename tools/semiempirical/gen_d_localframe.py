"""Transpile PYSEQM's local-frame d-orbital two-center reduced integrals to C++.

The oracle's two_elec_two_center_int_local_frame_d_orbitals (a vendored NumPy
port of PYSEQM, BSD-3) is ~4000 lines of one giant function: per masked case
(YH/YX/YY) it computes a long sequence of analytic multipole-interaction terms
(ev/sqrt(...) of the interatomic distance, the dipole/quadrupole/d charge
separations and the additive rho radii) and folds them into the local-frame
reduced two-electron integral vector (riYX has 450 slots, riYY 2025). Each line
is pure scalar arithmetic; hand-transcription at that scale is infeasible and
error-prone, so this script transpiles the active source lines mechanically into
device-callable C++ — bit-exact by construction. Validated against frozen oracle
captures (data/golden_yx_local.json etc.) by validate_d_localframe.py.

    python3 tools/semiempirical/gen_d_localframe.py

Writes rocmolkit/src/semiempirical/d_localframe_generated.h (do not hand-edit).
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
ORACLE = Path("/tmp/mlxmolkit_inspect/mlxmolkit/rm1/_pyseqm_port/"
              "two_elec_two_center_int_local_frame_d_orbitals_np.py")
OUT = ROOT / "rocmolkit" / "src" / "semiempirical" / "d_localframe_generated.h"

# Inputs each case-function takes (scalars, one pair). B carries d only for YY.
YX_INPUTS = ["r0", "da0", "db0", "qa0", "qb0", "dpa0", "dsa0", "dda0",
             "rho0a", "rho0b", "rho1a", "rho1b", "rho2a", "rho2b",
             "rho3a", "rho4a", "rho5a", "rho6a"]

# Case -> (1-based source line range of the term-defs + ri assignments, ri name,
# ri size, the input list). Ranges cover from just after the `ri = zeros` line to
# just before the `core...` packing, i.e. only the scalar reduced-integral math.
CASES = {
    "YX": {"lo": 730, "hi": 1413, "ri": "riYX", "size": 450, "inputs": YX_INPUTS},
}


def join_continuations(lines: list[str]) -> list[str]:
    out, buf = [], ""
    for ln in lines:
        s = ln.rstrip("\n")
        if buf:
            buf += " " + s.strip()
        else:
            buf = s
        if buf.rstrip().endswith("\\"):
            buf = buf.rstrip()[:-1]
        else:
            out.append(buf)
            buf = ""
    if buf:
        out.append(buf)
    return out


def strip_pow2(expr: str) -> str:
    """Replace each BASE**2 with sq(BASE), BASE = balanced ()-group or ident."""
    while True:
        k = expr.find("**2")
        if k < 0:
            return expr
        # find base ending at k-1
        j = k - 1
        if expr[j] == ")":
            depth = 0
            while j >= 0:
                if expr[j] == ")":
                    depth += 1
                elif expr[j] == "(":
                    depth -= 1
                    if depth == 0:
                        break
                j -= 1
            base = expr[j:k]
        else:
            s = k
            while s - 1 >= 0 and (expr[s - 1].isalnum() or expr[s - 1] in "_."):
                s -= 1
            j = s
            base = expr[j:k]
        expr = expr[:j] + "sq(" + base + ")" + expr[k + 3:]
    return expr


def transpile_expr(e: str) -> str:
    e = e.replace("torch.sqrt", "std::sqrt")
    e = e.replace("math.sqrt(2)", "kSqrt2").replace("math.sqrt(1)", "1.0")
    e = re.sub(r"math\.sqrt\(([^)]*)\)", r"std::sqrt((double)(\1))", e)
    e = strip_pow2(e)
    return e


def transpile_case(src: list[str], spec: dict) -> tuple[list[str], list[str]]:
    raw = src[spec["lo"] - 1: spec["hi"]]
    stmts = join_continuations(raw)
    body, decl_seen = [], set(spec["inputs"])
    riname = spec["ri"]
    skip_lhs = {f"{v}d" for v in spec["inputs"]} | {riname}
    for st in stmts:
        s = st.strip()
        if not s or s.startswith("#") or s.startswith("##"):
            continue
        s = s.split("##")[0].split("#")[0].strip()  # trailing comments
        if not s or "=" not in s:
            continue
        lhs, rhs = s.split("=", 1)
        lhs, rhs = lhs.strip(), rhs.strip()
        # Skip the per-pair input unpacking (r0 = r0d[YX], etc.) and zeros alloc.
        if "torch.zeros" in rhs or re.match(r"^\w+d\[", rhs) or rhs.endswith("[YX]") \
           or rhs.endswith("[YY]") or rhs.endswith("[YH]"):
            continue
        cpp_rhs = transpile_expr(rhs)
        m = re.match(rf"{riname}\[\.\.\.,(\d+)\]$", lhs)
        if m:
            body.append(f"  ri[{m.group(1)}] = {cpp_rhs};")
        elif re.match(r"^[A-Za-z_]\w*$", lhs):
            if lhs in decl_seen:
                body.append(f"  {lhs} = {cpp_rhs};")
            else:
                decl_seen.add(lhs)
                body.append(f"  double {lhs} = {cpp_rhs};")
        # else: indexed non-ri LHS — none expected in the scalar block
    return body, spec["inputs"]


HEADER = """// SPDX-FileCopyrightText: Copyright (c) 2025 InsilicAll. All rights reserved.
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
// GENERATED by tools/semiempirical/gen_d_localframe.py from PYSEQM's vendored
// two_elec_two_center_int_local_frame_d_orbitals (BSD-3) -- DO NOT HAND-EDIT.
//
// Local-frame d-orbital two-center reduced two-electron integrals: analytic
// multipole-interaction expansion (Dewar-Thiel) for the YX case (d-atom A + sp
// atom B). Inputs are the interatomic distance r0 (bohr), the dipole/quadrupole/
// d charge separations and the additive rho radii; outputs the reduced integral
// vector consumed by the d-rotation + Fock assembly. Validated bit-exact against
// the oracle by tools/semiempirical/validate_d_localframe.py.

#ifndef NVMOLKIT_SEMIEMPIRICAL_D_LOCALFRAME_GENERATED_H
#define NVMOLKIT_SEMIEMPIRICAL_D_LOCALFRAME_GENERATED_H

#include <cmath>

#include "device_macros.h"

namespace nvMolKit {
namespace semiempirical {
namespace dlocal {

constexpr double kSemiEVlf = 27.21;
"""

FOOTER = """
}  // namespace dlocal
}  // namespace semiempirical
}  // namespace nvMolKit

#endif  // NVMOLKIT_SEMIEMPIRICAL_D_LOCALFRAME_GENERATED_H
"""


def main() -> None:
    src = ORACLE.read_text().splitlines(keepends=False)
    out = [HEADER]
    for case, spec in CASES.items():
        body, inputs = transpile_case(src, spec)
        args = ", ".join(f"double {v}" for v in inputs)
        out.append(f"\n// {case}: d-atom A + sp atom B. ri must hold {spec['size']} doubles.")
        out.append(f"NVMOLKIT_HD inline void riLocal{case}(" + args + ", double* ri) {")
        out.append("  const double ev = kSemiEVlf, ev1 = ev / 2.0, ev2 = ev / 4.0,")
        out.append("               ev3 = ev / 8.0, ev4 = ev / 16.0;")
        out.append("  const double kSqrt2 = 1.4142135623730951;")
        out.append("  auto sq = [](double x) { return x * x; };")
        out.append(f"  for (int i = 0; i < {spec['size']}; ++i) ri[i] = 0.0;")
        out.append("  (void)ev4;")
        out.extend(body)
        out.append("}")
    out.append(FOOTER)
    OUT.write_text("\n".join(out))
    n = sum(1 for c in CASES)
    print(f"wrote {OUT.relative_to(ROOT)}  ({n} case(s))")


if __name__ == "__main__":
    main()
