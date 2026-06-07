"""Transpile PYSEQM's diatomic d-orbital overlap into a faithful C++ kernel.

The existing hand-written d-overlap (overlap_d_device.h) builds the s-d / p-d
block of a d-d pair via a "reverse dsBlock" trick, which only works when the
reversed orientation maps to a tabulated jcallds formula. For mixed heavy-halogen
pairs (Br-Cl, Br-S, I-Cl, I-S, I-Br) where BOTH atoms carry d-orbitals with
DIFFERENT principal quantum numbers, that trick breaks (e.g. I-Cl would need a
jcallds(dqn=3, partner qn=5) formula PYSEQM never tabulates — it uses jcallsd853
instead).

This generator transpiles PYSEQM/mlxmolkit's reference ``diatom_overlap_matrixD``
(LANL, BSD-3; vendored in guillaume-osmo/mlxmolkit as a pure-NumPy port) directly
into a self-contained C++ function ``interhalideOverlapDDev`` that builds the full
9x9 overlap for the ordered pair (heavier qn = atom A, so dqn_A >= dqn_B) using
PYSEQM's exact jcall / jcallds / jcallsd / jcalldd formulas for codes {7, 853, 9}.
Because it is a mechanical AST transpilation of the oracle, it is bit-exact by
construction; it is validated par-by-par against the oracle in
validate_interhalide_components.py.

    MLXMOLKIT=/tmp/mlxmolkit_inspect/mlxmolkit \\
      /tmp/semienv/bin/python tools/semiempirical/gen_interhalide_overlap.py

Writes rocmolkit/src/semiempirical/overlap_d_interhalide_device.h (do not hand-edit).
"""

from __future__ import annotations

import ast
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "rocmolkit" / "src" / "semiempirical" / "overlap_d_interhalide_device.h"
CODES = [7, 853, 9]

# ---------------------------------------------------------------------------
# AST -> C++ expression transpiler
# ---------------------------------------------------------------------------

# Arrays indexed as ARR[mask, k] -> ARR[k]; zeta_a/zeta_b[mask, k] -> za[k]/zb[k];
# rij[mask] -> Rb. A33/B33 also appear pre-sliced into A0..A7 scalar temporaries.
_ARRAYS = {"A111", "B111", "A211", "B211", "A121", "B121", "A22", "B22",
           "A311", "B311", "A321", "B321", "A131", "B131", "A231", "B231",
           "A33", "B33"}


class CppGen(ast.NodeVisitor):
    """Render a (masked, scalar-per-pair) NumPy arithmetic expression to C++."""

    def visit(self, node):
        m = getattr(self, "visit_" + type(node).__name__, None)
        if m is None:
            raise NotImplementedError(f"cannot transpile {ast.dump(node)}")
        return m(node)

    def visit_Expression(self, n):
        return self.visit(n.body)

    def visit_Constant(self, n):
        if isinstance(n.value, bool):
            raise NotImplementedError("bool constant")
        if isinstance(n.value, int):
            return f"{n.value}.0"
        return repr(float(n.value))

    def visit_Name(self, n):
        return n.id

    def visit_UnaryOp(self, n):
        if isinstance(n.op, ast.USub):
            return f"(-{self.visit(n.operand)})"
        if isinstance(n.op, ast.UAdd):
            return f"(+{self.visit(n.operand)})"
        raise NotImplementedError(ast.dump(n))

    _BIN = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/"}

    def visit_BinOp(self, n):
        if isinstance(n.op, ast.Pow):
            # x ** k : small positive integer -> explicit product (bit-exact to
            # NumPy's repeated-multiply); otherwise std::pow.
            base = self.visit(n.left)
            if isinstance(n.right, ast.Constant) and isinstance(n.right.value, int) \
                    and 0 < n.right.value <= 12:
                return "(" + "*".join([f"({base})"] * n.right.value) + ")"
            return f"std::pow({base}, {self.visit(n.right)})"
        op = self._BIN[type(n.op)]
        return f"({self.visit(n.left)} {op} {self.visit(n.right)})"

    def visit_Call(self, n):
        name = self._fname(n.func)
        if name in ("np.power", "math.pow", "torch.pow"):
            return f"std::pow({self.visit(n.args[0])}, {self.visit(n.args[1])})"
        if name in ("np.sqrt", "math.sqrt", "torch.sqrt"):
            return f"std::sqrt({self.visit(n.args[0])})"
        raise NotImplementedError(f"call {name}")

    def visit_Subscript(self, n):
        base = n.value
        sl = n.slice
        # Python 3.9+: slice is the index node directly (Tuple for [mask, k]).
        if isinstance(base, ast.Name):
            nm = base.id
            if isinstance(sl, ast.Tuple):  # ARR[mask, k] or zeta_x[mask, k]
                idx = sl.elts[-1]
                if not isinstance(idx, ast.Constant):
                    raise NotImplementedError(f"non-const index in {nm}")
                k = idx.value
                if nm == "zeta_a":
                    return f"za[{k}]"
                if nm == "zeta_b":
                    return f"zb[{k}]"
                if nm in _ARRAYS:
                    return f"{nm}[{k}]"
                raise NotImplementedError(f"subscript {nm}[..,{k}]")
            else:  # ARR[mask] (e.g. rij[mask])
                if nm == "rij":
                    return "Rb"
                raise NotImplementedError(f"single-index subscript {nm}")
        raise NotImplementedError(f"subscript base {ast.dump(base)}")

    @staticmethod
    def _fname(node):
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return CppGen._fname(node.value) + "." + node.attr
        raise NotImplementedError(ast.dump(node))


def cpp(expr_node) -> str:
    return CppGen().visit(expr_node)


# ---------------------------------------------------------------------------
# Source slicing
# ---------------------------------------------------------------------------

def load_src() -> list[str]:
    mlx = os.environ.get("MLXMOLKIT")
    if not mlx or not Path(mlx).is_dir():
        sys.exit("set MLXMOLKIT to the cloned mlxmolkit package dir")
    p = Path(mlx) / "rm1" / "_pyseqm_port" / "diat_overlapD_np.py"
    return p.read_text().split("\n")


def block_body(lines: list[str], header_re: str) -> str:
    """Return the dedented body under the first `if <mask>.sum() != 0:` that
    follows a line matching header_re."""
    start = None
    for i, l in enumerate(lines):
        if re.match(header_re, l):
            start = i
            break
    if start is None:
        raise RuntimeError(f"block {header_re!r} not found")
    # find the `if ...sum() != 0:` line right after
    j = start
    while j < len(lines) and ".sum()" not in lines[j]:
        j += 1
    indent = len(lines[j]) - len(lines[j].lstrip())
    body = []
    k = j + 1
    body_indent = None
    while k < len(lines):
        l = lines[k]
        if l.strip() == "":
            body.append("")
            k += 1
            continue
        ind = len(l) - len(l.lstrip())
        if ind <= indent:
            break
        if body_indent is None:
            body_indent = ind
        body.append(l[body_indent:])
        k += 1
    return "\n".join(body)


def assignments(body: str):
    """Yield (target_str, value_node) for each Assign in a body block."""
    tree = ast.parse(body)
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        tgt = node.targets[0]
        if isinstance(tgt, ast.Subscript) and isinstance(tgt.value, ast.Name):
            yield tgt.value.id, node.value  # Snnn[mask] = ... -> "Snnn"
        elif isinstance(tgt, ast.Name):
            yield tgt.id, node.value          # An = A33[mask,k]
        else:
            raise NotImplementedError(ast.dump(tgt))


# ---------------------------------------------------------------------------
# Emit the radial S-component dispatch for one block group.
# ---------------------------------------------------------------------------

def emit_group(lines, header_fmt, want_S, scalar_temps=False):
    """Return C++ lines: `if (jc == CODE) { <S assignments> }` per code.

    want_S: ordered list of S-component names to emit (e.g. ['S311','S321','S322']).
    scalar_temps: also emit the `double An = ...;` temporaries (d-d block)."""
    out = []
    for ci, code in enumerate(CODES):
        body = block_body(lines, header_fmt.format(code=code))
        stmts = list(assignments(body))
        kw = "if" if ci == 0 else "else if"
        out.append(f"  {kw} (jc == {code}) {{")
        # scalar temporaries first (preserve order), then S-components
        for name, val in stmts:
            if name in want_S:
                continue
            if not scalar_temps:
                raise RuntimeError(f"unexpected temp {name} in {header_fmt}")
            out.append(f"    const double {name} = {cpp(val)};")
        for sname in want_S:
            val = next(v for n, v in stmts if n == sname)
            out.append(f"    {sname} = {cpp(val)};")
        out.append("  }")
    return out


def emit_assembly(lines):
    """Transpile the di[...,i,j] = ... assembly (returns body lines for di[81])."""
    # locate the assembly region: from 'di[..., 0, 0] = S111' to 'return di'
    start = next(i for i, l in enumerate(lines) if re.match(r"\s*di\[\.\.\., 0, 0\]", l))
    end = next(i for i, l in enumerate(lines) if re.match(r"\s*return di", l))
    region = "\n".join(l.strip() and l[4:] if l.startswith("    ") else l
                       for l in lines[start:end])
    tree = ast.parse(region)
    out = []
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        tgt = node.targets[0]
        if not (isinstance(tgt, ast.Subscript) and isinstance(tgt.value, ast.Name)
                and tgt.value.id == "di"):
            continue
        elts = tgt.slice.elts  # [Ellipsis, i, j]
        i, j = elts[1].value, elts[2].value
        out.append(f"  di[{i} * 9 + {j}] = {cpp(node.value)};")
    return out


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
// GENERATED by tools/semiempirical/gen_interhalide_overlap.py from PYSEQM/mlxmolkit's
// diatom_overlap_matrixD (LANL, BSD-3) -- DO NOT HAND-EDIT.
//
// Faithful diatomic d-orbital overlap for mixed heavy-halogen pairs (Br-Cl, Br-S,
// I-Cl, I-S, I-Br) where both atoms carry d-orbitals with different principal
// quantum numbers. Builds the full 9x9 overlap of the ordered pair (atom A = the
// heavier-qn atom, so dqn_A >= dqn_B) via PYSEQM's exact jcall/jcallds/jcallsd/
// jcalldd formulas for codes {7, 853, 9}. za/zb are [zeta_s, zeta_p, zeta_d].

#ifndef NVMOLKIT_SEMIEMPIRICAL_OVERLAP_D_INTERHALIDE_DEVICE_H
#define NVMOLKIT_SEMIEMPIRICAL_OVERLAP_D_INTERHALIDE_DEVICE_H

#include <cmath>

#include "overlap_device.h"  // ovdetail::aintgs / bintgs / kOvNMax
#include "device_macros.h"

namespace nvMolKit {
namespace semiempirical {

// Full 9x9 local+rotated overlap for a hetero d-d pair. jc in {7, 853, 9} is the
// shared overlap code (jcall == jcallds == jcallsd == jcalldd for these pairs
// because atom A is the heavier-qn atom). ca/cb/sa/sb are the bond direction
// cosines (same convention as bondAngles). out is row-major 9x9 (81).
NVMOLKIT_HD inline void interhalideOverlapDDev(const double za[3], const double zb[3],
                                               double Rb, int jc, double ca, double cb,
                                               double sa, double sb, double* di) {
  using namespace ovdetail;
  const double sasb = sa * sb, sacb = sa * cb, casb = ca * sb, cacb = ca * cb;
  double S111 = 0, S211 = 0, S121 = 0, S221 = 0, S222 = 0;
  double S311 = 0, S321 = 0, S322 = 0, S131 = 0, S231 = 0, S232 = 0;
  double S331 = 0, S332 = 0, S333 = 0;
%(BLOCKS)s
  for (int i = 0; i < 81; ++i) di[i] = 0.0;
%(ASSEMBLY)s
}

}  // namespace semiempirical
}  // namespace nvMolKit

#endif  // NVMOLKIT_SEMIEMPIRICAL_OVERLAP_D_INTERHALIDE_DEVICE_H
"""


def aintgs_setup(arrA, arrB, za_i, zb_i):
    """C++ to fill A/B integral arrays for a (zeta_a[za_i], zeta_b[zb_i]) pair."""
    return [
        f"    double {arrA}[kOvNMax], {arrB}[kOvNMax];",
        f"    aintgs(0.5 * Rb * (za[{za_i}] + zb[{zb_i}]), {arrA});",
        f"    bintgs(0.5 * Rb * (za[{za_i}] - zb[{zb_i}]), {arrB});",
    ]


def main() -> int:
    lines = load_src()
    blocks = []

    # sp block: A111(za0,zb0) A211(za1,zb0) A121(za0,zb1) A22(za1,zb1)
    blocks.append("  {  // sp (jcall)")
    blocks += aintgs_setup("A111", "B111", 0, 0)
    blocks += aintgs_setup("A211", "B211", 1, 0)
    blocks += aintgs_setup("A121", "B121", 0, 1)
    blocks += aintgs_setup("A22", "B22", 1, 1)
    blocks += emit_group(lines, r"\s*jcall{code} = jcall == {code}\b",
                         ["S111", "S211", "S121", "S221", "S222"])
    blocks.append("  }")

    # d-s/d-p block (jcallds): A311(za2,zb0) A321(za2,zb1)
    blocks.append("  {  // d-s / d-p (jcallds)")
    blocks += aintgs_setup("A311", "B311", 2, 0)
    blocks += aintgs_setup("A321", "B321", 2, 1)
    blocks += emit_group(lines, r"\s*jcallds{code} = jcallds == {code}\b",
                         ["S311", "S321", "S322"])
    blocks.append("  }")

    # s-d/p-d block (jcallsd): A131(za0,zb2) A231(za1,zb2)
    blocks.append("  {  // s-d / p-d (jcallsd)")
    blocks += aintgs_setup("A131", "B131", 0, 2)
    blocks += aintgs_setup("A231", "B231", 1, 2)
    blocks += emit_group(lines, r"\s*jcallsd{code} = jcallsd == {code}\b",
                         ["S131", "S231", "S232"])
    blocks.append("  }")

    # d-d block (jcalldd): A33(za2,zb2), with An/Bn scalar temporaries
    blocks.append("  {  // d-d (jcalldd)")
    blocks += aintgs_setup("A33", "B33", 2, 2)
    blocks += emit_group(lines, r"\s*jcalldd{code} = jcalldd == {code}\b",
                         ["S331", "S332", "S333"], scalar_temps=True)
    blocks.append("  }")

    assembly = emit_assembly(lines)

    OUT.write_text(HEADER % {"BLOCKS": "\n".join(blocks), "ASSEMBLY": "\n".join(assembly)})
    print(f"wrote {OUT.relative_to(ROOT)}  ({len(blocks)} block lines, "
          f"{len(assembly)} assembly entries)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
