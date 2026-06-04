# Semi-empirical SCF on AMD (HIP/ROCm) — design & roadmap

Status: **in progress** (branch `feature/pm6-semiempirical`). New feature, not a
port of nvMolKit — nvMolKit has no quantum chemistry. We build an NDDO
semi-empirical SCF engine from scratch in HIP.

## Goal

Run the **entire SCF on the GPU** — not just the Fock build. The reference
Apple-Silicon implementation (guillaume-osmo/mlxmolkit, MIT) offloads only the
Fock matrix build to a Metal kernel and keeps the SCF loop, diagonalization and
most integrals in NumPy on the CPU. Our differentiator:

| Stage                       | mlxmolkit (Metal) | rocMolKit (target) |
| --------------------------- | ----------------- | ------------------ |
| One/two-electron integrals  | NumPy (CPU)       | HIP kernels (GPU)  |
| Core Hamiltonian build      | NumPy (CPU)       | HIP kernels (GPU)  |
| Fock build                  | Metal kernel      | HIP kernels (GPU)  |
| Diagonalization (per cycle) | NumPy (CPU)       | rocSOLVER (GPU)    |
| DIIS / density update       | NumPy (CPU)       | HIP kernels (GPU)  |
| Heat of formation, charges  | NumPy (CPU)       | HIP kernels (GPU)  |

We already have `src/symmetric_eigensolver.hip.cpp` — the per-cycle Fock
diagonalizer the SCF needs is in the tree.

## Methods (match the reference's coverage, in order)

1. **PM6 sp-only** — H, C, N, O, F (no d-orbitals). The whole SCF skeleton.
2. **PM6 sp-only heavy** — + P, S, Cl, Br, I treated sp-only (`PM6_SP`).
3. **PM6_D** — add d-orbitals on P, S, Cl, Br (22-integral local frame +
   Wigner-D rotation to the molecular frame).
4. **PM6-D3H4** — post-SCF Grimme D3 dispersion + Řezáč–Hobza H4 H-bond + HH
   repulsion corrections.
5. **AM1 / PM3 / RM1** — same NDDO skeleton, different parameter sets +
   Gaussian core-core terms.

## Validation anchor (non-negotiable)

Same discipline as the rest of rocMolKit: we do not trust our own numbers — we
anchor to an external reference. Here the anchor is **PYSEQM/MOPAC**.

- `tools/semiempirical/data/golden_pm6_charges.py` — 16 molecules with PM6_D
  Mulliken charges frozen from PYSEQM (sp-only + YH match to machine precision;
  YX to ~0.01 e). This is the bit-exactness target.
- PM6 parameters are public (Stewart PM6 / MOPAC), vendored as
  `tools/semiempirical/data/pm6_params_mopac.csv`.

A result counts only when it matches the golden charges (and, once added, the
heat of formation) to the stated tolerance — never timing alone.

## Two-step methodology

1. **CPU reference first** (`tools/semiempirical/pm6_reference.py`, NumPy):
   correct, readable, validated bit-exact against the golden anchor. This is the
   oracle we develop the GPU code against — fast to iterate, no Docker build.
2. **HIP port** (`rocmolkit/src/semiempirical/`): 100%-GPU production path,
   validated to reproduce the CPU reference, then measured for throughput vs the
   Metal-hybrid reference.

Never port to HIP before the CPU reference matches the anchor — that is how the
conformer numbers went wrong before.

## Layout

```
docs/SEMIEMPIRICAL_DESIGN.md          # this file
tools/semiempirical/
  data/pm6_params_mopac.csv           # vendored PM6 params (public, MOPAC)
  data/golden_pm6_charges.py          # PYSEQM-frozen validation anchor
  pm6_reference.py                    # NumPy reference SCF (oracle)
  test_pm6_reference.py               # reference vs golden
rocmolkit/src/semiempirical/          # HIP SCF engine (100% GPU) — to come
```
