# Semi-empirical SCF on AMD (HIP/ROCm) — design & roadmap

Status: **Phase 1 CPU reference complete** (branch `feature/pm6-semiempirical`).
New feature, not a port of nvMolKit — nvMolKit has no quantum chemistry. We build
an NDDO semi-empirical SCF engine from scratch in HIP.

## Progress

The Phase-1 (sp: H/C/N/O/F) CPU path is done and **bit-exact to PYSEQM** at every
stage (validated host-side against the frozen golden in
`tools/semiempirical/data/golden_intermediates.json`), and compiles + links
in-tree into `rocmolkit_core`:

| Stage | Module | vs PYSEQM |
| ----- | ------ | --------- |
| 1 Parameters | `pm6_params.{h,cpp}` (+ generated `_data.h`) | 17/17 |
| 2 Diatomic overlap | `overlap.{h,cpp}` | ΔS = 0.00 |
| 3 Core Hamiltonian | `core_hamiltonian.{h,cpp}` | ΔH = 0.00 |
| 4 Two-center integrals | `two_center.{h,cpp}` | Δri = 0.00 |
| 5 SCF loop | `scf.{h,cpp}` | ΔP=3e-8, ΔEig=1e-6 (golden rounding) |
| 6 Mulliken charges | `scf.cpp` | Δq = 0.00 |
| 6b Nuclear rep. + HoF | `scf.cpp`, `pm6_params.cpp` | ΔHoF = 0.00 kcal/mol |

Next: **Stage 7 — the HIP/GPU port** (below), then Phase 2/3 (d-orbitals, D3H4).

## GPU architecture (Stage 7)

The Phase-1 CPU code is the validated reference; the GPU path must reproduce it.
Matrices are small per molecule (nBasis ~ 6-30 for sp drug fragments), so the
throughput win is **batching**: one thread block per molecule, the whole SCF in
shared memory. Plan:

1. **Device-callable math.** Lift the integral math (`computeMultipoleParams`,
   `twoCenterLocal/Molecular`, the overlap A/B integrals) into `__host__
   __device__` headers so the same code feeds the CPU reference and the kernels —
   keeping the bit-exact guarantee.
2. **Batched Fock + integrals kernel.** One block per molecule builds H_core and
   each SCF cycle's Fock in shared memory; validate against the CPU `buildFock`.
3. **Diagonalization.** Per-cycle symmetric eigensolve — reuse
   `src/symmetric_eigensolver.hip.cpp` (already in the tree) or a block-local
   Jacobi for the small matrices; rocSOLVER for larger batched solves.
4. **Density + DIIS on device**, convergence test on device, charges + HoF
   reduction. Then measure throughput vs the Metal-hybrid reference, which keeps
   everything but the Fock build on the CPU.

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

- `tools/semiempirical/data/golden_pm6_charges.py` — 15 molecules with PM6_D
  Mulliken charges frozen from PYSEQM (sp-only + YH match to machine precision;
  YX/YY to ~0.01 e). This is the bit-exactness target.
- PM6 parameters are public (Stewart PM6 / MOPAC), vendored as
  `tools/semiempirical/data/pm6_params_mopac.csv`.

A result counts only when it matches the golden charges (and, once added, the
heat of formation) to the stated tolerance — never timing alone.

## Two-step methodology

1. **CPU oracle first.** Rather than write our own NumPy SCF, we reuse the
   already-validated reference (guillaume-osmo/mlxmolkit native PM6_D, MIT →
   PYSEQM NumPy port, BSD-3). It runs end-to-end in pure NumPy on Linux/x86 and
   reproduces every golden charge (worst |Δq| = 0.00045 e). `oracle_dump.py`
   freezes its outputs — charges, heat of formation, eigenvalues, and the
   H_core/density of the simplest molecules — into `golden_intermediates.json`,
   so the C++/HIP code validates against frozen numbers with no external
   dependency at build time.
2. **C++ then HIP port** (`rocmolkit/src/semiempirical/`): build each SCF stage
   as C++ host code validated against the frozen intermediates, then move the hot
   loops to HIP kernels for the 100%-GPU production path, then measure throughput
   vs the Metal-hybrid reference.

Each stage must reproduce the frozen golden before the next begins — that is the
discipline that was missing when the conformer numbers went wrong before.

## Layout

```
docs/SEMIEMPIRICAL_DESIGN.md          # this file
tools/semiempirical/
  data/pm6_params_mopac.csv           # vendored PM6 params (public, MOPAC)
  data/golden_pm6_charges.py          # PYSEQM-frozen charge anchor (15 mols)
  data/golden_intermediates.json      # frozen oracle intermediates (generated)
  oracle_dump.py                      # runs the MIT oracle, freezes targets
  gen_pm6_params_header.py            # CSV -> C++ pm6_params_data.h
rocmolkit/src/semiempirical/
  pm6_params.{h,cpp}                  # stage 1: parameter loader (done)
  pm6_params_data.h                   # generated PM6 table (107 elements)
  # overlap, H_core, two-electron, SCF, Mulliken, HIP kernels  — to come
```
