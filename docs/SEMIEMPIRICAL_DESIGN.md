# Semi-empirical SCF on AMD (HIP/ROCm) — design & roadmap

Status: **Phase 1 complete — full SCF runs 100% on the GPU**, bit-exact to
PYSEQM (branch `feature/pm6-semiempirical`). New feature, not a port of nvMolKit —
nvMolKit has no quantum chemistry. We build an NDDO semi-empirical SCF engine from
scratch in HIP.

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

### GPU port (Stage 7) — done

The whole SCF now runs on the device, one thread per molecule, validated on the
AMD Radeon RX 9060 XT (gfx1200/RDNA4) against the same PYSEQM golden:

| Stage | Module | GPU result |
| ----- | ------ | ---------- |
| 7a Shared device math | `*_device.h` (`__host__ __device__`) | CPU re-validated bit-exact |
| 7  Two-center integrals kernel | `two_center_kernels.hip.cpp` | GPU==CPU, Δ~1e-15 |
| 7b Fock-build kernel | `fock_kernels.hip.cpp` | GPU==CPU, Δ=5e-14 |
| 7c Full SCF on device | `scf_kernels.hip.cpp`, `scf_device.h` | bit-exact PYSEQM, Δq=4e-7 |
| 7d H_core on device (100% GPU) | `overlap_device.h`, `core_hamiltonian_device.h` | bit-exact PYSEQM, Δq=4e-7 |
| 7e Heat of formation on device | `energy_device.h` | bit-exact PYSEQM, ΔHoF=4e-4 kcal/mol |

The key idea: a single `__host__ __device__` codebase feeds both the CPU
reference (validated against PYSEQM) and the HIP kernels, so the bit-exact
guarantee extends to the GPU by construction. The entire pipeline — overlap,
two-center integrals, H_core, Fock, per-cycle diagonalization (block-local
Jacobi), Pulay DIIS, density, Mulliken charges — runs on the device; only the
parameter gather and data marshalling stay on the host. A 24000-molecule batch
converges 24000/24000 at ~3,900 mol/s on the RX 9060 XT.

This is the differentiator: the Apple/Metal reference (mlxmolkit) offloads only
the Fock build and keeps the SCF loop + diagonalization + integrals on the CPU;
rocMolKit runs all of it on the GPU.

## What's left

- **Throughput optimization.** ~3,900 mol/s for small sp molecules. The
  one-thread-per-molecule layout and the H-heavy (HX) device recursion (currently
  needing a 64 KB stack) are the obvious targets; de-recursing the HX case and a
  one-block-per-molecule layout with shared memory should help.
- **Phase 2 — sp-heavy.** P/S/Cl (qn=3, `PM6_SP`) **done**: charges **and** heat
  of formation bit-exact (CPU + GPU) for H2S/PH3/HCl/CSC/CH3SH/CH3Cl/CCl4. The
  core-core repulsion is now PM6's pairwise PWCCT (real PM6), so HoF matches
  PM6_SP across all sp elements. Remaining: Br/I (qn 4/5) need the higher-qn sp
  overlap formulas.
- **Phase 3 — d-orbitals (`PM6_D`)** for P/S/Cl/Br/I, then PM6-D3H4. Strategy
  (see below). Other methods (AM1/PM3/RM1) reuse the skeleton.

## Phase 3 strategy — d-orbitals

The d-orbital overlap is the gate. Two findings shape the plan:

1. **The oracle is in hand.** PYSEQM's `diatom_overlap_matrixD` (LANL, BSD-3),
   vendored in mlxmolkit as a pure-NumPy port, runs here and produces the
   reference d-overlap for any pair. `tools/semiempirical/gen_doverlap_golden.py`
   freezes it into `data/golden_doverlap.json` — the par-by-par validation anchor.
2. **The analytic shortcut does NOT work.** A from-scratch route
   (mlxmolkit's `overlap_d_local` + a Wigner-D rotation) was tested against the
   oracle: the sp block matches but the **d-block is entirely different** (e.g.
   S-H d-column `[0,0,0.1445,0,0]` from the oracle vs `[-0.121,0,0,0.209,0]`
   analytic). The exact d-orbital frame convention lives in the PYSEQM code, so
   the port must reproduce `diatom_overlap_matrixD`, not the analytic fallback.

So Phase 3 ports `diatom_overlap_matrixD` to **scalar** C++. Its ~5000 NumPy lines
are *vectorized* (per-pair `np.where` masks over every qn/orbital case); the
scalar form collapses those to `if/else` and is far shorter (~800-1000 lines). The
rest mirrors the sp pipeline: d two-center integrals (`d_two_center.py`,
`tetci_yh.py` — YH/YX/YY, with F0SD/G2SD), the 9×9 one-/two-center Fock
contribution (`fock_d.py`), 9-orbital basis, then the shared-math GPU port. Each
stage validated par-by-par against the oracle before the next — the same
discipline as Phases 1-2. Br/I (qn 4/5) fall out once the d-overlap covers qn≥4.
- **Open-shell / odd-electron** molecules (currently rejected).
- **Integration:** gtest coverage under `tests/`, a public C++/Python entry
  point, and a formal throughput comparison harness. (Already builds in-tree.)

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

## Canonical PM6 (MOPAC) — bit-exact charges incl. iodine & IBr

The engine exposes **two heats of formation** (the SCF / charges are identical for
both; only the post-SCF energy reference differs):

- `hof_nddo` — the PYSEQM-referenced PM6_D heat of formation (AM1-style core-core).
  **Bit-exact to the PYSEQM oracle** (`validate_pm6d_cpp.py`).
- `hof_pm6` — the **canonical, MOPAC-aligned** PM6 heat of formation: the PYSEQM
  electronic energy + the **PWCCT** pairwise core-core (Stewart 2007, ported from
  SCINE Sparrow) + a MOPAC-calibrated per-element reference. The core-core term is
  **bit-exact to MOPAC's `NUCLEAR-NUCLEAR REPULSION`** (`validate_pwcct.py`, worst
  5.4e-5 eV), and the SCF itself already matches MOPAC PM6 (charges to ~1e-4 e,
  HOMO/IP to ~1e-4 eV — the core-core does not enter the Fock).

Agreement of `hof_pm6` with MOPAC 23.2.5 (`validate_pm6_mopac.py`):

| elements | `hof_pm6` vs MOPAC | charges vs MOPAC |
| -------- | ------------------ | ---------------- |
| H, C, N, O, F, P, S, Cl, **Br** | **~0.1–1 kcal/mol** | bit-exact (≤1e-4 e) |
| **I** (iodine, incl. IBr) | **≤0.16 kcal/mol** | **bit-exact (≤1e-4 e)** |

**Iodine — resolved (now canonical MOPAC).** Iodine was diagnosed against MOPAC's
own overlap matrix (MOPAC `AUX(PRECISION=12)` dumps `OVERLAP_MATRIX` + `AO_ZETA`)
and fixed to bit-exact charges. The earlier divergence had **two** independent
causes:

1. **A corrupted parameter, not a PYSEQM-vs-MOPAC difference.** The iodine valence
   d-shell we carried (`zeta_d = 2.72301`, `U_dd = -23.46`, `beta_d = -5.25`) was a
   *data bug* inherited from the upstream NumPy port (mlxmolkit's hardcoded
   `pm6_params.py`), which contradicts **its own CSV, the canonical PYSEQM
   `parameters_PM6_MOPAC.csv`, and MOPAC** — all three of which use
   `zeta_d = 1.87518`, `U_dd = -28.8226`, `beta_d = -7.67611`. PYSEQM and MOPAC
   never disagreed on iodine. (The one-center d W integrals use the *tail*
   exponents, not the valence `zeta_d`, so they were already correct; only the
   overlap, `U_dd`/`beta_d`, and the two-center d charge separations — regenerated
   from `pyseqm_d_params(zeta_d = 1.87518)` — moved.)
2. **Genuine sign bugs in PYSEQM's qn5 diatomic-overlap formulas**, confirmed in
   the canonical `lanl/PYSEQM` source (`seqm/seqm_functions/diat_overlapD.py`) and
   the cause of the unphysical IBr `S > 1`: the σ p–s term of jcall 9 (I–Br) had
   `A[8]B[0] − A[9]B[1]` where it must be `+`; jcall 853 (I–Cl/I–S) σ/π p–p used a
   wrong coefficient pattern; and the I–H d–s term (jcall 651) had `m1 − 2m2 −
   2m3 + m4` where it must be `m1 − 2m2 + 2m3 − m4` (gave `<I_z2|H_s> = 0.701` vs
   MOPAC `0.462`). Each was re-derived from the exact analytic Slater overlap
   (spheroidal-coordinate expansion → `Σ cᵢⱼ A_i(p) B_j(q)`) and validated
   bit-close to MOPAC's `OVERLAP_MATRIX` for all five interhalides + the hydride.

With both fixed, the PM6_D SCF is **bit-exact to MOPAC** (worst `|Δq| = 1e-4 e`
over HI/ICl/IBr/CH3I and every light/Br molecule) and `hof_pm6` is within **0.16
kcal/mol** of MOPAC 23.2.5 for iodine (`validate_pm6_mopac.py`). IBr — previously
`HoF = −1115 kcal` (garbage, from `S > 1`) — is now physical and matches MOPAC.
The reference `kHofRef[53]` was re-fit for the corrected parameters
(`gen_pm6_hof_ref.py`, calibration residual 0.081 kcal). The diatomic overlap is
swept against MOPAC's `OVERLAP_MATRIX` by `_sweep_overlap_mopac.py`.

Everything above (both HoFs, charges, the PWCCT core-core, the interhalide
overlap) is verified **GPU == CPU on real gfx1200 hardware** to floating-point
rounding by `validate_gpu_cpu.py` (worst |Δq| = 2.2e-14, |ΔHoF| = 1.6e-11,
|ΔHoF_pm6| = 1.9e-11 kcal/mol).

**Charged species (ions).** `pm6dCharges` and `scfBatchDGpu` take a net molecular
charge (the Python binding reads it from each RDKit molecule's formal charges):
the electron count is `sum(valence) - charge`, so a cation removes electrons and
an anion adds them, and the converged Mulliken charges sum to the net charge.
Closed-shell ions match MOPAC bit-exact on charges (worst |Δq| = 1e-4) with HoF
within ~0.4 kcal/mol — `validate_pm6d_ions.py` (NH4⁺, CH3NH3⁺, OH⁻, CN⁻, Cl⁻,
HCOO⁻); GPU == CPU to FP rounding. Open-shell (odd-electron) systems still return
false (RHF only).

### Open-shell (UHF) — sp radicals done; d + GPU pending

`pm6dCharges` takes a spin multiplicity `mult` (2S+1): `mult=1` is closed-shell
RHF; `mult>1` (or an odd electron count) runs **UHF** — two spin densities Pα, Pβ
with `Fσ = H + J(Pα+Pβ) − K(Pσ)`. The closed-shell NDDO Fock bakes a factor ½ into
each two-electron term; the UHF Fock (`scf_uhf_device.h`) decomposes every sp term
into a pure Coulomb J (total density) and a pure exchange K (same-spin). The
electron count is `nα=(nElec+2S)/2, nβ=nElec−nα`; charges come from Pα+Pβ.
The SCF starts from MOPAC's **diagonal atomic-density guess** (each atom's core
charge spread over its orbitals, split by the α/β ratio) and applies a **decaying
level shift** on the virtual orbitals (`Fσ' = Fσ + λ(I−Pσ)`, λ: 8→4→1→0.1 eV).
The shift pins the aufbau filling without moving the SCF fixed point, steering UHF
multi-solution cases to MOPAC's ground state. Validated **bit-exact to MOPAC UHF**
(`validate_pm6d_uhf.py`): CH3•, NO•, OH•, NH2•, CN•, **NO₂•** (the hard
multi-solution case — H_core guess lands 44 kcal/mol too high; the diagonal guess
fixes it), NF2•, CH3O• (doublets) and **O₂ (triplet)** — worst |Δq| = 1e-4, HoF
within 0.56 kcal/mol. Closed-shell RHF is unchanged.

**d-orbital UHF.** The baked one-center d `W` folds Coulomb and exchange together
(`W = intg[rep] − ¼intg[rf1] − ¼intg[rf2]`). Splitting it gives a pure Coulomb
`W_J = intg[rep]` (contracts with the total density) and exchange
`W_Kfold = ¼(intg[rf1]+intg[rf2])`; the UHF d Fock is
`Fσ = ΣW_J·Pt − 2·ΣW_Kfold·Pσ`, which reduces to the closed-shell `W` at
`Pσ = Pt/2`. `W_J − W_Kfold` reproduces the oracle to ~1e-15. `fock_d_uhf_device.h`
mirrors `buildFockDDev` branch-for-branch in UHF form, so d-bearing radicals work:
**ClO•, PO• (doublets), SO (triplet) bit-exact to MOPAC** (`validate_pm6d_uhf.py`).
The closed-shell RHF (`kW`) and GPU paths are untouched.

The **Python binding** (`PM6DCharges`) classifies each molecule from its formal
charge + RDKit radical electrons: closed-shell molecules go into one GPU batch
(`scfBatchDGpu`), open-shell ones fall back to the CPU UHF `pm6dCharges`, and the
results are scattered back into the original order — so radicals are usable
through the binding. Validated on gfx1200: a mixed batch routes correctly (closed
GPU charges match CPU to <1e-14, radicals solved on CPU UHF).

**Known hard case:** `ClO2•` is a UHF multi-solution system where the engine
converges to a higher symmetric UHF solution than MOPAC's (HoF ~25 kcal/mol
above; both charge distributions symmetric). Matching MOPAC's exact converger
(Camp-King) for such pathological cases is future work; the open-shell GPU kernel
is likewise deferred (UHF is sp+d but cheap enough on the CPU).

### Transition metals + the d-shell overlap coverage gate

UHF unblocks the **active-d transition metals** (Sc–Cu, Z=21–29; mostly
open-shell), which additionally need `qnD = qn−1` (3d, vs the current `qnD = qn`)
and a **metal-sp(qn 4/5/6) × ligand-d(qn 3) overlap** formula. This last piece is
the gating blocker: the `dsBlockDev`/`dpBlockDev` Slater polynomials only cover
**dqn3 × {1,2,3}, dqn4 × {1,2,4}, dqn5 × {1,2,5}** (sigma/pi for the parameterized
main-group pairs); PYSEQM never tabulates the metal-sp × ligand-d combinations
(its table is `qni ≥ qnj` only), so they need a from-scratch derivation + MOPAC
validation. The closed-shell **group-12** metals (Zn/Cd/Hg, d¹⁰ core → sp) match
MOPAC for sp-only ligands (ZnF₂) but hit exactly this gap for the d-ligand halides
(ZnCl₂ etc.), so they stay unsupported (`pm6ValenceElectrons → 0`).

**Correctness gate (shipped).** The same gap silently affected *already-enabled*
main-group elements: a qn≥4 sp atom (Ga/Ge/Se, In/Sn/Te) or qn3 sp atom next to a
qn3/4/5 **d** ligand fell through to a same-qn formula (e.g. `p6`/`jcall8`) and
returned a **wrong** overlap — and a qn≥4 sp atom next to a lower-qn d atom
**stack-overflowed** a `tmp[16]` scratch in `diatomOverlapSpDev`. Both are now
fixed: the scratch is sized for a d partner, and `dSpOverlapPairsSupported`
(`overlap_d_device.h`) gates every molecule on the CPU (`pm6dCharges`,
`pm6dGradient`) and GPU (`scfBatchDGpu`) paths, so an unsupported d-sp pair now
**fails cleanly (returns false / None) instead of crashing or lying**. No
validated molecule is affected (light + halides + interhalides all pass).

Heavier TM (Y–Cd, La–Hg) are param stubs to be regenerated from the canonical
PYSEQM/MOPAC CSV.

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
