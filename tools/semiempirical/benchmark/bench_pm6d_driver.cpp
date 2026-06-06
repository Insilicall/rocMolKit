// SPDX-FileCopyrightText: Copyright (c) 2025 InsilicAll. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// PM6_D throughput benchmark driver (SECONDARY / perf only — not correctness).
//
// Reads a binary batch of real (ETKDG-embedded) molecules produced by
// bench_pm6d_gpu.py, then times:
//   * the batched GPU SCF  scfBatchDGpu  (the throughput path), and
//   * the CPU single-molecule loop pm6dCharges (the baseline),
// reporting molecules/sec, atoms/sec, and the GPU/CPU speedup. The GPU number
// is reported both as a cold (first call, includes one-time JIT/H2D/setup) and
// a steady-state (best of repeated calls) throughput.
//
// Binary batch format (little-endian, host order; written by numpy.tofile):
//   int32   nMol
//   int32   totalAtoms
//   int32[nMol]        molNAtoms
//   int32[totalAtoms]  atomsAll      (atomic numbers, concatenated)
//   float64[3*totalAtoms] coordsAll  (x,y,z per atom, Angstrom)
//   int32[nMol]        molCharge     (net charge per molecule)
//
// Results are printed as a single JSON line (prefixed RESULT_JSON:) for the
// Python harness to parse, plus a human-readable summary.

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "pm6_params.h"
#include "scf_d.h"
#include "scf_d_kernels.h"

using namespace nvMolKit::semiempirical;
using Clock = std::chrono::steady_clock;

static double secsSince(Clock::time_point t0) {
  return std::chrono::duration<double>(Clock::now() - t0).count();
}

template <typename T>
static void readN(FILE* f, T* dst, size_t n, const char* what) {
  if (std::fread(dst, sizeof(T), n, f) != n) {
    std::fprintf(stderr, "short read on %s\n", what);
    std::exit(2);
  }
}

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s batch.bin [gpu_repeats] [cpu_repeats]\n", argv[0]);
    return 2;
  }
  const int gpuRepeats = (argc > 2) ? std::atoi(argv[2]) : 3;
  const int cpuRepeats = (argc > 3) ? std::atoi(argv[3]) : 1;

  FILE* f = std::fopen(argv[1], "rb");
  if (!f) {
    std::fprintf(stderr, "cannot open %s\n", argv[1]);
    return 2;
  }
  int32_t nMol = 0, totalAtoms = 0;
  readN(f, &nMol, 1, "nMol");
  readN(f, &totalAtoms, 1, "totalAtoms");

  std::vector<int32_t> molNAtoms32(nMol), atomsAll32(totalAtoms), molCharge32(nMol);
  readN(f, molNAtoms32.data(), nMol, "molNAtoms");
  readN(f, atomsAll32.data(), totalAtoms, "atomsAll");
  std::vector<double> coordsAll(3 * (size_t)totalAtoms);
  readN(f, coordsAll.data(), coordsAll.size(), "coordsAll");
  readN(f, molCharge32.data(), nMol, "molCharge");
  std::fclose(f);

  // The engine takes `int`; build int views and derive nBasis.
  std::vector<int> molNAtoms(molNAtoms32.begin(), molNAtoms32.end());
  std::vector<int> atomsAll(atomsAll32.begin(), atomsAll32.end());
  std::vector<int> molCharge(molCharge32.begin(), molCharge32.end());
  std::vector<int> molNBasis(nMol, 0);
  {
    int off = 0;
    for (int m = 0; m < nMol; ++m) {
      int nb = 0;
      for (int a = 0; a < molNAtoms[m]; ++a) nb += pm6NumOrbitals(atomsAll[off + a]);
      molNBasis[m] = nb;
      off += molNAtoms[m];
    }
  }

  std::vector<double> charges(totalAtoms), hof(nMol), hofPm6(nMol);
  std::vector<int> conv(nMol);

  const int maxIter = 800;
  const double convTol = 1e-10;

  // ---- GPU: cold call (includes one-time HIP init / JIT / H2D) ----
  auto t0 = Clock::now();
  bool ok = scfBatchDGpu(nMol, molNAtoms.data(), molNBasis.data(), atomsAll.data(),
                         coordsAll.data(), charges.data(), hof.data(), conv.data(), maxIter,
                         convTol, hofPm6.data(), molCharge.data());
  double gpuCold = secsSince(t0);

  // ---- GPU: steady state (best of N repeats) ----
  double gpuBest = gpuCold;
  for (int r = 0; r < gpuRepeats; ++r) {
    auto t = Clock::now();
    ok = scfBatchDGpu(nMol, molNAtoms.data(), molNBasis.data(), atomsAll.data(),
                      coordsAll.data(), charges.data(), hof.data(), conv.data(), maxIter,
                      convTol, hofPm6.data(), molCharge.data());
    gpuBest = std::fmin(gpuBest, secsSince(t));
  }

  int converged = 0;
  for (int m = 0; m < nMol; ++m) converged += (conv[m] != 0);

  // Charge-conservation sanity: max |sum(q_mol) - Qnet| over molecules.
  double worstNet = 0.0;
  {
    int off = 0;
    for (int m = 0; m < nMol; ++m) {
      double s = 0.0;
      for (int a = 0; a < molNAtoms[m]; ++a) s += charges[off + a];
      worstNet = std::fmax(worstNet, std::fabs(s - (double)molCharge[m]));
      off += molNAtoms[m];
    }
  }

  // ---- CPU baseline: loop pm6dCharges over the same set ----
  double cpuBest = 1e30;
  for (int r = 0; r < cpuRepeats; ++r) {
    auto t = Clock::now();
    int off = 0;
    for (int m = 0; m < nMol; ++m) {
      std::vector<double> q(molNAtoms[m]);
      double ch = 0, cp = 0;
      pm6dCharges(molNAtoms[m], atomsAll.data() + off, coordsAll.data() + 3 * off, q.data(),
                  &ch, maxIter, convTol, &cp, molCharge[m]);
      off += molNAtoms[m];
    }
    cpuBest = std::fmin(cpuBest, secsSince(t));
  }

  long long atomsTotal = 0;
  for (int m = 0; m < nMol; ++m) atomsTotal += molNAtoms[m];

  const double gpuMolS = nMol / gpuBest;
  const double gpuAtomS = atomsTotal / gpuBest;
  const double gpuColdMolS = nMol / gpuCold;
  const double cpuMolS = nMol / cpuBest;
  const double cpuAtomS = atomsTotal / cpuBest;
  const double speedup = gpuBest > 0 ? (cpuBest / gpuBest) : 0;

  std::printf("=== PM6_D GPU batch throughput (gfx1200) ===\n");
  std::printf("molecules         : %d  (atoms total %lld, converged %d/%d)\n", nMol, atomsTotal,
              converged, nMol);
  std::printf("GPU cold (1st)    : %.4f s  -> %.1f mol/s\n", gpuCold, gpuColdMolS);
  std::printf("GPU steady (best) : %.4f s  -> %.1f mol/s, %.0f atoms/s\n", gpuBest, gpuMolS,
              gpuAtomS);
  std::printf("CPU loop (best)   : %.4f s  -> %.1f mol/s, %.0f atoms/s\n", cpuBest, cpuMolS,
              cpuAtomS);
  std::printf("GPU/CPU speedup   : %.1fx\n", speedup);
  std::printf("charge sanity     : worst |sum(q)-Qnet| = %.2e e\n", worstNet);
  std::printf("scfBatchDGpu ok   : %d\n", ok ? 1 : 0);

  std::printf(
      "RESULT_JSON:{\"nMol\":%d,\"atomsTotal\":%lld,\"converged\":%d,"
      "\"gpu_cold_s\":%.6f,\"gpu_best_s\":%.6f,\"cpu_best_s\":%.6f,"
      "\"gpu_mol_s\":%.4f,\"gpu_atom_s\":%.4f,\"gpu_cold_mol_s\":%.4f,"
      "\"cpu_mol_s\":%.4f,\"cpu_atom_s\":%.4f,\"speedup\":%.4f,"
      "\"worst_net_charge\":%.6e,\"ok\":%d}\n",
      nMol, atomsTotal, converged, gpuCold, gpuBest, cpuBest, gpuMolS, gpuAtomS, gpuColdMolS,
      cpuMolS, cpuAtomS, speedup, worstNet, ok ? 1 : 0);
  return ok ? 0 : 1;
}
