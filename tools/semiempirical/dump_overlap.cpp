// Dump the engine's diatomic 9x9 overlap (bond along z) for a Z-pair, to compare
// against MOPAC's AUX OVERLAP_MATRIX. Heavier atom is placed at the origin so the
// engine's "heavier-first" ordering keeps rows=atom A. Bond along +z.
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include "device_macros.h"
#include "core_hamiltonian.h"
#include "pm6_params.h"
#include "overlap_d_device.h"
using namespace nvMolKit::semiempirical;

int main(int argc, char** argv) {
  int zA = std::atoi(argv[1]);   // heavier (rows)
  int zB = std::atoi(argv[2]);   // lighter (cols)
  double R = std::atof(argv[3]); // Angstrom, bond along z
  // optional zeta_d override for atom A and/or B (to test MOPAC's 1.875175)
  AtomIntParams a, b;
  gatherAtomIntParamsD(zA, a);
  gatherAtomIntParamsD(zB, b);
  if (argc > 4) a.zetaD = std::atof(argv[4]);
  if (argc > 5) b.zetaD = std::atof(argv[5]);
  double cA[3] = {0, 0, 0}, cB[3] = {0, 0, R};
  double blk[81];
  int n = diatomOverlapDDev(a, cA, b, cB, blk);
  int nA = a.nOrb, nB = b.nOrb;
  const char* lab9[9] = {"s", "px", "py", "pz", "x2", "xz", "z2", "yz", "xy"};
  std::printf("# engine overlap Z%d(rows,%d orb) x Z%d(cols,%d orb) R=%.4f\n", zA, nA, zB, nB, R);
  std::printf("%6s", "");
  for (int j = 0; j < nB; ++j) std::printf("%11s", lab9[j]);
  std::printf("\n");
  for (int i = 0; i < nA; ++i) {
    std::printf("%6s", lab9[i]);
    for (int j = 0; j < nB; ++j) std::printf("%11.6f", blk[i * nB + j]);
    std::printf("\n");
  }
  return 0;
}
