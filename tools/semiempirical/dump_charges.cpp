// Read molecules (nAtoms; then Z x y z per line) from stdin, print engine PM6_D
// charges and the canonical (MOPAC-aligned) heat of formation. Used to compare
// against MOPAC's AUX ATOM_CHARGES / HEAT_OF_FORMATION after the overlap fix.
#include <cstdio>
#include <vector>
#include "scf_d.h"
using namespace nvMolKit::semiempirical;

int main() {
  int n;
  while (std::scanf("%d", &n) == 1) {
    std::vector<int> z(n);
    std::vector<double> c(3 * n), q(n);
    for (int i = 0; i < n; ++i)
      std::scanf("%d %lf %lf %lf", &z[i], &c[3 * i], &c[3 * i + 1], &c[3 * i + 2]);
    double hof = 0, hofPm6 = 0;
    bool ok = pm6dCharges(n, z.data(), c.data(), q.data(), &hof, 800, 1e-10, &hofPm6);
    std::printf("ok=%d hof_pyseqm=%.4f hof_pm6=%.4f q=", ok, hof, hofPm6);
    for (int i = 0; i < n; ++i) std::printf("%.6f ", q[i]);
    std::printf("\n");
  }
  return 0;
}
