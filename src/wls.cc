#include "wls.hh"

// Cowan p. 97
// https://en.wikipedia.org/wiki/Generalized_least_squares

// p = (At V^-1 A)^-1 At V^-1 y

void wls(
  const double* A, // matrix of functions values
  const double* ys, // measured values
  const double* us, // uncertainties
  double* ps, // fitted functions coefficients (parameters)
  unsigned nx, // number of measured values
  unsigned np // number of parameters
) {
  const unsigned N = utn(np);
  double* const L = new double[N];

  // V^-1 = (u^2)^-1
  // LL = At V^-1 A
  // p = At V^-1 y
  // solve  p = L^-1 p  twice

  delete[] L;
}
