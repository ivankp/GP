#ifndef WLS_HH
#define WLS_HH

void wls(
  const double* A, // matrix of functions values: p (row) then x (col)
  const double* y, // measured values
  const double* u, // uncertainties
  unsigned nx, // number of measured values
  unsigned np, // number of parameters
  double* p // fitted functions coefficients (parameters)
);

#endif