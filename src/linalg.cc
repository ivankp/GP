#include "linalg.hh"
#include <cmath>

namespace linalg {

void cholesky(double* A, unsigned N) noexcept {
  unsigned col = 0, row = 0;
  for (unsigned k=0; k<N; ++k) {
    double s = 0;
    if (col==row) {
      ++row;
      for (unsigned i=1; i<row; ++i)
        s += sq(A[k-i]);
      A[k] = std::sqrt(A[k]-s);
      col = 0;
    } else {
      const unsigned r = utn(col);
      for (unsigned i=0; i<col; ++i)
        s += A[r+i]*A[k+i-col];
      A[k] = (A[k]-s)/A[r+col];
      ++col;
    }
  }
}

void solve_triang(const double* L, double* v, unsigned n) noexcept {
  v[0] /= L[0];
  for (unsigned i=1, k=0; i<n; ++i) {
    for (unsigned j=0; j<i; ++j)
      v[i] -= L[++k] * v[j];
    v[i] /= L[++k];
  }
}

double dot(const double* a, const double* b, unsigned n) noexcept {
  double x = 0;
  for (unsigned i=0; i<n; ++i)
    x += a[i] * b[i];
  return x;
}

}
