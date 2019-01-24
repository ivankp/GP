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

void solve_triang_T(const double* L, double* v, unsigned n) noexcept {
  unsigned k = utn(n), i = n-1;
  v[i] /= L[--k];
  for (; i; ) {
    --i;
    unsigned k2 = --k;
    for (unsigned j=n-1; j>i; --j) {
      v[i] -= L[k2] * v[j];
      k2 -= j;
    }
    v[i] /= L[k2];
  }
}

double dot(const double* a, const double* b, unsigned n) noexcept {
  double x = 0;
  for (unsigned i=0; i<n; ++i)
    x += a[i] * b[i];
  return x;
}

void change_poly_coords(double* c, unsigned n, double a, double b) noexcept {
  double p, C;
  for (unsigned i=1; i<n; ++i) {
    C = 1;
    p = b;
    for (unsigned j=1; ; ) {
      C = C * (i+1-j) / j;
      c[i-j] += C * c[i] * p;
      if (j==i) break;
      ++j;
      p *= b;
    }
  }
  p = a;
  for (unsigned i=1; ; ) {
    c[i] *= p;
    if ((++i)==n) break;
    p *= a;
  }
}

}
