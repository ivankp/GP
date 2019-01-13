#ifndef GP_HH
#define GP_HH

// #include <iostream>
#include <array>
#include <vector>
#include <iterator>
#include <type_traits>
#include <cmath>

namespace gp {

template <typename T>
constexpr auto sq(T x) noexcept { return x*x; }
template <typename T, typename... TT>
constexpr auto sq(T x, TT... xx) noexcept { return sq(x)+sq(xx...); }

constexpr unsigned utn(unsigned n) noexcept { return n*(n+1) >> 1; }

void cholesky(double* A, unsigned N) noexcept;
void solve_triang(const double* L, double* v, unsigned n) noexcept;
double dot(const double* a, const double* b, unsigned n) noexcept;

template <typename Xs, typename Ys, typename Us, typename Ts, typename Kernel>
std::vector<std::array<double,2>> GP(
  const Xs& xs, // training points coordinates
  const Ys& ys, // training points values
  const Us& us, // uncertainties (add to diagonal)
  const Ts& ts, // test points
  Kernel&& kernel // kernel function
) {
  const auto x_begin = begin(xs);
  const auto x_end = end(xs);
  const auto nx = std::distance(x_begin, x_end);
  using nx_t = std::remove_const_t<decltype(nx)>;
  const auto N = utn(nx);
  double* L = new double[N];

  double* k = L;
  auto u = begin(us);
  // compute covariance matrix K
  for (auto a=x_begin; a!=x_end; ++a) {
    for (auto b=x_begin; ; ++b, ++k) {
      *k = kernel(*a,*b);
      if (b==a) { *k += sq(*u); ++u; ++k; break; }
    }
  }

  // std::cout << std::endl;
  // for (unsigned i1=0, i3=0; i1<n; ++i1) {
  //   for (unsigned i2=0; i2<=i1; ++i2, ++i3) {
  //     std::cout << L[i3] << ' ';
  //   }
  //   std::cout << std::endl;
  // }
  // std::cout << std::endl;

  cholesky(L,N); // K = L L

  // mean = k* (LL)^-1 y

  // std::cout << std::endl;
  // for (unsigned i1=0, i3=0; i1<n; ++i1) {
  //   for (unsigned i2=0; i2<=i1; ++i2, ++i3) {
  //     std::cout << L[i3] << ' ';
  //   }
  //   std::cout << std::endl;
  // }
  // std::cout << std::endl;

  // Solve L^-1 y
  const auto y_begin = begin(ys);
  const auto y_end = end(ys);
  const auto ny = std::distance(y_begin, y_end);
  using ny_t = std::remove_const_t<decltype(ny)>;

  double* y = new double[ny];
  for (ny_t i=0; i<ny; ++i)
    y[i] = *std::next(y_begin,i);

  solve_triang(L,y,ny);

  // for (ny_t i=0; i<ny; ++i)
  //   std::cout << y[i] << ' ';
  // std::cout << std::endl;
  // std::cout << std::endl;

  // Solve k* L^-1
  const auto t_begin = begin(ts);
  const auto t_end = end(ts);
  const auto nt = std::distance(t_begin, t_end);
  using nt_t = std::remove_const_t<decltype(nt)>;

  double* ks = new double[nt*nx];
  for (nt_t i=0; i<nt; ++i) {
    const auto t = std::next(t_begin,i);
    const auto k = ks + nx*i;
    for (nx_t j=0; j<nx; ++j)
      *(k+j) = kernel( *t, *std::next(x_begin,j) );

    solve_triang(L,k,nx);
  }

  // for (nt_t i=0; i<nt; ++i) {
  //   for (nx_t j=0; j<nx; ++j)
  //     std::cout << ks[i*nx+j] << ' ';
  //   std::cout << std::endl;
  // }
  // std::cout << std::endl;

  std::vector<std::array<double,2>> out(nt);
  for (nt_t i=0; i<nt; ++i) {
    const auto t = std::next(t_begin,i);
    const auto k = ks + nx*i;
    out[i] = {
      // mean = (k* L^-1) (L^-1 y)
      dot(k, y, nx),
      // var = k** - (k* L^-1) (L^-1 k*)
      std::sqrt(kernel(*t,*t) - dot(k, k, nx))
    };
  }

  delete[] ks;
  delete[] y;
  delete[] L;

  return out;
}

}

#endif
