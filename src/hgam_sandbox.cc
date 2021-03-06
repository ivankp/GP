#include <iostream>
#include <sstream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <random>
#include <chrono>

#include <nlohmann/json.hpp>

#include "gp.hh"
#include "wls.hh"
#include "generator.hh"

using std::cout;
using std::endl;
using std::string;
using std::vector;
using ivanp::linalg::sq;
using nlohmann::json;

using namespace ivanp;

auto time_seed() {
  const auto t = std::chrono::system_clock::now().time_since_epoch();
  return
    std::chrono::duration_cast<std::chrono::microseconds>(t).count()
    - std::chrono::duration_cast<std::chrono::seconds>(t).count()*1000000;
}

template <typename T>
const T& cref(const json& j) { return j.get_ref<const T&>(); }

template <typename T>
vector<T> split(const json& j) {
  vector<T> v;
  std::stringstream ss(cref<string>(j));
  T x;
  while (ss >> x) v.emplace_back(std::move(x));
  return v;
}

int main(int argc, char* argv[]) {
  json in, out;
  std::cin >> in;

  constexpr double x_min = 105, x_max = 160, x_range = x_max-x_min;

  vector<double> cs = split<double>(in["gen_cs"]);
  out["params"]["gen"] = cs;

  const unsigned n0 = atof(cref<string>(in["gen_n"]).c_str());
  const bool gen_exp = in["gen_exp"]=="true";
  const bool fit_exp = in["fit_exp"]=="true";

  // Generate pseudo-data ===========================================
  const auto& seed_str = cref<string>(in["gen_seed"]);
  const long seed = (seed_str.empty()
      || seed_str.find_first_not_of("0123456789")!=string::npos)
    ? time_seed()
    : atol(seed_str.c_str());
  std::mt19937 gen(seed);

  auto excl = split<double>(in["gen_excl"]);
  bool skipped = true;
  unsigned n = n0, n1 = 0, n_skip = 0;
  if (excl.size() >= 2) {
    if (excl[0] > excl[1]) std::swap(excl[0],excl[1]);
    if (excl[0] < x_min) excl[0] = x_min;
    if (excl[1] > x_max) excl[1] = x_max;
    const double s = x_range/n;
    n1 = (excl[0]-x_min) / s;
    n_skip = (excl[1]-excl[0]) / s;
    n -= n_skip;
    skipped = false;
  }

  vector<double> xs(n), ys(n), us(n), fit_ys(n), fit_us(n);
  for (unsigned i=0, xi=0; i<n; ++i, ++xi) {
    if (!skipped && i >= n1) {
      xi += n_skip;
      // gen.discard(n_skip);
      // for (auto i=n_skip; i; --i) std::poisson_distribution<unsigned>(100)(gen);
      skipped = true;
    }
    const double x = xs[i] = x_min + (xi + 0.5)*(x_range/n0);

    double f = 0; // sampled distribution
    for (auto j=cs.size(); j; ) --j, f += cs[j]*std::pow(x,j);
    if (gen_exp) f = std::exp(f);

    const double y = ys[i] = std::poisson_distribution<unsigned>(f)(gen);
    const bool gt0 = y > 0;
    const double u = us[i] = gt0 ? y : 1;

    if (fit_exp) {
      fit_ys[i] = gt0 ? std::log(y) : 0;
      fit_us[i] = 1./u;
    } else {
      fit_ys[i] = y;
      fit_us[i] = u;
    }
  }

  // Fit function ===================================================
  vector<double> ps(atoi(cref<string>(in["fit_deg"]).c_str())+1);

  vector<double> A;
  A.reserve(ps.size()*xs.size());
  for (unsigned i=0, n=ps.size(); i<n; ++i) {
    for (auto x : xs)
      A.push_back(std::pow(x,i));
  }

  vector<double> cov(linalg::utn(ps.size()));

  wls(A.data(), fit_ys.data(), fit_us.data(), xs.size(),
      ps.size(), ps.data(), cov.data());

  // turn cov matrix into corr and fractional unc
  for (unsigned i=0, k=0, n=ps.size(); i<n; ++i) {
    const unsigned a = linalg::utn(i+1)-1;
    cov[a] = std::sqrt(cov[a]);
    for (unsigned j=0; j<i; ++j, ++k) {
      const unsigned b = linalg::utn(j+1)-1;
      cov[k] /= cov[a]*cov[b];
    }
    ++k;
  }
  for (unsigned i=0, n=ps.size(); i<n; ++i) {
    const unsigned a = linalg::utn(i+1)-1;
    cov[a] = std::abs(cov[a]/ps[i]);
  }

  // Differences ====================================================
  vector<double> diff(n);
  for (unsigned i=0; i<n; ++i) {
    double f = 0; // fitted function
    for (auto j=ps.size(); j; ) --j, f += ps[j]*std::pow(xs[i],j);
    if (fit_exp) f = std::exp(f);

    // diff[i] = 100.*(ys[i] - f)/f;
    diff[i] = ys[i] - f;
  }

  // Fit Gaussian process to differences ============================
  const bool gp_diff = in["gp_diff"]=="true";
  const auto& unc_str = cref<string>(in["gp_u"]);
  if (!unc_str.empty()) {
    const double unc = sq(atof(unc_str.c_str()));
    for (double& u : us) u = unc;
  }
  const unsigned gp_n = atof(cref<string>(in["gp_n"]).c_str());
  const double kernel_coeff
    = -0.5/sq(atof(cref<string>(in["gp_l"]).c_str()));
  const auto gp = gp::regression(xs,(gp_diff ? diff : ys),us,
    generator(0,gp_n+1,[dx=x_range/gp_n](auto i){ return x_min+dx*i; }), // test points
    [=](auto a, auto b){ return std::exp(kernel_coeff*sq(a-b)); } // kernel
  );

  // Write output ===================================================

  out["params"]["fit"] = ps;
  out["seed"] = seed;
  out["xs"] = xs;
  out["ys"] = ys;
  out["gp"] = gp;
  out["cov"] = cov;
  cout << out;
}
