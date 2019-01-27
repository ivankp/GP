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
using linalg::sq;
using nlohmann::json;

auto time_seed() {
  const auto t = std::chrono::system_clock::now().time_since_epoch();
  return
    std::chrono::duration_cast<std::chrono::microseconds>(t).count()
    - std::chrono::duration_cast<std::chrono::seconds>(t).count()*1000000;
}

template <typename T>
const T& cref(const json& j) { return j.get_ref<const T&>(); }

int main(int argc, char* argv[]) {
  json in, out;
  std::cin >> in;

  std::vector<double> cs;
  { std::stringstream ss(cref<string>(in["cs"]));
    double x;
    while (ss >> x) cs.push_back(x);
  }
  out["params"]["input"] = cs;

  // [105, 160] -> [0, 1]
  linalg::change_poly_coords(cs.data(),cs.size(),55.,105.);

  const unsigned n = atof(cref<string>(in["gen_n"]).c_str());
  const bool gen_exp = in["gen_exp"]=="true";
  const bool fit_exp = in["fit_exp"]=="true";

  // Generate pseudo-data ===========================================
  const auto& seed_str = cref<string>(in["seed"]);
  const long seed = (seed_str.empty()
      || seed_str.find_first_not_of("0123456789")!=string::npos)
    ? time_seed()
    : atol(seed_str.c_str());
  std::mt19937 gen(seed);

  std::vector<double> xs(n), ys(n), fit_ys(n), us(n), res(n);
  for (unsigned i=0; i<n; ++i) {
    const double x = xs[i] = (i + 0.5)/n;

    double f = 0;
    for (auto j=cs.size(); j; ) --j, f += cs[j]*std::pow(x,j);
    if (gen_exp) f = std::exp(f);

    const double y = ys[i] = std::poisson_distribution<unsigned>(f)(gen);
    double& u = us[i] = std::sqrt(y);

    if (fit_exp) {
      u = 1./u;
      fit_ys[i] = std::log(y);
    } else {
      fit_ys[i] = y;
    }
  }

  // Fit function ===================================================
  std::vector<double> ps(atoi(cref<string>(in["fit_deg"]).c_str())+1);

  std::vector<double> A;
  A.reserve(ps.size()*xs.size());
  for (unsigned i=0, n=ps.size(); i<n; ++i) {
    for (auto x : xs)
      A.push_back(std::pow(x,i));
  }

  wls(A.data(),fit_ys.data(),us.data(),xs.size(),ps.size(),ps.data());

  // Residuals ======================================================
  for (unsigned i=0; i<n; ++i) {
    double f = 0;
    for (auto j=ps.size(); j; ) --j, f += ps[j]*std::pow(xs[i],j);
    if (fit_exp) f = std::exp(f);

    res[i] = 100.*(ys[i] - f)/f;
  }

  // Fit Gaussian process to residuals ==============================
  const unsigned gp_n = atof(cref<string>(in["gp_n"]).c_str());
  const double kernel_coeff
    = -0.5/sq(atof(cref<string>(in["gp_l"]).c_str()));
  const auto gp = GP(xs,res,
    generator(0,res.size(),[](auto){ return 1; }),
    generator(0,gp_n+1,[dx=1./gp_n](auto i){ return dx*i; }),
    [=](auto a, auto b){ return std::exp(kernel_coeff*sq(a-b)); }
  );

  // Write output ===================================================
  // [0, 1] -> [105, 160]
  linalg::change_poly_coords(ps.data(),ps.size(),1./55.,-105./55.);

  out["params"]["fitted"] = ps;
  out["seed"] = seed;
  out["ys"] = ys;
  out["res"] = res;
  out["gp"] = gp;
  cout << out;
}
