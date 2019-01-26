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
using linalg::sq;
using nlohmann::json;

auto time_seed() {
  const auto t = std::chrono::system_clock::now().time_since_epoch();
  return
    std::chrono::duration_cast<std::chrono::microseconds>(t).count()
    - std::chrono::duration_cast<std::chrono::seconds>(t).count()*1000000;
}

int main(int argc, char* argv[]) {
  json in;
  std::cin >> in;

  std::vector<double> cs;
  { std::stringstream ss(in["cs"].get_ref<const std::string&>());
    double x;
    while (ss >> x) cs.push_back(x);
  }

  // [105, 160] -> [0, 55]
  linalg::change_poly_coords(cs.data(),cs.size(),1.,105.);
  /*
  cout << "input parameters:\n";
  for (double c : cs) cout << c << '\n';
  cout << std::flush;
  */

  std::vector<double(*)(double)> fs {
    [](double x){ return 1.;  },
    [](double x){ return x;   },
    [](double x){ return x*x; }
  };
  std::vector<double> ps(fs.size());

  const auto f = [&](double x){
    double y = 0;
    for (auto i=fs.size(); i; ) --i, y += cs[i]*fs[i](x);
    return y;
  };

  const unsigned n = 55;
  const auto& seed_str = in["seed"].get_ref<const std::string&>();
  const long seed = (seed_str.empty()
      || seed_str.find_first_not_of("0123456789")!=std::string::npos)
    ? time_seed()
    : atol(seed_str.c_str());
  std::mt19937 gen(seed);
  std::vector<double> xs(n), ys(n), log_ys(n), us(n), res(n);
  for (unsigned i=0; i<n; ++i) {
    const double y = std::exp(f( xs[i] = i + 0.5 ));
    us[i] = 1./std::sqrt(
      ys[i] = std::poisson_distribution<unsigned>(y)(gen)
    );
    log_ys[i] = std::log(ys[i]);
    res[i] = ys[i] - y;
  }

  std::vector<double> A;
  A.reserve(fs.size()*xs.size());
  for (auto& f : fs)
    for (auto x : xs)
      A.push_back(f(x));

  wls(A.data(),log_ys.data(),us.data(),xs.size(),fs.size(),ps.data());

  /*
  cout << "\nfitted parameters:\n";
  for (auto p : ps)
    cout << p << '\n';
  cout << std::flush;
  */

  json out;
  out["seed"] = seed;
  out["params"]["input"] = in["cs"];
  out["params"]["transformed"] = cs;
  out["params"]["fitted"] = ps;
  // out["xs"] = xs;
  out["ys"] = ys;
  out["res"] = res;
  cout << out;
}
