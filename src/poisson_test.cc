#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <random>

#include "gp.hh"
#include "wls.hh"
#include "generator.hh"

using std::cout;
using std::endl;
using linalg::sq;

int main(int argc, char* argv[]) {
  if (argc==1) {
    cout << "usage: " << argv[0] << " c0 c1 c2" << endl;
    return 1;
  }

  std::vector<double> cs {
    atof(argv[1]),
    atof(argv[2]),
    atof(argv[3])
  };

  // [105, 160] -> [0, 55]
  linalg::change_poly_coords(cs.data(),cs.size(),1.,105.);
  cout << "input parameters:\n";
  for (double c : cs) cout << c << '\n';
  cout << std::flush;

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
  std::mt19937 gen(123);
  std::vector<double> xs(n), ys(n), us(n);
  for (unsigned i=0; i<n; ++i) {
    us[i] = 1./std::sqrt(
      ys[i] = std::poisson_distribution<unsigned>(std::exp(f(
        xs[i] = i + 0.5
      )))(gen)
    );
    ys[i] = std::log(ys[i]);
  }

  std::vector<double> A;
  A.reserve(fs.size()*xs.size());
  for (auto& f : fs)
    for (auto x : xs)
      A.push_back(f(x));

  wls(A.data(),ys.data(),us.data(),xs.size(),fs.size(),ps.data());

  cout << "\nfitted parameters:\n";
  for (auto p : ps)
    cout << p << '\n';
  cout << std::flush;
}
