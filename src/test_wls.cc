#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>

#include "wls.hh"

using std::cout;
using std::endl;

int main(int argc, char* argv[]) {
  std::vector<double> xs {0.1,0.2,0.3,0.4,0.5,1,2,3,4,5,6,7,8,9,10};
  std::vector<double(*)(double)> fs {
    [](double x){ return 1.;  },
    [](double x){ return x;   },
    [](double x){ return x*x; },
    [](double x){ return x*x*x; }
  };
  std::vector<double> ps(fs.size());

  std::vector<double> A;
  A.reserve(fs.size()*xs.size());
  for (auto& f : fs)
    for (auto x : xs)
      A.push_back(f(x));

  // for (auto x : A)
  //   cout << x << '\n';
  // cout << std::endl;

  std::vector<double> cs { 0.001, 2, 3, 0.5 };
  std::vector<double> ys(xs.size());
  std::vector<double> us(xs.size());
  for (unsigned i=0, nx=xs.size(); i<nx; ++i) {
    auto& y = ys[i] = 0;
    us[i] = 1;
    for (unsigned j=0, nf=fs.size(); j<nf; ++j)
      y += cs[j] * fs[j](xs[i]);
  }

  // for (auto y : ys)
  //   cout << y << '\n';
  // cout << std::endl;

  wls(A.data(),ys.data(),us.data(),xs.size(),fs.size(),ps.data());

  for (auto p : ps)
    cout << p << '\n';
  cout << std::flush;
}
