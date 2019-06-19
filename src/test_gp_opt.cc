#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <gsl/gsl_multimin.h>

#include "gp.hh"
#include "generator.hh"
#include "gsl_multimin.hh"

using std::cout;
using std::endl;
using ivanp::linalg::sq;

using namespace ivanp;

int main(int argc, char* argv[]) {
  std::vector<double> xs {
-7.288352, -6.354489, -6.343246, -5.873644, -4.744751, -4.092581, -3.743588,
-2.788551, -2.173762, -0.897309, 0.500722, 0.770547, 1.009111, 2.386,
2.475254, 4.245013, 4.332956, 4.908614, 5.837199, 6.143407
  };
  std::vector<double> ys {
-1.753582, -0.048711, 0.030086, 0.244986, -0.815186, -1.223496, -1.173352,
0.388252, 1.405444, 1.691977, -0.13467, -0.743553, -1.022923, -2.69914,
-2.226361, -1.287966, -1.116046, -1.574499, -1.080229, -0.84384
  };
  auto us = generator(xs.begin(),xs.end(),[](auto x){ return 0.5; });

  const auto hs = gsl_multimin({
      { 1., 0.1 },
      { 1., 0.1 }
    }, [&](const double* p){
      return gp::logml(xs, ys, us,
        [](auto a, auto b, double s, double l){
          return s * std::exp(-0.5*sq((a-b)/l));
        }, p[0], p[1]);
    }, { verbose: true, 1e-5 }
  );

  cout << hs[0] << ", " << hs[1] << endl;
}
