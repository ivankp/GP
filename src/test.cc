#include <iostream>
#include <vector>
#include <cmath>

#include "gp.hh"
#include "generator.hh"

using std::cout;
using std::endl;

int main(int argc, char* argv[]) {
  std::vector<double> xs {1,2,3,4,5,6,7,8,9,10};
  std::vector<double> ys {5.89, 4.91, 4.7, 3.61, 3.56, 3.17, 2.32, 1.85, 1.52, 1.09};
  std::vector<double> us {0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2,0.2};
  std::vector<double> ts {1.5,2.5,3.5};

  const auto gp = gp::GP(xs,ys,us,
    generator(0,1000,[](auto i){ return 1.+(9./1000)*i; }),
    [](auto a, auto b){
      return std::exp((-0.5/gp::sq(2))*gp::sq(a-b));
    }
  );

  for (const auto& p : gp)
    cout << p[0] << ' ' << p[1] << '\n';
  cout << std::flush;
}
