#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>

#include "gp.hh"
#include "wls.hh"
#include "generator.hh"

using std::cout;
using std::endl;
using linalg::sq;

int main(int argc, char* argv[]) {
  if (argc==1) {
    cout << "usage: " << argv[0] << " p0 p1 p2" << endl;
    return 1;
  }

  std::vector<double> p {
    atof(argv[1]),
    atof(argv[2]),
    atof(argv[3])
  };
  linalg::change_poly_coords(p.data(),p.size(),55.,105.);
  for (double p : p) cout << p << '\n';
}
