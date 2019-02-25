#include <iostream>
#include <string>
#include <vector>

using namespace std;

template <typename T>
void LT_L(T* L, unsigned n) {
  for (unsigned i=0, k=0; i<n; ++i) {
    for (unsigned j=0; j<=i; ++j, ++k) {
      cout << k << ' ';
      const unsigned r = i-j;
      cout << r << ' ';
      unsigned a = k;
      auto& l = L[k] = L[a] * L[a+r];
      cout << l << ' ';
      for (unsigned d=i+1; d<n; ++d) {
        a += d;
        l += "+" + L[a]*L[a+r];
      }
      cout << endl;
    }
  }
}

std::string operator*(const std::string& a, const std::string& b) {
  return a + b;
}

int main(int argc, char* argv[]) {
  std::vector<std::string> L {
    // "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"
    "a", "b", "c", "d", "e", "f", "g", "h", "i", "j"
  };
  const unsigned n = 4;

  LT_L(L.data(),n);

  for (unsigned i=0, k=0; i<n; ++i) {
    for (unsigned j=0; j<=i; ++j, ++k) {
      std::cout << ' ' << L[k];
    }
    std::cout << '\n';
  }

}
