#include <gsl/gsl_multimin.h>
#include <cstdio>
#include <array>
#include <vector>

namespace ivanp {

struct gsl_multimin_opts {
  bool verbose = false;
  double tolerance = 1e-3;
  size_t max_iter = 1000;
  const gsl_multimin_fminimizer_type* min_type
    = gsl_multimin_fminimizer_nmsimplex2;
};

template <typename F>
std::vector<double> gsl_multimin (
  const std::vector<std::array<double,2>>& start_step,
  F&& f,
  const gsl_multimin_opts& opts = { }
) {
  /* Initialize method and iterate */
  gsl_multimin_function minex_func;
  const unsigned n = start_step.size();
  minex_func.n = n;
  minex_func.params = &f;
  minex_func.f = [](const gsl_vector *v, void *p){
    return (*reinterpret_cast<F*>(p))(v->data);
  };

  gsl_vector *x = gsl_vector_alloc(n);
  gsl_vector *step = gsl_vector_alloc(n);
  for (unsigned i=0; i<n; ++i) {
    gsl_vector_set( x   , i, std::get<0>(start_step[i]) );
    gsl_vector_set( step, i, std::get<1>(start_step[i]) );
  }

  gsl_multimin_fminimizer *s = gsl_multimin_fminimizer_alloc(opts.min_type,n);
  gsl_multimin_fminimizer_set(s, &minex_func, x, step);

  size_t iter = 0;
  int status;
  double size;

  do {
    ++iter;
    status = gsl_multimin_fminimizer_iterate(s);

    if (status) break;

    size = gsl_multimin_fminimizer_size(s);
    status = gsl_multimin_test_size(size, opts.tolerance);

    if (opts.verbose) {
      printf("%5lu f = %10.3e size = %9.3e", iter, s->fval, size);
      for (size_t i=0; i<n; ++i)
        printf(" %10.3e",gsl_vector_get(s->x,i));
      printf("\n");
    }
  } while (status == GSL_CONTINUE && iter < opts.max_iter);

  std::vector<double> ret(n);
  for (auto i=n; i; ) { --i; ret[i] = gsl_vector_get(s->x,i); }

  gsl_vector_free(x);
  gsl_vector_free(step);
  gsl_multimin_fminimizer_free(s);

  return ret;
}

}
