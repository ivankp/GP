#include <vector>
#include <array>
#include <iterator>
#include <stdexcept>
#include <type_traits>
#include <memory>
#include <Python.h>

#include "gp.hh"
#include "gsl_multimin.hh"

// https://docs.python.org/2/c-api/concrete.html

/*
#include <iostream>
#define TEST(var) \
  std::cout << "\033[36m" #var "\033[0m = " << var << std::endl;
*/

namespace {

template <typename T> struct unpy_impl;

template <typename T> inline T unpy(PyObject* p) {
  return unpy_impl<T>::call(p);
}

template <> struct unpy_impl<double> {
  using type = double;
  static type call(PyObject* p) {
    return PyFloat_AsDouble(p);
  }
};

template <typename T>
struct unpy_impl<std::vector<T>> {
  using type = std::vector<T>;
  using size_type = size_t;
  static type call(PyObject* p) {
    const size_type n = PyList_Size(p);
    type v(n);
    for (size_type i=0; i<n; ++i)
      v[i] = unpy<T>(PyList_GET_ITEM(p,i));
    return v;
  }
};

template <typename T, size_t N>
struct unpy_impl<std::array<T,N>> {
  using type = std::array<T,N>;
  using size_type = decltype(N);
  static type call(PyObject* p) {
    const size_type n = std::min((size_type)PyList_Size(p),N);
    type v;
    size_type i = 0;
    for (; i<n; ++i) v[i] = unpy<T>(PyList_GET_ITEM(p,i));
    for (; i<N; ++i) v[i] = { };
    return v;
  }
};

/*
std::vector<double> PyObject2vector(PyObject* p) {
  std::vector<double> v;
  const auto n = PyObject_Size(p);
  if (n > -1) v.reserve(n);
  p = PyObject_GetIter(p);
  if (!p) throw std::runtime_error("Unable to iterate PyObject");
  PyObject *item;
  while ((item = PyIter_Next(p))) {
    v.push_back(PyFloat_AsDouble(item));
    Py_DECREF(item);
  }
  return v;
}
*/

PyObject* safe(PyObject* p) {
  if (p) return p;
  Py_DECREF(p);
  throw std::runtime_error("Unable to allocate memory for python object");
}

PyObject* py(double x) { return safe(PyFloat_FromDouble(x)); }
PyObject* py(float  x) { return safe(PyFloat_FromDouble((double)x)); }
PyObject* py(long   x) { return safe(PyInt_FromLong(x)); }
PyObject* py(size_t x) { return safe(PyInt_FromSize_t(x)); }

template <typename T>
PyObject* py(const T& x) {
  using std::begin;
  using std::end;
  using std::distance;
  auto a = begin(x);
  const auto b = end(x);
  PyObject *list = safe(PyList_New( distance(a,b) ));
  unsigned i = 0;
  for (; a!=b; ++a, ++i) PyList_SET_ITEM(list, i, py(*a));
  return list;
}

struct py_decref_deleter {
  void operator()(PyObject* ref) { Py_DECREF(ref); }
};

template <typename T, typename... Args>
T py_call(PyObject* f, const char* types, Args&&... args) {
  std::unique_ptr<PyObject,py_decref_deleter> ret(
    PyObject_CallFunction(f,const_cast<char*>(types),std::forward<Args>(args)...)
  );
  if (!ret) throw std::runtime_error("Unable to call python function");
  return unpy<T>(ret.get());
}

extern "C"
PyObject* ivanp_gp_regression(PyObject *self, PyObject *args) {
  PyObject
    *xs, // training points coordinates
    *ys, // training points values
    *us, // uncertainties (add to diagonal)
    *ts, // test points
    *kernel; // kernel function
  PyArg_ParseTuple(args, "OOOOO", &xs, &ys, &us, &ts, &kernel);

  return py(ivanp::gp::regression(
    unpy<std::vector<double>>(xs),
    unpy<std::vector<double>>(ys),
    unpy<std::vector<double>>(us),
    unpy<std::vector<double>>(ts),
    [=](double a, double b){ return py_call<double>(kernel,"dd",a,b); }
  ));
}

extern "C"
PyObject* ivanp_gp_logml(PyObject *self, PyObject *args) {
  PyObject
    *xs, // training points coordinates
    *ys, // training points values
    *us, // noise variances (add to diagonal)
    *kernel, // kernel function
    *hs; // kernel (hyper)parameters
  PyArg_ParseTuple(args, "OOOOO", &xs, &ys, &us, &kernel, &hs);

  return py(ivanp::gp::logml(
    unpy<std::vector<double>>(xs),
    unpy<std::vector<double>>(ys),
    unpy<std::vector<double>>(us),
    [=](double a, double b){ return py_call<double>(kernel,"ddO",a,b,hs); }
  ));
}

extern "C"
PyObject* ivanp_gp_opt(PyObject *self, PyObject *args) {
  PyObject
    *xs, // training points coordinates
    *ys, // training points values
    *us, // noise variances (add to diagonal)
    *kernel, // kernel function
    *hs; // kernel (hyper)parameters (start, step)
  PyArg_ParseTuple(args, "OOOOO", &xs, &ys, &us, &kernel, &hs);

  const auto& xs_v = unpy<std::vector<double>>(xs);
  const auto& ys_v = unpy<std::vector<double>>(ys);
  const auto& us_v = unpy<std::vector<double>>(us);
  const auto& start_step = unpy<std::vector<std::array<double,2>>>(hs);
  const unsigned np = start_step.size();

  return py(ivanp::gsl_multimin(
    start_step,
    [&](const double* p){
      return ivanp::gp::logml(
        xs_v,
        ys_v,
        us_v,
        [&](auto a, auto b){
          for (unsigned i=0; i<np; ++i)
            PyList_SET_ITEM(hs, i, py(p[i]));
          return py_call<double>(kernel,"ddO",a,b,hs);
        }
      );
    }, { verbose: true, 1e-5 }
  ));
}

PyMethodDef methods[] = {
  { "regression", ivanp_gp_regression, METH_VARARGS,
    "Gaussian Process Regression" },
  { "logml", ivanp_gp_logml, METH_VARARGS,
    "Log Marginal Likelihood for Gaussian Process Regression" },
  { "opt", ivanp_gp_opt, METH_VARARGS,
    "Optimize kernel parameters" },
  { NULL, NULL, 0, NULL }
};

} // end namespace

PyMODINIT_FUNC initivanp_gp(void) {
  (void) Py_InitModule("ivanp_gp", methods);
}

