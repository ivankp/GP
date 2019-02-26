#include <vector>
#include <iterator>
#include <stdexcept>
#include <type_traits>
#include <Python.h>
#include "gp.hh"

// https://docs.python.org/2/c-api/concrete.html

namespace {

std::vector<double> PyList2vector(PyObject* p) {
  const auto n = PyList_Size(p);
  std::vector<double> v(n);
  for (std::remove_const_t<decltype(n)> i=0; i<n; ++i)
    v[i] = PyFloat_AsDouble(PyList_GET_ITEM(p,i));
  return v;
}

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

extern "C"
PyObject* gp(PyObject *self, PyObject *args) {
  PyObject
    *xs, // training points coordinates
    *ys, // training points values
    *us, // uncertainties (add to diagonal)
    *ts, // test points
    *kernel; // kernel function
  PyArg_ParseTuple(args, "OOOOO", &xs, &ys, &us, &ts, &kernel);

  return py(GP(
    PyList2vector(xs),
    PyList2vector(ys),
    PyList2vector(us),
    PyList2vector(ts),
    [=](double a, double b){
      PyObject* ref = PyObject_CallFunction(kernel,"dd",a,b);
      if (!ref) throw std::runtime_error("Unable to call GP kernel function");
      const double k = PyFloat_AS_DOUBLE(ref);
      Py_DECREF(ref);
      return k;
    }
  ));
}

PyMethodDef methods[] = {
  { "gaussian_process", gp, METH_VARARGS, "Gaussian Process Regression" },
  { NULL, NULL, 0, NULL }
};

} // end namespace

PyMODINIT_FUNC initgaussian_process(void) {
  (void) Py_InitModule("gaussian_process", methods);
}

