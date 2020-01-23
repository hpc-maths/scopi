#include <iostream>                        // Standard library import for std::accumulate
#include <numeric>                        // Standard library import for std::accumulate

#include "pybind11/pybind11.h"            // Pybind11 import to define Python bindings

#include "xtensor/xmath.hpp"              // xtensor import for the C++ universal functions
#include "xtensor/xarray.hpp"              // xtensor import for the C++ universal functions

#include "fm/min_heap.hpp"

#include "xtensor-python/pyarray.hpp"     // Numpy bindings
#include "xtensor-python/pytensor.hpp"     // Numpy bindings

double eikonal(
  std::size_t i,
  std::size_t j,
  std::size_t k,
  double h,
  double infval,
  xt::pyarray<double>& img
);

xt::pyarray<double> compute_distance(
  double h,
  double infval,
  xt::pyarray<double>& img,
  xt::pyarray<std::size_t>& narrow_band
);


double xt_eikonal(
  std::size_t i,
  std::size_t j,
  std::size_t k,
  double h,
  double infval,
  xt::xarray<double>& img
);

xt::xarray<double> xt_compute_distance(
  double h,
  double infval,
  xt::xarray<double>& img,
  xt::xarray<std::size_t>& narrow_band
);

template <class E>
void test(
  xt::xexpression<E>& xd
);
