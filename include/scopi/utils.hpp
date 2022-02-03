#pragma once

#include <chrono>
#include <vector>

#include <xtensor-blas/xlinalg.hpp>

/////////////////////////////
// Functions for the timer //
/////////////////////////////

/// Timer used in tic & toc
static auto tic_timer = std::chrono::high_resolution_clock::now();
/// Launching the timer
void tic();
/// Stopping the timer and returning the duration in seconds
double toc();

// recursive function to initialize 2D newton (superellipsoid)
std::vector<double> create_binit(std::vector<double> binit, int n,
  double theta_g, double theta_d, double rx, double ry, double e);


// sign function (-1 if <0, +1 if >0 and 0 otherwise)
int sign(double val);

// newton method (with linesearch)
template<typename F, typename DF, typename U, typename A>
auto newton_method(U u0, F f, DF grad_f, A args, const int itermax, const double ftol, const double xtol){
  // std::cout << "newton_method : u0 = " << u0 << std::endl;
  // std::cout << "newton_method : args = " << args << std::endl;
  int cc = 0;
  auto u = u0;
  while (cc<itermax) {
    auto d = xt::linalg::solve(grad_f(u,args), -f(u,args));
    auto var = xt::linalg::norm(d);
    if (var < xtol) {
      // std::cout << "newton_method : cvgce (xtol) after " << cc << " iterations => RETURN u = " << u << std::endl;
      return std::make_tuple(u,cc);
    }
    // std::cout << "newton_method : iteration " << cc << " => d = " << d << " var = " << var << std::endl;
    // linesearch
    double t = 1;
    auto ferr = xt::linalg::norm(f(u,args));
    // std::cout << "newton_method : iteration " << cc << " => ferr = " << ferr << std::endl;
    if (ferr < ftol) {
      // std::cout << "newton_method : cvgce (ftol) after " << cc << " iterations => RETURN u = " << u << std::endl;
      return std::make_tuple(u,cc);
    }
    while ((xt::linalg::norm(f(u+t*d,args)) > ferr) && (t>0.01)){
        t -= 0.01;
    }
    // std::cout << "newton_method : iteration " << cc << " => t = " << t << std::endl;
    u += t * d;
    // std::cout << "newton_method : iteration " << cc << " => u = " << u << std::endl;
    cc += 1;
  }
  std::cout << "newton_method : !!!!!! FAILED !!!!!! after " << cc << " iterations => RETURN u = " << u << std::endl;

  return std::make_tuple(u,-1);
}
