#include <scopi/utils.hpp>

/////////////////////////////
// Functions for the timer //
/////////////////////////////

/// Launching the timer
void tic()
{
  tic_timer = std::chrono::high_resolution_clock::now();
}
/// Stopping the timer and returning the duration in seconds
double toc()
{
  const auto toc_timer = std::chrono::high_resolution_clock::now();
  const std::chrono::duration<double> time_span = toc_timer - tic_timer;
  return time_span.count();
}

// recursive function to initialize 2D newton (superellipsoid)
std::vector<double> create_binit(std::vector<double> binit, int n,
  double theta_g, double theta_d, double rx, double ry, double e) {
    if (n < 0)
        return binit;
    else
    {
      const double pi = 4*std::atan(1);
      double d1 = std::sqrt(2*(1-std::cos(theta_g-theta_d)));
      double xnew =  rx*( std::sin(theta_g)-std::sin(theta_d) )/d1;
      double ynew = -ry*( std::cos(theta_g)-std::cos(theta_d) )/d1;
      double ynew2 = ynew*rx*ry/( std::pow( std::pow(ry*xnew,2/e)+std::pow(rx*ynew,2/e), e/2) );
      double sinb = std::sqrt( std::pow( std::pow(ynew2/ry,2), 1/e) );
      double b = std::asin(sinb);
      binit.push_back(b);
      binit.push_back(b+pi/2);
      binit.push_back(b+pi);
      binit.push_back(b+3*pi/2);
      double theta_milieu = 0.5*(theta_g+theta_d);
      binit = create_binit(binit, n-1, theta_milieu, theta_d, rx, ry, e);
      binit = create_binit(binit, n-1, theta_g, theta_milieu, rx, ry, e);
      return binit;
    }
}


// sign function (-1 if <0, +1 if >0 and 0 otherwise)
int sign(double val) {
  return (0 < val) - (val < 0);
}
