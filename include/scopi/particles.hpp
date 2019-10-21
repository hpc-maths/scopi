#ifndef DEF_Particles
#define DEF_Particles

#include <iostream>
#include <vector>
#include <typeinfo>

#include <xtensor/xtensor.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xrandom.hpp>
#include "xtensor-python/pyarray.hpp"
#include "nanoflann/nanoflann.hpp"
#include "scopi/soa_xtensor.hpp"

struct particle {
  double x, y, z;
  double r;
};

SOA_DEFINE_TYPE(particle, x, y, z, r);


/// @class Particles
/// @bried This class manages particles
class Particles
{
public:

  /// @brief Constructor
  /// Instantiate np particles
  Particles(const std::string &name);

  /// @brief Destructor
  ~Particles();

  /// @brief Add particles in a ball
  void add_particles_in_ball(std::size_t n, double rmin, double rmax,
    double xc, double yc, double zc, double rball);

  /// @brief Add particles in a box
  void add_particles_in_box(std::size_t n,
      double rmin, double rmax,
      double xmin, double xmax,
      double ymin, double ymax,
      double zmin, double zmax);

  /// @brief Print the particles
  void print();

  xt::pyarray<double> xyzr();  // A REVOIR : copies...

private:

  soa::vector<particle> _data;
  std::string _name;

};

#endif
