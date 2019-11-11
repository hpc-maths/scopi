#ifndef DEF_Particles
#define DEF_Particles

#include <xtensor/xtensor.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xdynamic_view.hpp>
#include <xtensor/xio.hpp>
#include "xtensor-python/pytensor.hpp"
#include "scopi/soa_xtensor.hpp"

struct particle {
  std::size_t id;
  double x, y, z;
  double r, m;
  double vapx, vapy, vapz;
  double vx, vy, vz;
};

SOA_DEFINE_TYPE(particle, id, x, y, z, r, m, vapx, vapy, vapz, vx, vy, vz);

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

  soa::vector<particle> data;

  /// @brief Add particles in a ball
  void add_particles_in_ball(
    std::size_t n,
    double rmin, double rmax,
    double mmin, double mmax,
    double xc, double yc, double zc,
    double rball
  );

  /// @brief Add particles in a box
  void add_particles_in_box(
    std::size_t n,
    double rmin, double rmax,
    double mmin, double mmax,
    double xmin, double xmax,
    double ymin, double ymax,
    double zmin, double zmax
  );

  /// @brief Print the particles
  void print() const;

  /// @brief Set the a priori velocities : only from python
  void set_vap(xt::pytensor<double, 2> vap);

  /// @brief Get coordinates and radius of particles : only for access from python
  xt::xtensor<double, 2> get_data() const;
  xt::xtensor<double, 2> get_positions() const;
  xt::xtensor<double, 2> get_v() const;
  xt::xtensor<double, 2> get_vap() const;
  xt::xtensor<double, 1> get_id() const;
  xt::xtensor<double, 1> get_x() const;
  xt::xtensor<double, 1> get_y() const;
  xt::xtensor<double, 1> get_z() const;
  xt::xtensor<double, 1> get_r() const;
  xt::xtensor<double, 1> get_m() const;

  /// @brief Move the particles
  void move(double dt);

private:

  std::string _name;
  std::size_t _idmax;

};

#endif
