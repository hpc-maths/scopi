#include <iostream>
#include <typeinfo>

#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xrandom.hpp>

#include <mkl_service.h>
#include <mkl_spblas.h>

#include <scopi/particles.hpp>
#include <scopi/obstacle.hpp>
#include <scopi/contacts.hpp>
#include <scopi/projection.hpp>


int main()
{

  // Particles
  Particles part("particules");
  std::size_t n = 10;
  double rmin = 0.1;
  double rmax = 0.2;
  double mmin = 0.8;
  double mmax = 1.2;
  double xc = 0;
  double yc = 0;
  double zc = 0;
  double rball = 2;
  part.add_particles_in_ball(n, rmin, rmax, mmin, mmax, xc, yc, zc, rball);
  part.print();

  // obstacles
  Obstacle obs("../objets3D/box.stl",2.5,1.);

  // Contacts
  double dxc = 1.2*rmax;
  Contacts cont(dxc);
  cont.compute_contacts(part);
  cont.print();

  // Projection
  std::size_t maxiter = 40000;
  double dmin = 0.1;
  double dt = 0.1;
  double rho = 0.2;
  double tol = 1.0e-2;
  Projection proj(maxiter, rho, dmin, tol, dt);
  proj.run(part,cont);

  // Move Particles
  part.move(dt);
  part.print();

  return 0;

}
