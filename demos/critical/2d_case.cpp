#include <xtensor/xmath.hpp>
#include <scopi/objects/types/superellipsoid.hpp>
#include <scopi/solvers/mosek.hpp>
#include <scopi/container.hpp>
#include <random>

// cmake --build . --target critical_2d

int main()
{
    constexpr std::size_t dim = 2;
    double PI = xt::numeric_constants<double>::PI;
    double dt = .01;
    std::size_t total_it = 1000;//50000;
    scopi::scopi_container<dim> particles;

    std::default_random_engine generator;
    // std::uniform_real_distribution<double> distrib_e(0.6,1.);
    // std::uniform_real_distribution<double> distrib_r(0.1,1.0);
    // std::uniform_real_distribution<double> distrib_gp1(2.0,10.0);
    // std::uniform_real_distribution<double> distrib_gp2(-10.0,-2.0);
    // std::uniform_real_distribution<double> distrib_y(-10.0,10.0);
    // std::uniform_real_distribution<double> distrib_rot(0,PI);
    std::uniform_real_distribution<double> distrib_e(0.6,1.);
    std::uniform_real_distribution<double> distrib_r(0.2,1.0);
    std::uniform_real_distribution<double> distrib_r2(0.2,1.0);
    std::uniform_real_distribution<double> distrib_x(-400.0,400.0);
    std::uniform_real_distribution<double> distrib_y(-400.0,400.0);
    std::uniform_real_distribution<double> distrib_rot(0,PI);
    // std::uniform_real_distribution<double> distrib_rot(PI/4,PI/4);

    int n = 2000;
    // int n = 10;

    for (int i=0;i<n;++i){

      // double e = distrib_e(generator);
      double r = distrib_r(generator);
      double r2 = distrib_r2(generator);
      double x = distrib_x(generator);
      double y = distrib_y(generator);
      double rot = distrib_rot(generator);
      double dist_orig = std::sqrt(x*x+y*y);

      scopi::superellipsoid<dim> s1({ {x, y}}, {scopi::quaternion(rot)}, {{r, r2}}, {{1}});
      particles.push_back(s1, {{0, 0}}, {{-x/dist_orig, -y/dist_orig}}, 0, 0, {{0, 0}});

      // e = distrib_e(generator);
      r = distrib_r(generator);
      r2 = distrib_r2(generator);
      x = distrib_x(generator);
      y = distrib_y(generator);
      rot = distrib_rot(generator);
      dist_orig = std::sqrt(x*x+y*y);
      // scopi::sphere<dim> s2( {{x, y}}, {scopi::quaternion(rot)}, r);
      // particles.push_back(s2, {{0, 0}}, {{-x/dist_orig, -y/dist_orig}}, 0, 0, {{0, 0}});
      scopi::superellipsoid<dim> s2({ {x, y}}, {scopi::quaternion(rot)}, {{r, r}}, {{1}});
      particles.push_back(s2, {{0, 0}}, {{-x/dist_orig, -y/dist_orig}}, 0, 0, {{0, 0}});

      // e = distrib_e(generator);
      // r = distrib_r(generator);
      // r2 = distrib_r2(generator);
      // x = distrib_x(generator);
      // y = distrib_y(generator);
      // rot = distrib_rot(generator);
      // dist_orig = std::sqrt(x*x+y*y);
      // scopi::superellipsoid<dim> s3({ {x, y}}, {scopi::quaternion(rot)}, {{r, r2}}, {{e}});
      // particles.push_back(s3, {{0, 0}}, {{-x/dist_orig, -y/dist_orig}}, 0, 0, {{0, 0}});

    }






    // for (int i=0;i<n;++i){
    //
    //   const double e = distrib_e(generator);
    //
    //   std::cout << "i = " << i << " e = " << e << std::endl;
    //
    //   // scopi::sphere<dim> s1(
    //   //   {{distrib_gp1(generator), distrib_y(generator)}},
    //   //   {scopi::quaternion(distrib_rot(generator))},
    //   //   distrib_r(generator));
    //   scopi::superellipsoid<dim> s1({{distrib_gp1(generator), distrib_y(generator)}},
    //     {scopi::quaternion(distrib_rot(generator))}, {{distrib_r(generator), distrib_r(generator)}},
    //     {{e}});
    //   // s1.print();
    //   particles.push_back(s1, {{0, 0}}, {{-0.25, 0}}, 0, 0, {{0, 0}});
    //
    //   // scopi::sphere<dim> s2(
    //   //   {{distrib_gp2(generator), distrib_y(generator)}},
    //   //   {scopi::quaternion(distrib_rot(generator))},
    //   //   distrib_r(generator));
    //   scopi::superellipsoid<dim> s2({{distrib_gp2(generator), distrib_y(generator)}},
    //     {scopi::quaternion(distrib_rot(generator))}, {{distrib_r(generator), distrib_r(generator)}},
    //     {{e}});
    //   // s2.print();
    //   particles.push_back(s2, {{0, 0}}, {{0.25, 0}}, 0, 0, {{0, 0}});
    //
    // }

    // std::size_t active_ptr = 1;
    std::size_t active_ptr = 0; // pas d'obstacles

    // scopi::mosek_solver(particles, dt, total_it, active_ptr);
    mosek_solver(particles, dt, total_it, active_ptr);

    return 0;
}
