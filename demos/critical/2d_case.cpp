#include <xtensor/xmath.hpp>
#include <scopi/objects/types/superellipsoid.hpp>
#include <scopi/solvers/mosek.hpp>
#include <scopi/container.hpp>
#include <random>
#include <scopi/vap/base.hpp>

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
    std::uniform_real_distribution<double> distrib_x(-1400.0,1400.0);
    std::uniform_real_distribution<double> distrib_y(-1400.0,1400.0);
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

      scopi::sphere<dim> s1( {{x, y}}, {scopi::quaternion(rot)}, r);
      // scopi::superellipsoid<dim> s1({ {x, y}}, {scopi::quaternion(rot)}, {{r, r2}}, {{1}});
      particles.push_back(s1, {{0, 0}}, {{-x/dist_orig, -y/dist_orig}}, 0, 0, {{0, 0}});

      // e = distrib_e(generator);
      r = distrib_r(generator);
      r2 = distrib_r2(generator);
      x = distrib_x(generator);
      y = distrib_y(generator);
      rot = distrib_rot(generator);
      dist_orig = std::sqrt(x*x+y*y);
      scopi::sphere<dim> s2( {{x, y}}, {scopi::quaternion(rot)}, r);
      // particles.push_back(s2, {{0, 0}}, {{-x/dist_orig, -y/dist_orig}}, 0, 0, {{0, 0}});
      // scopi::superellipsoid<dim> s2({ {x, y}}, {scopi::quaternion(rot)}, {{r, r}}, {{1}});
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
    scopi::mosek_solver(particles, dt, total_it, active_ptr);

    return 0;
}

// ----> CONTACTS : run implementation contact_brute_force
// ----> CPUTIME : compute 30 contacts = 7.07612
// ----> CPUTIME : sort 30 contacts = 6.202e-06
// ----> CONTACTS : i j = 197 1678 d = 1.16859
// ----> CONTACTS : i j = 267 1530 d = -0.296426
// ----> CONTACTS : i j = 349 3739 d = 0.0430042
// ----> CONTACTS : i j = 356 1135 d = 0.576964
// ----> CONTACTS : i j = 364 819 d = 0.0680541
// ----> CONTACTS : i j = 451 2247 d = 0.763122
// ----> CONTACTS : i j = 454 1522 d = 1.08576
// ----> CONTACTS : i j = 475 629 d = 1.33887
// ----> CONTACTS : i j = 667 2356 d = 1.90684
// ----> CONTACTS : i j = 745 3009 d = 0.456119
// ----> CONTACTS : i j = 896 1404 d = 1.53092
// ----> CONTACTS : i j = 907 1512 d = 0.832471
// ----> CONTACTS : i j = 912 992 d = 0.853473
// ----> CONTACTS : i j = 994 1528 d = 0.388348
// ----> CONTACTS : i j = 1076 2117 d = 1.77413
// ----> CONTACTS : i j = 1407 2955 d = 0.760695
// ----> CONTACTS : i j = 1643 1692 d = -0.150049
// ----> CONTACTS : i j = 1801 3872 d = 0.685851
// ----> CONTACTS : i j = 1830 2339 d = 1.40466
// ----> CONTACTS : i j = 1832 3856 d = 0.689811
// ----> CONTACTS : i j = 1877 1938 d = 1.79114
// ----> CONTACTS : i j = 1899 3581 d = 1.75176
// ----> CONTACTS : i j = 1949 2418 d = 1.6662
// ----> CONTACTS : i j = 1998 2513 d = 1.61932
// ----> CONTACTS : i j = 2066 3545 d = 1.55091
// ----> CONTACTS : i j = 2283 3055 d = 0.253956
// ----> CONTACTS : i j = 2330 3934 d = 1.50639
// ----> CONTACTS : i j = 2342 3009 d = 1.85813
// ----> CONTACTS : i j = 2346 3669 d = 1.98497
// ----> CONTACTS : i j = 3307 3885 d = 1.73152
// ----> MAIN : contacts.size() = 30
