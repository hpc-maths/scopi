#include <xtensor/xmath.hpp>
#include <scopi/object/superellipsoid.hpp>
// #include "../mosek_solver.hpp"
#include "../mosek_solver_sparse.hpp"
#include <random>

int main()
{
    constexpr std::size_t dim = 2;
    double PI = xt::numeric_constants<double>::PI;
    double dt = .05;
    std::size_t total_it = 10000;
    scopi::scopi_container<dim> particles;

    std::default_random_engine generator;
    std::uniform_real_distribution<double> distrib_e(0.6,1.);
    std::uniform_real_distribution<double> distrib_r(0.1,1.0);
    std::uniform_real_distribution<double> distrib_gp1(2.0,10.0);
    std::uniform_real_distribution<double> distrib_gp2(-10.0,-2.0);
    std::uniform_real_distribution<double> distrib_y(-10.0,10.0);
    std::uniform_real_distribution<double> distrib_rot(0,PI);

    int n = 200;

    for (int i=0;i<n;++i){

      const double e = distrib_e(generator);

      std::cout << "i = " << i << " e = " << e << std::endl;

      scopi::sphere<dim> s1(
        {{distrib_gp1(generator), distrib_y(generator)}},
        {scopi::quaternion(distrib_rot(generator))},
        distrib_r(generator));
      // scopi::superellipsoid<dim> s1({{distrib_gp1(generator), distrib_y(generator)}},
      //   {scopi::quaternion(distrib_rot(generator))}, {{distrib_r(generator), distrib_r(generator)}},
      //   {{e}});
      // s1.print();
      particles.push_back(s1, {{0, 0}}, {{-0.25, 0}}, 0, 0, {{0, 0}});

      scopi::sphere<dim> s2(
        {{distrib_gp2(generator), distrib_y(generator)}},
        {scopi::quaternion(distrib_rot(generator))},
        distrib_r(generator));
      // scopi::superellipsoid<dim> s2({{distrib_gp2(generator), distrib_y(generator)}},
      //   {scopi::quaternion(distrib_rot(generator))}, {{distrib_r(generator), distrib_r(generator)}},
      //   {{e}});
      // s2.print();
      particles.push_back(s2, {{0, 0}}, {{0.25, 0}}, 0, 0, {{0, 0}});

    }

    // std::size_t active_ptr = 1;
    std::size_t active_ptr = 0; // pas d'obstacles

    mosek_solver(particles, dt, total_it, active_ptr);

    return 0;
}
