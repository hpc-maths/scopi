#include <random>

#include <xtensor/xmath.hpp>

#include <scopi/container.hpp>
#include <scopi/objects/types/superellipsoid.hpp>
#include <scopi/property.hpp>
#include <scopi/solver.hpp>

// cmake --build . --target critical_2d_no_overlap

int main()
{

    constexpr std::size_t dim = 2;
    double PI = xt::numeric_constants<double>::PI;
    double dt = .01;
    std::size_t total_it = 100;
    scopi::scopi_container<dim> particles;

    int n = 20; // 2*n*n particles

    std::default_random_engine generator;
    std::uniform_real_distribution<double> distrib_r(0.2, 0.4);
    std::uniform_real_distribution<double> distrib_r2(0.2, 0.4);
    std::uniform_real_distribution<double> distrib_move_x(-0.1, 0.1);
    std::uniform_real_distribution<double> distrib_move_y(-0.1, 0.1);
    std::uniform_real_distribution<double> distrib_rot(0, PI);
    std::uniform_real_distribution<double> distrib_velocity(2., 5.);

    for(int i = 0; i < n; ++i)
    {
        for(int j = 0; j < n; ++j)
        {
            double rot = distrib_rot(generator);
            double r = distrib_r(generator);
            double r2 = distrib_r2(generator);
            double x = (i + 0.5) + distrib_move_x(generator);
            double y = (j + 0.5) + distrib_move_y(generator);
            double velocity = distrib_velocity(generator);

            scopi::superellipsoid<dim> s1({ {x, y}}, {scopi::quaternion(rot)}, {{r, r2}}, 1);
            particles.push_back(s1, scopi::property<dim>().desired_velocity({{velocity, 0.}}).mass(1.));

            rot = distrib_rot(generator);
            r = distrib_r(generator);
            r2 = distrib_r2(generator);
            x = (n + i + 0.5) + distrib_move_x(generator);
            y = (j + 0.5) + distrib_move_y(generator);
            velocity = distrib_velocity(generator);

            scopi::superellipsoid<dim> s2({ {x, y}}, {scopi::quaternion(rot)}, {{r, r2}}, 1);
            particles.push_back(s2, scopi::property<dim>().desired_velocity({{-velocity, 0.}}).mass(1.));
        }
    }

    scopi::ScopiSolver<dim> solver(particles, dt);
    solver.solve(total_it);
}
