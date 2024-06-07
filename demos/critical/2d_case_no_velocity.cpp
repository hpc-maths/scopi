#include <random>

#include <xtensor/xmath.hpp>

#include <scopi/container.hpp>
#include <scopi/objects/types/sphere.hpp>
#include <scopi/property.hpp>
#include <scopi/solver.hpp>

// cmake --build . --target critical_2d_no_overlap

int main()
{
    constexpr std::size_t dim = 2;
    double dt                 = .01;
    std::size_t total_it      = 100;
    scopi::scopi_container<dim> particles;
    auto prop = scopi::property<dim>().mass(1.).moment_inertia(0.1);

    int n = 20; // 2*n*n particles

    std::default_random_engine generator;
    std::uniform_real_distribution<double> distrib_r(0.2, 0.4);
    std::uniform_real_distribution<double> distrib_move_x(-0.1, 0.1);
    std::uniform_real_distribution<double> distrib_move_y(-0.1, 0.1);

    scopi::sphere<dim> s(
        {
            {0., 0.}
    },
        0.01);
    particles.push_back(s, scopi::property<dim>().deactivate());

    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            double r = distrib_r(generator);
            double x = -n + (i + 0.5) + distrib_move_x(generator);
            double y = -n / 2. + (j + 0.5) + distrib_move_y(generator);
            scopi::sphere<dim> s1(
                {
                    {x, y}
            },
                r);
            particles.push_back(s1);

            r = distrib_r(generator);
            x = (i + 0.5) + distrib_move_x(generator);
            y = -n / 2. + (j + 0.5) + distrib_move_y(generator);
            scopi::sphere<dim> s2(
                {
                    {x, y}
            },
                r);
            particles.push_back(s2);
        }
    }

    scopi::ScopiSolver<dim> solver(particles);
    solver.run(dt, total_it);
}
