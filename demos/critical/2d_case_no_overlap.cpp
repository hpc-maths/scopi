#include <xtensor/xmath.hpp>
#include <scopi/objects/types/superellipsoid.hpp>
#include <scopi/solvers/mosek.hpp>
#include <scopi/container.hpp>
#include <random>

// cmake --build . --target critical_2d_no_overlap

int main()
{

    constexpr std::size_t dim = 2;
    double PI = xt::numeric_constants<double>::PI;
    double dt = .01;
    std::size_t total_it = 100;
    scopi::scopi_container<dim> particles;

    std::seed_seq seed;
    std::default_random_engine generator(seed);
    std::uniform_real_distribution<double> distrib_r(0.8,1.);
    std::uniform_real_distribution<double> distrib_r2(0.8,1.);
    std::uniform_real_distribution<double> distrib_move_x(-0.9,0.9);
    std::uniform_real_distribution<double> distrib_move_y(-0.9,0.9);
    std::uniform_real_distribution<double> distrib_rot(0,PI);

    int n = 200; // 2*n*n particles
    double velocity = 2.0;

    for(int i = 0; i < n; ++i)
    {
        for(int j = 0; j < n; ++j)
        {
            double rot = distrib_rot(generator);
            double r = distrib_r(generator);
            double r2 = distrib_r2(generator);
            double x = (i + 0.5)*velocity + distrib_move_x(generator);
            double y = (j + 0.5)*velocity + distrib_move_y(generator);
            scopi::superellipsoid<dim> s1({ {x, y}}, {scopi::quaternion(rot)}, {{r, r2}}, {{1}});
            particles.push_back(s1, {{0, 0}}, {{1., 0.}}, 0, 0, {{0, 0}});

            rot = distrib_rot(generator);
            r = distrib_r(generator);
            r2 = distrib_r2(generator);
            x = (n + i + 0.5)*velocity + distrib_move_x(generator);
            y = (j + 0.5)*velocity + distrib_move_y(generator);
            scopi::superellipsoid<dim> s2({ {x, y}}, {scopi::quaternion(rot)}, {{r, r2}}, {{1}});
            particles.push_back(s2, {{0, 0}}, {{-1., 0.}}, 0, 0, {{0, 0}});
        }
    }
 
    std::size_t active_ptr = 0; // pas d'obstacles

    scopi::MosekSolver<dim, scopi::useOsqpCppSolver> mosek_solver(particles, dt, active_ptr);
    mosek_solver.solve(total_it);
}
