#include <cstddef>
#include <scopi/objects/types/worm.hpp>
#include <scopi/property.hpp>
#include <scopi/solver.hpp>
#include <scopi/solvers/OptimMosek.hpp>
#include <xtensor/xmath.hpp>

int main()
{
    plog::init(plog::error, "critical_worms.log");

    constexpr std::size_t dim = 2;
    double dt                 = .005;
    std::size_t total_it      = 2000;
    scopi::scopi_container<dim> particles;
    auto prop = scopi::property<dim>().mass(1.).moment_inertia(0.1);

    int n = 10; // 2*n worms
    std::default_random_engine generator;
    std::uniform_real_distribution<double> distrib_r(0.1, 0.8);
    std::uniform_real_distribution<double> distrib_move_x(-0.5, 0.5);
    std::uniform_real_distribution<double> distrib_move_y(-0.1, 0.1);
    std::uniform_real_distribution<double> distrib_velocity(-0.3, 0.3);

    for (int i = 0; i < n; ++i)
    {
        double r  = distrib_r(generator);
        double x0 = 2. + distrib_move_x(generator);
        double y  = 2. * (i + 0.5);
        ;
        scopi::worm<dim> w1(
            {
                {x0,           y},
                {x0 + 2. * r,  y},
                {x0 + 4. * r,  y},
                {x0 + 6. * r,  y},
                {x0 + 8. * r,  y},
                {x0 + 10. * r, y}
        },
            {{scopi::quaternion(0.)},
             {scopi::quaternion(0.)},
             {scopi::quaternion(0.)},
             {scopi::quaternion(0.)},
             {scopi::quaternion(0.)},
             {scopi::quaternion(0.)}},
            r,
            6);
        double v_x = distrib_velocity(generator);
        double v_y = distrib_velocity(generator);
        particles.push_back(w1,
                            prop.desired_velocity({
                                {-1. + v_x, v_y}
        }));

        r  = distrib_r(generator);
        x0 = -2. + distrib_move_x(generator);
        y  = 2. * i;
        scopi::worm<dim> w2(
            {
                {x0,           y},
                {x0 - 2. * r,  y},
                {x0 - 4. * r,  y},
                {x0 - 6. * r,  y},
                {x0 - 8. * r,  y},
                {x0 - 10. * r, y}
        },
            {{scopi::quaternion(0.)},
             {scopi::quaternion(0.)},
             {scopi::quaternion(0.)},
             {scopi::quaternion(0.)},
             {scopi::quaternion(0.)},
             {scopi::quaternion(0.)}},
            r,
            6);
        v_x = distrib_velocity(generator);
        v_y = distrib_velocity(generator);
        particles.push_back(w2,
                            prop.desired_velocity({
                                {1. + v_x, v_y}
        }));
    }

    scopi::ScopiSolver<dim, scopi::OptimMosek<>> solver(particles, dt);
    solver.run(total_it);

    return 0;
}
