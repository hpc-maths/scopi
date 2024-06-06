#include <random>

#include <CLI/CLI.hpp>

#include <scopi/contact/contact_brute_force.hpp>
#include <scopi/objects/types/plan.hpp>
#include <scopi/objects/types/segment.hpp>
#include <scopi/objects/types/sphere.hpp>
#include <scopi/property.hpp>
#include <scopi/solver.hpp>
#include <scopi/vap/vap_fpd.hpp>

#include <plog/Appenders/ColorConsoleAppender.h>
#include <plog/Formatters/TxtFormatter.h>
#include <plog/Init.h>
#include <plog/Log.h>

int main(int argc, char** argv)
{
    constexpr std::size_t dim = 2;
    double dt                 = .001;
    const double max_radius   = 0.06;
    std::size_t total_it      = 100;
    std::size_t n_parts       = 20;

    static plog::ColorConsoleAppender<plog::TxtFormatter> consoleAppender;
    plog::init(plog::info, &consoleAppender);

    CLI::App app("two segments");
    app.add_option("--nparts", n_parts, "Number of particles")->capture_default_str();
    app.add_option("--nite", total_it, "Number of iterations")->capture_default_str();
    app.add_option("--dt", dt, "Time step")->capture_default_str();
    CLI11_PARSE(app, argc, argv);

    scopi::segment<dim> seg1(scopi::type::position_t<dim>{0., 1.}, scopi::type::position_t<dim>{0.4, 1.});
    scopi::segment<dim> seg2(scopi::type::position_t<dim>{0.6, 1.}, scopi::type::position_t<dim>{1., 1.});

    scopi::scopi_container<dim> particles;
    particles.push_back(seg1, scopi::property<dim>().deactivate());
    particles.push_back(seg2, scopi::property<dim>().deactivate());

    std::default_random_engine generator;
    std::uniform_real_distribution<double> distrib_x(0.1, 0.9);
    std::uniform_real_distribution<double> distrib_y(0, 0.7);
    std::uniform_real_distribution<double> distrib_r(0.01, max_radius);

    for (std::size_t i = 0; i < n_parts; ++i)
    {
        auto x      = distrib_x(generator);
        auto y      = distrib_y(generator);
        auto radius = distrib_r(generator);
        auto prop   = scopi::property<dim>()
                        .desired_velocity({
                            {0.5 - x, 2 - y}
        })
                        .mass(1.)
                        .moment_inertia(0.1);

        particles.push_back(scopi::sphere<dim>(
                                {
                                    {x, y}
        },
                                radius),
                            prop);
    }

    using problem_t    = scopi::NoFriction;
    using optim_solver = scopi::OptimGradient<scopi::apgd>;
    using contact_t    = scopi::contact_kdtree;
    using vap_t        = scopi::vap_fixed;
    using solver_type  = scopi::ScopiSolver<dim, problem_t, optim_solver, contact_t, vap_t>;
    solver_type solver(particles, dt);

    auto params = solver.get_params();
    // params.optim_params.alpha           = 0.2 / (dt * dt);
    params.optim_params.dynamic_descent = true;
    params.contact_params.dmax          = 4 * max_radius;

    // solver.init_options(app);
    // CLI11_PARSE(app, argc, argv);

    solver.run(total_it);

    return 0;
}
