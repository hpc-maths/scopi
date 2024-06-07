#include <random>

#include <scopi/objects/types/segment.hpp>
#include <scopi/objects/types/sphere.hpp>
#include <scopi/solver.hpp>

int main(int argc, char** argv)
{
    constexpr std::size_t dim = 2;
    double dt                 = .001;
    const double max_radius   = 0.06;
    std::size_t total_it      = 1000;
    std::size_t n_parts       = 50;

    scopi::initialize("spheres passing between two segments");
    auto& app = scopi::get_app();
    app.add_option("--nparts", n_parts, "Number of particles")->capture_default_str();
    app.add_option("--nite", total_it, "Number of iterations")->capture_default_str();
    app.add_option("--dt", dt, "Time step")->capture_default_str();

    scopi::scopi_container<dim> particles;
    scopi::ScopiSolver<dim> solver(particles);
    SCOPI_PARSE(argc, argv);

    scopi::segment<dim> seg1(scopi::type::position_t<dim>{0., 1.}, scopi::type::position_t<dim>{0.4, 1.});
    scopi::segment<dim> seg2(scopi::type::position_t<dim>{0.6, 1.}, scopi::type::position_t<dim>{1., 1.});
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

    auto params                                 = solver.get_params();
    params.contact_method_params.dmax           = 2 * dt;
    params.contact_method_params.kd_tree_radius = 4 * max_radius;

    solver.run(dt, total_it);

    return 0;
}
