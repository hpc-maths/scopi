#include <CLI/CLI.hpp>

#include <scopi/objects/types/segment.hpp>
#include <scopi/objects/types/sphere.hpp>
#include <scopi/property.hpp>
#include <scopi/solver.hpp>
#include <scopi/contact/contact_brute_force.hpp>


int main(int argc, char **argv)
{
    constexpr std::size_t dim = 2;
    double dt = .001;
    std::size_t total_it = 1000;

    plog::init(plog::info, "two_segments.log");

    CLI::App app("two segments");

    scopi::segment<dim> seg1(scopi::type::position_t<dim>{0., 1.}, scopi::type::position_t<dim>{0.4, 1.});
    scopi::segment<dim> seg2(scopi::type::position_t<dim>{0.6, 1.}, scopi::type::position_t<dim>{1., 1.});

    scopi::scopi_container<dim> particles;
    particles.push_back(seg1, scopi::property<dim>().deactivate());
    particles.push_back(seg2, scopi::property<dim>().deactivate());

    std::default_random_engine generator;
    std::uniform_real_distribution<double> distrib_x(0.1, 0.9);
    std::uniform_real_distribution<double> distrib_y(0, 0.7);
    std::uniform_real_distribution<double> distrib_r(0.01, 0.06);

    auto s_prop = scopi::property<dim>().desired_velocity({{.5, 2}}).mass(1.).moment_inertia(0.1);
    for (std::size_t i = 0; i < 20; ++i)
    {
        auto x = distrib_x(generator);
        auto y = distrib_y(generator);
        auto radius = distrib_r(generator);
        auto prop = scopi::property<dim>().desired_velocity({{0.5-x, 2-y}}).mass(1.).moment_inertia(0.1);

        particles.push_back(scopi::sphere<dim>({{x, y}}, radius), prop);
    }

    using solver_type = scopi::ScopiSolver<dim>;
    solver_type solver(particles, dt);
    solver.init_options(app);
    CLI11_PARSE(app, argc, argv);
    solver.run(total_it);

    return 0;
}
