#include <CLI/CLI.hpp>
#include <xtensor/xmasked_view.hpp>
#include <xtensor/xmath.hpp>
#include <xtensor/xnoalias.hpp>
#include <xtensor/xview.hpp>

#include <scopi/box.hpp>
#include <scopi/objects/types/sphere.hpp>
#include <scopi/objects/types/superellipsoid.hpp>
#include <scopi/property.hpp>
#include <scopi/solver.hpp>

int main(int argc, char** argv)
{
    plog::init(plog::info, "two_spheres_periodic.log", 0, 0);

    CLI::App app("two spheres with periodic boundary conditions");
    constexpr std::size_t dim = 2;
    double dt                 = .005;
    std::size_t total_it      = 1000;

    double vel = -0.25;

    scopi::scopi_container<dim> particles;

    scopi::sphere<dim> s1(
        {
            {0.4, 0.45}
    },
        0.1);
    scopi::sphere<dim> s2(
        {
            {0.6, 0.55}
    },
        0.1);
    particles.push_back(s1,
                        scopi::property<dim>()
                            .desired_velocity({
                                {vel, 0}
    })
                            .mass(1.)
                            .moment_inertia(0.1));
    particles.push_back(s2,
                        scopi::property<dim>()
                            .desired_velocity({
                                {-vel, 0}
    })
                            .mass(1.)
                            .moment_inertia(0.1));

    auto domain = scopi::BoxDomain<dim>({0, 0}, {1, 1}).with_periodicity(0);
    scopi::ScopiSolver<dim> solver(domain, particles, dt);
    solver.init_options(app);
    CLI11_PARSE(app, argc, argv);
    solver.run(total_it);

    return 0;
}
