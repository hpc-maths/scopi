#include <CLI/CLI.hpp>

#include <scopi/objects/types/worm.hpp>
#include <scopi/property.hpp>
#include <scopi/solver.hpp>

int main(int argc, char** argv)
{
    plog::init(plog::info, "three_worms.log");

    CLI::App app("three worms with periodic boundary conditions");

    constexpr std::size_t dim = 2;

    double dt            = .01;
    std::size_t total_it = 500;
    scopi::scopi_container<dim> particles;
    auto prop = scopi::property<dim>().mass(1.).moment_inertia(0.1);

    double radius          = .01;
    std::size_t nb_spheres = 6;
    std::vector<scopi::type::position_t<dim>> pos1(nb_spheres), pos2(nb_spheres), pos3(nb_spheres);
    std::vector<scopi::type::quaternion_t> q(nb_spheres);
    for (std::size_t i = 0; i < pos1.size(); ++i)
    {
        pos1[i](0) = 0.42 + 0.52 - i * 2 * radius;
        pos1[i][1] = .495;
        pos2[i][0] = 0.58 - 0.48 + i * 2 * radius;
        pos2[i][1] = .505;
        pos3[i][0] = .5 - 0.48;
        pos3[i][1] = 0.5 - i * 2 * radius;

        // pos1[i](0) = 0.42 - i*2*radius;
        // pos1[i][1] = .495;
        // pos2[i][0] = 0.58 + i*2*radius;
        // pos2[i][1] = .505;
        // pos3[i][0] = .5 ;
        // pos3[i][1] = 0.5 - i*2*radius;
        q[i] = scopi::quaternion(0);
    }

    scopi::worm<dim> w1(pos1, q, radius, nb_spheres);
    scopi::worm<dim> w2(pos2, q, radius, nb_spheres);
    scopi::worm<dim> w3(pos3, q, radius, nb_spheres);
    particles.push_back(w1, prop.desired_velocity({.5, 0.}));
    particles.push_back(w2, prop.desired_velocity({-.5, 0.}));
    particles.push_back(w3, prop.desired_velocity({0., 0.}));

    auto domain = scopi::BoxDomain<dim>({0, 0}, {1, 1}).with_periodicity(0);
    scopi::ScopiSolver<dim, scopi::OptimUzawaMatrixFreeOmp<scopi::DryWithoutFriction>> solver(domain, particles, dt);

    solver.init_options(app);
    CLI11_PARSE(app, argc, argv);
    solver.run(total_it);

    return 0;
}
