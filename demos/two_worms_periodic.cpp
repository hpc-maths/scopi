#include <cstddef>
#include <CLI/CLI.hpp>

#include <xtensor/xmath.hpp>

#include <xtensor/xmath.hpp>
#include <scopi/objects/types/worm.hpp>
#include <scopi/solver.hpp>
#include <scopi/property.hpp>
#include <scopi/solvers/OptimMosek.hpp>
#include <scopi/contact/contact_brute_force.hpp>

int main(int argc, char **argv)
{
    plog::init(plog::error, "two_worms.log");

    CLI::App app("two spheres with periodic boundary conditions");

    constexpr std::size_t dim = 2;
    double PI = xt::numeric_constants<double>::PI;

    double dt = .01;
    std::size_t total_it = 1000;
    scopi::scopi_container<dim> particles;
    auto prop = scopi::property<dim>().mass(1.).moment_inertia(0.1);

    particles.push_back(scopi::plan<dim>({{0., 0.}}, PI/2.), scopi::property<dim>().deactivate());
    particles.push_back(scopi::plan<dim>({{0., 1.}}, PI/2.), scopi::property<dim>().deactivate());

    double radius = .01;
    std::vector<scopi::type::position_t<dim>> pos1(6), pos2(6), pos3(6);
    std::vector<scopi::type::quaternion_t> q(6);
    for(std::size_t i = 0; i < pos1.size(); ++i)
    {
        pos1[i](0) = 0.42 - i*2*radius;
        pos1[i][1] = .495;
        pos2[i][0] = 0.58 + i*2*radius;
        pos2[i][1] = .505;
        pos3[i][0] = .5;
        pos3[i][1] = 0.6 - i*2*radius;
        q[i] = scopi::quaternion(0);
    }

    scopi::worm<dim> w1(pos1, q, radius, 6);
    scopi::worm<dim> w2(pos2, q, radius, 6);
    scopi::worm<dim> w3(pos3, q, radius, 6);
    particles.push_back(w1, prop.desired_velocity({.5, 0.}));
    particles.push_back(w2, prop.desired_velocity({-.5, 0.}));
    particles.push_back(w3, prop.desired_velocity({0., 0.}));

    auto domain = scopi::BoxDomain<dim>({0, 0}, {1, 1}).with_periodicity(0);

    scopi::ScopiSolver<dim> solver(domain, particles, dt);
    solver.init_options(app);
    CLI11_PARSE(app, argc, argv);
    solver.run(total_it);

    return 0;
}
