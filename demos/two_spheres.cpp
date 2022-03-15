#include <xtensor/xmath.hpp>
#include <scopi/objects/types/sphere.hpp>
#include <scopi/solver.hpp>
#include <scopi/property.hpp>

int main()
{
    constexpr std::size_t dim = 2;
    double dt = .005;
    std::size_t total_it = 1000;
    scopi::scopi_container<dim> particles;


    scopi::sphere<dim> s1({{-0.2, -0.05}}, 0.1);
    scopi::sphere<dim> s2({{ 0.2,  0.05}}, 0.1);
    particles.push_back(s1, scopi::property<dim>().desired_velocity({{0.25, 0}}).mass(1.).moment_inertia(0.1);
    particles.push_back(s2, scopi::property<dim>().desired_velocity({{-0.25, 0}}).mass(1.).moment_inertia(0.1);

    scopi::ScopiSolver<dim> solver(particles, dt);
    solver.solve(total_it);

    return 0;
}
