#include <xtensor/xmath.hpp>
#include <scopi/objects/types/sphere.hpp>
#include <scopi/solver.hpp>
#include <scopi/property.hpp>

int main()
{
    plog::init(plog::error, "two_spheres.log");

    constexpr std::size_t dim = 2;
    double dt = .005;
    std::size_t total_it = 1000;
    scopi::scopi_container<dim> particles;

    scopi::sphere<dim> s1({{-0.2, -0.05}}, 0.1);
    scopi::sphere<dim> s2({{ 0.2,  0.05}}, 0.1);
    particles.push_back(s1, scopi::property<dim>().desired_velocity({{0.25, 0}}).mass(1.).moment_inertia(0.1));
    particles.push_back(s2, scopi::property<dim>().desired_velocity({{-0.25, 0}}).mass(1.).moment_inertia(0.1));

    scopi::OptimParams<scopi::OptimUzawaMatrixFreeOmp<scopi::DryWithoutFriction>> optim_params;
    optim_params.m_max_iter = 1000;
    scopi::ProblemParams<scopi::DryWithoutFriction> problem_params;

    scopi::ScopiSolver<dim> solver(particles, dt, optim_params, problem_params);
    solver.solve(total_it);

    return 0;
}
