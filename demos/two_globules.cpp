#include <cstddef>
#include <xtensor/xmath.hpp>
#include <scopi/objects/types/globule.hpp>
#include <scopi/solver.hpp>
#include <scopi/property.hpp>
#include <scopi/solvers/OptimUzawaMkl.hpp>
#include <scopi/problems/ViscousGlobule.hpp>

int main()
{
    plog::init(plog::verbose, "two_globules.log");

    constexpr std::size_t dim = 2;
    double dt = .005;
    std::size_t total_it = 500;
    scopi::scopi_container<dim> particles;
    auto prop = scopi::property<dim>().mass(1.).moment_inertia(0.1);

    scopi::globule<dim> g1({{2., 0.}, {4., 0.}, {6., 0.}, {8., 0.}, {10., 0.}, {12., 0.}}, 1.);
    scopi::globule<dim> g2({{-2., -0.}, {-4., -0.}, {-6., -0.}, {-8., -0.}, {-10., -0.}, {-12., -0.}}, 1.);
    particles.push_back(g1, prop.desired_velocity({-1., 0.}));
    particles.push_back(g2, prop.desired_velocity({1., 0.}));

    scopi::OptimParams<scopi::OptimUzawaMkl> optim_params;
    // optim_params.m_rho = 35.;
    // optim_params.m_tol = 1e-5;
    // optim_params.m_max_iter = 100000.;
    scopi::ProblemParams<scopi::ViscousGlobule> problem_params;
    scopi::ScopiSolver<dim, scopi::ViscousGlobule, scopi::OptimUzawaMkl> solver(particles, dt, optim_params, problem_params);
    solver.solve(total_it);

    return 0;
}
