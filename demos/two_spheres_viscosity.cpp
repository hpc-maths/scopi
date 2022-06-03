#include <xtensor/xmath.hpp>
#include <scopi/objects/types/sphere.hpp>
#include <scopi/solver.hpp>
#include <scopi/property.hpp>
#include <scopi/solvers/OptimMosek.hpp>
#include <scopi/vap/vap_fpd.hpp>
#include <scopi/problems/ViscousWithFriction.hpp>

int main()
{
    plog::init(plog::warning, "two_spheres_viscosity.log");
    constexpr std::size_t dim = 2;
    double dt = .005;
    std::size_t total_it = 1000;
    double r = 0.1;
    scopi::scopi_container<dim> particles;
    auto prop = scopi::property<dim>().mass(1.).moment_inertia(1.*r*r/2.);

    scopi::sphere<dim> s1({{0., 0.}}, r);
    scopi::sphere<dim> s2({{ 0.5, 1.}}, r);
    particles.push_back(s1, prop.velocity({0.3, 0.75}));
    particles.push_back(s2, prop.velocity({0., 0.}));

    scopi::OptimParams<scopi::OptimMosek> optim_params;
    scopi::ProblemParams<ViscousWithFriction<dim>> problem_params;
    scopi::ScopiSolver<dim, scopi::ViscousWithFriction<dim>, scopi::OptimMosek, scopi::contact_kdtree, scopi::vap_fpd> solver(particles, dt, optim_params, problem_params);
    solver.solve(total_it);

    return 0;
}
