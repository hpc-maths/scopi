#include <cstddef>
#include <xtensor/xmath.hpp>
#include <scopi/objects/types/globule.hpp>
#include <scopi/solver.hpp>
#include <scopi/property.hpp>
#include <scopi/solvers/OptimMosek.hpp>
#include <scopi/problems/ViscousGlobule.hpp>

int main()
{
    plog::init(plog::error, "two_globules.log");

    constexpr std::size_t dim = 2;
    double dt = .005;
    std::size_t total_it = 1;
    scopi::scopi_container<dim> particles;
    auto prop = scopi::property<dim>().mass(1.).moment_inertia(0.1);

    scopi::globule<dim> g1({{3., 0.}, {5., 0.}, {7., 0.}, {9., 0.}, {11., 0.}, {13., 0.}}, 1.);
    scopi::globule<dim> g2({{-3., 0.}, {-5., 0.}, {-7., 0.}, {-9., 0.}, {-11., 0.}, {-13., 0.}}, 1.);
    particles.push_back(g1, prop);
    particles.push_back(g2, prop);

    scopi::OptimParams<scopi::OptimMosek> optim_params;
    scopi::ProblemParams<scopi::ViscousGlobule> problem_params;
    scopi::ScopiSolver<dim, scopi::ViscousGlobule, scopi::OptimMosek> solver(particles, dt, optim_params, problem_params);
    solver.solve(total_it);

    return 0;
}
