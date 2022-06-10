#include <cstddef>
#include <xtensor/xmath.hpp>
#include <scopi/objects/types/globule.hpp>
#include <scopi/solver.hpp>
#include <scopi/property.hpp>

int main()
{
    plog::init(plog::error, "two_globules.log");

    constexpr std::size_t dim = 2;
    double dt = .005;
    std::size_t total_it = 1;
    scopi::scopi_container<dim> particles;
    auto prop = scopi::property<dim>().mass(1.).moment_inertia(0.1);

    scopi::globule<dim> g1({{0., 0.}, {1., 0.}, {2., 0.}, {3., 0.}, {4., 0.}, {5., 0.}}, 0.5);
    particles.push_back(g1, prop);

    scopi::OptimParams<scopi::OptimUzawaMatrixFreeOmp> optim_params;
    scopi::ProblemParams<scopi::DryWithoutFriction> problem_params;
    scopi::ScopiSolver<dim> solver(particles, dt, optim_params, problem_params);
    solver.solve(total_it);

    for (std::size_t i = 0; i < particles.nb_active(); ++i)
    {
        particles.pos()(i)(0) *= -1.;
    }

    solver.solve(2*total_it, total_it);

    return 0;
}
