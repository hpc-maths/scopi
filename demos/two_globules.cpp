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

    scopi::sphere<dim> s2({{ -0.5,  0.}}, 0.1);
    particles.push_back(s2);

    scopi::globule<dim> g1({{0., 0.}}, 0.1);
    particles.push_back(g1);
    for (std::size_t i = 0; i < g1.size(); ++i)
    {
        particles[1]->pos(i)(0) = 2.*i*g1.radius();
    }

    scopi::OptimParams<scopi::OptimUzawaMatrixFreeOmp> optim_params;
    scopi::ProblemParams<scopi::DryWithoutFriction> problem_params;

    scopi::ScopiSolver<dim> solver(particles, dt, optim_params, problem_params);
    solver.solve(total_it);

    return 0;
}
