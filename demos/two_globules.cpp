#include <cstddef>
#include <vector>
#include <xtensor/xmath.hpp>
#include <scopi/objects/types/globule.hpp>
// #include <scopi/solver.hpp>
#include <scopi/property.hpp>
#include <xtensor/xadapt.hpp>

int main()
{
    plog::init(plog::error, "two_globules.log");

    constexpr std::size_t dim = 2;
    double dt = .005;
    std::size_t total_it = 1;
    // scopi::scopi_container<dim> particles;

    std::vector<scopi::type::position_t<dim>> pos(6);
    for (auto& p : pos)
        p.fill(1.);
    // std::cout << xt::adapt(pos) << std::endl;
    scopi::globule<dim> g1(pos, 1.);
    g1.pos(1)(0) = 2.;
    g1.pos(1)(1) = 3.;
    for (std::size_t i = 0; i < g1.size(); ++i)
    {
        std::cout << g1.pos(i) << std::endl;
    }
    // particles.push_back(g1);
    // scopi::sphere<dim> s1({{2., 3.}}, 1.);
    // std::cout << s1.pos(0)(0) << std::endl;
    // particles.push_back(s1);

    // for (std::size_t i = 0; i < particles.nb_active(); ++i)
    // {
    //     std::cout << particles.pos()(i) << std::endl;
    // }

    // scopi::globule<dim> g2({{-2., -1.}}, 1.);
    // particles.push_back(g2);

    // scopi::OptimParams<scopi::OptimUzawaMatrixFreeOmp> optim_params;
    // scopi::ProblemParams<scopi::DryWithoutFriction> problem_params;
    // scopi::ScopiSolver<dim> solver(particles, dt, optim_params, problem_params);
    // solver.solve(total_it);


    // for (std::size_t i = 0; i < particles.nb_active(); ++i)
    // {
    //     particles.pos()(i)(1) *= -1.;
    // }

    // solver.solve(2*total_it, total_it);

    return 0;
}
