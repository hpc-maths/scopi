#include <xtensor/xmath.hpp>
#include <scopi/objects/types/sphere.hpp>
#include <scopi/solvers/mosek.hpp>

int main()
{
    constexpr std::size_t dim = 2;
    double dt = .005;
    std::size_t total_it = 1000;
    scopi::scopi_container<dim> particles;


    scopi::sphere<dim> s1({{-0.2, -0.05}}, 0.1);
    scopi::sphere<dim> s2({{ 0.2,  0.05}}, 0.1);
    particles.push_back(s1, {{0, 0}}, {{0.25, 0}}, 0, 0, {{0, 0}});
    particles.push_back(s2, {{0, 0}}, {{-0.25, 0}}, 0, 0, {{0, 0}});

    std::size_t active_ptr = 0; // without obstacles
    scopi::MosekSolver<dim, scopi::useScsSolver> mosek_solver(particles, dt, active_ptr);
    mosek_solver.solve(total_it);

    return 0;
}
