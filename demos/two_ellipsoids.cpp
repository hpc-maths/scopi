#include <xtensor/xmath.hpp>
#include <scopi/objects/types/superellipsoid.hpp>
#include <scopi/solvers/mosek.hpp>

int main()
{
    constexpr std::size_t dim = 2;
    double PI = xt::numeric_constants<double>::PI;
    double dt = .005;
    std::size_t total_it = 1000;
    scopi::scopi_container<dim> particles;

    scopi::superellipsoid<dim> s1({{-0.2, 0., 0.}}, {scopi::quaternion(PI/4)}, {{.1, .05, .05}}, {{1, 1}});
    scopi::superellipsoid<dim> s2({{0.2, 0., 0.}}, {scopi::quaternion(-PI/4)}, {{.1, .05, .05}}, {{1, 1}});
    particles.push_back(s1, {{0, 0, 0}}, {{0.25, 0, 0}}, 0, 0, {{0, 0, 0}});
    particles.push_back(s2, {{0, 0, 0}}, {{-0.25, 0, 0}}, 0, 0, {{0, 0, 0}});

    std::size_t active_ptr = 0; // without obstacles

    scopi::ScopiSolver<dim, scopi::MosekSolver<dim>> mosek_solver(particles, dt, active_ptr);
    // scopi::ScopiSolver<dim, scopi::ScsSolver<dim>> mosek_solver(particles, dt, active_ptr);
    mosek_solver.solve(total_it);
    return 0;
}
