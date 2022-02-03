#include <xtensor/xmath.hpp>
#include <scopi/objects/types/superellipsoid.hpp>
#include <scopi/solver.hpp>

int main()
{
    constexpr std::size_t dim = 2;
    double PI = xt::numeric_constants<double>::PI;
    double dt = .005;
    std::size_t total_it = 1000;
    scopi::scopi_container<dim> particles;

    scopi::superellipsoid<dim> s1({{-0.2, 0., 0.}}, {scopi::quaternion(PI/4)}, {{.1, .1, .1}}, {{1, 1}});
    scopi::superellipsoid<dim> s2({{0.2, 0., 0.}}, {scopi::quaternion(-PI/4)}, {{.1, .1, .1}}, {{1, 1}});
    particles.push_back(s1, {{0, 0, 0}}, {{0.25, 0, 0}}, 0, 0, {{0, 0, 0}});
    particles.push_back(s2, {{0, 0, 0}}, {{-0.25, 0, 0}}, 0, 0, {{0, 0, 0}});

    std::size_t active_ptr = 0; // without obstacles

    scopi::ScopiSolver<dim> solver(particles, dt, active_ptr);
    solver.solve(total_it);

    return 0;
}
