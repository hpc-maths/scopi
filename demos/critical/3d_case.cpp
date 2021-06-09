#include <xtensor/xmath.hpp>
#include <scopi/object/superellipsoid.hpp>
#include "../mosek_solver.hpp"

int main()
{
    constexpr std::size_t dim = 3;
    double PI = xt::numeric_constants<double>::PI;
    double dt = .005;
    std::size_t total_it = 600;
    scopi::scopi_container<dim> particles;

    // ellipsoids
    scopi::superellipsoid<dim> s1({{-0.2, 0., 0.}}, {scopi::quaternion(PI/4)}, {{.1, .05, .05}}, {{1, 1}});
    scopi::superellipsoid<dim> s2({{0.2, 0., 0.}}, {scopi::quaternion(-PI/4)}, {{.1, .05, .05}}, {{1, 1}});
    particles.push_back(s1, {{0, 0, 0}}, {{0.25, 0, 0}}, 0, 0, {{0, 0, 0}});
    particles.push_back(s2, {{0, 0, 0}}, {{-0.25, 0, 0}}, 0, 0, {{0, 0, 0}});
    std::size_t active_ptr = 0;

    mosek_solver(particles, dt, total_it, active_ptr);

    return 0;
}