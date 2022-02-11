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

    scopi::superellipsoid<dim> s1({{-0.2, 0.}}, {scopi::quaternion(PI/4)}, {{.1, .05}}, 1);
    scopi::superellipsoid<dim> s2({{0.2, 0.02}}, {scopi::quaternion(-PI/4)}, {{.1, .05}}, 1);

    auto prop1 = scopi::property<dim>().desired_velocity({{0.25, 0}});
    auto prop2 = scopi::property<dim>().desired_velocity({{-0.25, 0}});

    particles.push_back(s1, prop1);
    particles.push_back(s2, prop2);

    scopi::ScopiSolver<dim> solver(particles, dt);
    solver.solve(total_it);

    return 0;
}
