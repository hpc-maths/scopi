#include <xtensor/xmath.hpp>

#include <plog/Log.h>
#include "plog/Initializers/RollingFileInitializer.h"

#include <scopi/objects/types/superellipsoid.hpp>
#include <scopi/property.hpp>
#include <scopi/solver.hpp>


int main()
{
    plog::init(plog::info, "two_ellipsoids.log");

    constexpr std::size_t dim = 2;
    double PI = xt::numeric_constants<double>::PI;
    double dt = .005;
    std::size_t total_it = 1000;
    scopi::scopi_container<dim> particles;

    scopi::superellipsoid<dim> s1({{-0.2, 0.}}, {scopi::quaternion(PI/4)}, {{.1, .05}}, 1);
    scopi::superellipsoid<dim> s2({{0.2, 0.}}, {scopi::quaternion(-PI/4)}, {{.1, .05}}, 1);

    auto prop1 = scopi::property<dim>().desired_velocity({{0.25, 0}}).mass(1.);
    auto prop2 = scopi::property<dim>().desired_velocity({{-0.25, 0}}).mass(1.);

    particles.push_back(s1, prop1);
    particles.push_back(s2, prop2);

    scopi::ScopiSolver<dim> mosek_solver(particles, dt);
    mosek_solver.solve(total_it);
    return 0;
}
