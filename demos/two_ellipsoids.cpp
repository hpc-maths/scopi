#include <xtensor/xmath.hpp>

#include <scopi/objects/types/superellipsoid.hpp>
#include <scopi/scopi.hpp>
#include <scopi/solver.hpp>

int main()
{
    scopi::initialize("Two ellipsoids");

    constexpr std::size_t dim = 2;
    double PI                 = xt::numeric_constants<double>::PI;
    double dt                 = .005;
    std::size_t total_it      = 1000;
    scopi::scopi_container<dim> particles;

    scopi::superellipsoid<dim> s1(
        {
            {-0.2, 0.}
    },
        {scopi::quaternion(PI / 4)},
        {{.1, .05}},
        1);
    scopi::superellipsoid<dim> s2(
        {
            {0.2, 0.}
    },
        {scopi::quaternion(-PI / 4)},
        {{.1, .05}},
        1);

    auto prop1 = scopi::property<dim>()
                     .desired_velocity({
                         {0.25, 0}
    })
                     .mass(1.)
                     .moment_inertia(0.1);
    auto prop2 = scopi::property<dim>()
                     .desired_velocity({
                         {-0.25, 0}
    })
                     .mass(1.)
                     .moment_inertia(0.1);

    particles.push_back(s1, prop1);
    particles.push_back(s2, prop2);

    scopi::ScopiSolver<dim> solver(particles);
    solver.run(dt, total_it);
    return 0;
}
