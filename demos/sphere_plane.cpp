#include <scopi/objects/types/plane.hpp>
#include <scopi/objects/types/sphere.hpp>
#include <scopi/scopi.hpp>
#include <scopi/solver.hpp>
#include <scopi/vap/vap_fpd.hpp>
#include <xtensor/xmath.hpp>

int main(int argc, char** argv)
{
    scopi::initialize("sphere plane simulation");

    constexpr std::size_t dim = 2;
    double PI                 = xt::numeric_constants<double>::PI;

    double radius = 1.;
    double g      = 1.;
    double mass   = 1.;
    double h      = 2. * radius;
    auto prop     = scopi::property<dim>().mass(mass).moment_inertia(mass * radius * radius / 2.);

    double dt            = 0.05;
    std::size_t total_it = 200;
    double alpha         = PI / 6.;

    scopi::scopi_container<dim> particles;
    scopi::plane<dim> p(
        {
            {0., 0.}
    },
        PI / 2. - alpha);
    scopi::sphere<dim> s(
        {
            {h * std::sin(alpha), h * std::cos(alpha)}
    },
        radius);
    particles.push_back(p, scopi::property<dim>().deactivate());
    particles.push_back(s,
                        prop.force({
                            {0., -g}
    }));

    scopi::ScopiSolver<dim, scopi::FrictionFixedPoint, scopi::OptimGradient<scopi::apgd>, scopi::contact_kdtree, scopi::vap_fpd> solver(
        particles);
    auto params                        = solver.get_params();
    params.default_contact_property.mu = 0.1;

    SCOPI_PARSE(argc, argv);
    solver.run(dt, total_it);

    return 0;
}
