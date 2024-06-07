#include <scopi/objects/types/plane.hpp>
#include <scopi/objects/types/sphere.hpp>
#include <scopi/solver.hpp>
#include <scopi/vap/vap_fpd.hpp>
#include <xtensor/xmath.hpp>

#include <random>

int main(int argc, char** argv)
{
    scopi::initialize("Pile of sand");

    constexpr std::size_t dim = 3;
    double PI                 = xt::numeric_constants<double>::PI;

    double dt            = 0.01;
    std::size_t total_it = 1000;
    double width_box     = 10.;
    std::size_t n        = 3; // n^3 spheres
    double g             = 1.;

    scopi::scopi_container<dim> particles;
    auto prop = scopi::property<dim>().force({
        {0., -g, 0.}
    });

    scopi::plane<dim> p_horizontal(
        {
            {0., 0., 0.}
    },
        PI / 2.);
    particles.push_back(p_horizontal, scopi::property<dim>().deactivate());

    std::default_random_engine generator;
    std::uniform_real_distribution<double> distrib_m(1., 2.);
    double r = width_box / 2. / (n + 1);

    for (std::size_t i = 0; i < n; ++i)
    {
        for (std::size_t j = 1; j < n; ++j)
        {
            for (std::size_t k = 0; k < n; ++k)
            {
                double m = distrib_m(generator);
                scopi::sphere<dim> s(
                    {
                        {i * 2. * r, r + j * 2. * r, k * 2. * r}
                },
                    r);
                particles.push_back(s, prop.mass(m).moment_inertia({m * r * r / 2., m * r * r / 2., m * r * r / 2.}));
            }
        }
    }

    // j = 0
    double dec_x = 0.5 * r;
    for (std::size_t i = 0; i < n; ++i)
    {
        for (std::size_t k = 0; k < n; ++k)
        {
            double m = distrib_m(generator);
            scopi::sphere<dim> s(
                {
                    {i * 2. * r + dec_x, r, k * 2. * r}
            },
                r);
            particles.push_back(s, prop.mass(m).moment_inertia({m * r * r / 2., m * r * r / 2., m * r * r / 2.}));
        }
    }

    scopi::ScopiSolver<dim, scopi::FrictionFixedPoint, scopi::OptimGradient<scopi::apgd>, scopi::contact_kdtree, scopi::vap_fpd> solver(
        particles);
    auto params                        = solver.get_params();
    params.default_contact_property.mu = 0.1;

    SCOPI_PARSE(argc, argv);
    solver.run(dt, total_it);

    return 0;
}
