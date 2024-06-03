#include <cstddef>
#include <scopi/objects/types/plan.hpp>
#include <scopi/objects/types/sphere.hpp>
#include <scopi/objects/types/superellipsoid.hpp>
#include <scopi/property.hpp>
#include <scopi/solver.hpp>
#include <vector>
#include <xtensor/xmath.hpp>

#include <scopi/contact/contact_brute_force.hpp>
#include <scopi/solvers/OptimMosek.hpp>
#include <scopi/solvers/OptimProjectedGradient.hpp>
#include <scopi/vap/vap_fpd.hpp>

template <std::size_t dim>
void add_obstacle(scopi::scopi_container<dim>& particles, double x, double r)
{
    // scopi::sphere<dim> s({{i*2.*r, -r}}, r);
    scopi::superellipsoid<dim> s(
        {
            {x, -r}
    },
        {r, r},
        1.);
    particles.push_back(s, scopi::property<dim>().deactivate());
}

int main()
{
#ifdef SCOPI_USE_MKL

    plog::init(plog::info, "pile_of_sand_superellipsoids.log");

    constexpr std::size_t dim = 2;
    double PI                 = xt::numeric_constants<double>::PI;

    double dt            = 0.01;
    std::size_t total_it = 1000;
    double width_box     = 10.;
    std::size_t n        = 10; // n^2 ellipses
    double g             = 1.;

    double r     = width_box / 2. / (n + 1);
    double r_obs = r / 10.;

    scopi::scopi_container<dim> particles;
    auto prop = scopi::property<dim>().force({
        {0., -g}
    });

    // obstacles
    // scopi::plan<dim> p({{0., 0}}, PI/2.);
    // particles.push_back(p, scopi::property<dim>().deactivate());
    double dist_obs = -width_box;
    while (dist_obs < 2. * width_box)
    {
        add_obstacle(particles, dist_obs, r_obs);
        dist_obs += 2 * r_obs;
    }
    PLOG_INFO << particles.size() << " obstacles";

    std::default_random_engine generator;
    std::uniform_real_distribution<double> distrib_m(1., 2.);
    std::uniform_real_distribution<double> distrib_rx(0.5 * r, 1.5 * r);

    for (std::size_t j = 1; j < n; ++j)
    {
        double x = 0.;
        while (x < width_box)
        // for (std::size_t i = 0; i < n; ++i)
        {
            double m  = distrib_m(generator);
            double rx = distrib_rx(generator);
            x += rx;
            // scopi::sphere<dim> s({{x, r + j*2.*r}}, rx);
            scopi::superellipsoid<dim> s(
                {
                    {x, r + j * 2. * r}
            },
                {rx, r},
                1.);
            x += rx;
            particles.push_back(s, prop.mass(m).moment_inertia(m * PI / 4. * 2. * rx * r * r * r));
        }
    }

    // j = 0
    double dec_x = 0.5 * r;
    double x     = 0.;
    // for (std::size_t i = 0; i < n; ++i)
    while (x < width_box)
    {
        double m  = distrib_m(generator);
        double rx = distrib_rx(generator);
        x += rx;
        // scopi::sphere<dim> s({{x + dec_x, r}}, rx);
        scopi::superellipsoid<dim> s(
            {
                {x + dec_x, r}
        },
            {rx, r},
            1.);
        x += rx;
        particles.push_back(s, prop.mass(m).moment_inertia(m * PI / 4. * 2. * rx * r * r * r));
    }

    scopi::ScopiSolver<dim, scopi::OptimProjectedGradient<scopi::DryWithoutFriction>, scopi::contact_brute_force, scopi::vap_fpd> solver(
        particles,
        dt);
    auto params                = solver.get_params();
    params.optim_params.tol_l  = 1e-6;
    params.contact_params.dmax = 2. * 1.5 * r;
    // params.optim_params.rho = 2.;
    // params.solver_params.output_frequency = 2;
    // params.optim_params.change_default_tol_mosek = false;
    // params.problem_params.mu = 0.1;
    solver.run(total_it);
#endif
    return 0;
}
