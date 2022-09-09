#include <cstddef>
#include <vector>
#include <xtensor/xmath.hpp>
#include <scopi/objects/types/superellipsoid.hpp>
#include <scopi/solver.hpp>
#include <scopi/property.hpp>

#include <scopi/solvers/OptimProjectedGradient.hpp>
#include <scopi/solvers/gradient/nesterov_restart.hpp>
#include <scopi/vap/vap_fpd.hpp>

template <std::size_t dim>
void add_obstacle(scopi::scopi_container<dim>& particles, double x, double r)
{
    scopi::superellipsoid<dim> s({{x, -r}}, {r, r}, 1.);
    particles.push_back(s, scopi::property<dim>().deactivate());
}

int main()
{
    // Figure 10: pile of sand with ellipses.
    plog::init(plog::info, "pile_of_sand_ellipses.log");

    constexpr std::size_t dim = 2;
    double PI = xt::numeric_constants<double>::PI;
    std::size_t total_it = 1000;
    double width_box = 10.;
    std::size_t n = 100; // n^2 ellipses
    double g = 1.;

    double r = width_box/2./(n+1);
    double r_obs = r/10.;
    double dt = 0.2*r/(std::sqrt(2.*width_box*g));

    scopi::Params<scopi::OptimProjectedGradient<scopi::DryWithoutFriction, scopi::nesterov_restart<>>, scopi::DryWithoutFriction, scopi::contact_kdtree, scopi::vap_fpd> params;
    params.optim_params.tol_l = 1e-6;
    params.scopi_params.output_frequency = 20;
    params.scopi_params.filename = "/mnt/beegfs/workdir/helene.bloch/scopi/proceeding/220909_ellipses/scopi_objects_"
    params.contacts_params.dmax = r;
    params.contacts_params.kd_tree_radius = params.contacts_params.dmax + 2.*1.5*r;

    scopi::scopi_container<dim> particles;
    auto prop = scopi::property<dim>().force({{0., -g}});

    // obstacles
    double dist_obs = - width_box;
    while (dist_obs < 2.*width_box)
    {
        add_obstacle(particles, dist_obs, r_obs);
        dist_obs += 2*r_obs;
    }
    PLOG_INFO << particles.size() << " obstacles" ;

    std::default_random_engine generator;
    std::uniform_real_distribution<double> distrib_m(1., 2.);
    std::uniform_real_distribution<double> distrib_rx(0.5*r, 1.5*r);

    for (std::size_t j = 1; j < n; ++j)
    {
        double x = 0.;
        while (x < width_box)
        {
            double m = distrib_m(generator);
            double rx = distrib_rx(generator);
            x += rx;
            scopi::superellipsoid<dim> s({{x, r + j*2.*r}}, {rx, r}, 1.);
            x += rx;
            particles.push_back(s, prop.mass(m).moment_inertia(m*PI/4.*2.*rx*r*r*r));
        }
    }

    // j = 0
    double dec_x = 0.5*r;
    double x = 0.;
    while (x < width_box)
    {
        double m = distrib_m(generator);
        double rx = distrib_rx(generator);
        x += rx;
        scopi::superellipsoid<dim> s({{x + dec_x, r}}, {rx, r}, 1.);
        x += rx;
        particles.push_back(s, prop.mass(m).moment_inertia(m*PI/4.*2.*rx*r*r*r));
    }

    scopi::ScopiSolver<dim, scopi::OptimProjectedGradient<scopi::DryWithoutFriction, scopi::nesterov_restart<>>, scopi::contact_kdtree, scopi::vap_fpd> solver(particles, dt, params);
    solver.solve(total_it);

    return 0;
}
