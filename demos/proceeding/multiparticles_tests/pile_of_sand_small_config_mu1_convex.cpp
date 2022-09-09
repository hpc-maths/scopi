#include <cstddef>
#include <vector>
#include <xtensor/xmath.hpp>
#include <scopi/objects/types/sphere.hpp>
#include <scopi/objects/types/plan.hpp>
#include <scopi/solver.hpp>
#include <scopi/property.hpp>

#include <scopi/solvers/OptimMosek.hpp>
#include <scopi/problems/DryWithFriction.hpp>
#include <scopi/vap/vap_fpd.hpp>

int main()
{
    // Table 4: 8^3 spheres falling on a plane with friction.
    // mu = 1, convex scheme.
    plog::init(plog::info, "pile_of_sand_spheres_small_config_mu1_convex.log");

    constexpr std::size_t dim = 3;
    double PI = xt::numeric_constants<double>::PI;

    double width_box = 10.;
    std::size_t n = 8; // n^3 spheres
    std::size_t total_it = 1000;
    double g = 1.;

    double r = width_box/2./(n+1);
    double dt = 0.2*r/(std::sqrt(2.*width_box*g));

    scopi::Params<scopi::OptimMosek<scopi::DryWithFriction>, scopi::DryWithFriction, scopi::contact_kdtree, scopi::vap_fpd> params;
    params.optim_params.change_default_tol_mosek = false;
    params.problem_params.mu = 1.;
    params.contacts_params.dmax = r;
    params.contacts_params.kd_tree_radius = params.contacts_params.dmax + 2.*r;

    scopi::scopi_container<dim> particles;
    auto prop = scopi::property<dim>().force({{0., -g, 0.}});

    scopi::plan<dim> p_horizontal({{0., 0., 0.}}, PI/2.);
    particles.push_back(p_horizontal, scopi::property<dim>().deactivate());

    std::default_random_engine generator;
    std::uniform_real_distribution<double> distrib_m(1., 2.);

    for (std::size_t i = 0; i < n; ++i)
    {
        for (std::size_t j = 1; j < n; ++j)
        {
            for (std::size_t k = 0; k < n; ++k)
            {
                double m = distrib_m(generator);
                scopi::sphere<dim> s({{i*2.*r, r + j*2.*r, k*2.*r}}, r);
                particles.push_back(s, prop.mass(m).moment_inertia({m*r*r/2., m*r*r/2., m*r*r/2.}));
            }
        }
    }

    // j = 0
    double dec_x = 0.5*r;
    for (std::size_t i = 0; i < n; ++i)
    {
        for (std::size_t k = 0; k < n; ++k)
        {
            double m = distrib_m(generator);
            scopi::sphere<dim> s({{i*2.*r + dec_x, r, k*2.*r}}, r);
            particles.push_back(s, prop.mass(m).moment_inertia({m*r*r/2., m*r*r/2., m*r*r/2.}));
        }
    }

    scopi::ScopiSolver<dim, scopi::OptimMosek<scopi::DryWithFriction>, scopi::contact_kdtree, scopi::vap_fpd> solver(particles, dt, params);
    solver.solve(total_it);

    return 0;
}
