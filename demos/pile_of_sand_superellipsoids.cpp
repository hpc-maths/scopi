#include <cstddef>
#include <vector>
#include <xtensor/xmath.hpp>
#include <scopi/objects/types/superellipsoid.hpp>
#include <scopi/objects/types/plan.hpp>
#include <scopi/solver.hpp>
#include <scopi/property.hpp>

#include <scopi/solvers/OptimProjectedGradient.hpp>
#include <scopi/solvers/OptimMosek.hpp>
#include <scopi/vap/vap_fpd.hpp>
#include <scopi/contact/contact_brute_force.hpp>

template <std::size_t dim>
void add_obstacle(scopi::scopi_container<dim>& particles, double r, std::size_t i)
{
    scopi::superellipsoid<dim> s({{i*2.*r, -r}}, {r, r}, 1.);
    particles.push_back(s, scopi::property<dim>().deactivate());
}

int main()
{
    plog::init(plog::info, "pile_of_sand_superellipsoids.log");

    constexpr std::size_t dim = 2;
    double PI = xt::numeric_constants<double>::PI;

    double dt = 0.01;
    std::size_t total_it = 100;
    double width_box = 10.;
    std::size_t n = 3; // n^3 spheres
    double g = 1.;

    scopi::Params<scopi::OptimProjectedGradient<scopi::DryWithoutFriction>, scopi::DryWithoutFriction, scopi::contact_brute_force, scopi::vap_fpd> params;
    params.optim_params.tol_l = 1e-3;
    params.optim_params.rho = 2.;
    // params.optim_params.change_default_tol_mosek = false;
    // params.problem_params.mu = 0.1;
    double r = width_box/2./(n+1);

    scopi::scopi_container<dim> particles;
    auto prop = scopi::property<dim>().force({{0., -g}});

    // obstacles
    scopi::plan<dim> p({{0., 0}}, PI/2.);
    particles.push_back(p, scopi::property<dim>().deactivate());
    // for (std::size_t i = 0; i < n; ++i)
    // {
    //     add_obstacle(particles, r, -i   );
    //     add_obstacle(particles, r,  i   );
    //     add_obstacle(particles, r, (i+n));
    // }

    std::default_random_engine generator;
    std::uniform_real_distribution<double> distrib_m(1., 2.);

    for (std::size_t i = 0; i < n; ++i)
    {
        for (std::size_t j = 1; j < n; ++j)
        {
            double m = distrib_m(generator);
            scopi::superellipsoid<dim> s({{i*2.*2.*r, r + j*2.*r}}, {2.*r, r}, 1.);
            particles.push_back(s, prop.mass(m).moment_inertia(m*PI/4.*2.*r*r*r*r));
        }
    }

    // j = 0
    double dec_x = 0.5*r;
    for (std::size_t i = 0; i < n; ++i)
    {
        double m = distrib_m(generator);
        scopi::superellipsoid<dim> s({{i*2.*2.*r + dec_x, r}}, {2.*r, r}, 1.);
        particles.push_back(s, prop.mass(m).moment_inertia(m*PI/4.*2.*r*r*r*r));
    }

    scopi::ScopiSolver<dim, scopi::OptimProjectedGradient<scopi::DryWithoutFriction>, scopi::contact_brute_force, scopi::vap_fpd> solver(particles, dt, params);
    solver.solve(total_it);

    return 0;
}
