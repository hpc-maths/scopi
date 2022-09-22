#include <vector>
#include <xtensor/xmath.hpp>
#include <scopi/objects/types/sphere.hpp>
#include <scopi/objects/types/plan.hpp>
#include <scopi/solver.hpp>
#include <scopi/property.hpp>

#include <scopi/solvers/OptimProjectedGradient.hpp>
#include <scopi/solvers/gradient/apgd_as.hpp>
#include <scopi/vap/vap_fpd.hpp>

int main()
{
    // Figure 2: Disk placed on a plane without friction. Evolution of the constraint and cost as a function of the iterations for the APGD-AS algorithm.
    // Table 1: Disk placed on a plane without friction. Number of iterations for the APGD-AS algorithm and tolerance 10^{-9}.
    // The constraint is the first column in the log, the cost is the second column.
    plog::init(plog::verbose, "disk_on_inclined_plane_oneStep_apdgas_smallTol.log");

    constexpr std::size_t dim = 2;
    double PI = xt::numeric_constants<double>::PI;

    double radius = 1.;
    double g = 1.;
    double mass = 1.;
    double h = radius;
    auto prop = scopi::property<dim>().mass(mass).moment_inertia(mass*radius*radius/2.);
    scopi::Params<scopi::OptimProjectedGradient<scopi::DryWithoutFriction, scopi::apgd_as>, scopi::contact_kdtree, scopi::vap_fpd> params;
    params.optim_params.tol_l = 1e-9;
    params.optim_params.rho = 2.;
    params.optim_params.verbose = true;

    double dt = 0.05;
    std::size_t total_it = 1;
    double alpha = PI/6.;

    scopi::scopi_container<dim> particles;
    scopi::plan<dim> p({{0., 0.}}, PI/2.-alpha);
    scopi::sphere<dim> s({{h*std::sin(alpha), h*std::cos(alpha)}}, radius);
    particles.push_back(p, scopi::property<dim>().deactivate());
    particles.push_back(s, prop.force({{0., -g}}));

    scopi::ScopiSolver<dim, scopi::OptimProjectedGradient<scopi::DryWithoutFriction, scopi::apgd_as>, scopi::contact_kdtree, scopi::vap_fpd> solver(particles, dt, params);
    solver.run(total_it);

    return 0;
}
