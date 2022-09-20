#include <vector>
#include <xtensor/xmath.hpp>
#include <scopi/objects/types/sphere.hpp>
#include <scopi/objects/types/plan.hpp>
#include <scopi/solver.hpp>
#include <scopi/property.hpp>

#include <scopi/solvers/OptimMosek.hpp>
#include <scopi/problems/DryWithFrictionFixedPoint.hpp>
#include <scopi/vap/vap_fpd.hpp>

int main()
{
    // Figure 5: Disk falling on a plane with friction. Distance, tangential velocity, normal velocity, and rotation.
    // mu = 0.1, fixed point algorithm
    plog::init(plog::error, "distance_and_velocity_with_friction_mu01_convex.log");

    constexpr std::size_t dim = 2;
    double PI = xt::numeric_constants<double>::PI;

    double radius = 1.;
    double g = 1.;
    double mass = 1.;
    double h = 2.*radius;
    auto prop = scopi::property<dim>().mass(mass).moment_inertia(mass*radius*radius/2.);
    scopi::Params<scopi::OptimMosek<scopi::DryWithFrictionFixedPoint>, scopi::contact_kdtree, scopi::vap_fpd> params;
    params.problem_params.mu = 0.1;
    params.problem_params.tol_fixed_point = 1e-6;
    params.optim_params.change_default_tol_mosek = false;
    params.scopi_params.write_velocity = true;

    double dt = 0.05;
    std::size_t total_it = 200;
    double alpha = PI/6.;

    scopi::scopi_container<dim> particles;
    scopi::plan<dim> p({{0., 0.}}, PI/2.-alpha);
    scopi::sphere<dim> s({{h*std::sin(alpha), h*std::cos(alpha)}}, radius);
    particles.push_back(p, scopi::property<dim>().deactivate());
    particles.push_back(s, prop.force({{0., -g}}));

    scopi::ScopiSolver<dim, scopi::OptimMosek<scopi::DryWithFrictionFixedPoint>, scopi::contact_kdtree, scopi::vap_fpd> solver(particles, dt, params);
    solver.run(total_it);

    return 0;
}
