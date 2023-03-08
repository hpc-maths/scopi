#include <vector>
#include <xtensor/xmath.hpp>
#include <scopi/objects/types/sphere.hpp>
#include <scopi/objects/types/plan.hpp>
#include <scopi/solver.hpp>
#include <scopi/property.hpp>

#include <scopi/solvers/OptimProjectedGradient.hpp>
#include <scopi/solvers/gradient/apgd_asr.hpp>
#include <scopi/vap/vap_fpd.hpp>

int main()
{
    // Figure 3: Disk falling on a plane without friction. Distance, tangential velocity, and normal velocity.
    plog::init(plog::error, "distance_and_velocity_without_friction.log");

    constexpr std::size_t dim = 2;
    double PI = xt::numeric_constants<double>::PI;

    double radius = 1.;
    double g = 1.;
    double mass = 1.;
    double h = 2.*radius;
    auto prop = scopi::property<dim>().mass(mass).moment_inertia(mass*radius*radius/2.);

    double dt = 0.05;
    std::size_t total_it = 200;
    double alpha = PI/6.;

    scopi::scopi_container<dim> particles;
    scopi::plan<dim> p({{0., 0.}}, PI/2.-alpha);
    scopi::sphere<dim> s({{h*std::sin(alpha), h*std::cos(alpha)}}, radius);
    particles.push_back(p, scopi::property<dim>().deactivate());
    particles.push_back(s, prop.force({{0., -g}}));

    scopi::ScopiSolver<dim, scopi::OptimProjectedGradient<scopi::DryWithoutFriction, scopi::apgd_asr>, scopi::contact_kdtree, scopi::vap_fpd> solver(particles, dt);
    auto params = solver.get_params();
    params.optim_params.tol_l = 1e-9;
    params.optim_params.rho = 2.;
    params.solver_params.write_velocity = true;

    solver.run(total_it);

    return 0;
}
