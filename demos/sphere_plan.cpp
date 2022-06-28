#include <vector>
#include <xtensor/xmath.hpp>
#include <scopi/objects/types/sphere.hpp>
#include <scopi/objects/types/plan.hpp>
#include <scopi/solver.hpp>
#include <scopi/property.hpp>

#include <scopi/solvers/OptimProjectedGradient.hpp>
#include <scopi/vap/vap_fpd.hpp>

int main()
{
    plog::init(plog::verbose, "sphere_plan.log");

    constexpr std::size_t dim = 2;
    double PI = xt::numeric_constants<double>::PI;

    double radius = 1.;
    double g = 1.;
    double mass = 1.;
    double h = radius;
    auto prop = scopi::property<dim>().mass(mass).moment_inertia(mass*radius*radius/2.);
    scopi::OptimParams<scopi::OptimProjectedGradient<scopi::DryWithoutFriction>> params;

    double dt = 0.05;
    std::size_t total_it = 1;
    double alpha = PI/6.;

    scopi::scopi_container<dim> particles;
    scopi::plan<dim> p({{0., 0.}}, PI/2.-alpha);
    scopi::sphere<dim> s({{h*std::sin(alpha), h*std::cos(alpha)}}, radius);
    particles.push_back(p, scopi::property<dim>().deactivate());
    particles.push_back(s, prop.force({{0., -g}}));

    params.rho = 2.;
    scopi::ScopiSolver<dim, scopi::OptimProjectedGradient<scopi::DryWithoutFriction>, scopi::contact_kdtree, scopi::vap_fpd> solver(particles, dt, params);
    solver.solve(total_it);

    return 0;
}
