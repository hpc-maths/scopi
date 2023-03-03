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
    plog::init(plog::info, "sphere_plan.log");

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

    scopi::ScopiSolver<dim, scopi::OptimMosek<scopi::DryWithFrictionFixedPoint>, scopi::contact_kdtree, scopi::vap_fpd> solver(particles, dt);
    auto params = solver.get_params();
    params.problem_params.mu = 0.1;
    solver.run(total_it);

    return 0;
}
