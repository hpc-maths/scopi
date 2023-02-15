#include <vector>
#include <xtensor/xmath.hpp>
#include <scopi/objects/types/sphere.hpp>
#include <scopi/objects/types/plan.hpp>
#include <scopi/solver.hpp>
#include <scopi/property.hpp>

#include <scopi/solvers/OptimMosek.hpp>
#include <scopi/solvers/OptimUzawaMkl.hpp>
#include <scopi/vap/vap_fpd.hpp>
#include <scopi/problems/ViscousWithoutFriction.hpp>
#include <scopi/problems/ViscousWithFriction.hpp>

int main()
{
    plog::init(plog::info, "viscosity.log");

    constexpr std::size_t dim = 2;
    using solver_t = scopi::ScopiSolver<dim, scopi::OptimMosek<scopi::ViscousWithoutFriction<dim>>, scopi::contact_kdtree, scopi::vap_fpd>;
    using params_t = typename solver_t::params_t;
    double PI = xt::numeric_constants<double>::PI;

    double radius = 1.;
    double g = 1.;
    double h = 1.5*radius;
    double alpha = PI/4.;
    auto prop = scopi::property<dim>().mass(1.).moment_inertia(1.*radius*radius/2.);

    double dt = 0.05;
    std::size_t total_it = 60;
    scopi::scopi_container<dim> particles;
    scopi::plan<dim> p({{0., 0.}}, PI/2.);
    scopi::sphere<dim> s({{0., 2*h}}, radius);
    particles.push_back(p, scopi::property<dim>().deactivate());
    particles.push_back(s, prop.force({{g*std::cos(alpha), -g*std::sin(alpha)}}));
    params_t params;
    params.contacts_params.dmax = 2;
    solver_t solver(particles, dt, params);
    solver.run(total_it);
    particles.f()(1)(1) *= -1.;
    solver.run(3*total_it, total_it);
    particles.f()(1)(1) *= -1.;
    solver.run(8*total_it, 3*total_it);

    return 0;
}
