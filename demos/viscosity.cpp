#include <vector>
#include <xtensor/xmath.hpp>
#include <scopi/objects/types/sphere.hpp>
#include <scopi/objects/types/plan.hpp>
#include <scopi/solver.hpp>
#include <scopi/property.hpp>

#include <scopi/solvers/OptimMosek.hpp>
#include <scopi/vap/vap_fpd.hpp>
#include <scopi/solvers/MatrixOptimSolverViscosity.hpp>

int main()
{
    plog::init(plog::warning, "viscosity.log");

    constexpr std::size_t dim = 2;
    double PI = xt::numeric_constants<double>::PI;

    double radius = 1.;
    double g = radius;
    double h = 2.*radius;
    auto prop = scopi::property<dim>().mass(1.).moment_inertia(1.*radius*radius/2.);

    double dt = 0.05;
    std::size_t total_it = 100;

    scopi::scopi_container<dim> particles;
    scopi::plan<dim> p({{0., 0.}}, PI/2.);
    scopi::sphere<dim> s({{0., h}}, radius);
    particles.push_back(p, scopi::property<dim>().deactivate());
    particles.push_back(s, prop.force({{0., -g}}));

    {
        scopi::ScopiSolver<dim, scopi::OptimMosek<scopi::MatrixOptimSolverViscosity<dim>>, scopi::contact_kdtree, scopi::vap_fpd> solver(particles, dt);
        solver.solve(total_it);
    }
    {
        particles.f()(1)(1) *= -1.;
        scopi::ScopiSolver<dim, scopi::OptimMosek<scopi::MatrixOptimSolverViscosity<dim>>, scopi::contact_kdtree, scopi::vap_fpd> solver(particles, dt);
        solver.solve(2*total_it, total_it);
    }

    return 0;
}
