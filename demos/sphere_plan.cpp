#include <xtensor/xmath.hpp>
#include <scopi/objects/types/sphere.hpp>
#include <scopi/objects/types/plan.hpp>
#include <scopi/solver.hpp>
#include <scopi/property.hpp>

#include <scopi/solvers/OptimMosek.hpp>
#include <scopi/vap/vap_fpd.hpp>

int main()
{
    plog::init(plog::debug, "sphere_plan.log");

    constexpr std::size_t dim = 2;
    scopi::scopi_container<dim> particles;

    double h = 10.;
    double r = 1.;
    double PI = xt::numeric_constants<double>::PI;
    double alpha = PI/4.;
    double L = h/std::tan(alpha);
    double dt = std::sqrt(2.*std::sqrt(2.)*L)/100.;
    PLOG_INFO << "dt = " << dt << "  L = " << L;
    std::size_t total_it = 100;

    scopi::sphere<dim> s({{0., h+r}}, r);
    scopi::plan<dim> p({{ L*std::cos(alpha),  0.}}, alpha);

    particles.push_back(p, scopi::property<dim>().deactivate());
    particles.push_back(s, scopi::property<dim>().force({{0., -1.}}));

    scopi::ScopiSolver<dim, scopi::OptimMosek<>, scopi::contact_kdtree, scopi::vap_fpd> solver(particles, dt);
    solver.solve(total_it);

    return 0;
}
