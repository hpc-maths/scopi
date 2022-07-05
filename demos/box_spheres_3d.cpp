#include <vector>
#include <xtensor/xmath.hpp>
#include <scopi/objects/types/sphere.hpp>
#include <scopi/objects/types/plan.hpp>
#include <scopi/solver.hpp>
#include <scopi/property.hpp>

#include <scopi/solvers/OptimMosek.hpp>
#include <scopi/vap/vap_fpd.hpp>

int main()
{
    plog::init(plog::info, "box_spheres_3d.log");

    constexpr std::size_t dim = 3;
    double PI = xt::numeric_constants<double>::PI;

    double radius = 1.;
    double mass = 1.;
    double width_box = 10.;
    double g = 1.;

    auto prop = scopi::property<dim>().mass(mass).moment_inertia(mass*radius*radius/2.);
    scopi::OptimParams<scopi::OptimMosek<>> params;
    params.change_default_tol_mosek = false;

    double dt = 0.005;
    std::size_t total_it = 1;

    scopi::scopi_container<dim> particles;
    scopi::plan<dim> p_left({{0., 0.}}, 0.);
    scopi::plan<dim> p_right({{width_box, 0.}}, 0.);
    scopi::plan<dim> p_horizontal({{0., 0.}}, PI/2.);
    particles.push_back(p_left, scopi::property<dim>().deactivate());
    particles.push_back(p_right, scopi::property<dim>().deactivate());
    particles.push_back(p_horizontal, scopi::property<dim>().deactivate());

    scopi::sphere<dim> s({{5., 5.}}, 1.);
    particles.push_back(s, prop.force({{0., -g}}));

    scopi::ScopiSolver<dim, scopi::OptimMosek<>, scopi::contact_kdtree, scopi::vap_fpd> solver(particles, dt, params);
    solver.solve(total_it);

    return 0;
}
