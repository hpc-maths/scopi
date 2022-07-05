#include <cstddef>
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

    double width_box = 10.;
    std::size_t n = 10; // n^3 spheres
    double dt = 0.005;
    std::size_t total_it = 1000;
    double g = 1.;
    double r0 = 0.8; // 0 < r0 <= 1

    scopi::OptimParams<scopi::OptimMosek<>> params;
    params.change_default_tol_mosek = false;

    scopi::scopi_container<dim> particles;
    auto prop = scopi::property<dim>().force({{0., -g, 0.}});

    scopi::plan<dim> p_left({{0., 0., 0.}}, 0.);
    scopi::plan<dim> p_right({{width_box, 0., 0.}}, 0.);
    // scopi::plan<dim> p_front({{0., 0., 0.}}, 0.);
    // scopi::plan<dim> p_back({{width_box, 0., 0.}}, 0.);
    scopi::plan<dim> p_horizontal({{0., 0., 0.}}, PI/2.);
    particles.push_back(p_left, scopi::property<dim>().deactivate());
    particles.push_back(p_right, scopi::property<dim>().deactivate());
    particles.push_back(p_horizontal, scopi::property<dim>().deactivate());

    std::default_random_engine generator;
    std::uniform_real_distribution<double> distrib_r(r0*(width_box-0.2)/2./n - 0.1, r0*(width_box-0.2)/2./n + 0.1);
    std::uniform_real_distribution<double> distrib_m(1., 2.);
    std::uniform_real_distribution<double> distrib_mouve_center(-0.1, 0.1);

    for (std::size_t i = 0; i < n; ++i)
    {
        for (std::size_t j = 0; j < n; ++j)
        {
            for (std::size_t k = 0; k < n; ++k)
            {
                double r = distrib_r(generator);
                double m = distrib_m(generator);
                double dx = distrib_mouve_center(generator);
                double dy = distrib_mouve_center(generator);
                double dz = distrib_mouve_center(generator);
                scopi::sphere<dim> s({{(width_box/2./n+0.1) + i*width_box/n+dx, (width_box/2./n + 1.) + j*width_box/n+dy, -width_box/2./n + k*width_box/n+dz}}, r);
                particles.push_back(s, prop.mass(m).moment_inertia({m*r*r/2., m*r*r/2., m*r*r/2.}));
            }
        }
    }

    scopi::ScopiSolver<dim, scopi::OptimMosek<>, scopi::contact_kdtree, scopi::vap_fpd> solver(particles, dt, params);
    solver.solve(total_it);

    return 0;
}
