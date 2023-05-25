#include <cstddef>
#include <vector>
#include <xtensor/xmath.hpp>
#include <scopi/objects/types/sphere.hpp>
#include <scopi/objects/types/plan.hpp>
#include <scopi/solver.hpp>
#include <scopi/property.hpp>

#include <scopi/solvers/OptimProjectedGradient.hpp>
#include <scopi/solvers/gradient/apgd_ar.hpp>
#include <scopi/vap/vap_fpd.hpp>

int main()
{
    // Table 2: 8^3 spheres falling in a box.
    // APGD-AR algorithm.
    plog::init(plog::info, "spheres_in_box_small_config_apgdar.log");

    constexpr std::size_t dim = 3;
    double PI = xt::numeric_constants<double>::PI;

    std::size_t total_it = 1000;
    double width_box = 10.;
    std::size_t n = 8; // n^3 spheres
    double g = 1.;

    double r0 = width_box/n/2.;
    double dt = 0.2*0.9*r0/(std::sqrt(2.*width_box*g));
    double rho = 0.2/(dt*dt);

    scopi::scopi_container<dim> particles;
    auto prop = scopi::property<dim>().force({{0., -g, 0.}});

    const xt::xtensor_fixed<double, xt::xshape<dim>> axes_y({0., 1., 0.});
    const xt::xtensor_fixed<double, xt::xshape<dim>> axes_z({0., 0., 1.});

    scopi::plan<dim> p_left({{0., 0., 0.}}, {scopi::quaternion(0., axes_z)});
    particles.push_back(p_left, scopi::property<dim>().deactivate());
    scopi::plan<dim> p_right({{width_box+2*r0, 0., 0.}}, {scopi::quaternion(0., axes_z)});
    particles.push_back(p_right, scopi::property<dim>().deactivate());
    scopi::plan<dim> p_horizontal({{0., 0., 0.}}, {scopi::quaternion(PI/2., axes_z)});
    particles.push_back(p_horizontal, scopi::property<dim>().deactivate());
    scopi::plan<dim> p_front({{0., 0., 0.}}, {scopi::quaternion(PI/2., axes_y)});
    particles.push_back(p_front, scopi::property<dim>().deactivate());
    scopi::plan<dim> p_back({{0., 0., width_box+2*r0}}, {scopi::quaternion(PI/2., axes_y)});
    particles.push_back(p_back, scopi::property<dim>().deactivate());

    std::default_random_engine generator;
    std::uniform_real_distribution<double> distrib_r(0.8*r0, 0.9*r0);
    std::uniform_real_distribution<double> distrib_m(1., 2.);
    std::uniform_real_distribution<double> distrib_mouve_center(-0.05*r0, 0.05*r0);

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
                scopi::sphere<dim> s({{r0+0.1 + i*2.*r0 + dx, 1.1*r0 + 2.1*r0*j + dy, r0+0.1 + k*2.*r0 + dz}}, r);
                particles.push_back(s, prop.mass(m).moment_inertia({m*r*r/2., m*r*r/2., m*r*r/2.}));
            }
        }
    }

    scopi::ScopiSolver<dim, scopi::OptimProjectedGradient<scopi::DryWithoutFriction, scopi::apgd_ar>, scopi::contact_kdtree, scopi::vap_fpd> solver(particles, dt);
    auto params = solver.get_params();
    params.solver_params.output_frequency = std::size_t(-1);
    params.optim_params.tol_l = 1e-3;
    params.optim_params.rho = rho;
    params.contact_params.dmax = 0.9*r0;
    params.contact_params.kd_tree_radius = params.contact_params.dmax + 2.*0.9*r0;

    solver.run(total_it);

    return 0;
}
