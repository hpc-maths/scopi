#include <cstddef>
#include <vector>
#include <xtensor/xmath.hpp>
#include <scopi/objects/types/sphere.hpp>
#include <scopi/objects/types/plan.hpp>
#include <scopi/solver.hpp>
#include <scopi/property.hpp>

#include <scopi/solvers/OptimProjectedGradient.hpp>
#include <scopi/solvers/gradient/nesterov_dynrho_restart.hpp>
#include <scopi/vap/vap_fpd.hpp>

int main()
{
    // Figure 4: Disk falling on a plane without friction. L^2 error on the position of the center of the disk as a function of dt.
    plog::init(plog::warning, "error_without_friction.log");

    constexpr std::size_t dim = 2;
    double PI = xt::numeric_constants<double>::PI;

    double radius = 1.;
    double g = 1.;
    double mass = 1.;
    double h = 2.*radius;
    double alpha = PI/6.;
    auto prop = scopi::property<dim>().mass(mass).moment_inertia(mass*radius*radius/2.);
    scopi::Params<scopi::OptimProjectedGradient<scopi::DryWithoutFriction, scopi::nesterov_dynrho_restart<>>, scopi::DryWithoutFriction, scopi::contact_kdtree, scopi::vap_fpd> params;
    params.optim_params.tol_l = 1e-9;
    params.optim_params.rho = 2.;
    params.scopi_params.write_velocity = true;

    std::vector<double> dt({0.1, 0.05, 0.01, 0.005, 0.001});
    std::vector<std::size_t> total_it({100, 200, 1000, 2000, 10000});

    for (std::size_t i = 0; i < dt.size(); ++i)
    {
        scopi::scopi_container<dim> particles;
        scopi::plan<dim> p({{0., 0.}}, PI/2.-alpha);
        scopi::sphere<dim> s({{h*std::sin(alpha), h*std::cos(alpha)}}, radius);
        particles.push_back(p, scopi::property<dim>().deactivate());
        particles.push_back(s, prop.force({{0., -g}}));

        double error_pos = 0.;
        for (std::size_t n = 1; n < total_it[i]; ++n)
        {
            scopi::ScopiSolver<dim, scopi::OptimProjectedGradient<scopi::DryWithoutFriction, scopi::nesterov_dynrho_restart<>>, scopi::contact_kdtree, scopi::vap_fpd> solver(particles, dt[i], params);
            solver.solve(n, n-1);

            auto pos = particles.pos();
            auto tmp = scopi::analytical_solution_sphere_plan(alpha, 0., dt[i]*n, radius, g, h);
            auto pos_analytical = tmp.first;
            error_pos += xt::linalg::norm(pos(1) - pos_analytical) / xt::linalg::norm(pos_analytical);
        }
        PLOG_WARNING << "dt = " << dt << " err = " << error_pos;
    }

    return 0;
}
