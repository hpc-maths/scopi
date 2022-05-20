#include <vector>
#include <xtensor/xmath.hpp>
#include <scopi/objects/types/sphere.hpp>
#include <scopi/objects/types/plan.hpp>
#include <scopi/solver.hpp>
#include <scopi/property.hpp>

#include <scopi/solvers/OptimMosek.hpp>
#include <scopi/vap/vap_fpd.hpp>
#include <scopi/problems/DryWithFriction.hpp>

int main()
{
    plog::init(plog::warning, "sphere_plan.log");

    constexpr std::size_t dim = 2;
    double PI = xt::numeric_constants<double>::PI;

    double radius = 1.;
    double g = 1.;
    double mass = 1.;
    double h = 2.*radius;
    auto prop = scopi::property<dim>().mass(mass).moment_inertia(mass*radius*radius/2.);

    std::vector<double> dt({0.1, 0.05, 0.01, 0.005, 0.001});
    std::vector<std::size_t> total_it({100, 200, 1000, 2000, 10000});
    std::vector<double> mu_vec({0., 0.1, 0.5, 1.});
    std::vector<double> alpha_vec({PI/3.});
    // std::vector<double> alpha_vec({PI/6., PI/4., PI/3.});

    for(auto mu : mu_vec)
    {
        for(auto alpha : alpha_vec)
        {
            for (std::size_t i = 0; i < dt.size(); ++i)
            {
                scopi::scopi_container<dim> particles;
                scopi::plan<dim> p({{0., 0.}}, PI/2.-alpha);
                scopi::sphere<dim> s({{h*std::sin(alpha), h*std::cos(alpha)}}, radius);
                particles.push_back(p, scopi::property<dim>().deactivate());
                particles.push_back(s, prop.force({{0., -g}}));

                scopi::ScopiSolver<dim, scopi::DryWithFriction, scopi::OptimMosek, scopi::contact_kdtree, scopi::vap_fpd> solver(particles, dt[i]);
                solver.set_coeff_friction(mu);
                solver.solve(total_it[i]);

                auto pos = particles.pos();
                auto q = particles.q();
                auto tmp = scopi::analytical_solution_sphere_plan(alpha, mu, dt[i]*(total_it[i]+1), radius, g, h);
                auto pos_analytical = tmp.first;
                auto q_analytical = scopi::quaternion(tmp.second);
                double error_pos = xt::linalg::norm(pos(1) - pos_analytical) / xt::linalg::norm(pos_analytical);
                double error_q = xt::linalg::norm(q(1) - q_analytical) / xt::linalg::norm(q_analytical);

                auto v = particles.v();
                auto omega = particles.omega();
                tmp = scopi::analytical_solution_sphere_plan_velocity(alpha, mu, dt[i]*(total_it[i]+1), radius, g, h);
                auto v_analytical = tmp.first;
                auto omega_analytical = tmp.second;
                double error_v = xt::linalg::norm(v(1) - v_analytical) / xt::linalg::norm(v_analytical);
                double error_omega = std::abs((omega(1) - omega_analytical) / omega_analytical);

                PLOG_WARNING << "mu = " << mu << "  alpha = " << alpha << "   dt = " << dt[i] << '\t' << error_pos << '\t' << error_q << '\t' << error_v << '\t' << error_omega;

            }
            PLOG_WARNING << std::endl;
        }
    }

    return 0;
}
