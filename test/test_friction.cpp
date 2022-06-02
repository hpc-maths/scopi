#include <cstddef>
#include "doctest/doctest.h"
#include <random>

#include "test_common.hpp"
#include "utils.hpp"

#include <scopi/objects/types/sphere.hpp>
#include <scopi/vap/vap_fpd.hpp>
#include <scopi/container.hpp>
#include <scopi/solver.hpp>
#include <tuple>

namespace scopi {

    TEST_CASE_TEMPLATE("two spheres asymetrical friction", SolverAndParams, SOLVER_DRY_WITH_FRICTION(2, contact_kdtree, vap_fixed), SOLVER_DRY_WITH_FRICTION(2, contact_brute_force, vap_fixed))
    {
        using SolverType = typename SolverAndParams::SolverType;
        using OptimParamsType = typename SolverAndParams::OptimParamsType;

        static constexpr std::size_t dim = 2;
        double dt = .005;
        std::size_t total_it = 1000;
        double mu = 1./2.;

        sphere<dim> s1({{-0.2, -0.05}}, 0.1);
        sphere<dim> s2({{ 0.2,  0.05}}, 0.1);
        auto p = property<dim>().mass(1.).moment_inertia(0.1);

        scopi_container<dim>particles;
        particles.push_back(s1, p.desired_velocity({{0.25, 0}}));
        particles.push_back(s2, p.desired_velocity({{-0.25, 0}}));

        OptimParamsType params;
        SolverType solver(particles, dt, params);
        solver.set_coeff_friction(mu);
        solver.solve(total_it);

        CHECK(diffFile("./Results/scopi_objects_0999.json", "../test/references/two_spheres_asymmetrical_friction.json", tolerance));
    }

    TEST_CASE_TEMPLATE("critical 2d spheres friction", SolverAndParams, SOLVER_DRY_WITH_FRICTION(2, contact_kdtree, vap_fixed), SOLVER_DRY_WITH_FRICTION(2, contact_brute_force, vap_fixed))
    {
        using SolverType = typename SolverAndParams::SolverType;
        using OptimParamsType = typename SolverAndParams::OptimParamsType;

        static constexpr std::size_t dim = 2;
        double dt = .01;
        std::size_t total_it = 100;
        double mu = 1./2.;
        scopi_container<dim> particles;

        int n = 2; // 2*n*n particles
        std::minstd_rand0 generator(0);
        std::uniform_real_distribution<double> distrib_r(0.2, 0.4);
        std::uniform_real_distribution<double> distrib_move_x(-0.1, 0.1);
        std::uniform_real_distribution<double> distrib_move_y(-0.1, 0.1);
        std::uniform_real_distribution<double> distrib_velocity(2., 5.);
        auto prop = property<dim>().mass(1.).moment_inertia(0.1);

        for(int i = 0; i < n; ++i)
        {
            for(int j = 0; j < n; ++j)
            {
                double r = distrib_r(generator);
                double x = (i + 0.5) + distrib_move_x(generator);
                double y = (j + 0.5) + distrib_move_y(generator);
                double velocity = distrib_velocity(generator);
                sphere<dim> s1({{x, y}}, r);
                particles.push_back(s1, prop.desired_velocity({{velocity, 0.}}));

                r = distrib_r(generator);
                x = (n + i + 0.5) + distrib_move_x(generator);
                y = (j + 0.5) + distrib_move_y(generator);
                velocity = distrib_velocity(generator);
                sphere<dim> s2({{x, y}}, r);
                particles.push_back(s2, prop.desired_velocity({{-velocity, 0.}}));
            }
        }

        OptimParamsType params;
        SolverType solver(particles, dt, params);
        solver.set_coeff_friction(mu);
        solver.solve(total_it);

        CHECK(diffFile("./Results/scopi_objects_0099.json", "../test/references/2d_case_spheres_friction.json", tolerance));
    }

    TEST_CASE_TEMPLATE("sphere inclined plan friction", SolverAndParams, SOLVER_DRY_WITH_FRICTION(2, contact_kdtree, vap_fpd), SOLVER_DRY_WITH_FRICTION(2, contact_brute_force, vap_fpd))
    {
        using SolverType = typename SolverAndParams::SolverType;
        using OptimParamsType = typename SolverAndParams::OptimParamsType;

        std::tuple<double, double, double, double, double, double> data;
        std::vector<std::tuple<double, double, double, double, double, double>>
            data_container({std::make_tuple(0.1, PI/6., 0.00101117, 0.00409424, 0.00100648, 0.000971288),
                            std::make_tuple(0.1, PI/4., 0.00102263, 0.00304414, 0.00100977, 0.000929086),
                            std::make_tuple(0.1, PI/3., 0.00102867, 0.00178655, 0.00101159, 0.000856009),
                            std::make_tuple(0.5, PI/6., 0.00105583, 0.00715824, 0.86414, 0.000999002),
                            std::make_tuple(0.5, PI/4., 0.00104746, 0.0102814, 0.859236, 0.000999008),
                            std::make_tuple(0.5, PI/3., 0.0010907, 0.0096141, 0.00105025, 0.000883162),
                            std::make_tuple(1., PI/6., 0.00107652, 0.00680805, 0.86414, 0.000999002),
                            std::make_tuple(1., PI/4., 0.00112413, 0.00844355, 0.859236, 0.000999003), 
                            std::make_tuple(1., PI/3., 0.0011583, 0.00915214, 0.848851, 0.000999013)});

        DOCTEST_VALUE_PARAMETERIZED_DATA(data, data_container);

        static constexpr std::size_t dim = 2;
        double radius = 1.;
        double g = 1.;
        double dt = 0.01;
        std::size_t total_it = 1000;
        double h = 2.*radius;
        double mu = std::get<0>(data);
        double alpha = std::get<1>(data);

        auto prop = property<dim>().mass(1.).moment_inertia(1.*radius*radius/2.);
        plan<dim> p({{0., 0.}}, PI/2.-alpha);
        sphere<dim> s({{h*std::sin(alpha), h*std::cos(alpha)}}, radius);

        scopi_container<dim> particles;
        particles.push_back(p, property<dim>().deactivate());
        particles.push_back(s, prop.force({{0., -g}}));

        OptimParamsType params;
        SolverType solver(particles, dt, params);
        solver.set_coeff_friction(mu);
        solver.solve(total_it);

        auto pos = particles.pos();
        auto q = particles.q();
        auto tmp = analytical_solution_sphere_plan(alpha, mu, dt*(total_it+1), radius, g, h);
        auto pos_analytical = tmp.first;
        auto q_analytical = quaternion(tmp.second);
        double error_pos = xt::linalg::norm(pos(1) - pos_analytical) / xt::linalg::norm(pos_analytical);
        double error_q = xt::linalg::norm(q(1) - q_analytical) / xt::linalg::norm(q_analytical);
        auto v = particles.v();
        auto omega = particles.omega();
        tmp = analytical_solution_sphere_plan_velocity(alpha, mu, dt*(total_it+1), radius, g, h);
        auto v_analytical = tmp.first;
        double error_v = xt::linalg::norm(v(1) - v_analytical) / xt::linalg::norm(v_analytical);
        auto omega_analytical = tmp.second;
        double error_omega = std::abs((omega(1) - omega_analytical) / omega_analytical);

        REQUIRE(error_pos == doctest::Approx(std::get<2>(data)));
        REQUIRE(error_q == doctest::Approx(std::get<3>(data)));
        REQUIRE(error_v == doctest::Approx(std::get<4>(data)));
        REQUIRE(error_omega == doctest::Approx(std::get<5>(data)));
    }

}
