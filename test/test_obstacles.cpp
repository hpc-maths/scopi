#include <doctest/doctest.h>
#include <random>
#include <xtensor/xmath.hpp>

// #include "analytical_solution.hpp"
#include "test_common.hpp"
#include "utils.hpp"

#include <scopi/container.hpp>
#include <scopi/objects/types/plane.hpp>
#include <scopi/objects/types/sphere.hpp>
#include <scopi/solver.hpp>
#include <scopi/vap/vap_fpd.hpp>

namespace scopi
{
    void check_result_sphere_plan(const scopi_container<2>& particles)
    {
        auto pos = particles.pos();
        REQUIRE(pos(0)(0) == doctest::Approx(0.));
        REQUIRE(pos(0)(1) == doctest::Approx(0.));
        REQUIRE(pos(1)(0) == doctest::Approx(0.));
        REQUIRE(pos(1)(1) == doctest::Approx(1.));

        auto q = particles.q();
        REQUIRE(q(0)(0) == doctest::Approx(std::sqrt(2.) / 2.));
        REQUIRE(q(0)(1) == doctest::Approx(0.));
        REQUIRE(q(0)(2) == doctest::Approx(0.));
        REQUIRE(q(0)(3) == doctest::Approx(std::sqrt(2.) / 2.));
        REQUIRE(q(1)(0) == doctest::Approx(1.));
        REQUIRE(q(1)(1) == doctest::Approx(0.));
        REQUIRE(q(1)(2) == doctest::Approx(0.));
        REQUIRE(q(1)(3) == doctest::Approx(0.));
    }

    template <class solver_t>
    void set_params(solver_t& solver, std::size_t total_it)
    {
        auto params                           = solver.get_params();
        params.solver_params.output_frequency = total_it - 1;
    }

    TEST_CASE_TEMPLATE_DEFINE("sphere plane", SolverType, sphere_plane)
    {
        static constexpr std::size_t dim = 2;

        double dt            = .005;
        std::size_t total_it = 100;
        double radius        = 1.;

        sphere<dim> s(
            {
                {0., radius}
        },
            radius);
        plane<dim> p(
            {
                {0., 0.}
        },
            PI / 2.);
        auto prop = property<dim>().mass(1.).moment_inertia(0.1);

        scopi_container<dim> particles;
        particles.push_back(p, property<dim>().deactivate());

        SUBCASE("fixed")
        {
            particles.push_back(s, prop);
            SolverType solver(particles);
            solver.run(dt, total_it);
            set_params(solver, total_it);
            check_result_sphere_plan(particles);
        }

        SUBCASE("velocity")
        {
            particles.push_back(s,
                                prop.desired_velocity({
                                    {0., -1.}
            }));
            SolverType solver(particles);
            solver.run(dt, total_it);
            set_params(solver, total_it);
            check_result_sphere_plan(particles);
        }
    }

    TEST_CASE_TEMPLATE_DEFINE("sphere plane force", SolverType, sphere_plane_force)
    {
        static constexpr std::size_t dim = 2;

        double dt            = .005;
        std::size_t total_it = 100;
        double radius        = 1.;

        sphere<dim> s(
            {
                {0., radius}
        },
            radius);
        plane<dim> p(
            {
                {0., 0.}
        },
            PI / 2.);
        auto prop = property<dim>().mass(1.).moment_inertia(0.1);

        scopi_container<dim> particles;
        particles.push_back(p, property<dim>().deactivate());

        particles.push_back(s,
                            prop.force({
                                {0., -1.}
        }));
        SolverType solver(particles);
        solver.run(dt, total_it);
        set_params(solver, total_it);
        check_result_sphere_plan(particles);
    }

    void check_result_sphere_sphere(const scopi_container<2>& particles)
    {
        auto pos = particles.pos();
        REQUIRE(pos(0)(0) == doctest::Approx(0.));
        REQUIRE(pos(0)(1) == doctest::Approx(0.));
        REQUIRE(pos(1)(0) == doctest::Approx(0.));
        REQUIRE(pos(1)(1) == doctest::Approx(2.));

        auto q = particles.q();
        REQUIRE(q(0)(0) == doctest::Approx(1.));
        REQUIRE(q(0)(1) == doctest::Approx(0.));
        REQUIRE(q(0)(2) == doctest::Approx(0.));
        REQUIRE(q(0)(3) == doctest::Approx(0.));
        REQUIRE(q(1)(0) == doctest::Approx(1.));
        REQUIRE(q(1)(1) == doctest::Approx(0.));
        REQUIRE(q(1)(2) == doctest::Approx(0.));
        REQUIRE(q(1)(3) == doctest::Approx(0.));
    }

    TEST_CASE_TEMPLATE_DEFINE("sphere sphere fixed", SolverType, sphere_sphere_fixed)
    {
        static constexpr std::size_t dim = 2;
        double dt                        = .005;
        std::size_t total_it             = 100;
        double radius                    = 1.;

        sphere<dim> obstacle(
            {
                {0., 0.}
        },
            radius);
        sphere<dim> sphere(
            {
                {0., 2.}
        },
            radius);
        auto prop = property<dim>().mass(1.).moment_inertia(0.1);

        scopi_container<dim> particles;
        particles.push_back(obstacle, property<dim>().deactivate());

        SUBCASE("fixed")
        {
            particles.push_back(sphere, prop);
            SolverType solver(particles);
            solver.run(dt, total_it);
            set_params(solver, total_it);
            check_result_sphere_sphere(particles);
        }

        SUBCASE("velocity")
        {
            particles.push_back(sphere,
                                prop.desired_velocity({
                                    {0., -1.}
            }));
            SolverType solver(particles);
            solver.run(dt, total_it);
            set_params(solver, total_it);
            check_result_sphere_sphere(particles);
        }
    }

    TEST_CASE_TEMPLATE_DEFINE("sphere sphere fixed force", SolverType, sphere_sphere_fixed_force)
    {
        static constexpr std::size_t dim = 2;
        double dt                        = .005;
        std::size_t total_it             = 100;
        double radius                    = 1.;

        sphere<dim> obstacle(
            {
                {0., 0.}
        },
            radius);
        sphere<dim> sphere(
            {
                {0., 2.}
        },
            radius);
        auto prop = property<dim>().mass(1.).moment_inertia(0.1);

        scopi_container<dim> particles;
        particles.push_back(obstacle, property<dim>().deactivate());

        particles.push_back(sphere,
                            prop.force({
                                {0., -1.}
        }));
        SolverType solver(particles);
        solver.run(dt, total_it);
        set_params(solver, total_it);
        check_result_sphere_sphere(particles);
    }

    TEST_CASE_TEMPLATE_DEFINE("sphere sphere moving", SolverType, sphere_sphere_moving)
    {
        static constexpr std::size_t dim = 2;
        double dt                        = .005;
        std::size_t total_it             = 100;
        double radius                    = 1.;

        sphere<dim> obstacle(
            {
                {0., 0.}
        },
            radius);
        sphere<dim> sphere(
            {
                {1., std::sqrt(3.)}
        },
            radius);
        auto prop = property<dim>().mass(1.).moment_inertia(0.1);

        scopi_container<dim> particles;
        particles.push_back(obstacle, property<dim>().deactivate());
        particles.push_back(sphere,
                            prop.force({
                                {0., -10.}
        }));

        SolverType solver(particles);
        set_params(solver, total_it);
        auto params                           = solver.get_params();
        params.solver_params.output_frequency = 1;
        solver.run(dt, total_it);

        CHECK(diffFile("./Results/scopi_objects_0099.json", "../test/references/obstacles_sphere_sphere_moving.json", tolerance));
    }

    TEST_CASE_TEMPLATE_DEFINE("sphere inclined plane", SolverType, sphere_inclined_plane)
    {
        std::tuple<double, double, double, double> data;
        std::vector<std::tuple<double, double, double, double>> data_container({std::make_tuple(PI / 6., 0.000998789, 0., 0.0010002),
                                                                                std::make_tuple(PI / 4., 0.00100008, 0., 0.00100119),
                                                                                std::make_tuple(PI / 3., 0.00100105, 0., 0.0010027)});

        DOCTEST_VALUE_PARAMETERIZED_DATA(data, data_container);

        static constexpr std::size_t dim = 2;
        double radius                    = 1.;
        double g                         = 1.;
        double dt                        = 0.01;
        std::size_t total_it             = 1000;
        double h                         = 2. * radius;
        double alpha                     = std::get<0>(data);

        auto prop = property<dim>().mass(1.).moment_inertia(1. * radius * radius / 2.);
        plane<dim> p(
            {
                {0., 0.}
        },
            PI / 2. - alpha);
        sphere<dim> s(
            {
                {h * std::sin(alpha), h * std::cos(alpha)}
        },
            radius);

        scopi_container<dim> particles;
        particles.push_back(p, property<dim>().deactivate());
        particles.push_back(s,
                            prop.force({
                                {0., -g}
        }));

        SolverType solver(particles);
        solver.run(dt, total_it);
        set_params(solver, total_it);

        auto pos            = particles.pos();
        auto q              = particles.q();
        auto tmp            = analytical_solution_sphere_plane(alpha, 0., dt * (total_it + 1), radius, g, h);
        auto pos_analytical = tmp.first;
        auto q_analytical   = quaternion(tmp.second);
        double error_pos    = xt::linalg::norm(pos(1) - pos_analytical) / xt::linalg::norm(pos_analytical);
        double error_q      = xt::linalg::norm(q(1) - q_analytical) / xt::linalg::norm(q_analytical);
        auto v              = particles.v();
        auto omega          = particles.omega();
        tmp                 = analytical_solution_sphere_plane_velocity(alpha, 0., dt * (total_it + 1), radius, g, h);
        auto v_analytical   = tmp.first;
        double error_v      = xt::linalg::norm(v(1) - v_analytical) / xt::linalg::norm(v_analytical);

        REQUIRE(error_pos == doctest::Approx(std::get<1>(data)));
        REQUIRE(error_q == doctest::Approx(std::get<2>(data)));
        REQUIRE(error_v == doctest::Approx(std::get<3>(data)).epsilon(1e-4));
        REQUIRE(omega(1) == doctest::Approx(0.));
    }

    TEST_CASE_TEMPLATE_APPLY(sphere_plane, solver_dry_without_friction_t<2>);
    TEST_CASE_TEMPLATE_APPLY(sphere_plane_force, solver_dry_without_friction_t<2, vap_fpd>);
    TEST_CASE_TEMPLATE_APPLY(sphere_sphere_fixed, solver_dry_without_friction_t<2>);
    TEST_CASE_TEMPLATE_APPLY(sphere_sphere_fixed_force, solver_dry_without_friction_t<2, vap_fpd>);
    TEST_CASE_TEMPLATE_APPLY(sphere_sphere_moving, solver_dry_without_friction_t<2, vap_fpd>);
    TEST_CASE_TEMPLATE_APPLY(sphere_inclined_plane, solver_dry_without_friction_t<2, vap_fpd>);

}
