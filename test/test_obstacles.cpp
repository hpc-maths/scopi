#include "doctest/doctest.h"
#include <random>
#include<xtensor/xmath.hpp>

#include "test_common.hpp"
#include "utils.hpp"

#include <scopi/objects/types/sphere.hpp>
#include <scopi/objects/types/plan.hpp>
#include <scopi/vap/vap_fpd.hpp>
#include <scopi/container.hpp>
#include <scopi/solver.hpp>

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
        REQUIRE(q(0)(0) == doctest::Approx(std::sqrt(2.)/2.));
        REQUIRE(q(0)(1) == doctest::Approx(0.));
        REQUIRE(q(0)(2) == doctest::Approx(0.));
        REQUIRE(q(0)(3) == doctest::Approx(std::sqrt(2.)/2.));
        REQUIRE(q(1)(0) == doctest::Approx(1.));
        REQUIRE(q(1)(1) == doctest::Approx(0.));
        REQUIRE(q(1)(2) == doctest::Approx(0.));
        REQUIRE(q(1)(3) == doctest::Approx(0.));
    }

    TEST_CASE_TEMPLATE("sphere plan", SolverType, SOLVER_WITH_CONTACT(2, contact_kdtree, vap_fixed), SOLVER_WITH_CONTACT(2, contact_brute_force, vap_fixed))
    {
        static constexpr std::size_t dim = 2;

        double dt = .005;
        std::size_t total_it = 100;
        double radius = 1.;

        sphere<dim> s({{0., radius}}, radius);
        plan<dim> p({{ 0.,  0.}}, PI/2.);
        auto prop = property<dim>().mass(1.).moment_inertia(0.1);

        scopi_container<dim> particles;
        particles.push_back(p, property<dim>().deactivate());

        SUBCASE("fixed")
        {
            particles.push_back(s, prop);
            SolverType solver(particles, dt);
            solver.solve(total_it);
            check_result_sphere_plan(particles);
        }

        SUBCASE("velocity")
        {
            particles.push_back(s, prop.desired_velocity({{0., -1.}}));
            SolverType solver(particles, dt);
            solver.solve(total_it);
            check_result_sphere_plan(particles);
        }
    }

    TEST_CASE_TEMPLATE("sphere plan force", SolverType, SOLVER_WITH_CONTACT(2, contact_kdtree, vap_fpd), SOLVER_WITH_CONTACT(2, contact_brute_force, vap_fpd))
    {
        static constexpr std::size_t dim = 2;

        double dt = .005;
        std::size_t total_it = 100;
        double radius = 1.;

        sphere<dim> s({{0., radius}}, radius);
        plan<dim> p({{ 0.,  0.}}, PI/2.);
        auto prop = property<dim>().mass(1.).moment_inertia(0.1);

        scopi_container<dim> particles;
        particles.push_back(p, property<dim>().deactivate());

        particles.push_back(s, prop.force({{0., -1.}}));
        SolverType solver(particles, dt);
        solver.solve(total_it);
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

    TEST_CASE_TEMPLATE("sphere sphere fixed", SolverType, SOLVER_WITH_CONTACT(2, contact_kdtree, vap_fixed), SOLVER_WITH_CONTACT(2, contact_brute_force, vap_fixed))
    {
        static constexpr std::size_t dim = 2;
        double dt = .005;
        std::size_t total_it = 100;
        double radius = 1.;

        sphere<dim> obstacle({{ 0.,  0.}}, radius);
        sphere<dim> sphere({{ 0.,  2.}}, radius);
        auto prop = property<dim>().mass(1.).moment_inertia(0.1);

        scopi_container<dim> particles;
        particles.push_back(obstacle, property<dim>().deactivate());

        SUBCASE("fixed")
        {
            particles.push_back(sphere, prop);
            SolverType solver(particles, dt);
            solver.solve(total_it);
            check_result_sphere_sphere(particles);
        }

        SUBCASE("velocity")
        {
            particles.push_back(sphere, prop.desired_velocity({{0., -1.}}));
            SolverType solver(particles, dt);
            solver.solve(total_it);
            check_result_sphere_sphere(particles);
        }
    }

    TEST_CASE_TEMPLATE("sphere sphere fixed force", SolverType, SOLVER_WITH_CONTACT(2, contact_kdtree, vap_fpd), SOLVER_WITH_CONTACT(2, contact_brute_force, vap_fpd))
    {
        static constexpr std::size_t dim = 2;
        double dt = .005;
        std::size_t total_it = 100;
        double radius = 1.;

        sphere<dim> obstacle({{ 0.,  0.}}, radius);
        sphere<dim> sphere({{ 0.,  2.}}, radius);
        auto prop = property<dim>().mass(1.).moment_inertia(0.1);

        scopi_container<dim> particles;
        particles.push_back(obstacle, property<dim>().deactivate());

        particles.push_back(sphere, prop.force({{0., -1.}}));
        SolverType solver(particles, dt);
        solver.solve(total_it);
        check_result_sphere_sphere(particles);
    }

    TEST_CASE_TEMPLATE("sphere sphere moving", SolverType, SOLVER_WITH_CONTACT(2, contact_kdtree, vap_fpd), SOLVER_WITH_CONTACT(2, contact_brute_force, vap_fpd))
    {
        static constexpr std::size_t dim = 2;
        double dt = .005;
        std::size_t total_it = 100;
        double radius = 1.;

        sphere<dim> obstacle({{ 0.,  0.}}, radius);
        sphere<dim> sphere({{ 1.,  std::sqrt(3.)}}, radius);
        auto prop = property<dim>().mass(1.).moment_inertia(0.1);

        scopi_container<dim> particles;
        particles.push_back(obstacle, property<dim>().deactivate());
        particles.push_back(sphere, prop.force({{0., -10.}}));

        SolverType solver(particles, dt);
        solver.solve(total_it);

        CHECK(diffFile("./Results/scopi_objects_0099.json", "../test/references/obstacles_sphere_sphere_moving.json", tolerance));
    }

    TEST_CASE_TEMPLATE("sphere inclined plan", SolverType, SOLVER_WITH_CONTACT(2, contact_kdtree, vap_fpd), SOLVER_WITH_CONTACT(2, contact_brute_force, vap_fpd))
    {
        std::tuple<double, double> data;
        std::vector<std::tuple<double, double>>
            data_container({std::make_tuple(PI/6., 0.000499746),
                            std::make_tuple(PI/4., 0.000499748),
                            std::make_tuple(PI/3., 0.000499729)});

        DOCTEST_VALUE_PARAMETERIZED_DATA(data, data_container);

        static constexpr std::size_t dim = 2;
        double radius = 1.;
        double g = 1.;
        double dt = 0.005;
        std::size_t total_it = 2000;
        double alpha = std::get<0>(data);

        auto prop = property<dim>().mass(1.).moment_inertia(1.*radius*radius/2.);
        plan<dim> p({{-radius*std::cos(PI/2.-alpha), -radius*std::sin(PI/2.-alpha)}}, PI/2.-alpha);
        sphere<dim> s({{0., 0.}}, radius);

        scopi_container<dim> particles;
        particles.push_back(p, property<dim>().deactivate());
        particles.push_back(s, prop.force({{0., -g}}));

        SolverType solver(particles, dt);
        solver.solve(total_it);

        auto pos = particles.pos();
        auto omega = particles.omega();
        auto sol = scopi::analytical_solution_sphere_plan(alpha, 0., dt*(total_it+1), radius, g);
        auto pos_analytical = sol.first;
        double err_pos = xt::linalg::norm(pos(1) - pos_analytical) / xt::linalg::norm(pos_analytical);

        REQUIRE(err_pos == doctest::Approx(std::get<1>(data)));
        REQUIRE(omega(1) == doctest::Approx(0.));
    }
}
