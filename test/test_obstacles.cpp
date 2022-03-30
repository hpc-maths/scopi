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




# if 0
    template <class S>
    class TestInclinedPlan  : public ::testing::Test 
    {
        protected:
        static constexpr std::size_t dim = 2;
        TestInclinedPlan()
        : m_r(1.)
        , m_alpha(PI/4.)
        , prop(property<dim>().mass(1.).moment_inertia(0.1))
        {
            plan<dim> p({{-m_r*std::cos(m_alpha), -m_r*std::sin(m_alpha)}}, m_alpha);
            sphere<dim> s({{0., 0.}}, m_r);

            particles.push_back(p, property<dim>().deactivate());
            particles.push_back(s, prop.force({{0., -m_g}}));
        }

        double m_r;
        double m_alpha;
        double m_g = 1.;
        scopi_container<dim> particles;
        double dt = .001;
        std::size_t total_it = 1000;
        property<dim> prop; 
    };

    TYPED_TEST_SUITE(TestInclinedPlan, solver_types_vap, );

    TYPED_TEST(TestInclinedPlan, inclined_plan)
    {
        GTEST_SKIP();
        TypeParam solver(this->particles, this->dt);
        solver.solve(this->total_it);

        auto pos = this->particles.pos();
        double tf = this->dt*(this->total_it+1);
        auto analytical_sol = this->m_g/2.*std::sin(this->m_alpha)*tf*tf * xt::xtensor<double, 1>({std::cos(this->m_alpha), -std::sin(this->m_alpha)});
        PLOG_DEBUG << "pos = " << pos(1);
        PLOG_DEBUG << "sol = " << analytical_sol;
        double error = xt::linalg::norm(pos(1) - analytical_sol, 2) / xt::linalg::norm(analytical_sol);
        PLOG_INFO << "error = " << error;
        EXPECT_NEAR(error, 0., 1e-2);
        // EXPECT_NEAR(pos(1)(0), x*std::cos(this->m_alpha), tolerance);
        // EXPECT_NEAR(pos(1)(1), -x*std::sin(this->m_alpha), tolerance);
    }
#endif
}
