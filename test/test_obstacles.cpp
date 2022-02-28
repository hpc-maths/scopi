#include <gtest/gtest.h>
#include <random>

#include "test_common.hpp"
#include "utils.hpp"

#include <scopi/objects/types/sphere.hpp>
#include <scopi/objects/types/plan.hpp>
#include <scopi/vap/vap_fpd.hpp>
#include <scopi/container.hpp>
#include <scopi/solver.hpp>

namespace scopi
{
    // sphere - plan
    class TestSpherePlanBase {
        protected:
        static constexpr std::size_t dim = 2;
        TestSpherePlanBase() 
        : m_radius(1.)
        , m_s({{0., m_radius}}, m_radius)
        , m_p({{ 0.,  0.}}, PI/2.)
        {
            m_particles.push_back(m_p, scopi::property<dim>().deactivate());
        }

        void check_result()
        {
            auto pos = m_particles.pos();
            EXPECT_DOUBLE_EQ(pos(0)(0), 0.);
            EXPECT_DOUBLE_EQ(pos(0)(1), 0.);
            EXPECT_NEAR(pos(1)(0), 0., tolerance);
            EXPECT_NEAR(pos(1)(1), m_radius, tolerance);

            auto q = m_particles.q();
            EXPECT_DOUBLE_EQ(q(0)(0), std::sqrt(2.)/2.);
            EXPECT_DOUBLE_EQ(q(0)(1), 0.);
            EXPECT_DOUBLE_EQ(q(0)(2), 0.);
            EXPECT_DOUBLE_EQ(q(0)(3), std::sqrt(2.)/2.);
            EXPECT_DOUBLE_EQ(q(1)(0), 1.);
            EXPECT_DOUBLE_EQ(q(1)(1), 0.);
            EXPECT_DOUBLE_EQ(q(1)(2), 0.);
            EXPECT_NEAR(q(1)(3), 0., tolerance);
        }

        double m_dt = .005;
        std::size_t m_total_it = 100;

        double m_radius;
        sphere<dim> m_s;
        plan<dim> m_p;
        scopi_container<2> m_particles;
    };

    template <class S>
    class TestSpherePlan  : public ::testing::Test 
                          , public TestSpherePlanBase
    {
        protected:
        void SetUp() override {
            m_particles.push_back(m_s);
        }
    };

    TYPED_TEST_SUITE(TestSpherePlan, solver_with_contact_types<2>);

    TYPED_TEST(TestSpherePlan, sphere_plan)
    {
        TypeParam solver(this->m_particles, this->m_dt);
        solver.solve(this->m_total_it);
        this->check_result();
    }

    template <class S>
    class TestSpherePlanVelocity  : public ::testing::Test 
                                  , public TestSpherePlanBase
    {
        protected:
        void SetUp() override {
            m_particles.push_back(m_s, scopi::property<dim>().desired_velocity({{0., -1.}}));
        }
    };

    TYPED_TEST_SUITE(TestSpherePlanVelocity, solver_with_contact_types<2>);

    TYPED_TEST(TestSpherePlanVelocity, sphere_plan_velocity)
    {
        TypeParam solver(this->m_particles, this->m_dt);
        solver.solve(this->m_total_it);
        this->check_result();
    }

    template <class S>
    class TestSpherePlanForce  : public ::testing::Test 
                               , public TestSpherePlanBase
    {
        protected:
        void SetUp() override {
            m_particles.push_back(m_s, scopi::property<dim>().force({{0., -1.}}));
        }
    };

    using solver_types_vap = solver_with_contact_types<2, vap_fpd>; // TODO does not compile without using
    TYPED_TEST_SUITE(TestSpherePlanForce, solver_types_vap);

    TYPED_TEST(TestSpherePlanForce, sphere_plan_force)
    {
        TypeParam solver(this->m_particles, this->m_dt);
        solver.solve(this->m_total_it);
        this->check_result();
    }

    // sphere - sphere
    class TestSphereSphereBase {
        protected:
        static constexpr std::size_t dim = 2;
        TestSphereSphereBase(double radius, double pos_x, double pos_y) 
        : m_radius(radius)
        , m_sphere({{ pos_x,  pos_y}}, m_radius)
        , m_obstacle({{ 0.,  0.}}, m_radius)
        {
            m_particles.push_back(m_obstacle, scopi::property<dim>().deactivate());
        }

        void check_result_fixed()
        {
            auto pos = m_particles.pos();
            EXPECT_DOUBLE_EQ(pos(0)(0), 0.);
            EXPECT_DOUBLE_EQ(pos(0)(1), 0.);
            EXPECT_DOUBLE_EQ(pos(1)(0), 0.);
            EXPECT_NEAR(pos(1)(1), 2., tolerance);

            auto q = m_particles.q();
            EXPECT_DOUBLE_EQ(q(0)(0), 1.);
            EXPECT_DOUBLE_EQ(q(0)(1), 0.);
            EXPECT_DOUBLE_EQ(q(0)(2), 0.);
            EXPECT_DOUBLE_EQ(q(0)(3), 0.);
            EXPECT_DOUBLE_EQ(q(1)(0), 1.);
            EXPECT_DOUBLE_EQ(q(1)(1), 0.);
            EXPECT_DOUBLE_EQ(q(1)(2), 0.);
            EXPECT_DOUBLE_EQ(q(1)(3), 0.);
        }

        double m_dt = .005;
        std::size_t m_total_it = 100;

        double m_radius;
        sphere<dim> m_sphere;
        sphere<dim> m_obstacle;
        scopi_container<2> m_particles;
    };

    template <class S>
    class TestSphereSphereFixed  : public ::testing::Test 
                                 , public TestSphereSphereBase
    {
        protected:
        TestSphereSphereFixed()
        : TestSphereSphereBase(1., 0., 2.)
        {
            m_particles.push_back(m_sphere);
        }
    };

    TYPED_TEST_SUITE(TestSphereSphereFixed, solver_with_contact_types<2>);

    TYPED_TEST(TestSphereSphereFixed, sphere_sphere_fixed)
    {
        TypeParam solver(this->m_particles, this->m_dt);
        solver.solve(this->m_total_it);
        this->check_result_fixed();
    }

    template <class S>
    class TestSphereSphereFixedVelocity  : public ::testing::Test 
                                         , public TestSphereSphereBase
    {
        protected:
        TestSphereSphereFixedVelocity()
        : TestSphereSphereBase(1., 0., 2.)
        {
            m_particles.push_back(m_sphere, scopi::property<dim>().desired_velocity({{0., -1.}}));
        }
    };

    TYPED_TEST_SUITE(TestSphereSphereFixedVelocity, solver_with_contact_types<2>);

    TYPED_TEST(TestSphereSphereFixedVelocity, sphere_sphere_fixed_velocity)
    {
        TypeParam solver(this->m_particles, this->m_dt);
        solver.solve(this->m_total_it);
        this->check_result_fixed();
    }

    template <class S>
    class TestSphereSphereFixedForce  : public ::testing::Test 
                                      , public TestSphereSphereBase
    {
        protected:
        TestSphereSphereFixedForce()
        : TestSphereSphereBase(1., 0., 2.)
        {
            m_particles.push_back(m_sphere, scopi::property<dim>().force({{0., -1.}}));
        }
    };

    using solver_types_vap = solver_with_contact_types<2, vap_fpd>; // TODO does not compile without using
    TYPED_TEST_SUITE(TestSphereSphereFixedForce, solver_types_vap);

    TYPED_TEST(TestSphereSphereFixedForce, sphere_sphere_fixed_force)
    {
        TypeParam solver(this->m_particles, this->m_dt);
        solver.solve(this->m_total_it);
        this->check_result_fixed();
    }

    template <class S>
    class TestSphereSphereMoving  : public ::testing::Test 
                                  , public TestSphereSphereBase
    {
        protected:
        TestSphereSphereMoving()
        : TestSphereSphereBase(1., std::sqrt(3.), std::sqrt(3.)-3./2.)
        {
            m_particles.push_back(m_sphere, scopi::property<dim>().force({{0., -1.}}));
        }
    };

    using solver_types_vap = solver_with_contact_types<2, vap_fpd>; // TODO does not compile without using
    TYPED_TEST_SUITE(TestSphereSphereMoving, solver_types_vap);

    TYPED_TEST(TestSphereSphereMoving, sphere_sphere_moving)
    {
        TypeParam solver(this->m_particles, this->m_dt);
        solver.solve(this->m_total_it);
    }
}
