#include <gtest/gtest.h>
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
    // sphere - plan
    class TestSpherePlanBase {
        protected:
        static constexpr std::size_t dim = 2;
        TestSpherePlanBase() 
        : m_radius(1.)
        , m_s({{0., m_radius}}, m_radius)
        , m_p({{ 0.,  0.}}, PI/2.)
        , m_prop(property<dim>().mass(1.).moment_inertia({{0.1, 0.1, 0.1}}))
        {
            m_particles.push_back(m_p, property<dim>().deactivate());
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
        property<dim> m_prop; 
    };

    template <class S>
    class TestSpherePlan  : public ::testing::Test 
                          , public TestSpherePlanBase
    {
        protected:
        void SetUp() override {
            m_particles.push_back(m_s, m_prop);
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
            m_particles.push_back(m_s, m_prop.desired_velocity({{0., -1.}}));
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
            m_particles.push_back(m_s, m_prop.force({{0., -1.}}));
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
        , m_prop(property<dim>().mass(1.).moment_inertia({{0.1, 0.1, 0.1}}))
        {
            m_particles.push_back(m_obstacle, property<dim>().deactivate());
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
        property<dim> m_prop; 
    };

    template <class S>
    class TestSphereSphereFixed  : public ::testing::Test 
                                 , public TestSphereSphereBase
    {
        protected:
        TestSphereSphereFixed()
        : TestSphereSphereBase(1., 0., 2.)
        {
            m_particles.push_back(m_sphere, m_prop);
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
            m_particles.push_back(m_sphere, m_prop.desired_velocity({{0., -1.}}));
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
            m_particles.push_back(m_sphere, m_prop.force({{0., -1.}}));
        }
    };

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
        : TestSphereSphereBase(1., 2.*1./2., 2.*std::sqrt(3.)/2.)
        {
            m_particles.push_back(m_sphere, m_prop.force({{0., -10.}}));
        }
    };

    TYPED_TEST_SUITE(TestSphereSphereMoving, solver_types_vap);

    TYPED_TEST(TestSphereSphereMoving, sphere_sphere_moving)
    {
        TypeParam solver(this->m_particles, this->m_dt);
        solver.solve(this->m_total_it);
        EXPECT_PRED3(diffFile, "./Results/scopi_objects_0099.json", "../test/references/obstacles_sphere_sphere_moving.json", tolerance);
    }

    template <class S>
    class TestInclinedPlan  : public ::testing::Test 
    {
        protected:
        static constexpr std::size_t dim = 2;
        TestInclinedPlan()
        : m_r(1.)
        , m_alpha(PI/4.)
        , m_prop(property<dim>().mass(1.).moment_inertia({{0.1, 0.1, 0.1}}))
        {
            plan<dim> p({{-m_r*std::cos(m_alpha), -m_r*std::sin(m_alpha)}}, m_alpha);
            sphere<dim> s({{0., 0.}}, m_r);

            m_particles.push_back(p, property<dim>().deactivate());
            m_particles.push_back(s, m_prop.force({{0., -m_g}}));
        }

        double m_r;
        double m_alpha;
        double m_g = 1.;
        scopi_container<dim> m_particles;
        double m_dt = .001;
        std::size_t m_total_it = 1000;
        property<dim> m_prop; 
    };

    TYPED_TEST_SUITE(TestInclinedPlan, solver_types_vap);

    TYPED_TEST(TestInclinedPlan, inclined_plan)
    {
        GTEST_SKIP();
        TypeParam solver(this->m_particles, this->m_dt);
        solver.solve(this->m_total_it);

        auto pos = this->m_particles.pos();
        double tf = this->m_dt*(this->m_total_it+1);
        auto analytical_sol = this->m_g/2.*std::sin(this->m_alpha)*tf*tf * xt::xtensor<double, 1>({std::cos(this->m_alpha), -std::sin(this->m_alpha)});
        PLOG_DEBUG << "pos = " << pos(1);
        PLOG_DEBUG << "sol = " << analytical_sol;
        double error = xt::linalg::norm(pos(1) - analytical_sol, 2) / xt::linalg::norm(analytical_sol);
        PLOG_INFO << "error = " << error;
        EXPECT_NEAR(error, 0., 1e-2);
        // EXPECT_NEAR(pos(1)(0), x*std::cos(this->m_alpha), tolerance);
        // EXPECT_NEAR(pos(1)(1), -x*std::sin(this->m_alpha), tolerance);
    }
}
