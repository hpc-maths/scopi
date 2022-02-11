#include <gtest/gtest.h>
#include "utils.hpp"

#include <scopi/objects/types/sphere.hpp>
#include <scopi/objects/types/superellipsoid.hpp>
#include <scopi/container.hpp>

namespace scopi
{
    class Container2dTest  : public ::testing::Test {
        static constexpr std::size_t dim = 2;
        protected:
            Container2dTest()
            {
                superellipsoid<dim> s1({{-0.2, 0.1}}, {quaternion(PI/3)}, {{.2, .05}}, 1);
                sphere<dim> s2({{ 0.2,  0.05}}, {quaternion(PI/2)}, 0.1);
                auto p = property<dim>().omega(PI/3)
                                        .desired_omega(PI/12);
                m_particles.push_back(s1, p.velocity({{0.1, 0.2}})
                                           .desired_velocity({{0.01, 0.02}})
                                           .force({{1., 2.}}));
                m_particles.push_back(s2, p.velocity({{0.4, 0.5}})
                                           .desired_velocity({{0.04, 0.05}})
                                           .force({{4., 5.}}));
            }
            scopi_container<dim> m_particles;
    };

    class Container3dTest  : public ::testing::Test {
        static constexpr std::size_t dim = 3;
        protected:
            Container3dTest()
            {
                superellipsoid<dim> s1({{-0.2, 0.1, 0.}}, {quaternion(PI/3)}, {{.2, .05, 0.01}}, {{1, 1}});
                sphere<dim> s2({{ 0.2,  0.05, 0.}}, {quaternion(PI/2)}, 0.1);
                auto p = property<dim>().omega({{PI/3, PI/3}})
                                        .desired_omega({{PI/12, PI/12}});
                m_particles.push_back(s1, p.velocity({{0.1, 0.2, 0.3}})
                                           .desired_velocity({{0.01, 0.02, 0.03}})
                                           .force({{1., 2., 3,}}));

                m_particles.push_back(s2, p.velocity({{0.4, 0.5, 0.6}})
                                           .desired_velocity({{0.04, 0.05, 0.06}})
                                           .force({{4., 5., 6,}}));
            }
            scopi_container<dim> m_particles;
    };

    //size
    TEST_F(Container2dTest, size_2d)
    {
        EXPECT_EQ(m_particles.size(), 2);
    }

    TEST_F(Container3dTest, size_3d)
    {
        EXPECT_EQ(m_particles.size(), 2);
    }

    // pos
    TEST_F(Container2dTest, pos_2d)
    {
        auto pos = m_particles.pos();
        EXPECT_DOUBLE_EQ(pos(0)(0), -0.2);
        EXPECT_DOUBLE_EQ(pos(0)(1), 0.1);
        EXPECT_DOUBLE_EQ(pos(1)(0), 0.2);
        EXPECT_DOUBLE_EQ(pos(1)(1), 0.05);
    }

    TEST_F(Container3dTest, pos_3d)
    {
        auto pos = m_particles.pos();
        EXPECT_DOUBLE_EQ(pos(0)(0), -0.2);
        EXPECT_DOUBLE_EQ(pos(0)(1), 0.1);
        EXPECT_DOUBLE_EQ(pos(0)(2), 0.);
        EXPECT_DOUBLE_EQ(pos(1)(0), 0.2);
        EXPECT_DOUBLE_EQ(pos(1)(1), 0.05);
        EXPECT_DOUBLE_EQ(pos(1)(2), 0.);
    }

    // q
    TEST_F(Container2dTest, q_2d)
    {
        auto q = m_particles.q();
        EXPECT_DOUBLE_EQ(q(0)(0), std::sqrt(3.)/2.);
        EXPECT_DOUBLE_EQ(q(0)(1), 0.);
        EXPECT_DOUBLE_EQ(q(0)(2), 0.);
        EXPECT_DOUBLE_EQ(q(0)(3), 1./2.);
        EXPECT_DOUBLE_EQ(q(1)(0), std::sqrt(2.)/2.);
        EXPECT_DOUBLE_EQ(q(1)(1), 0.);
        EXPECT_DOUBLE_EQ(q(1)(2), 0.);
        EXPECT_DOUBLE_EQ(q(1)(3), std::sqrt(2.)/2.);
    }

    TEST_F(Container3dTest, q_3d)
    {
        auto q = m_particles.q();
        EXPECT_DOUBLE_EQ(q(0)(0), std::sqrt(3.)/2.);
        EXPECT_DOUBLE_EQ(q(0)(1), 0.);
        EXPECT_DOUBLE_EQ(q(0)(2), 0.);
        EXPECT_DOUBLE_EQ(q(0)(3), 1./2.);
        EXPECT_DOUBLE_EQ(q(1)(0), std::sqrt(2.)/2.);
        EXPECT_DOUBLE_EQ(q(1)(1), 0.);
        EXPECT_DOUBLE_EQ(q(1)(2), 0.);
        EXPECT_DOUBLE_EQ(q(1)(3), std::sqrt(2.)/2.);
    }

    // f
    TEST_F(Container2dTest, f_2d)
    {
        auto f = m_particles.f();
        EXPECT_DOUBLE_EQ(f(0)(0), 1.);
        EXPECT_DOUBLE_EQ(f(0)(1), 2.);
        EXPECT_DOUBLE_EQ(f(1)(0), 4.);
        EXPECT_DOUBLE_EQ(f(1)(1), 5.);
    }

    TEST_F(Container3dTest, f_3d)
    {
        auto f = m_particles.f();
        EXPECT_DOUBLE_EQ(f(0)(0), 1.);
        EXPECT_DOUBLE_EQ(f(0)(1), 2.);
        EXPECT_DOUBLE_EQ(f(0)(2), 3.);
        EXPECT_DOUBLE_EQ(f(1)(0), 4.);
        EXPECT_DOUBLE_EQ(f(1)(1), 5.);
        EXPECT_DOUBLE_EQ(f(1)(2), 6.);
    }

    // v
    TEST_F(Container2dTest, v_2d)
    {
        auto v = m_particles.v();
        EXPECT_DOUBLE_EQ(v(0)(0), 0.1);
        EXPECT_DOUBLE_EQ(v(0)(1), 0.2);
        EXPECT_DOUBLE_EQ(v(1)(0), 0.4);
        EXPECT_DOUBLE_EQ(v(1)(1), 0.5);
    }

    TEST_F(Container3dTest, v_3d)
    {
        auto v = m_particles.v();
        EXPECT_DOUBLE_EQ(v(0)(0), 0.1);
        EXPECT_DOUBLE_EQ(v(0)(1), 0.2);
        EXPECT_DOUBLE_EQ(v(0)(2), 0.3);
        EXPECT_DOUBLE_EQ(v(1)(0), 0.4);
        EXPECT_DOUBLE_EQ(v(1)(1), 0.5);
        EXPECT_DOUBLE_EQ(v(1)(2), 0.6);
    }

    // omega
    TEST_F(Container2dTest, omega_2d)
    {
        auto omega = m_particles.omega();
        EXPECT_DOUBLE_EQ(omega(0), PI/3.);
    }

    TEST_F(Container3dTest, omega_3d)
    {
        auto omega = m_particles.omega();
        EXPECT_DOUBLE_EQ(omega(0)[0], PI/3.);
        EXPECT_DOUBLE_EQ(omega(0)[1], PI/3.);
    }

    // desired_omega
    TEST_F(Container2dTest, desired_omega_2d)
    {
        auto desired_omega = m_particles.desired_omega();
        EXPECT_DOUBLE_EQ(desired_omega(0), PI/12.);
    }

    TEST_F(Container3dTest, desired_omega_3d)
    {
        auto desired_omega = m_particles.desired_omega();
        EXPECT_DOUBLE_EQ(desired_omega(0)[0], PI/12.);
        EXPECT_DOUBLE_EQ(desired_omega(0)[1], PI/12.);
    }

    // vd
    TEST_F(Container2dTest, vd_2d)
    {
        auto vd = m_particles.vd();
        EXPECT_DOUBLE_EQ(vd(0)(0), 0.01);
        EXPECT_DOUBLE_EQ(vd(0)(1), 0.02);
        EXPECT_DOUBLE_EQ(vd(1)(0), 0.04);
        EXPECT_DOUBLE_EQ(vd(1)(1), 0.05);
    }

    TEST_F(Container3dTest, vd_3d)
    {
        auto vd = m_particles.vd();
        EXPECT_DOUBLE_EQ(vd(0)(0), 0.01);
        EXPECT_DOUBLE_EQ(vd(0)(1), 0.02);
        EXPECT_DOUBLE_EQ(vd(0)(2), 0.03);
        EXPECT_DOUBLE_EQ(vd(1)(0), 0.04);
        EXPECT_DOUBLE_EQ(vd(1)(1), 0.05);
        EXPECT_DOUBLE_EQ(vd(1)(2), 0.06);
    }

}
