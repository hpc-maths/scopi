#include <gtest/gtest.h>
#include <random>

#include "test_common.hpp"
#include "utils.hpp"

#include <scopi/objects/types/sphere.hpp>
#include <scopi/vap/vap_fpd.hpp>
#include <scopi/container.hpp>
#include <scopi/solver.hpp>
#include <scopi/property.hpp>

namespace scopi
{
    class Sphere2dTest  : public ::testing::Test {
        static constexpr std::size_t dim = 2;
        protected:
            Sphere2dTest()
                : m_s({{-0.2, 0.3}}, 0.1)
                , m_p(property<dim>().desired_velocity({{0.25, 0}}))
                {}
            sphere<dim> m_s;
            property<dim> m_p;
    };

    class Sphere2dConstTest  : public ::testing::Test {
        static constexpr std::size_t dim = 2;
        protected:
            Sphere2dConstTest()
                : m_s({{-0.2, 0.3}}, 0.1)
                , m_p(property<dim>().desired_velocity({{0.25, 0}}))
                {}
            const sphere<dim> m_s;
            const property<dim> m_p;
    };

    class Sphere2dRotationTest  : public ::testing::Test {
        static constexpr std::size_t dim = 2;
        protected:
            Sphere2dRotationTest()
                : m_s({{-0.2, 0.3}}, {quaternion(PI/3)}, 0.1)
                , m_p(property<dim>().desired_velocity({{0.25, 0}}))
                {}
            sphere<dim> m_s;
            property<dim> m_p;
    };

    class Sphere3dTest  : public ::testing::Test {
        static constexpr std::size_t dim = 3;
        protected:
            Sphere3dTest()
                : m_s({{-0.2, 0.3, 0.1}}, 0.1)
                , m_p(property<dim>().desired_velocity({{0.25, 0, 0}}))
                {}
            sphere<dim> m_s;
            property<dim> m_p;
    };

    class Sphere3dConstTest  : public ::testing::Test {
        static constexpr std::size_t dim = 3;
        protected:
            Sphere3dConstTest()
                : m_s({{-0.2, 0.3, 0.1}}, 0.1)
                , m_p(property<dim>().desired_velocity({{0.25, 0, 0}}))
                {}
            const sphere<dim> m_s;
            const property<dim> m_p;
    };

    class Sphere3dRotationTest  : public ::testing::Test {
        static constexpr std::size_t dim = 3;
        protected:
            Sphere3dRotationTest()
                : m_s({{-0.2, 0.3, 0.1}}, {quaternion(PI/3)}, 0.1)
                , m_p(property<dim>().desired_velocity({{0.25, 0, 0}}))
                {}
            sphere<dim> m_s;
            property<dim> m_p;
    };

    // pos
    TEST_F(Sphere2dTest, pos_2d)
    {
        EXPECT_EQ(m_s.pos()(0), -0.2);
        EXPECT_EQ(m_s.pos()(1), 0.3);
    }

    TEST_F(Sphere3dTest, pos_3d)
    {
        EXPECT_EQ(m_s.pos()(0), -0.2);
        EXPECT_EQ(m_s.pos()(1), 0.3);
        EXPECT_EQ(m_s.pos()(2), 0.1);
    }

    TEST_F(Sphere2dConstTest, pos_2d_const)
    {
        EXPECT_EQ(m_s.pos()(0), -0.2);
        EXPECT_EQ(m_s.pos()(1), 0.3);
    }

    TEST_F(Sphere3dConstTest, pos_3d_const)
    {
        EXPECT_EQ(m_s.pos()(0), -0.2);
        EXPECT_EQ(m_s.pos()(1), 0.3);
        EXPECT_EQ(m_s.pos()(2), 0.1);
    }

    TEST_F(Sphere2dTest, pos_2d_index)
    {
        EXPECT_EQ(m_s.pos(0)(), -0.2);
        EXPECT_EQ(m_s.pos(1)(), 0.3);
    }

    TEST_F(Sphere3dTest, pos_3d_index)
    {
        EXPECT_EQ(m_s.pos(0)(), -0.2);
        EXPECT_EQ(m_s.pos(1)(), 0.3);
        EXPECT_EQ(m_s.pos(2)(), 0.1);
    }

    TEST_F(Sphere2dConstTest, pos_2d_index_const)
    {
        EXPECT_EQ(m_s.pos(0)(), -0.2);
        EXPECT_EQ(m_s.pos(1)(), 0.3);
    }

    TEST_F(Sphere3dConstTest, pos_3d_index_const)
    {
        EXPECT_EQ(m_s.pos(0)(), -0.2);
        EXPECT_EQ(m_s.pos(1)(), 0.3);
        EXPECT_EQ(m_s.pos(2)(), 0.1);
    }

    TEST_F(Sphere2dTest, pos_2d_container)
    {
        constexpr std::size_t dim = 2;
        scopi_container<dim> particles;
        particles.push_back(m_s, m_p);

        EXPECT_EQ(particles[0]->pos()(0), -0.2);
        EXPECT_EQ(particles[0]->pos()(1), 0.3);
    }

    TEST_F(Sphere3dTest, pos_3d_container)
    {
        constexpr std::size_t dim = 3;
        scopi_container<dim> particles;
        particles.push_back(m_s, m_p);

        EXPECT_EQ(particles[0]->pos()(0), -0.2);
        EXPECT_EQ(particles[0]->pos()(1), 0.3);
        EXPECT_EQ(particles[0]->pos()(2), 0.1);
    }

    TEST_F(Sphere2dConstTest, pos_2d_const_container)
    {
        constexpr std::size_t dim = 2;
        scopi_container<dim> particles;
        particles.push_back(m_s, m_p);

        EXPECT_EQ(particles[0]->pos()(0), -0.2);
        EXPECT_EQ(particles[0]->pos()(1), 0.3);
    }

    TEST_F(Sphere3dConstTest, pos_3d_const_container)
    {
        constexpr std::size_t dim = 3;
        scopi_container<dim> particles;
        particles.push_back(m_s, m_p);

        EXPECT_EQ(particles[0]->pos()(0), -0.2);
        EXPECT_EQ(particles[0]->pos()(1), 0.3);
        EXPECT_EQ(particles[0]->pos()(2), 0.1);
    }

    TEST_F(Sphere2dTest, pos_2d_index_container)
    {
        constexpr std::size_t dim = 2;
        scopi_container<dim> particles;
        particles.push_back(m_s, m_p);

        EXPECT_EQ(particles[0]->pos(0)(), -0.2);
        EXPECT_EQ(particles[0]->pos(1)(), 0.3);
    }

    TEST_F(Sphere3dTest, pos_3d_index_container)
    {
        constexpr std::size_t dim = 3;
        scopi_container<dim> particles;
        particles.push_back(m_s, m_p);

        EXPECT_EQ(particles[0]->pos(0)(), -0.2);
        EXPECT_EQ(particles[0]->pos(1)(), 0.3);
        EXPECT_EQ(particles[0]->pos(2)(), 0.1);
    }

    TEST_F(Sphere2dConstTest, pos_2d_index_const_container)
    {
        constexpr std::size_t dim = 2;
        scopi_container<dim> particles;
        particles.push_back(m_s, m_p);

        EXPECT_EQ(particles[0]->pos(0)(), -0.2);
        EXPECT_EQ(particles[0]->pos(1)(), 0.3);
    }

    TEST_F(Sphere3dConstTest, pos_3d_index_const_container)
    {
        constexpr std::size_t dim = 3;
        scopi_container<dim> particles;
        particles.push_back(m_s, m_p);

        EXPECT_EQ(particles[0]->pos(0)(), -0.2);
        EXPECT_EQ(particles[0]->pos(1)(), 0.3);
        EXPECT_EQ(particles[0]->pos(2)(), 0.1);
    }

    // q
    TEST_F(Sphere2dTest, q)
    {
        EXPECT_EQ(m_s.q()(0), 1.);
        EXPECT_EQ(m_s.q()(1), 0.);
        EXPECT_EQ(m_s.q()(2), 0.);
        EXPECT_EQ(m_s.q()(3), 0.);
    }

    TEST_F(Sphere2dConstTest, q_const)
    {
        EXPECT_EQ(m_s.q()(0), 1.);
        EXPECT_EQ(m_s.q()(1), 0.);
        EXPECT_EQ(m_s.q()(2), 0.);
        EXPECT_EQ(m_s.q()(3), 0.);
    }

    TEST_F(Sphere2dTest, q_index)
    {
        EXPECT_EQ(m_s.q(0)(), 1.);
        EXPECT_EQ(m_s.q(1)(), 0.);
        EXPECT_EQ(m_s.q(2)(), 0.);
        EXPECT_EQ(m_s.q(3)(), 0.);
    }

    TEST_F(Sphere2dConstTest, q_index_const)
    {
        EXPECT_EQ(m_s.q(0)(), 1.);
        EXPECT_EQ(m_s.q(1)(), 0.);
        EXPECT_EQ(m_s.q(2)(), 0.);
        EXPECT_EQ(m_s.q(3)(), 0.);
    }

    TEST_F(Sphere2dTest, q_container)
    {
        constexpr std::size_t dim = 2;
        scopi_container<dim> particles;
        particles.push_back(m_s, m_p);

        EXPECT_EQ(particles[0]->q()(0), 1.);
        EXPECT_EQ(particles[0]->q()(1), 0.);
        EXPECT_EQ(particles[0]->q()(2), 0.);
        EXPECT_EQ(particles[0]->q()(3), 0.);
    }

    TEST_F(Sphere2dConstTest, q_const_container)
    {
        constexpr std::size_t dim = 2;
        scopi_container<dim> particles;
        particles.push_back(m_s, m_p);

        EXPECT_EQ(particles[0]->q()(0), 1.);
        EXPECT_EQ(particles[0]->q()(1), 0.);
        EXPECT_EQ(particles[0]->q()(2), 0.);
        EXPECT_EQ(particles[0]->q()(3), 0.);
    }

    TEST_F(Sphere2dTest, q_index_container)
    {
        constexpr std::size_t dim = 2;
        scopi_container<dim> particles;
        particles.push_back(m_s, m_p);

        EXPECT_EQ(particles[0]->q(0)(), 1.);
        EXPECT_EQ(particles[0]->q(1)(), 0.);
        EXPECT_EQ(particles[0]->q(2)(), 0.);
        EXPECT_EQ(particles[0]->q(3)(), 0.);
    }

    TEST_F(Sphere2dConstTest, q_index_const_container)
    {
        constexpr std::size_t dim = 2;
        scopi_container<dim> particles;
        particles.push_back(m_s, m_p);

        EXPECT_EQ(particles[0]->q(0)(), 1.);
        EXPECT_EQ(particles[0]->q(1)(), 0.);
        EXPECT_EQ(particles[0]->q(2)(), 0.);
        EXPECT_EQ(particles[0]->q(3)(), 0.);
    }

    // radius
    TEST_F(Sphere2dTest, radius)
    {
        EXPECT_EQ(m_s.radius(), 0.1);
    }

    // rotation
    TEST_F(Sphere2dRotationTest, rotation_2d)
    {
        auto rotation_matrix = m_s.rotation();
        EXPECT_DOUBLE_EQ(rotation_matrix(0, 0), 1./2.);
        EXPECT_DOUBLE_EQ(rotation_matrix(0, 1), -std::sqrt(3.)/2.);
        EXPECT_DOUBLE_EQ(rotation_matrix(1, 0), std::sqrt(3.)/2.);
        EXPECT_DOUBLE_EQ(rotation_matrix(1, 1), 1./2.);
    }


    TEST_F(Sphere3dRotationTest, rotation_3d)
    {
        auto rotation_matrix = m_s.rotation();
        EXPECT_DOUBLE_EQ(rotation_matrix(0, 0), 1./2.);
        EXPECT_DOUBLE_EQ(rotation_matrix(0, 1), -std::sqrt(3.)/2.);
        EXPECT_DOUBLE_EQ(rotation_matrix(0, 2), 0.);
        EXPECT_DOUBLE_EQ(rotation_matrix(1, 0), std::sqrt(3.)/2.);
        EXPECT_DOUBLE_EQ(rotation_matrix(1, 1), 1./2.);
        EXPECT_DOUBLE_EQ(rotation_matrix(1, 2), 0.);
        EXPECT_DOUBLE_EQ(rotation_matrix(2, 0), 0.);
        EXPECT_DOUBLE_EQ(rotation_matrix(2, 1), 0.);
        EXPECT_DOUBLE_EQ(rotation_matrix(2, 2), 1.);
    }

    // point
    TEST_F(Sphere2dTest, point_x_2d)
    {
        auto point = m_s.point(0.);
        EXPECT_DOUBLE_EQ(point(0), -0.1);
        EXPECT_DOUBLE_EQ(point(1), 0.3);
    }

    TEST_F(Sphere2dTest, point_y_2d)
    {
        auto point = m_s.point(PI/2.);
        EXPECT_DOUBLE_EQ(point(0), -0.2);
        EXPECT_DOUBLE_EQ(point(1), 0.4);
    }

    TEST_F(Sphere3dTest, point_x_3d)
    {
        auto point = m_s.point(0., 0.);
        EXPECT_DOUBLE_EQ(point(0), -0.1);
        EXPECT_DOUBLE_EQ(point(1), 0.3);
        EXPECT_DOUBLE_EQ(point(2), 0.1);
    }

    TEST_F(Sphere3dTest, point_y_3d)
    {
        auto point = m_s.point(0., PI/2.);
        EXPECT_DOUBLE_EQ(point(0), -0.2);
        EXPECT_DOUBLE_EQ(point(1), 0.4);
        EXPECT_DOUBLE_EQ(point(2), 0.1);
    }

    TEST_F(Sphere3dTest, point_z_3d)
    {
        auto point = m_s.point(PI/2., 0.);
        EXPECT_DOUBLE_EQ(point(0), -0.2);
        EXPECT_DOUBLE_EQ(point(1), 0.3);
        EXPECT_DOUBLE_EQ(point(2), 0.2);
    }

    // normal
    TEST_F(Sphere2dTest, normal_2d)
    {
        auto normal = m_s.normal(0.);
        EXPECT_EQ(normal(0), 1.);
        EXPECT_EQ(normal(1), 0.);
    }

    TEST_F(Sphere3dTest, normal_3d)
    {
        auto normal = m_s.normal(0., 0.);
        EXPECT_EQ(normal(0), 1.);
        EXPECT_EQ(normal(1), 0.);
        EXPECT_EQ(normal(2), 0.);
    }

    // two_spheres
    template <class S>
    class TestTwoSpheresAsymmetrical  : public ::testing::Test {
        static constexpr std::size_t dim = 2;
        protected:
            void SetUp() override {
                sphere<dim> s1({{-0.2, -0.05}}, 0.1);
                sphere<dim> s2({{ 0.2,  0.05}}, 0.1);
                auto p = property<dim>().mass(1.);
                m_particles.push_back(s1, p.desired_velocity({{0.25, 0}}));
                m_particles.push_back(s2, p.desired_velocity({{-0.25, 0}}));
            }

            double m_dt = .005;
            std::size_t m_total_it = 1000;
            scopi_container<dim> m_particles;
    };

    TYPED_TEST_SUITE(TestTwoSpheresAsymmetrical, solver_with_contact_types<2>);

    TYPED_TEST(TestTwoSpheresAsymmetrical, two_spheres_asymmetrical)
    {
        TypeParam solver(this->m_particles, this->m_dt);
        solver.solve(this->m_total_it);

        EXPECT_PRED3(diffFile, "./Results/scopi_objects_0999.json", "../test/references/two_spheres_asymmetrical.json", tolerance);
    }

    template <class S>
    class TestTwoSpheresSymmetrical  : public ::testing::Test {
        static constexpr std::size_t dim = 2;
        protected:
            void SetUp() override {
                sphere<dim> s1({{-0.2, 0.}}, 0.1);
                sphere<dim> s2({{ 0.2, 0.}}, 0.1);
                auto p = property<dim>().desired_velocity({{0.25, 0}}).mass(1.);
                m_particles.push_back(s1, p);
                m_particles.push_back(s2, p.desired_velocity({{-0.25, 0}}));
            }

            double m_dt = .005;
            std::size_t m_total_it = 1000;
            scopi_container<dim> m_particles;
    };

    TYPED_TEST_SUITE(TestTwoSpheresSymmetrical, solver_with_contact_types<2>);

    TYPED_TEST(TestTwoSpheresSymmetrical, two_spheres_symmetrical)
    {
        TypeParam solver(this->m_particles, this->m_dt);
        solver.solve(this->m_total_it);

        EXPECT_PRED3(diffFile, "./Results/scopi_objects_0999.json", "../test/references/two_spheres_symmetrical.json", tolerance);
    }

    // critical_2d
    template <class S>
    class Test2dCaseSpheres  : public ::testing::Test {
        static constexpr std::size_t dim = 2;
        protected:
            void SetUp() override {
                int n = 3; // 2*n*n particles
                std::minstd_rand0 generator(123);
                std::uniform_real_distribution<double> distrib_r(0.2, 0.4);
                std::uniform_real_distribution<double> distrib_move_x(-0.1, 0.1);
                std::uniform_real_distribution<double> distrib_move_y(-0.1, 0.1);
                std::uniform_real_distribution<double> distrib_velocity(2., 5.);

                for(int i = 0; i < n; ++i)
                {
                    for(int j = 0; j < n; ++j)
                    {
                        double r = distrib_r(generator);
                        double x = (i + 0.5) + distrib_move_x(generator);
                        double y = (j + 0.5) + distrib_move_y(generator);
                        double velocity = distrib_velocity(generator);
                        sphere<dim> s1({{x, y}}, r);
                        m_particles.push_back(s1, scopi::property<dim>().desired_velocity({{velocity, 0.}}).mass(1.));

                        r = distrib_r(generator);
                        x = (n + i + 0.5) + distrib_move_x(generator);
                        y = (j + 0.5) + distrib_move_y(generator);
                        velocity = distrib_velocity(generator);
                        sphere<dim> s2({{x, y}}, r);
                        m_particles.push_back(s2, scopi::property<dim>().desired_velocity({{-velocity, 0.}}).mass(1.));
                    }
                }
            }

            double m_dt = .01;
            std::size_t m_total_it = 20;
            scopi_container<dim> m_particles;
    };

    TYPED_TEST_SUITE(Test2dCaseSpheres, solver_with_contact_types<2>);

    TYPED_TEST(Test2dCaseSpheres, 2d_case_spheres)
    {
        TypeParam solver(this->m_particles, this->m_dt);
        solver.solve(this->m_total_it);

        EXPECT_PRED3(diffFile, "./Results/scopi_objects_0019.json", "../test/references/2d_case_spheres.json", tolerance);
    }

    template <class S>
    class Test2dCaseSpheresVap  : public ::testing::Test {
        static constexpr std::size_t dim = 2;
        protected:
            void SetUp() override {
                int n = 2; // 2*n*n particles
                std::default_random_engine generator(0);
                std::uniform_real_distribution<double> distrib_r(0.2, 0.4);
                std::uniform_real_distribution<double> distrib_move_x(-0.1, 0.1);
                std::uniform_real_distribution<double> distrib_move_y(-0.1, 0.1);

                sphere<dim> s({ {0., 0.}}, 0.1);
                m_particles.push_back(s, scopi::property<dim>().deactivate());

                for(int i = 0; i < n; ++i)
                {
                    for(int j = 0; j < n; ++j)
                    {
                        double r = distrib_r(generator);
                        double x = -n + (i + 0.5) + distrib_move_x(generator);
                        double y = -n/2. + (j + 0.5) + distrib_move_y(generator);
                        sphere<dim> s1({ {x, y}}, r);
                        m_particles.push_back(s1);

                        r = distrib_r(generator);
                        x = (i + 0.5) + distrib_move_x(generator);
                        y = -n/2. + (j + 0.5) + distrib_move_y(generator);
                        sphere<dim> s2({ {x, y}}, r);
                        m_particles.push_back(s2);
                    }
                }
            }

            double m_dt = .01;
            std::size_t m_total_it = 30;
            scopi_container<dim> m_particles;
    };

    using solver_types_vap = solver_with_contact_types<2, vap_fpd>; // TODO does not compile without using
    TYPED_TEST_SUITE(Test2dCaseSpheresVap, solver_types_vap);

    TYPED_TEST(Test2dCaseSpheresVap, 2d_case_spheres_vap)
    {
        TypeParam solver(this->m_particles, this->m_dt);
        solver.solve(this->m_total_it);

        EXPECT_PRED3(diffFile, "./Results/scopi_objects_0029.json", "../test/references/2d_case_spheres_vap.json", tolerance);
    }
}
