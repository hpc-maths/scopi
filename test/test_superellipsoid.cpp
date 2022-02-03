#include <gtest/gtest.h>
#include "utils.hpp"

#include <scopi/objects/types/superellipsoid.hpp>
#include <scopi/container.hpp>
#include <scopi/solvers/mosek.hpp>

namespace scopi
{
    class Superellipsoid2dTest  : public ::testing::Test {
        protected:
            Superellipsoid2dTest()
                : m_s({{-0.2, 0.3}}, {quaternion(0)}, {{0.1, 0.2}}, 1.5)
                {}
            superellipsoid<2> m_s;
    };

    class Superellipsoid2dConstTest  : public ::testing::Test {
        protected:
            Superellipsoid2dConstTest()
                : m_s({{-0.2, 0.3}}, {quaternion(0)}, {{0.1, 0.2}}, 1)
                {}
            const superellipsoid<2> m_s;
    };

    class Superellipsoid2dRotationTest  : public ::testing::Test {
        protected:
            Superellipsoid2dRotationTest()
                : m_s({{-0.2, 0.3}}, {quaternion(PI/3)}, {{0.1, 0.2}}, 1.5)
                {}
            superellipsoid<2> m_s;
    };

    class Superellipsoid3dTest  : public ::testing::Test {
        protected:
            Superellipsoid3dTest()
                : m_s({{-0.2, 0.3, 0.1}}, {quaternion(0)}, {{0.1, 0.2, 0.3}}, {{0.5, 1.5}})
                {}
            superellipsoid<3> m_s;
    };

    class Superellipsoid3dConstTest  : public ::testing::Test {
        protected:
            Superellipsoid3dConstTest()
                : m_s({{-0.2, 0.3, 0.1}}, {quaternion(0)}, {{0.1, 0.2, 0.3}}, {{1, 1}})
                {}
            const superellipsoid<3> m_s;
    };

    class Superellipsoid3dRotationTest  : public ::testing::Test {
        protected:
            Superellipsoid3dRotationTest()
                : m_s({{-0.2, 0.3, 0.1}}, {quaternion(PI/3)}, {{0.1, 0.2, 0.3}}, {{0.5, 1.5}})
                {}
            superellipsoid<3> m_s;
    };


    // pos
    TEST_F(Superellipsoid2dTest, pos_2d)
    {
        EXPECT_EQ(m_s.pos()(0), -0.2);
        EXPECT_EQ(m_s.pos()(1), 0.3);
    }

    TEST_F(Superellipsoid3dTest, pos_3d)
    {
        EXPECT_EQ(m_s.pos()(0), -0.2);
        EXPECT_EQ(m_s.pos()(1), 0.3);
        EXPECT_EQ(m_s.pos()(2), 0.1);
    }

    TEST_F(Superellipsoid2dConstTest, pos_2d_const)
    {
        EXPECT_EQ(m_s.pos()(0), -0.2);
        EXPECT_EQ(m_s.pos()(1), 0.3);
    }

    TEST_F(Superellipsoid3dConstTest, pos_3d_const)
    {
        EXPECT_EQ(m_s.pos()(0), -0.2);
        EXPECT_EQ(m_s.pos()(1), 0.3);
        EXPECT_EQ(m_s.pos()(2), 0.1);
    }

    TEST_F(Superellipsoid2dTest, pos_2d_index)
    {
        EXPECT_EQ(m_s.pos(0)(), -0.2);
        EXPECT_EQ(m_s.pos(1)(), 0.3);
    }

    TEST_F(Superellipsoid3dTest, pos_3d_index)
    {
        EXPECT_EQ(m_s.pos(0)(), -0.2);
        EXPECT_EQ(m_s.pos(1)(), 0.3);
        EXPECT_EQ(m_s.pos(2)(), 0.1);
    }

    TEST_F(Superellipsoid2dConstTest, pos_2d_index_const)
    {

        EXPECT_EQ(m_s.pos(0)(), -0.2);
        EXPECT_EQ(m_s.pos(1)(), 0.3);
    }

    TEST_F(Superellipsoid3dConstTest, pos_3d_index_const)
    {
        EXPECT_EQ(m_s.pos(0)(), -0.2);
        EXPECT_EQ(m_s.pos(1)(), 0.3);
        EXPECT_EQ(m_s.pos(2)(), 0.1);
    }

    TEST_F(Superellipsoid2dTest, pos_2d_container)
    {
        constexpr std::size_t dim = 2;
        scopi_container<dim> particles;
        particles.push_back(m_s, {{0, 0}}, {{0.25, 0}}, 0, 0, {{0, 0}});

        EXPECT_EQ(particles[0]->pos()(0), -0.2);
        EXPECT_EQ(particles[0]->pos()(1), 0.3);
    }

    TEST_F(Superellipsoid3dTest, pos_3d_container)
    {
        constexpr std::size_t dim = 3;
        scopi_container<dim> particles;
        particles.push_back(m_s, {{0, 0, 0}}, {{0.25, 0, 0}}, 0, 0, {{0, 0, 0}});

        EXPECT_EQ(particles[0]->pos()(0), -0.2);
        EXPECT_EQ(particles[0]->pos()(1), 0.3);
        EXPECT_EQ(particles[0]->pos()(2), 0.1);
    }

    TEST_F(Superellipsoid2dConstTest, pos_2d_const_container)
    {
        constexpr std::size_t dim = 2;
        scopi_container<dim> particles;
        particles.push_back(m_s, {{0, 0}}, {{0.25, 0}}, 0, 0, {{0, 0}});

        EXPECT_EQ(particles[0]->pos()(0), -0.2);
        EXPECT_EQ(particles[0]->pos()(1), 0.3);
    }

    TEST_F(Superellipsoid3dConstTest, pos_3d_const_container)
    {
        constexpr std::size_t dim = 3;
        scopi_container<dim> particles;
        particles.push_back(m_s, {{0, 0, 0}}, {{0.25, 0, 0}}, 0, 0, {{0, 0, 0}});

        EXPECT_EQ(particles[0]->pos()(0), -0.2);
        EXPECT_EQ(particles[0]->pos()(1), 0.3);
        EXPECT_EQ(particles[0]->pos()(2), 0.1);
    }

    TEST_F(Superellipsoid2dTest, pos_2d_index_container)
    {
        constexpr std::size_t dim = 2;
        scopi_container<dim> particles;
        particles.push_back(m_s, {{0, 0}}, {{0.25, 0}}, 0, 0, {{0, 0}});

        EXPECT_EQ(particles[0]->pos(0)(), -0.2);
        EXPECT_EQ(particles[0]->pos(1)(), 0.3);
    }

    TEST_F(Superellipsoid3dTest, pos_3d_index_container)
    {
        constexpr std::size_t dim = 3;
        scopi_container<dim> particles;
        particles.push_back(m_s, {{0, 0, 0}}, {{0.25, 0, 0}}, 0, 0, {{0, 0, 0}});

        EXPECT_EQ(particles[0]->pos(0)(), -0.2);
        EXPECT_EQ(particles[0]->pos(1)(), 0.3);
        EXPECT_EQ(particles[0]->pos(2)(), 0.1);
    }

    TEST_F(Superellipsoid2dConstTest, pos_2d_index_const_container)
    {
        constexpr std::size_t dim = 2;
        scopi_container<dim> particles;
        particles.push_back(m_s, {{0, 0}}, {{0.25, 0}}, 0, 0, {{0, 0}});

        EXPECT_EQ(particles[0]->pos(0)(), -0.2);
        EXPECT_EQ(particles[0]->pos(1)(), 0.3);
    }

    TEST_F(Superellipsoid3dConstTest, pos_3d_index_const_container)
    {
        constexpr std::size_t dim = 3;
        scopi_container<dim> particles;
        particles.push_back(m_s, {{0, 0, 0}}, {{0.25, 0, 0}}, 0, 0, {{0, 0, 0}});

        EXPECT_EQ(particles[0]->pos(0)(), -0.2);
        EXPECT_EQ(particles[0]->pos(1)(), 0.3);
        EXPECT_EQ(particles[0]->pos(2)(), 0.1);
    }

    // q
    TEST_F(Superellipsoid2dTest, q)
    {
        EXPECT_EQ(m_s.q()(0), 1.);
        EXPECT_EQ(m_s.q()(1), 0.);
        EXPECT_EQ(m_s.q()(2), 0.);
        EXPECT_EQ(m_s.q()(3), 0.);
    }

    TEST_F(Superellipsoid2dConstTest, q_const)
    {
        EXPECT_EQ(m_s.q()(0), 1.);
        EXPECT_EQ(m_s.q()(1), 0.);
        EXPECT_EQ(m_s.q()(2), 0.);
        EXPECT_EQ(m_s.q()(3), 0.);
    }

    TEST_F(Superellipsoid2dTest, q_index)
    {
        EXPECT_EQ(m_s.q(0)(), 1.);
        EXPECT_EQ(m_s.q(1)(), 0.);
        EXPECT_EQ(m_s.q(2)(), 0.);
        EXPECT_EQ(m_s.q(3)(), 0.);
    }

    TEST_F(Superellipsoid2dConstTest, q_index_const)
    {
        EXPECT_EQ(m_s.q(0)(), 1.);
        EXPECT_EQ(m_s.q(1)(), 0.);
        EXPECT_EQ(m_s.q(2)(), 0.);
        EXPECT_EQ(m_s.q(3)(), 0.);
    }

    TEST_F(Superellipsoid2dTest, q_container)
    {
        constexpr std::size_t dim = 2;
        scopi_container<dim> particles;
        particles.push_back(m_s, {{0, 0}}, {{0.25, 0}}, 0, 0, {{0, 0}});

        EXPECT_EQ(particles[0]->q()(0), 1.);
        EXPECT_EQ(particles[0]->q()(1), 0.);
        EXPECT_EQ(particles[0]->q()(2), 0.);
        EXPECT_EQ(particles[0]->q()(3), 0.);
    }

    TEST_F(Superellipsoid2dConstTest, q_const_container)
    {
        constexpr std::size_t dim = 2;
        scopi_container<dim> particles;
        particles.push_back(m_s, {{0, 0}}, {{0.25, 0}}, 0, 0, {{0, 0}});

        EXPECT_EQ(particles[0]->q()(0), 1.);
        EXPECT_EQ(particles[0]->q()(1), 0.);
        EXPECT_EQ(particles[0]->q()(2), 0.);
        EXPECT_EQ(particles[0]->q()(3), 0.);
    }

    TEST_F(Superellipsoid2dTest, q_index_container)
    {
        constexpr std::size_t dim = 2;
        scopi_container<dim> particles;
        particles.push_back(m_s, {{0, 0}}, {{0.25, 0}}, 0, 0, {{0, 0}});

        EXPECT_EQ(particles[0]->q(0)(), 1.);
        EXPECT_EQ(particles[0]->q(1)(), 0.);
        EXPECT_EQ(particles[0]->q(2)(), 0.);
        EXPECT_EQ(particles[0]->q(3)(), 0.);
    }

    TEST_F(Superellipsoid2dConstTest, q_index_const_container)
    {
        constexpr std::size_t dim = 2;
        scopi_container<dim> particles;
        particles.push_back(m_s, {{0, 0}}, {{0.25, 0}}, 0, 0, {{0, 0}});

        EXPECT_EQ(particles[0]->q(0)(), 1.);
        EXPECT_EQ(particles[0]->q(1)(), 0.);
        EXPECT_EQ(particles[0]->q(2)(), 0.);
        EXPECT_EQ(particles[0]->q(3)(), 0.);
    }

    // radius
    TEST_F(Superellipsoid2dTest, radius_2d)
    {
        auto radius = m_s.radius();
        EXPECT_EQ(radius(0), 0.1);
        EXPECT_EQ(radius(1), 0.2);
    }

    TEST_F(Superellipsoid3dTest, radius_3d)
    {
        auto radius = m_s.radius();
        EXPECT_EQ(radius(0), 0.1);
        EXPECT_EQ(radius(1), 0.2);
        EXPECT_EQ(radius(2), 0.3);
    }

    //squareness
    TEST_F(Superellipsoid2dTest, squareness_2d)
    {
        auto squareness = m_s.squareness();
        EXPECT_EQ(squareness(0), 1.5);
    }

    TEST_F(Superellipsoid3dTest, squareness_3d)
    {
        auto squareness = m_s.squareness();
        EXPECT_EQ(squareness(0), 0.5);
        EXPECT_EQ(squareness(1), 1.5);
    }

    // rotation
    TEST_F(Superellipsoid2dRotationTest, rotation_2d)
    {
        auto rotation_matrix = m_s.rotation();
        EXPECT_DOUBLE_EQ(rotation_matrix(0, 0), 1./2.);
        EXPECT_DOUBLE_EQ(rotation_matrix(0, 1), -std::sqrt(3.)/2.);
        EXPECT_DOUBLE_EQ(rotation_matrix(1, 0), std::sqrt(3.)/2.);
        EXPECT_DOUBLE_EQ(rotation_matrix(1, 1), 1./2.);
    }

    TEST_F(Superellipsoid3dRotationTest, rotation_3d)
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
    TEST_F(Superellipsoid2dTest, point_x_2d)
    {
        auto point = m_s.point(0.);
        EXPECT_DOUBLE_EQ(point(0), -0.1);
        EXPECT_DOUBLE_EQ(point(1), 0.3);
    }

    TEST_F(Superellipsoid2dTest, point_y_2d)
    {
        auto point = m_s.point(PI/2.);
        EXPECT_DOUBLE_EQ(point(0), -0.2);
        EXPECT_DOUBLE_EQ(point(1), 0.5);
    }

    TEST_F(Superellipsoid3dTest, point_x_3d)
    {
        auto point = m_s.point(0., 0.);
        EXPECT_DOUBLE_EQ(point(0), -0.1);
        EXPECT_DOUBLE_EQ(point(1), 0.3);
        EXPECT_DOUBLE_EQ(point(2), 0.1);
    }

    TEST_F(Superellipsoid3dTest, point_y_3d)
    {
        auto point = m_s.point(0., PI/2.);
        EXPECT_NEAR(point(0), -0.2, 1e-9); // EXPECT_DOUBLE_EQ fails because cos(PI/2) != 0
        EXPECT_DOUBLE_EQ(point(1), 0.5);
        EXPECT_DOUBLE_EQ(point(2), 0.1);
    }

    TEST_F(Superellipsoid3dTest, point_z_3d)
    {
        auto point = m_s.point(PI/2., 0.);
        EXPECT_DOUBLE_EQ(point(0), -0.2);
        EXPECT_DOUBLE_EQ(point(1), 0.3);
        EXPECT_DOUBLE_EQ(point(2), 0.4);
    }

    // normal
    TEST_F(Superellipsoid2dTest, normal_x_2d)
    {
        auto normal = m_s.normal(0.);
        EXPECT_DOUBLE_EQ(normal(0), 1.);
        EXPECT_DOUBLE_EQ(normal(1), 0.);
    }

    TEST_F(Superellipsoid2dTest, normal_y_2d)
    {
        auto normal = m_s.normal(PI/2.);
        EXPECT_NEAR(normal(0), 0., 1e-7); // EXPECT_DOUBLE_EQ fails because cos(PI/2) != 0
        EXPECT_DOUBLE_EQ(normal(1), 1.);
    }

    TEST_F(Superellipsoid3dTest, normal_x_3d)
    {
        auto normal = m_s.normal(0., 0.);
        EXPECT_DOUBLE_EQ(normal(0), 1.);
        EXPECT_DOUBLE_EQ(normal(1), 0.);
        EXPECT_DOUBLE_EQ(normal(2), 0.);
    }

    TEST_F(Superellipsoid3dTest, normal_y_3d)
    {
        auto normal = m_s.normal(0., PI/2.);
        EXPECT_NEAR(normal(0), 0., 1e-17); // EXPECT_DOUBLE_EQ fails because cos(PI/2) != 0
        EXPECT_DOUBLE_EQ(normal(1), 1.);
        EXPECT_DOUBLE_EQ(normal(2), 0.);
    }

    TEST_F(Superellipsoid3dTest, normal_z_3d)
    {
        auto normal = m_s.normal(PI/2., 0.);
        EXPECT_NEAR(normal(0), 0., 1e-7); // EXPECT_DOUBLE_EQ fails because cos(PI/2) != 0
        EXPECT_DOUBLE_EQ(normal(1), 0.);
        EXPECT_DOUBLE_EQ(normal(2), 1.);
    }
    //

    // tangent
    TEST_F(Superellipsoid2dTest, tangent_x_2d)
    {
        auto tangent = m_s.tangent(0.);
        EXPECT_DOUBLE_EQ(tangent(0), 0.);
        EXPECT_DOUBLE_EQ(tangent(1), 1.);
    }

    TEST_F(Superellipsoid2dTest, tangent_y_2d)
    {
        auto tangent = m_s.tangent(PI/2.);
        EXPECT_DOUBLE_EQ(tangent(0), -1.);
        EXPECT_NEAR(tangent(1), 0., 1e-7); // EXPECT_DOUBLE_EQ fails because cos(PI/2) != 0
    }

    TEST_F(Superellipsoid3dTest, tangent_x_3d)
    {
        auto tangent = m_s.tangents(0., 0.);
        EXPECT_DOUBLE_EQ(tangent.first(0), 0.);
        EXPECT_DOUBLE_EQ(tangent.first(1), 1.);
        EXPECT_DOUBLE_EQ(tangent.first(2), 0.);

        EXPECT_DOUBLE_EQ(tangent.second(0), 0.);
        EXPECT_DOUBLE_EQ(tangent.second(1), 0.);
        EXPECT_DOUBLE_EQ(tangent.second(2), 1.);
    }

    TEST_F(Superellipsoid3dTest, tangent_y_3d)
    {
        auto tangent = m_s.tangents(0., PI/2.);
        EXPECT_DOUBLE_EQ(tangent.first(0), -1.);
        EXPECT_NEAR(tangent.first(1), 0., 1e-17); // EXPECT_DOUBLE_EQ fails because cos(PI/2) != 0
        EXPECT_DOUBLE_EQ(tangent.first(2), 0.);

        EXPECT_DOUBLE_EQ(tangent.second(0), 0.);
        EXPECT_DOUBLE_EQ(tangent.second(1), 0.);
        EXPECT_DOUBLE_EQ(tangent.second(2), 1.);
    }

    TEST_F(Superellipsoid3dTest, tangent_z_3d)
    {
        auto tangent = m_s.tangents(PI/2., 0.);
        EXPECT_DOUBLE_EQ(tangent.first(0), 0.);
        EXPECT_DOUBLE_EQ(tangent.first(1), 1.);
        EXPECT_DOUBLE_EQ(tangent.first(2), 0.);

        EXPECT_DOUBLE_EQ(tangent.second(0), -1.);
        EXPECT_DOUBLE_EQ(tangent.second(1), 0.);
        EXPECT_NEAR(tangent.second(2), 0., 1e-7); // EXPECT_DOUBLE_EQ fails because cos(PI/2) != 0
    }
    //

    // two ellipsoids
    class TestTwoEllipsoidsSymmetrical  : public ::testing::Test {
        protected:
            void SetUp() override {
                superellipsoid<2> s1({{-0.2, 0.}}, {scopi::quaternion(PI/4)}, {{.1, .05}}, 1);
                superellipsoid<2> s2({{0.2, 0.}}, {scopi::quaternion(-PI/4)}, {{.1, .05}}, 1);
                m_particles.push_back(s1, {{0, 0}}, {{0.25, 0}}, 0, 0, {{0, 0}});
                m_particles.push_back(s2, {{0, 0}}, {{-0.25, 0}}, 0, 0, {{0, 0}});
            }

            double m_dt = .005;
            std::size_t m_total_it = 200;
            scopi_container<2> m_particles;
            std::size_t m_active_ptr = 0; // without obstacles
    };

    class TestTwoEllipsoidsSpheresSymmetrical  : public ::testing::Test {
        protected:
            void SetUp() override {
                superellipsoid<2> s1({{-0.2, 0.}}, {scopi::quaternion(PI/4)}, {{.1, .1}}, 1);
                superellipsoid<2> s2({{0.2, 0.}}, {scopi::quaternion(-PI/4)}, {{.1, .1}}, 1);
                m_particles.push_back(s1, {{0, 0}}, {{0.25, 0}}, 0, 0, {{0, 0}});
                m_particles.push_back(s2, {{0, 0}}, {{-0.25, 0}}, 0, 0, {{0, 0}});
            }

            double m_dt = .005;
            std::size_t m_total_it = 1000;
            scopi_container<2> m_particles;
            std::size_t m_active_ptr = 0; // without obstacles
    };

    class TestTwoEllipsoidsAsymmetrical  : public ::testing::Test {
        protected:
            void SetUp() override {
                superellipsoid<2> s1({{-0.2, -0.05}}, {scopi::quaternion(PI/4)}, {{.1, .05}}, 1);
                superellipsoid<2> s2({{0.2, 0.05}}, {scopi::quaternion(-PI/4)}, {{.1, .05}}, 1);
                m_particles.push_back(s1, {{0, 0}}, {{0.25, 0}}, 0, 0, {{0, 0}});
                m_particles.push_back(s2, {{0, 0}}, {{-0.25, 0}}, 0, 0, {{0, 0}});
            }

            double m_dt = .005;
            std::size_t m_total_it = 1000;
            scopi_container<2> m_particles;
            std::size_t m_active_ptr = 0; // without obstacles
    };

    class TestTwoEllipsoidsSpheresAsymmetrical  : public ::testing::Test {
        protected:
            void SetUp() override {
                superellipsoid<2> s1({{-0.2, -0.05}}, {scopi::quaternion(PI/4)}, {{.1, .1}}, 1);
                superellipsoid<2> s2({{0.2, 0.05}}, {scopi::quaternion(-PI/4)}, {{.1, .1}}, 1);
                m_particles.push_back(s1, {{0, 0}}, {{0.25, 0}}, 0, 0, {{0, 0}});
                m_particles.push_back(s2, {{0, 0}}, {{-0.25, 0}}, 0, 0, {{0, 0}});
            }

            double m_dt = .005;
            std::size_t m_total_it = 1000;
            scopi_container<2> m_particles;
            std::size_t m_active_ptr = 0; // without obstacles
    };

    TEST_F(TestTwoEllipsoidsSymmetrical, two_ellipsoids_symmetrical)
    {
        // TODO set the optimization solver (Mosek, Uzawa, ...) here and duplicate this test for all solver 
        constexpr std::size_t dim = 2;
        ScopiSolver<dim> solver(m_particles, m_dt, m_active_ptr);
        solver.solve(m_total_it);

        std::string filenameRef;
        if(solver.getOptimSolverName() == "OptimMosek")
            filenameRef = "../test/references/two_ellipsoids_symmetrical_mosek.json"; 
        else if(solver.getOptimSolverName() == "OptimUzawaMkl")
            filenameRef = "../test/references/two_ellipsoids_symmetrical_uzawaMkl.json"; 
        else if(solver.getOptimSolverName() == "OptimScs")
            filenameRef = "../test/references/two_ellipsoids_symmetrical_scs.json"; 
        else if(solver.getOptimSolverName() == "OptimUzawaMatrixFreeTbb")
            filenameRef = "../test/references/two_ellipsoids_symmetrical_uzawaMatrixFreeTbb.json"; 
        else if(solver.getOptimSolverName() == "OptimUzawaMatrixFreeOmp")
            filenameRef = "../test/references/two_ellipsoids_symmetrical_uzawaMatrixFreeOmp.json"; 

        EXPECT_PRED2(diffFile, "./Results/scopi_objects_0199.json", filenameRef);
    }

    TEST_F(TestTwoEllipsoidsSpheresSymmetrical, two_ellipsoids_spheres_symmetrical)
    {
        // TODO set the optimization solver (Mosek, Uzawa, ...) here and duplicate this test for all solver 
        constexpr std::size_t dim = 2;
        ScopiSolver<dim> solver(m_particles, m_dt, m_active_ptr);
        solver.solve(m_total_it);

        std::string filenameRef;
        if(solver.getOptimSolverName() == "OptimMosek")
            filenameRef = "../test/references/two_ellipsoids_spheres_symmetrical_mosek.json"; 
        else if(solver.getOptimSolverName() == "OptimUzawaMkl")
            filenameRef = "../test/references/two_ellipsoids_spheres_symmetrical_uzawaMkl.json"; 
        else if(solver.getOptimSolverName() == "OptimScs")
            filenameRef = "../test/references/two_ellipsoids_spheres_symmetrical_scs.json"; 
        else if(solver.getOptimSolverName() == "OptimUzawaMatrixFreeTbb")
            filenameRef = "../test/references/two_ellipsoids_spheres_symmetrical_uzawaMatrixFreeTbb.json"; 
        else if(solver.getOptimSolverName() == "OptimUzawaMatrixFreeOmp")
            filenameRef = "../test/references/two_ellipsoids_spheres_symmetrical_uzawaMatrixFreeOmp.json"; 

        EXPECT_PRED2(diffFile, "./Results/scopi_objects_0999.json", filenameRef);
    }

    TEST_F(TestTwoEllipsoidsAsymmetrical, two_ellipsoids_symmetrical)
    {
        // TODO set the optimization solver (Mosek, Uzawa, ...) here and duplicate this test for all solver 
        constexpr std::size_t dim = 2;
        ScopiSolver<dim> solver(m_particles, m_dt, m_active_ptr);
        solver.solve(m_total_it);

        std::string filenameRef;
        if(solver.getOptimSolverName() == "OptimMosek")
            filenameRef = "../test/references/two_ellipsoids_asymmetrical_mosek.json"; 
        else if(solver.getOptimSolverName() == "OptimUzawaMkl")
            filenameRef = "../test/references/two_ellipsoids_asymmetrical_uzawaMkl.json"; 
        else if(solver.getOptimSolverName() == "OptimScs")
            filenameRef = "../test/references/two_ellipsoids_asymmetrical_scs.json"; 
        else if(solver.getOptimSolverName() == "OptimUzawaMatrixFreeTbb")
            filenameRef = "../test/references/two_ellipsoids_asymmetrical_uzawaMatrixFreeTbb.json"; 
        else if(solver.getOptimSolverName() == "OptimUzawaMatrixFreeOmp")
            filenameRef = "../test/references/two_ellipsoids_asymmetrical_uzawaMatrixFreeOmp.json"; 

        EXPECT_PRED2(diffFile, "./Results/scopi_objects_0999.json", filenameRef);
    }

    TEST_F(TestTwoEllipsoidsSpheresAsymmetrical, two_ellipsoids_symmetrical)
    {
        // TODO set the optimization solver (Mosek, Uzawa, ...) here and duplicate this test for all solver 
        constexpr std::size_t dim = 2;
        ScopiSolver<dim> solver(m_particles, m_dt, m_active_ptr);
        solver.solve(m_total_it);

        std::string filenameRef;
        if(solver.getOptimSolverName() == "OptimMosek")
            filenameRef = "../test/references/two_ellipsoids_spheres_asymmetrical_mosek.json"; 
        else if(solver.getOptimSolverName() == "OptimUzawaMkl")
            filenameRef = "../test/references/two_ellipsoids_spheres_asymmetrical_uzawaMkl.json"; 
        else if(solver.getOptimSolverName() == "OptimScs")
            filenameRef = "../test/references/two_ellipsoids_spheres_asymmetrical_scs.json"; 
        else if(solver.getOptimSolverName() == "OptimUzawaMatrixFreeTbb")
            filenameRef = "../test/references/two_ellipsoids_spheres_asymmetrical_uzawaMatrixFreeTbb.json"; 
        else if(solver.getOptimSolverName() == "OptimUzawaMatrixFreeOmp")
            filenameRef = "../test/references/two_ellipsoids_spheres_asymmetrical_uzawaMatrixFreeOmp.json"; 

        EXPECT_PRED2(diffFile, "./Results/scopi_objects_0999.json", filenameRef);
    }

}
