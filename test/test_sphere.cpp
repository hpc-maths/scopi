#include <gtest/gtest.h>
#include "utils.hpp"

#include <scopi/objects/types/sphere.hpp>
#include <scopi/container.hpp>
#include <scopi/solvers/mosek.hpp>

namespace scopi
{
    // pos
    TEST(sphere, pos_2d)
    {
        constexpr std::size_t dim = 2;
        sphere<dim> s({{-0.2, 0.0}}, 0.1);

        EXPECT_EQ(s.pos()(0), -0.2);
        EXPECT_EQ(s.pos()(1), 0.);
    }

    TEST(sphere, pos_3d)
    {
        constexpr std::size_t dim = 3;
        sphere<dim> s({{-0.2, 0.0, 0.1}}, 0.1);

        EXPECT_EQ(s.pos()(0), -0.2);
        EXPECT_EQ(s.pos()(1), 0.);
        EXPECT_EQ(s.pos()(2), 0.1);
    }

    TEST(sphere, pos_2d_const)
    {
        constexpr std::size_t dim = 2;
        const sphere<dim> s({{-0.2, 0.0}}, 0.1);

        EXPECT_EQ(s.pos()(0), -0.2);
        EXPECT_EQ(s.pos()(1), 0.);
    }

    TEST(sphere, pos_3d_const)
    {
        constexpr std::size_t dim = 3;
        const sphere<dim> s({{-0.2, 0.0, 0.1}}, 0.1);

        EXPECT_EQ(s.pos()(0), -0.2);
        EXPECT_EQ(s.pos()(1), 0.);
        EXPECT_EQ(s.pos()(2), 0.1);
    }

    TEST(sphere, pos_2d_index)
    {
        constexpr std::size_t dim = 2;
        sphere<dim> s({{-0.2, 0.0}}, 0.1);

        EXPECT_EQ(s.pos(0)(), -0.2);
        EXPECT_EQ(s.pos(1)(), 0.);
    }

    TEST(sphere, pos_3d_index)
    {
        constexpr std::size_t dim = 3;
        sphere<dim> s({{-0.2, 0.0, 0.1}}, 0.1);

        EXPECT_EQ(s.pos(0)(), -0.2);
        EXPECT_EQ(s.pos(1)(), 0.);
        EXPECT_EQ(s.pos(2)(), 0.1);
    }

    TEST(sphere, pos_2d_index_const)
    {
        constexpr std::size_t dim = 2;
        const sphere<dim> s({{-0.2, 0.0}}, 0.1);

        EXPECT_EQ(s.pos(0)(), -0.2);
        EXPECT_EQ(s.pos(1)(), 0.);
    }

    TEST(sphere, pos_3d_index_const)
    {
        constexpr std::size_t dim = 3;
        const sphere<dim> s({{-0.2, 0.0, 0.1}}, 0.1);

        EXPECT_EQ(s.pos(0)(), -0.2);
        EXPECT_EQ(s.pos(1)(), 0.);
        EXPECT_EQ(s.pos(2)(), 0.1);
    }

    TEST(sphere, pos_2d_container)
    {
        constexpr std::size_t dim = 2;
        sphere<dim> s({{-0.2, 0.0}}, 0.1);

        scopi_container<dim> particles;
        particles.push_back(s, {{0, 0}}, {{0.25, 0}}, 0, 0, {{0, 0}});

        EXPECT_EQ(particles[0]->pos()(0), -0.2);
        EXPECT_EQ(particles[0]->pos()(1), 0.);
    }

    TEST(sphere, pos_3d_container)
    {
        constexpr std::size_t dim = 3;
        sphere<dim> s({{-0.2, 0.0, 0.1}}, 0.1);

        scopi_container<dim> particles;
        particles.push_back(s, {{0, 0, 0}}, {{0.25, 0, 0}}, 0, 0, {{0, 0, 0}});

        EXPECT_EQ(particles[0]->pos()(0), -0.2);
        EXPECT_EQ(particles[0]->pos()(1), 0.);
        EXPECT_EQ(particles[0]->pos()(2), 0.1);
    }

    TEST(sphere, pos_2d_const_container)
    {
        constexpr std::size_t dim = 2;
        const sphere<dim> s({{-0.2, 0.0}}, 0.1);

        scopi_container<dim> particles;
        particles.push_back(s, {{0, 0}}, {{0.25, 0}}, 0, 0, {{0, 0}});

        EXPECT_EQ(particles[0]->pos()(0), -0.2);
        EXPECT_EQ(particles[0]->pos()(1), 0.);
    }

    TEST(sphere, pos_3d_const_container)
    {
        constexpr std::size_t dim = 3;
        const sphere<dim> s({{-0.2, 0.0, 0.1}}, 0.1);

        scopi_container<dim> particles;
        particles.push_back(s, {{0, 0, 0}}, {{0.25, 0, 0}}, 0, 0, {{0, 0, 0}});

        EXPECT_EQ(particles[0]->pos()(0), -0.2);
        EXPECT_EQ(particles[0]->pos()(1), 0.);
        EXPECT_EQ(particles[0]->pos()(2), 0.1);
    }

    TEST(sphere, pos_2d_index_container)
    {
        constexpr std::size_t dim = 2;
        sphere<dim> s({{-0.2, 0.0}}, 0.1);

        scopi_container<dim> particles;
        particles.push_back(s, {{0, 0}}, {{0.25, 0}}, 0, 0, {{0, 0}});

        EXPECT_EQ(particles[0]->pos(0)(), -0.2);
        EXPECT_EQ(particles[0]->pos(1)(), 0.);
    }

    TEST(sphere, pos_3d_index_container)
    {
        constexpr std::size_t dim = 3;
        sphere<dim> s({{-0.2, 0.0, 0.1}}, 0.1);

        scopi_container<dim> particles;
        particles.push_back(s, {{0, 0, 0}}, {{0.25, 0, 0}}, 0, 0, {{0, 0, 0}});

        EXPECT_EQ(particles[0]->pos(0)(), -0.2);
        EXPECT_EQ(particles[0]->pos(1)(), 0.);
        EXPECT_EQ(particles[0]->pos(2)(), 0.1);
    }

    TEST(sphere, pos_2d_index_const_container)
    {
        constexpr std::size_t dim = 2;
        const sphere<dim> s({{-0.2, 0.0}}, 0.1);

        scopi_container<dim> particles;
        particles.push_back(s, {{0, 0}}, {{0.25, 0}}, 0, 0, {{0, 0}});

        EXPECT_EQ(particles[0]->pos(0)(), -0.2);
        EXPECT_EQ(particles[0]->pos(1)(), 0.);
    }

    TEST(sphere, pos_3d_index_const_container)
    {
        constexpr std::size_t dim = 3;
        const sphere<dim> s({{-0.2, 0.0, 0.1}}, 0.1);

        scopi_container<dim> particles;
        particles.push_back(s, {{0, 0, 0}}, {{0.25, 0, 0}}, 0, 0, {{0, 0, 0}});

        EXPECT_EQ(particles[0]->pos(0)(), -0.2);
        EXPECT_EQ(particles[0]->pos(1)(), 0.);
        EXPECT_EQ(particles[0]->pos(2)(), 0.1);
    }

    // q
    TEST(sphere, q)
    {
        constexpr std::size_t dim = 2;
        sphere<dim> s({{-0.2, 0.0}}, 0.1);

        EXPECT_EQ(s.q()(0), 1.);
        EXPECT_EQ(s.q()(1), 0.);
        EXPECT_EQ(s.q()(2), 0.);
        EXPECT_EQ(s.q()(3), 0.);
    }

    TEST(sphere, q_const)
    {
        constexpr std::size_t dim = 2;
        const sphere<dim> s({{-0.2, 0.0}}, 0.1);

        EXPECT_EQ(s.q()(0), 1.);
        EXPECT_EQ(s.q()(1), 0.);
        EXPECT_EQ(s.q()(2), 0.);
        EXPECT_EQ(s.q()(3), 0.);
    }

    TEST(sphere, q_index)
    {
        constexpr std::size_t dim = 2;
        sphere<dim> s({{-0.2, 0.0}}, 0.1);

        EXPECT_EQ(s.q(0)(), 1.);
        EXPECT_EQ(s.q(1)(), 0.);
        EXPECT_EQ(s.q(2)(), 0.);
        EXPECT_EQ(s.q(3)(), 0.);
    }

    TEST(sphere, q_index_const)
    {
        constexpr std::size_t dim = 2;
        const sphere<dim> s({{-0.2, 0.0}}, 0.1);

        EXPECT_EQ(s.q(0)(), 1.);
        EXPECT_EQ(s.q(1)(), 0.);
        EXPECT_EQ(s.q(2)(), 0.);
        EXPECT_EQ(s.q(3)(), 0.);
    }

    TEST(sphere, q_container)
    {
        constexpr std::size_t dim = 2;
        sphere<dim> s({{-0.2, 0.0}}, 0.1);

        scopi_container<dim> particles;
        particles.push_back(s, {{0, 0}}, {{0.25, 0}}, 0, 0, {{0, 0}});

        EXPECT_EQ(particles[0]->q()(0), 1.);
        EXPECT_EQ(particles[0]->q()(1), 0.);
        EXPECT_EQ(particles[0]->q()(2), 0.);
        EXPECT_EQ(s.q()(3), 0.);
    }

    TEST(sphere, q_const_container)
    {
        constexpr std::size_t dim = 2;
        const sphere<dim> s({{-0.2, 0.0}}, 0.1);

        scopi_container<dim> particles;
        particles.push_back(s, {{0, 0}}, {{0.25, 0}}, 0, 0, {{0, 0}});

        EXPECT_EQ(particles[0]->q()(0), 1.);
        EXPECT_EQ(particles[0]->q()(1), 0.);
        EXPECT_EQ(particles[0]->q()(2), 0.);
        EXPECT_EQ(particles[0]->q()(3), 0.);
    }

    TEST(sphere, q_index_container)
    {
        constexpr std::size_t dim = 2;
        sphere<dim> s({{-0.2, 0.0}}, 0.1);

        scopi_container<dim> particles;
        particles.push_back(s, {{0, 0}}, {{0.25, 0}}, 0, 0, {{0, 0}});

        EXPECT_EQ(particles[0]->q(0)(), 1.);
        EXPECT_EQ(particles[0]->q(1)(), 0.);
        EXPECT_EQ(particles[0]->q(2)(), 0.);
        EXPECT_EQ(particles[0]->q(3)(), 0.);
    }

    TEST(sphere, q_index_const_container)
    {
        constexpr std::size_t dim = 2;
        const sphere<dim> s({{-0.2, 0.0}}, 0.1);

        scopi_container<dim> particles;
        particles.push_back(s, {{0, 0}}, {{0.25, 0}}, 0, 0, {{0, 0}});

        EXPECT_EQ(particles[0]->q(0)(), 1.);
        EXPECT_EQ(particles[0]->q(1)(), 0.);
        EXPECT_EQ(particles[0]->q(2)(), 0.);
        EXPECT_EQ(particles[0]->q(3)(), 0.);
    }

    // radius
    TEST(sphere, radius)
    {
        constexpr std::size_t dim = 2;
        sphere<dim> s({{-0.2, 0.0}}, 0.1);

        EXPECT_EQ(s.radius(), 0.1);
    }

    // rotation
    TEST(sphere, rotation_2d)
    {
        constexpr std::size_t dim = 2;
        sphere<dim> s({{-0.2, 0.0}}, {quaternion(PI/3)}, 0.1);

        auto rotation_matrix = s.rotation();

        EXPECT_DOUBLE_EQ(rotation_matrix(0, 0), 1./2.);
        EXPECT_DOUBLE_EQ(rotation_matrix(0, 1), -std::sqrt(3.)/2.);
        EXPECT_DOUBLE_EQ(rotation_matrix(1, 0), std::sqrt(3.)/2.);
        EXPECT_DOUBLE_EQ(rotation_matrix(1, 1), 1./2.);
    }

    TEST(sphere, rotation_3d)
    {
        constexpr std::size_t dim = 3;
        sphere<dim> s({{-0.2, 0.0, 0.0}}, {quaternion(PI/3)}, 0.1);

        auto rotation_matrix = s.rotation();

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
    TEST(sphere, point_2d)
    {
        constexpr std::size_t dim = 2;
        sphere<dim> s({{0.0, 0.0}}, 0.1);

        auto point = s.point(0.);

        EXPECT_EQ(point(0), 0.1);
        EXPECT_EQ(point(1), 0.);
    }

    TEST(sphere, point_3d)
    {
        constexpr std::size_t dim = 3;
        sphere<dim> s({{0.0, 0.0, 0.0}}, 0.1);

        auto point = s.point(0., 0.);

        EXPECT_EQ(point(0), 0.1);
        EXPECT_EQ(point(1), 0.);
        EXPECT_EQ(point(2), 0.);
    }

    // normal
    TEST(sphere, normal_2d)
    {
        constexpr std::size_t dim = 2;
        sphere<dim> s({{0.0, 0.0}}, 0.1);

        auto normal = s.normal(0.);

        EXPECT_EQ(normal(0), 1.);
        EXPECT_EQ(normal(1), 0.);
    }

    TEST(sphere, normal_3d)
    {
        constexpr std::size_t dim = 3;
        sphere<dim> s({{0.0, 0.0, 0.0}}, 0.1);

        auto normal = s.normal(0., 0.);

        EXPECT_EQ(normal(0), 1.);
        EXPECT_EQ(normal(1), 0.);
        EXPECT_EQ(normal(2), 0.);
    }

    // two_spheres
    TEST(sphere, two_spheres_asymetrical)
    {
        constexpr std::size_t dim = 2;
        double dt = .005;
        std::size_t total_it = 1000;
        scopi_container<dim> particles;

        sphere<dim> s1({{-0.2, -0.05}}, 0.1);
        sphere<dim> s2({{ 0.2,  0.05}}, 0.1);
        particles.push_back(s1, {{0, 0}}, {{0.25, 0}}, 0, 0, {{0, 0}});
        particles.push_back(s2, {{0, 0}}, {{-0.25, 0}}, 0, 0, {{0, 0}});

        std::size_t active_ptr = 0; // without obstacles
        ScopiSolver<dim> solver(particles, dt, active_ptr);
        solver.solve(total_it);

        std::string filenameRef;
        if(solver.getOptimSolverName() == "OptimMosek")
            filenameRef = "../test/two_spheres_asymetrical_mosek.json"; 
        else if(solver.getOptimSolverName() == "OptimUzawaMkl")
            filenameRef = "../test/two_spheres_asymetrical_uzawaMkl.json"; 
        else if(solver.getOptimSolverName() == "OptimScs")
            filenameRef = "../test/two_spheres_asymetrical_scs.json"; 
        else if(solver.getOptimSolverName() == "OptimUzawaMatrixFreeTbb")
            filenameRef = "../test/two_spheres_asymetrical_uzawaMatrixFreeTbb.json"; 
        else if(solver.getOptimSolverName() == "OptimUzawaMatrixFreeOmp")
            filenameRef = "../test/two_spheres_asymetrical_uzawaMatrixFreeOmp.json"; 

        EXPECT_PRED2(diffFile, "./Results/scopi_objects_0999.json", filenameRef);
    }

}
