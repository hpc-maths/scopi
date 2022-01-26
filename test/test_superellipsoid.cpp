#include <gtest/gtest.h>

#include <scopi/objects/types/superellipsoid.hpp>
#include <scopi/container.hpp>

namespace scopi
{
    // pos
    TEST(superellipsoid, pos_2d)
    {
        constexpr std::size_t dim = 2;
        superellipsoid<dim> s({{-0.2, 0.}}, {scopi::quaternion(0)}, {{0.1, 0.2}}, {{1}});

        EXPECT_EQ(s.pos()(0), -0.2);
        EXPECT_EQ(s.pos()(1), 0.);
    }

    TEST(superellipsoid, pos_3d)
    {
        constexpr std::size_t dim = 3;
        superellipsoid<dim> s({{-0.2, 0., 0.1}}, {scopi::quaternion(0)}, {{0.1, 0.2, 0.3}}, {{1, 1}});

        EXPECT_EQ(s.pos()(0), -0.2);
        EXPECT_EQ(s.pos()(1), 0.);
        EXPECT_EQ(s.pos()(2), 0.1);
    }

    TEST(superellipsoid, pos_2d_const)
    {
        constexpr std::size_t dim = 2;
        const superellipsoid<dim> s({{-0.2, 0.}}, {scopi::quaternion(0)}, {{0.1, 0.2}}, {{1}});

        EXPECT_EQ(s.pos()(0), -0.2);
        EXPECT_EQ(s.pos()(1), 0.);
    }

    TEST(superellipsoid, pos_3d_const)
    {
        constexpr std::size_t dim = 3;
        const superellipsoid<dim> s({{-0.2, 0., 0.1}}, {scopi::quaternion(0)}, {{0.1, 0.2, 0.3}}, {{1, 1}});

        EXPECT_EQ(s.pos()(0), -0.2);
        EXPECT_EQ(s.pos()(1), 0.);
        EXPECT_EQ(s.pos()(2), 0.1);
    }

    TEST(superellipsoid, pos_2d_index)
    {
        constexpr std::size_t dim = 2;
        superellipsoid<dim> s({{-0.2, 0.}}, {scopi::quaternion(0)}, {{0.1, 0.2}}, {{1}});

        EXPECT_EQ(s.pos(0)(), -0.2);
        EXPECT_EQ(s.pos(1)(), 0.);
    }

    TEST(superellipsoid, pos_3d_index)
    {
        constexpr std::size_t dim = 3;
        superellipsoid<dim> s({{-0.2, 0., 0.1}}, {scopi::quaternion(0)}, {{0.1, 0.2, 0.3}}, {{1, 1}});

        EXPECT_EQ(s.pos(0)(), -0.2);
        EXPECT_EQ(s.pos(1)(), 0.);
        EXPECT_EQ(s.pos(2)(), 0.1);
    }

    TEST(superellipsoid, pos_2d_index_const)
    {
        constexpr std::size_t dim = 2;
        const superellipsoid<dim> s({{-0.2, 0.}}, {scopi::quaternion(0)}, {{0.1, 0.2}}, {{1}});

        EXPECT_EQ(s.pos(0)(), -0.2);
        EXPECT_EQ(s.pos(1)(), 0.);
    }

    TEST(superellipsoid, pos_3d_index_const)
    {
        constexpr std::size_t dim = 3;
        const superellipsoid<dim> s({{-0.2, 0., 0.1}}, {scopi::quaternion(0)}, {{0.1, 0.2, 0.3}}, {{1, 1}});

        EXPECT_EQ(s.pos(0)(), -0.2);
        EXPECT_EQ(s.pos(1)(), 0.);
        EXPECT_EQ(s.pos(2)(), 0.1);
    }

    TEST(superellipsoid, pos_2d_container)
    {
        constexpr std::size_t dim = 2;
        superellipsoid<dim> s({{-0.2, 0.}}, {scopi::quaternion(0)}, {{0.1, 0.2}}, {{1}});

        scopi::scopi_container<dim> particles;
        particles.push_back(s, {{0, 0}}, {{0.25, 0}}, 0, 0, {{0, 0}});

        EXPECT_EQ(particles[0]->pos()(0), -0.2);
        EXPECT_EQ(particles[0]->pos()(1), 0.);
    }

    TEST(superellipsoid, pos_3d_container)
    {
        constexpr std::size_t dim = 3;
        superellipsoid<dim> s({{-0.2, 0., 0.1}}, {scopi::quaternion(0)}, {{0.1, 0.2, 0.3}}, {{1, 1}});

        scopi::scopi_container<dim> particles;
        particles.push_back(s, {{0, 0, 0}}, {{0.25, 0, 0}}, 0, 0, {{0, 0, 0}});

        EXPECT_EQ(particles[0]->pos()(0), -0.2);
        EXPECT_EQ(particles[0]->pos()(1), 0.);
        EXPECT_EQ(particles[0]->pos()(2), 0.1);
    }

    TEST(superellipsoid, pos_2d_const_container)
    {
        constexpr std::size_t dim = 2;
        const superellipsoid<dim> s({{-0.2, 0.}}, {scopi::quaternion(0)}, {{0.1, 0.2}}, {{1}});

        scopi::scopi_container<dim> particles;
        particles.push_back(s, {{0, 0}}, {{0.25, 0}}, 0, 0, {{0, 0}});

        EXPECT_EQ(particles[0]->pos()(0), -0.2);
        EXPECT_EQ(particles[0]->pos()(1), 0.);
    }

    TEST(superellipsoid, pos_3d_const_container)
    {
        constexpr std::size_t dim = 3;
        const superellipsoid<dim> s({{-0.2, 0., 0.1}}, {scopi::quaternion(0)}, {{0.1, 0.2, 0.3}}, {{1, 1}});

        scopi::scopi_container<dim> particles;
        particles.push_back(s, {{0, 0, 0}}, {{0.25, 0, 0}}, 0, 0, {{0, 0, 0}});

        EXPECT_EQ(particles[0]->pos()(0), -0.2);
        EXPECT_EQ(particles[0]->pos()(1), 0.);
        EXPECT_EQ(particles[0]->pos()(2), 0.1);
    }

    TEST(superellipsoid, pos_2d_index_container)
    {
        constexpr std::size_t dim = 2;
        superellipsoid<dim> s({{-0.2, 0.}}, {scopi::quaternion(0)}, {{0.1, 0.2}}, {{1}});

        scopi::scopi_container<dim> particles;
        particles.push_back(s, {{0, 0}}, {{0.25, 0}}, 0, 0, {{0, 0}});

        EXPECT_EQ(particles[0]->pos(0)(), -0.2);
        EXPECT_EQ(particles[0]->pos(1)(), 0.);
    }

    TEST(superellipsoid, pos_3d_index_container)
    {
        constexpr std::size_t dim = 3;
        superellipsoid<dim> s({{-0.2, 0., 0.1}}, {scopi::quaternion(0)}, {{0.1, 0.2, 0.3}}, {{1, 1}});

        scopi::scopi_container<dim> particles;
        particles.push_back(s, {{0, 0, 0}}, {{0.25, 0, 0}}, 0, 0, {{0, 0, 0}});

        EXPECT_EQ(particles[0]->pos(0)(), -0.2);
        EXPECT_EQ(particles[0]->pos(1)(), 0.);
        EXPECT_EQ(particles[0]->pos(2)(), 0.1);
    }

    TEST(superellipsoid, pos_2d_index_const_container)
    {
        constexpr std::size_t dim = 2;
        const superellipsoid<dim> s({{-0.2, 0.}}, {scopi::quaternion(0)}, {{0.1, 0.2}}, {{1}});

        scopi::scopi_container<dim> particles;
        particles.push_back(s, {{0, 0}}, {{0.25, 0}}, 0, 0, {{0, 0}});

        EXPECT_EQ(particles[0]->pos(0)(), -0.2);
        EXPECT_EQ(particles[0]->pos(1)(), 0.);
    }

    TEST(superellipsoid, pos_3d_index_const_container)
    {
        constexpr std::size_t dim = 3;
        const superellipsoid<dim> s({{-0.2, 0., 0.1}}, {scopi::quaternion(0)}, {{0.1, 0.2, 0.3}}, {{1, 1}});

        scopi::scopi_container<dim> particles;
        particles.push_back(s, {{0, 0, 0}}, {{0.25, 0, 0}}, 0, 0, {{0, 0, 0}});

        EXPECT_EQ(particles[0]->pos(0)(), -0.2);
        EXPECT_EQ(particles[0]->pos(1)(), 0.);
        EXPECT_EQ(particles[0]->pos(2)(), 0.1);
    }

    // q
    TEST(superellipsoid, q)
    {
        constexpr std::size_t dim = 2;
        superellipsoid<dim> s({{-0.2, 0.}}, {scopi::quaternion(0)}, {{0.1, 0.2}}, {{1}});

        EXPECT_EQ(s.q()(0), 1.);
        EXPECT_EQ(s.q()(1), 0.);
        EXPECT_EQ(s.q()(2), 0.);
        EXPECT_EQ(s.q()(3), 0.);
    }

    TEST(superellipsoid, q_const)
    {
        constexpr std::size_t dim = 2;
        const superellipsoid<dim> s({{-0.2, 0.}}, {scopi::quaternion(0)}, {{0.1, 0.2}}, {{1}});

        EXPECT_EQ(s.q()(0), 1.);
        EXPECT_EQ(s.q()(1), 0.);
        EXPECT_EQ(s.q()(2), 0.);
        EXPECT_EQ(s.q()(3), 0.);
    }

    TEST(superellipsoid, q_index)
    {
        constexpr std::size_t dim = 2;
        superellipsoid<dim> s({{-0.2, 0.}}, {scopi::quaternion(0)}, {{0.1, 0.2}}, {{1}});

        EXPECT_EQ(s.q(0)(), 1.);
        EXPECT_EQ(s.q(1)(), 0.);
        EXPECT_EQ(s.q(2)(), 0.);
        EXPECT_EQ(s.q(3)(), 0.);
    }

    TEST(superellipsoid, q_index_const)
    {
        constexpr std::size_t dim = 2;
        const superellipsoid<dim> s({{-0.2, 0.}}, {scopi::quaternion(0)}, {{0.1, 0.2}}, {{1}});

        EXPECT_EQ(s.q(0)(), 1.);
        EXPECT_EQ(s.q(1)(), 0.);
        EXPECT_EQ(s.q(2)(), 0.);
        EXPECT_EQ(s.q(3)(), 0.);
    }

    TEST(superellipsoid, q_container)
    {
        constexpr std::size_t dim = 2;
        superellipsoid<dim> s({{-0.2, 0.}}, {scopi::quaternion(0)}, {{0.1, 0.2}}, {{1}});

        scopi::scopi_container<dim> particles;
        particles.push_back(s, {{0, 0}}, {{0.25, 0}}, 0, 0, {{0, 0}});

        EXPECT_EQ(particles[0]->q()(0), 1.);
        EXPECT_EQ(particles[0]->q()(1), 0.);
        EXPECT_EQ(particles[0]->q()(2), 0.);
        EXPECT_EQ(s.q()(3), 0.);
    }

    TEST(superellipsoid, q_const_container)
    {
        constexpr std::size_t dim = 2;
        const superellipsoid<dim> s({{-0.2, 0.}}, {scopi::quaternion(0)}, {{0.1, 0.2}}, {{1}});

        scopi::scopi_container<dim> particles;
        particles.push_back(s, {{0, 0}}, {{0.25, 0}}, 0, 0, {{0, 0}});

        EXPECT_EQ(particles[0]->q()(0), 1.);
        EXPECT_EQ(particles[0]->q()(1), 0.);
        EXPECT_EQ(particles[0]->q()(2), 0.);
        EXPECT_EQ(particles[0]->q()(3), 0.);
    }

    TEST(superellipsoid, q_index_container)
    {
        constexpr std::size_t dim = 2;
        superellipsoid<dim> s({{-0.2, 0.}}, {scopi::quaternion(0)}, {{0.1, 0.2}}, {{1}});

        scopi::scopi_container<dim> particles;
        particles.push_back(s, {{0, 0}}, {{0.25, 0}}, 0, 0, {{0, 0}});

        EXPECT_EQ(particles[0]->q(0)(), 1.);
        EXPECT_EQ(particles[0]->q(1)(), 0.);
        EXPECT_EQ(particles[0]->q(2)(), 0.);
        EXPECT_EQ(particles[0]->q(3)(), 0.);
    }

    TEST(superellipsoid, q_index_const_container)
    {
        constexpr std::size_t dim = 2;
        const superellipsoid<dim> s({{-0.2, 0.}}, {scopi::quaternion(0)}, {{0.1, 0.2}}, {{1}});

        scopi::scopi_container<dim> particles;
        particles.push_back(s, {{0, 0}}, {{0.25, 0}}, 0, 0, {{0, 0}});

        EXPECT_EQ(particles[0]->q(0)(), 1.);
        EXPECT_EQ(particles[0]->q(1)(), 0.);
        EXPECT_EQ(particles[0]->q(2)(), 0.);
        EXPECT_EQ(particles[0]->q(3)(), 0.);
    }

    // radius
    TEST(superellipsoid, radius_2d)
    {
        constexpr std::size_t dim = 2;
        superellipsoid<dim> s({{-0.2, 0.}}, {scopi::quaternion(0)}, {{0.1, 0.2}}, {{1}});

        auto radius = s.radius();

        EXPECT_EQ(radius(0), 0.1);
        EXPECT_EQ(radius(1), 0.2);
    }

    TEST(superellipsoid, radius_3d)
    {
        constexpr std::size_t dim = 3;
        superellipsoid<dim> s({{-0.2, 0., 0.1}}, {scopi::quaternion(0)}, {{0.1, 0.2, 0.3}}, {{1, 1}});

        auto radius = s.radius();

        EXPECT_EQ(radius(0), 0.1);
        EXPECT_EQ(radius(1), 0.2);
        EXPECT_EQ(radius(2), 0.3);
    }

    //squareness
    TEST(superellipsoid, squareness_2d)
    {
        constexpr std::size_t dim = 2;
        superellipsoid<dim> s({{-0.2, 0.}}, {scopi::quaternion(0)}, {{0.1, 0.2}}, {{1.5}});

        auto squareness = s.squareness();

        EXPECT_EQ(squareness(0), 1.5);
    }

    TEST(superellipsoid, squareness_3d)
    {
        constexpr std::size_t dim = 3;
        superellipsoid<dim> s({{-0.2, 0., 0.1}}, {scopi::quaternion(0)}, {{0.1, 0.2, 0.3}}, {{0.5, 1.5}});

        auto squareness = s.squareness();

        EXPECT_EQ(squareness(0), 0.5);
        EXPECT_EQ(squareness(1), 1.5);
    }

    // rotation
    TEST(superellipsoid, rotation_2d)
    {
        constexpr std::size_t dim = 2;
        double PI = xt::numeric_constants<double>::PI;
        superellipsoid<dim> s({{-0.2, 0.}}, {scopi::quaternion(PI/3)}, {{0.1, 0.2}}, {{1}});

        auto rotation_matrix = s.rotation();

        EXPECT_DOUBLE_EQ(rotation_matrix(0, 0), 1./2.);
        EXPECT_DOUBLE_EQ(rotation_matrix(0, 1), -std::sqrt(3.)/2.);
        EXPECT_DOUBLE_EQ(rotation_matrix(1, 0), std::sqrt(3.)/2.);
        EXPECT_DOUBLE_EQ(rotation_matrix(1, 1), 1./2.);
    }

    TEST(superellipsoid, rotation_3d)
    {
        constexpr std::size_t dim = 3;
        double PI = xt::numeric_constants<double>::PI;
        superellipsoid<dim> s({{-0.2, 0., 0.1}}, {scopi::quaternion(PI/3    )}, {{0.1, 0.2, 0.3}}, {{1, 1}});

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
}
