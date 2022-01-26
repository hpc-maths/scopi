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

}
