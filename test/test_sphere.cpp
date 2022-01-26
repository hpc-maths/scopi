#include <gtest/gtest.h>

#include <scopi/objects/types/sphere.hpp>
#include <scopi/container.hpp>
#include <scopi/objects/methods/closest_points.hpp>

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

        scopi::scopi_container<dim> particles;
        particles.push_back(s, {{0, 0}}, {{0.25, 0}}, 0, 0, {{0, 0}});

        EXPECT_EQ(particles[0]->pos()(0), -0.2);
        EXPECT_EQ(particles[0]->pos()(1), 0.);
    }

    TEST(sphere, pos_3d_container)
    {
        constexpr std::size_t dim = 3;
        sphere<dim> s({{-0.2, 0.0, 0.1}}, 0.1);

        scopi::scopi_container<dim> particles;
        particles.push_back(s, {{0, 0, 0}}, {{0.25, 0, 0}}, 0, 0, {{0, 0, 0}});

        EXPECT_EQ(particles[0]->pos()(0), -0.2);
        EXPECT_EQ(particles[0]->pos()(1), 0.);
        EXPECT_EQ(particles[0]->pos()(2), 0.1);
    }

    TEST(sphere, pos_2d_const_container)
    {
        constexpr std::size_t dim = 2;
        const sphere<dim> s({{-0.2, 0.0}}, 0.1);

        scopi::scopi_container<dim> particles;
        particles.push_back(s, {{0, 0}}, {{0.25, 0}}, 0, 0, {{0, 0}});

        EXPECT_EQ(particles[0]->pos()(0), -0.2);
        EXPECT_EQ(particles[0]->pos()(1), 0.);
    }

    TEST(sphere, pos_3d_const_container)
    {
        constexpr std::size_t dim = 3;
        const sphere<dim> s({{-0.2, 0.0, 0.1}}, 0.1);

        scopi::scopi_container<dim> particles;
        particles.push_back(s, {{0, 0, 0}}, {{0.25, 0, 0}}, 0, 0, {{0, 0, 0}});

        EXPECT_EQ(particles[0]->pos()(0), -0.2);
        EXPECT_EQ(particles[0]->pos()(1), 0.);
        EXPECT_EQ(particles[0]->pos()(2), 0.1);
    }

    TEST(sphere, pos_2d_index_container)
    {
        constexpr std::size_t dim = 2;
        sphere<dim> s({{-0.2, 0.0}}, 0.1);

        scopi::scopi_container<dim> particles;
        particles.push_back(s, {{0, 0}}, {{0.25, 0}}, 0, 0, {{0, 0}});

        EXPECT_EQ(particles[0]->pos(0)(), -0.2);
        EXPECT_EQ(particles[0]->pos(1)(), 0.);
    }

    TEST(sphere, pos_3d_index_container)
    {
        constexpr std::size_t dim = 3;
        sphere<dim> s({{-0.2, 0.0, 0.1}}, 0.1);

        scopi::scopi_container<dim> particles;
        particles.push_back(s, {{0, 0, 0}}, {{0.25, 0, 0}}, 0, 0, {{0, 0, 0}});

        EXPECT_EQ(particles[0]->pos(0)(), -0.2);
        EXPECT_EQ(particles[0]->pos(1)(), 0.);
        EXPECT_EQ(particles[0]->pos(2)(), 0.1);
    }

    TEST(sphere, pos_2d_index_const_container)
    {
        constexpr std::size_t dim = 2;
        const sphere<dim> s({{-0.2, 0.0}}, 0.1);

        scopi::scopi_container<dim> particles;
        particles.push_back(s, {{0, 0}}, {{0.25, 0}}, 0, 0, {{0, 0}});

        EXPECT_EQ(particles[0]->pos(0)(), -0.2);
        EXPECT_EQ(particles[0]->pos(1)(), 0.);
    }

    TEST(sphere, pos_3d_index_const_container)
    {
        constexpr std::size_t dim = 3;
        const sphere<dim> s({{-0.2, 0.0, 0.1}}, 0.1);

        scopi::scopi_container<dim> particles;
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

        scopi::scopi_container<dim> particles;
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

        scopi::scopi_container<dim> particles;
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

        scopi::scopi_container<dim> particles;
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

        scopi::scopi_container<dim> particles;
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
        const sphere<dim> s({{-0.2, 0.0}}, 0.1);

        EXPECT_EQ(s.radius(), 0.1);
    }

    // rotation
    TEST(sphere, rotation_2d)
    {
        constexpr std::size_t dim = 2;
        const sphere<dim> s({{-0.2, 0.0}}, 0.1);

        auto rotation_matrix = s.rotation();

        EXPECT_EQ(rotation_matrix(0, 0), 1.);
        EXPECT_EQ(rotation_matrix(0, 1), 0.);
        EXPECT_EQ(rotation_matrix(1, 0), 0.);
        EXPECT_EQ(rotation_matrix(1, 1), 1.);
    }

    TEST(sphere, rotation_3d)
    {
        constexpr std::size_t dim = 3;
        const sphere<dim> s({{-0.2, 0.0, 0.0}}, 0.1);

        auto rotation_matrix = s.rotation();

        EXPECT_EQ(rotation_matrix(0, 0), 1.);
        EXPECT_EQ(rotation_matrix(0, 1), 0.);
        EXPECT_EQ(rotation_matrix(0, 2), 0.);
        EXPECT_EQ(rotation_matrix(1, 0), 0.);
        EXPECT_EQ(rotation_matrix(1, 1), 1.);
        EXPECT_EQ(rotation_matrix(1, 2), 0.);
        EXPECT_EQ(rotation_matrix(2, 0), 0.);
        EXPECT_EQ(rotation_matrix(2, 1), 0.);
        EXPECT_EQ(rotation_matrix(2, 2), 1.);
    }

    // distance sphere - sphere
    TEST(sphere, closest_points_sphere_2d)
    {
        constexpr std::size_t dim = 2;
        sphere<dim> s1({{-0.2, 0.0}}, 0.1);
        sphere<dim> s2({{ 0.2, 0.0}}, 0.1);

        auto out = closest_points(s1, s2);

        EXPECT_EQ(out.dij, 0.2);
    }

    TEST(sphere, closest_points_dispatch_sphere_2d)
    {
        constexpr std::size_t dim = 2;
        sphere<dim> s1({{-0.2, 0.0}}, 0.1);
        sphere<dim> s2({{ 0.2, 0.0}}, 0.1);

        scopi::scopi_container<dim> particles;
        particles.push_back(s1, {{0, 0}}, {{0.25, 0}}, 0, 0, {{0, 0}});
        particles.push_back(s2, {{0, 0}}, {{-0.25, 0}}, 0, 0, {{0, 0}});

        auto out = closest_points_dispatcher<dim>::dispatch(*particles[0], *particles[1]);

        EXPECT_EQ(out.dij, 0.2);
    }

    TEST(sphere, closest_points_sphere_3d)
    {
        constexpr std::size_t dim = 3;
        sphere<dim> s1({{-0.2, 0.0, 0.0}}, 0.1);
        sphere<dim> s2({{ 0.2, 0.0, 0.0}}, 0.1);

        auto out = closest_points(s1, s2);

        EXPECT_EQ(out.dij, 0.2);
    }

    TEST(sphere, closest_points_dispatch_sphere_3d)
    {
        constexpr std::size_t dim = 3;
        sphere<dim> s1({{-0.2, 0.0, 0.0}}, 0.1);
        sphere<dim> s2({{ 0.2, 0.0, 0.0}}, 0.1);

        scopi::scopi_container<dim> particles;
        particles.push_back(s1, {{0, 0, 0}}, {{0.25, 0, 0}}, 0, 0, {{0, 0, 0}});
        particles.push_back(s2, {{0, 0, 0}}, {{-0.25, 0, 0}}, 0, 0, {{0, 0, 0}});

        auto out = closest_points_dispatcher<dim>::dispatch(*particles[0], *particles[1]);

        EXPECT_EQ(out.dij, 0.2);
    }

    // distance sphere - plan
    TEST(sphere, closest_points_plan_2d)
    {
        constexpr std::size_t dim = 2;
        sphere<dim> s({{0.0, 0.0}}, 0.1);
        plan<dim> p({{ 0.3, 0.0}}, 0.);

        auto out = closest_points(s, p);

        EXPECT_DOUBLE_EQ(out.dij, 0.2);
    }

    TEST(sphere, closest_points_dispatch_plan_2d)
    {
        constexpr std::size_t dim = 2;
        sphere<dim> s({{0.0, 0.0}}, 0.1);
        plan<dim> p({{ 0.3, 0.0}}, 0.);

        scopi::scopi_container<dim> particles;
        particles.push_back(s, {{0, 0}}, {{0.25, 0}}, 0, 0, {{0, 0}});
        particles.push_back(p, {{0, 0}}, {{-0.25, 0}}, 0, 0, {{0, 0}});

        auto out = closest_points_dispatcher<dim>::dispatch(*particles[0], *particles[1]);

        EXPECT_DOUBLE_EQ(out.dij, 0.2);
    }

    TEST(sphere, closest_points_plan_3d)
    {
        constexpr std::size_t dim = 3;
        sphere<dim> s({{0.0, 0.0, 0.0}}, 0.1);
        plan<dim> p({{ 0.3, 0.0, 0.0}}, 0.);

        auto out = closest_points(s, p);

        EXPECT_DOUBLE_EQ(out.dij, 0.2);
    }

    TEST(sphere, closest_points_dispatch_plan_3d)
    {
        constexpr std::size_t dim = 3;
        sphere<dim> s({{0.0, 0.0, 0.0}}, 0.1);
        plan<dim> p({{ 0.3, 0.0, 0.0}}, 0.);

        scopi::scopi_container<dim> particles;
        particles.push_back(s, {{0, 0, 0}}, {{0.25, 0, 0}}, 0, 0, {{0, 0, 0}});
        particles.push_back(p, {{0, 0, 0}}, {{-0.25, 0, 0}}, 0, 0, {{0, 0, 0}});

        auto out = closest_points_dispatcher<dim>::dispatch(*particles[0], *particles[1]);

        EXPECT_DOUBLE_EQ(out.dij, 0.2);
    }

    
    // distance sphere - superellipsoid
    TEST(sphere, closest_points_superellipsoid_2d)
    {
        constexpr std::size_t dim = 2;
        sphere<dim> s({{0.2, 0.0}}, 0.1);
        superellipsoid<dim> e({{-0.2, 0.0}}, {scopi::quaternion(0.)}, {{0.1, 0.2}}, {{1}});

        auto out = closest_points(s, e);

        EXPECT_NEAR(out.dij, 0.2, 1e-7);
    }

    TEST(sphere, closest_points_dispatch_superellipsoid_2d)
    {
        constexpr std::size_t dim = 2;
        sphere<dim> s({{0.2, 0.0}}, 0.1);
        superellipsoid<dim> e({{-0.2, 0.0}}, {scopi::quaternion(0.)}, {{0.1, 0.2}}, {{1}});

        scopi::scopi_container<dim> particles;
        particles.push_back(s, {{0, 0}}, {{0.25, 0}}, 0, 0, {{0, 0}});
        particles.push_back(e, {{0, 0}}, {{-0.25, 0}}, 0, 0, {{0, 0}});

        auto out = closest_points_dispatcher<dim>::dispatch(*particles[0], *particles[1]);

        EXPECT_NEAR(out.dij, 0.2, 1e-7);
    }

    TEST(sphere, closest_points_superellipsoid_3d)
    {
        GTEST_SKIP();
        constexpr std::size_t dim = 3;
        sphere<dim> s({{0.2, 0.0, 0.0}}, 0.1);
        superellipsoid<dim> e({{-0.2, 0.0, 0.0}}, {scopi::quaternion(0.)}, {{0.1, 0.1, 0.1}}, {{1}});

        auto out = closest_points(s, e);

        EXPECT_NEAR(out.dij, 0.2, 1e-7);
    }

    TEST(sphere, closest_points_dispatch_superellipsoid_3d)
    {
        GTEST_SKIP();
        constexpr std::size_t dim = 3;
        sphere<dim> s({{0.2, 0.0, 0.0}}, 0.1);
        superellipsoid<dim> e({{-0.2, 0.0, 0.0}}, {scopi::quaternion(0.)}, {{0.1, 0.2, 0.3}}, {{1}});

        scopi::scopi_container<dim> particles;
        particles.push_back(s, {{0, 0, 0}}, {{0.25, 0, 0}}, 0, 0, {{0, 0, 0}});
        particles.push_back(e, {{0, 0, 0}}, {{-0.25, 0, 0}}, 0, 0, {{0, 0, 0}});

        auto out = closest_points_dispatcher<dim>::dispatch(*particles[0], *particles[1]);

        EXPECT_NEAR(out.dij, 0.2, 1e-7);
    }

}
