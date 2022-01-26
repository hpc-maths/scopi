#include <gtest/gtest.h>

#include <scopi/objects/types/sphere.hpp>
#include <scopi/container.hpp>
#include <scopi/objects/methods/closest_points.hpp>

namespace scopi
{
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
