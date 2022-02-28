#include <gtest/gtest.h>
#include "utils.hpp"

#include <scopi/objects/types/sphere.hpp>
#include <scopi/objects/types/superellipsoid.hpp>
#include <scopi/container.hpp>
#include <scopi/objects/methods/closest_points.hpp>

namespace scopi
{
    // distance sphere - sphere
    TEST(closest_points, sphere_sphere_2d)
    {
        constexpr std::size_t dim = 2;
        sphere<dim> s1({{-0.2, 0.0}}, 0.1);
        sphere<dim> s2({{ 0.2, 0.0}}, 0.1);

        auto out = closest_points(s1, s2);

        EXPECT_EQ(out.pi(0), -0.1);
        EXPECT_EQ(out.pi(1), 0.);
        EXPECT_EQ(out.pj(0), 0.1);
        EXPECT_EQ(out.pj(1), 0.);
        EXPECT_EQ(out.nij(0), -1.);
        EXPECT_EQ(out.nij(1), 0.);
        EXPECT_EQ(out.dij, 0.2);
    }

    TEST(closest_points, sphere_sphere_2d_rotation_30_deg)
    {
        constexpr std::size_t dim = 2;
        double dist = 0.3;
        double cosRot = std::sqrt(3.)/2.;
        double sinRot = 1./2.;
        sphere<dim> s1({{-dist*cosRot,  dist*sinRot}}, 0.1);
        sphere<dim> s2({{ dist*cosRot, -dist*sinRot}}, 0.1);

        auto out = closest_points(s1, s2);

        EXPECT_DOUBLE_EQ(out.pi(0), -0.2*cosRot);
        EXPECT_DOUBLE_EQ(out.pi(1), 0.2*sinRot);
        EXPECT_DOUBLE_EQ(out.pj(0), 0.2*cosRot);
        EXPECT_DOUBLE_EQ(out.pj(1), -0.2*sinRot);
        EXPECT_DOUBLE_EQ(out.nij(0), -cosRot);
        EXPECT_DOUBLE_EQ(out.nij(1), sinRot);
        EXPECT_DOUBLE_EQ(out.dij, 0.4);
    }

    TEST(closest_points, sphere_sphere_2d_dispatch)
    {
        constexpr std::size_t dim = 2;
        sphere<dim> s1({{-0.2, 0.0}}, 0.1);
        sphere<dim> s2({{ 0.2, 0.0}}, 0.1);

        scopi_container<dim> particles;
        particles.push_back(s1);
        particles.push_back(s2);

        auto out = closest_points_dispatcher<dim>::dispatch(*particles[0], *particles[1]);

        EXPECT_EQ(out.pi(0), -0.1);
        EXPECT_EQ(out.pi(1), 0.);
        EXPECT_EQ(out.pj(0), 0.1);
        EXPECT_EQ(out.pj(1), 0.);
        EXPECT_EQ(out.nij(0), -1.);
        EXPECT_EQ(out.nij(1), 0.);
        EXPECT_EQ(out.dij, 0.2);
    }

    TEST(closest_points, sphere_sphere_2d_dispatch_rotation_30_deg)
    {
        constexpr std::size_t dim = 2;
        double dist = 0.3;
        double cosRot = std::sqrt(3.)/2.;
        double sinRot = 1./2.;
        sphere<dim> s1({{-dist*cosRot,  dist*sinRot}}, 0.1);
        sphere<dim> s2({{ dist*cosRot, -dist*sinRot}}, 0.1);

        scopi_container<dim> particles;
        particles.push_back(s1);
        particles.push_back(s2);

        auto out = closest_points_dispatcher<dim>::dispatch(*particles[0], *particles[1]);

        EXPECT_DOUBLE_EQ(out.pi(0), -0.2*cosRot);
        EXPECT_DOUBLE_EQ(out.pi(1), 0.2*sinRot);
        EXPECT_DOUBLE_EQ(out.pj(0), 0.2*cosRot);
        EXPECT_DOUBLE_EQ(out.pj(1), -0.2*sinRot);
        EXPECT_DOUBLE_EQ(out.nij(0), -cosRot);
        EXPECT_DOUBLE_EQ(out.nij(1), sinRot);
        EXPECT_DOUBLE_EQ(out.dij, 0.4);
    }

    TEST(closest_points, sphere_sphere_3d)
    {
        constexpr std::size_t dim = 3;
        sphere<dim> s1({{-0.2, 0.0, 0.0}}, 0.1);
        sphere<dim> s2({{ 0.2, 0.0, 0.0}}, 0.1);

        auto out = closest_points(s1, s2);

        EXPECT_EQ(out.pi(0), -0.1);
        EXPECT_EQ(out.pi(1), 0.);
        EXPECT_EQ(out.pi(2), 0.);
        EXPECT_EQ(out.pj(0), 0.1);
        EXPECT_EQ(out.pj(1), 0.);
        EXPECT_EQ(out.pj(2), 0.);
        EXPECT_EQ(out.nij(0), -1.);
        EXPECT_EQ(out.nij(1), 0.);
        EXPECT_EQ(out.nij(2), 0.);
        EXPECT_EQ(out.dij, 0.2);
    }

    TEST(closest_points, sphere_sphere_3d_rotation_30_deg)
    {
        constexpr std::size_t dim = 3;
        double dist = 0.3;
        double cosRot = std::sqrt(3.)/2.;
        double sinRot = 1./2.;
        sphere<dim> s1({{-dist*cosRot,  dist*sinRot, 0.}}, 0.1);
        sphere<dim> s2({{ dist*cosRot, -dist*sinRot, 0.}}, 0.1);

        auto out = closest_points(s1, s2);

        EXPECT_DOUBLE_EQ(out.pi(0), -0.2*cosRot);
        EXPECT_DOUBLE_EQ(out.pi(1), 0.2*sinRot);
        EXPECT_DOUBLE_EQ(out.pi(2), 0.);
        EXPECT_DOUBLE_EQ(out.pj(0), 0.2*cosRot);
        EXPECT_DOUBLE_EQ(out.pj(1), -0.2*sinRot);
        EXPECT_DOUBLE_EQ(out.pj(2), 0.);
        EXPECT_DOUBLE_EQ(out.nij(0), -cosRot);
        EXPECT_DOUBLE_EQ(out.nij(1), sinRot);
        EXPECT_DOUBLE_EQ(out.nij(2), 0.);
        EXPECT_DOUBLE_EQ(out.dij, 0.4);
    }

    TEST(closest_points, sphere_sphere_3d_dispatch)
    {
        constexpr std::size_t dim = 3;
        sphere<dim> s1({{-0.2, 0.0, 0.0}}, 0.1);
        sphere<dim> s2({{ 0.2, 0.0, 0.0}}, 0.1);

        scopi_container<dim> particles;
        particles.push_back(s1);
        particles.push_back(s2);

        auto out = closest_points_dispatcher<dim>::dispatch(*particles[0], *particles[1]);

        EXPECT_EQ(out.pi(0), -0.1);
        EXPECT_EQ(out.pi(1), 0.);
        EXPECT_EQ(out.pi(2), 0.);
        EXPECT_EQ(out.pj(0), 0.1);
        EXPECT_EQ(out.pj(1), 0.);
        EXPECT_EQ(out.pj(2), 0.);
        EXPECT_EQ(out.nij(0), -1.);
        EXPECT_EQ(out.nij(1), 0.);
        EXPECT_EQ(out.nij(2), 0.);
        EXPECT_EQ(out.dij, 0.2);
    }

    TEST(closest_points, sphere_sphere_3d_dispatch_rotation_30_deg)
    {
        constexpr std::size_t dim = 3;
        double dist = 0.3;
        double cosRot = std::sqrt(3.)/2.;
        double sinRot = 1./2.;
        sphere<dim> s1({{-dist*cosRot,  dist*sinRot, 0.}}, 0.1);
        sphere<dim> s2({{ dist*cosRot, -dist*sinRot, 0.}}, 0.1);

        scopi_container<dim> particles;
        particles.push_back(s1);
        particles.push_back(s2);

        auto out = closest_points_dispatcher<dim>::dispatch(*particles[0], *particles[1]);

        EXPECT_DOUBLE_EQ(out.pi(0), -0.2*cosRot);
        EXPECT_DOUBLE_EQ(out.pi(1), 0.2*sinRot);
        EXPECT_DOUBLE_EQ(out.pi(2), 0.);
        EXPECT_DOUBLE_EQ(out.pj(0), 0.2*cosRot);
        EXPECT_DOUBLE_EQ(out.pj(1), -0.2*sinRot);
        EXPECT_DOUBLE_EQ(out.pj(2), 0.);
        EXPECT_DOUBLE_EQ(out.nij(0), -cosRot);
        EXPECT_DOUBLE_EQ(out.nij(1), sinRot);
        EXPECT_DOUBLE_EQ(out.nij(2), 0.);
        EXPECT_DOUBLE_EQ(out.dij, 0.4);
    }

    // distance sphere - plan
    TEST(closest_points, sphere_plan_2d)
    {
        constexpr std::size_t dim = 2;
        sphere<dim> s({{0.0, 0.0}}, 0.1);
        plan<dim> p({{ 0.3, 0.0}}, 0.);

        auto out = closest_points(s, p);

        EXPECT_DOUBLE_EQ(out.pi(0), 0.1);
        EXPECT_DOUBLE_EQ(out.pi(1), 0.);
        EXPECT_DOUBLE_EQ(out.pj(0), 0.3);
        EXPECT_DOUBLE_EQ(out.pj(1), 0.);
        EXPECT_DOUBLE_EQ(out.nij(0), -1.);
        EXPECT_DOUBLE_EQ(out.nij(1), 0.);
        EXPECT_DOUBLE_EQ(out.dij, 0.2);
    }

    TEST(closest_points, sphere_plan_2d_rotation_30_deg)
    {
        constexpr std::size_t dim = 2;
        sphere<dim> s({{0.0, 0.0}}, 0.1);
        double dist = 0.3;
        double cosRot = std::sqrt(3.)/2.;
        double sinRot = 1./2.;
        plan<dim> p({{dist*cosRot, dist*sinRot}}, PI/6.);

        auto out = closest_points(s, p);

        EXPECT_DOUBLE_EQ(out.pi(0), 0.1*cosRot);
        EXPECT_DOUBLE_EQ(out.pi(1), 0.1*sinRot);
        EXPECT_DOUBLE_EQ(out.pj(0), 0.3*cosRot);
        EXPECT_DOUBLE_EQ(out.pj(1), 0.3*sinRot);
        EXPECT_DOUBLE_EQ(out.nij(0), -cosRot);
        EXPECT_DOUBLE_EQ(out.nij(1), -sinRot);
        EXPECT_DOUBLE_EQ(out.dij, 0.2);
    }

    TEST(closest_points, sphere_plan_2d_rotation_90_deg)
    {
        constexpr std::size_t dim = 2;
        sphere<dim> s({{0.0, 0.0}}, 0.1);
        plan<dim> p({{0., -0.2}}, PI/2.);

        auto out = closest_points(s, p);

        EXPECT_NEAR(out.pi(0), 0., 1e-7);
        EXPECT_DOUBLE_EQ(out.pi(1), -0.1);
        EXPECT_NEAR(out.pj(0), 0., 1e-7);
        EXPECT_DOUBLE_EQ(out.pj(1), -0.2);
        EXPECT_NEAR(out.nij(0), 0., 1e-7);
        EXPECT_DOUBLE_EQ(out.nij(1), 1.);
        EXPECT_DOUBLE_EQ(out.dij, 0.1);
    }

    TEST(closest_points, sphere_plan_2d_dispatch)
    {
        constexpr std::size_t dim = 2;
        sphere<dim> s({{0.0, 0.0}}, 0.1);
        plan<dim> p({{ 0.3, 0.0}}, 0.);

        scopi_container<dim> particles;
        particles.push_back(s);
        particles.push_back(p);

        auto out = closest_points_dispatcher<dim>::dispatch(*particles[0], *particles[1]);

        EXPECT_DOUBLE_EQ(out.pi(0), 0.1);
        EXPECT_DOUBLE_EQ(out.pi(1), 0.);
        EXPECT_DOUBLE_EQ(out.pj(0), 0.3);
        EXPECT_DOUBLE_EQ(out.pj(1), 0.);
        EXPECT_DOUBLE_EQ(out.nij(0), -1.);
        EXPECT_DOUBLE_EQ(out.nij(1), 0.);
        EXPECT_DOUBLE_EQ(out.dij, 0.2);
    }

    TEST(closest_points, sphere_plan_2d_dispatch_rotation_30_deg)
    {
        constexpr std::size_t dim = 2;
        sphere<dim> s({{0.0, 0.0}}, 0.1);
        double dist = 0.3;
        double cosRot = std::sqrt(3.)/2.;
        double sinRot = 1./2.;
        plan<dim> p({{dist*cosRot, dist*sinRot}}, PI/6.);

        scopi_container<dim> particles;
        particles.push_back(s);
        particles.push_back(p);

        auto out = closest_points_dispatcher<dim>::dispatch(*particles[0], *particles[1]);

        EXPECT_DOUBLE_EQ(out.pi(0), 0.1*cosRot);
        EXPECT_DOUBLE_EQ(out.pi(1), 0.1*sinRot);
        EXPECT_DOUBLE_EQ(out.pj(0), 0.3*cosRot);
        EXPECT_DOUBLE_EQ(out.pj(1), 0.3*sinRot);
        EXPECT_DOUBLE_EQ(out.nij(0), -cosRot);
        EXPECT_DOUBLE_EQ(out.nij(1), -sinRot);
        EXPECT_DOUBLE_EQ(out.dij, 0.2);
    }

    TEST(closest_points, sphere_plan_2d_dispatch_rotation_90_deg)
    {
        constexpr std::size_t dim = 2;
        sphere<dim> s({{0.0, 0.0}}, 0.1);
        plan<dim> p({{0., -0.2}}, PI/2.);

        scopi_container<dim> particles;
        particles.push_back(s);
        particles.push_back(p);

        auto out = closest_points_dispatcher<dim>::dispatch(*particles[0], *particles[1]);

        EXPECT_NEAR(out.pi(0), 0., 1e-7);
        EXPECT_DOUBLE_EQ(out.pi(1), -0.1);
        EXPECT_NEAR(out.pj(0), 0., 1e-7);
        EXPECT_DOUBLE_EQ(out.pj(1), -0.2);
        EXPECT_NEAR(out.nij(0), 0., 1e-7);
        EXPECT_DOUBLE_EQ(out.nij(1), 1.);
        EXPECT_DOUBLE_EQ(out.dij, 0.1);
    }

    TEST(closest_points, sphere_plan_3d)
    {
        constexpr std::size_t dim = 3;
        sphere<dim> s({{0.0, 0.0, 0.0}}, 0.1);
        plan<dim> p({{ 0.3, 0.0, 0.0}}, 0.);

        auto out = closest_points(s, p);

        EXPECT_EQ(out.pi(0), 0.1);
        EXPECT_EQ(out.pi(1), 0.);
        EXPECT_EQ(out.pi(2), 0.);
        EXPECT_EQ(out.pj(0), 0.3);
        EXPECT_EQ(out.pj(1), 0.);
        EXPECT_EQ(out.pj(2), 0.);
        EXPECT_EQ(out.nij(0), -1.);
        EXPECT_EQ(out.nij(1), 0.);
        EXPECT_EQ(out.nij(2), 0.);
        EXPECT_DOUBLE_EQ(out.dij, 0.2);
    }

    TEST(closest_points, sphere_plan_3d_rotation_30_deg)
    {
        constexpr std::size_t dim = 3;
        sphere<dim> s({{0.0, 0.0, 0.0}}, 0.1);
        double dist = 0.3;
        double cosRot = std::sqrt(3.)/2.;
        double sinRot = 1./2.;
        plan<dim> p({{dist*cosRot, dist*sinRot, 0.}}, PI/6.);

        auto out = closest_points(s, p);

        EXPECT_DOUBLE_EQ(out.pi(0), 0.1*cosRot);
        EXPECT_DOUBLE_EQ(out.pi(1), 0.1*sinRot);
        EXPECT_DOUBLE_EQ(out.pi(2), 0.);
        EXPECT_DOUBLE_EQ(out.pj(0), 0.3*cosRot);
        EXPECT_DOUBLE_EQ(out.pj(1), 0.3*sinRot);
        EXPECT_DOUBLE_EQ(out.pj(2), 0.);
        EXPECT_DOUBLE_EQ(out.nij(0), -cosRot);
        EXPECT_DOUBLE_EQ(out.nij(1), -sinRot);
        EXPECT_DOUBLE_EQ(out.nij(2), 0.);
        EXPECT_DOUBLE_EQ(out.dij, 0.2);
    }

    TEST(closest_points, sphere_plan_3d_rotation_90_deg)
    {
        constexpr std::size_t dim = 3;
        sphere<dim> s({{0.0, 0.0, 0,}}, 0.1);
        plan<dim> p({{0., -0.2, 0.}}, PI/2.);

        auto out = closest_points(s, p);

        EXPECT_NEAR(out.pi(0), 0., 1e-7);
        EXPECT_DOUBLE_EQ(out.pi(1), -0.1);
        EXPECT_DOUBLE_EQ(out.pi(2), 0.);
        EXPECT_NEAR(out.pj(0), 0., 1e-7);
        EXPECT_DOUBLE_EQ(out.pj(1), -0.2);
        EXPECT_DOUBLE_EQ(out.pj(2), 0.);
        EXPECT_NEAR(out.nij(0), 0., 1e-7);
        EXPECT_DOUBLE_EQ(out.nij(1), 1.);
        EXPECT_DOUBLE_EQ(out.nij(2), 0.);
        EXPECT_DOUBLE_EQ(out.dij, 0.1);
    }

    TEST(closest_points, sphere_plan_3d_dispatch)
    {
        constexpr std::size_t dim = 3;
        sphere<dim> s({{0.0, 0.0, 0.0}}, 0.1);
        plan<dim> p({{ 0.3, 0.0, 0.0}}, 0.);

        scopi_container<dim> particles;
        particles.push_back(s);
        particles.push_back(p);

        auto out = closest_points_dispatcher<dim>::dispatch(*particles[0], *particles[1]);

        EXPECT_EQ(out.pi(0), 0.1);
        EXPECT_EQ(out.pi(1), 0.);
        EXPECT_EQ(out.pi(2), 0.);
        EXPECT_EQ(out.pj(0), 0.3);
        EXPECT_EQ(out.pj(1), 0.);
        EXPECT_EQ(out.pj(2), 0.);
        EXPECT_EQ(out.nij(0), -1.);
        EXPECT_EQ(out.nij(1), 0.);
        EXPECT_EQ(out.nij(2), 0.);
        EXPECT_DOUBLE_EQ(out.dij, 0.2);
    }

    TEST(closest_points, sphere_plan_3d_dispatch_rotation_30_deg)
    {
        constexpr std::size_t dim = 3;
        sphere<dim> s({{0.0, 0.0, 0.0}}, 0.1);
        double dist = 0.3;
        double cosRot = std::sqrt(3.)/2.;
        double sinRot = 1./2.;
        plan<dim> p({{dist*cosRot, dist*sinRot, 0.}}, PI/6.);

        scopi_container<dim> particles;
        particles.push_back(s);
        particles.push_back(p);

        auto out = closest_points_dispatcher<dim>::dispatch(*particles[0], *particles[1]);

        EXPECT_DOUBLE_EQ(out.pi(0), 0.1*cosRot);
        EXPECT_DOUBLE_EQ(out.pi(1), 0.1*sinRot);
        EXPECT_DOUBLE_EQ(out.pi(2), 0.);
        EXPECT_DOUBLE_EQ(out.pj(0), 0.3*cosRot);
        EXPECT_DOUBLE_EQ(out.pj(1), 0.3*sinRot);
        EXPECT_DOUBLE_EQ(out.pj(2), 0.);
        EXPECT_DOUBLE_EQ(out.nij(0), -cosRot);
        EXPECT_DOUBLE_EQ(out.nij(1), -sinRot);
        EXPECT_DOUBLE_EQ(out.nij(2), 0.);
        EXPECT_DOUBLE_EQ(out.dij, 0.2);
    }

    TEST(closest_points, sphere_plan_3d_dispatch_rotation_90_deg)
    {
        constexpr std::size_t dim = 3;
        sphere<dim> s({{0.0, 0.0, 0,}}, 0.1);
        plan<dim> p({{0., -0.2, 0.}}, PI/2.);

        scopi_container<dim> particles;
        particles.push_back(s);
        particles.push_back(p);

        auto out = closest_points_dispatcher<dim>::dispatch(*particles[0], *particles[1]);

        EXPECT_NEAR(out.pi(0), 0., 1e-7);
        EXPECT_DOUBLE_EQ(out.pi(1), -0.1);
        EXPECT_DOUBLE_EQ(out.pi(2), 0.);
        EXPECT_NEAR(out.pj(0), 0., 1e-7);
        EXPECT_DOUBLE_EQ(out.pj(1), -0.2);
        EXPECT_DOUBLE_EQ(out.pj(2), 0.);
        EXPECT_NEAR(out.nij(0), 0., 1e-7);
        EXPECT_DOUBLE_EQ(out.nij(1), 1.);
        EXPECT_DOUBLE_EQ(out.nij(2), 0.);
        EXPECT_DOUBLE_EQ(out.dij, 0.1);
    }

    // distance plan - sphere
    TEST(closest_points, plan_sphere_2d)
    {
        constexpr std::size_t dim = 2;
        sphere<dim> s({{0.0, 0.0}}, 0.1);
        plan<dim> p({{ 0.3, 0.0}}, 0.);

        auto out = closest_points(p, s);

        EXPECT_DOUBLE_EQ(out.pi(0), 0.3);
        EXPECT_DOUBLE_EQ(out.pi(1), 0.);
        EXPECT_DOUBLE_EQ(out.pj(0), 0.1);
        EXPECT_DOUBLE_EQ(out.pj(1), 0.);
        EXPECT_DOUBLE_EQ(out.nij(0), 1.);
        EXPECT_DOUBLE_EQ(out.nij(1), 0.);
        EXPECT_DOUBLE_EQ(out.dij, 0.2);
    }

    TEST(closest_points, plan_sphere_2d_rotation_30_deg)
    {
        constexpr std::size_t dim = 2;
        sphere<dim> s({{0.0, 0.0}}, 0.1);
        double dist = 0.3;
        double cosRot = std::sqrt(3.)/2.;
        double sinRot = 1./2.;
        plan<dim> p({{dist*cosRot, dist*sinRot}}, PI/6.);

        auto out = closest_points(p, s);

        EXPECT_DOUBLE_EQ(out.pi(0), 0.3*cosRot);
        EXPECT_DOUBLE_EQ(out.pi(1), 0.3*sinRot);
        EXPECT_DOUBLE_EQ(out.pj(0), 0.1*cosRot);
        EXPECT_DOUBLE_EQ(out.pj(1), 0.1*sinRot);
        EXPECT_DOUBLE_EQ(out.nij(0), cosRot);
        EXPECT_DOUBLE_EQ(out.nij(1), sinRot);
        EXPECT_DOUBLE_EQ(out.dij, 0.2);
    }

    TEST(closest_points, plan_sphere_2d_rotation_90_deg)
    {
        constexpr std::size_t dim = 2;
        sphere<dim> s({{0.0, 0.0}}, 0.1);
        plan<dim> p({{0., -0.2}}, PI/2.);

        auto out = closest_points(p, s);

        EXPECT_NEAR(out.pi(0), 0., 1e-7);
        EXPECT_DOUBLE_EQ(out.pi(1), -0.2);
        EXPECT_NEAR(out.pj(0), 0., 1e-7);
        EXPECT_DOUBLE_EQ(out.pj(1), -0.1);
        EXPECT_NEAR(out.nij(0), 0., 1e-7);
        EXPECT_DOUBLE_EQ(out.nij(1), -1.);
        EXPECT_DOUBLE_EQ(out.dij, 0.1);
    }

    TEST(closest_points, plan_sphere_2d_dispatch)
    {
        constexpr std::size_t dim = 2;
        sphere<dim> s({{0.0, 0.0}}, 0.1);
        plan<dim> p({{ 0.3, 0.0}}, 0.);

        scopi_container<dim> particles;
        particles.push_back(p);
        particles.push_back(s);

        auto out = closest_points_dispatcher<dim>::dispatch(*particles[0], *particles[1]);

        EXPECT_DOUBLE_EQ(out.pi(0), 0.3);
        EXPECT_DOUBLE_EQ(out.pi(1), 0.);
        EXPECT_DOUBLE_EQ(out.pj(0), 0.1);
        EXPECT_DOUBLE_EQ(out.pj(1), 0.);
        EXPECT_DOUBLE_EQ(out.nij(0), 1.);
        EXPECT_DOUBLE_EQ(out.nij(1), 0.);
        EXPECT_DOUBLE_EQ(out.dij, 0.2);
    }

    TEST(closest_points, plan_sphere_2d_dispatch_rotation_30_deg)
    {
        constexpr std::size_t dim = 2;
        sphere<dim> s({{0.0, 0.0}}, 0.1);
        double dist = 0.3;
        double cosRot = std::sqrt(3.)/2.;
        double sinRot = 1./2.;
        plan<dim> p({{dist*cosRot, dist*sinRot}}, PI/6.);

        scopi_container<dim> particles;
        particles.push_back(p);
        particles.push_back(s);

        auto out = closest_points_dispatcher<dim>::dispatch(*particles[0], *particles[1]);

        EXPECT_DOUBLE_EQ(out.pi(0), 0.3*cosRot);
        EXPECT_DOUBLE_EQ(out.pi(1), 0.3*sinRot);
        EXPECT_DOUBLE_EQ(out.pj(0), 0.1*cosRot);
        EXPECT_DOUBLE_EQ(out.pj(1), 0.1*sinRot);
        EXPECT_DOUBLE_EQ(out.nij(0), cosRot);
        EXPECT_DOUBLE_EQ(out.nij(1), sinRot);
        EXPECT_DOUBLE_EQ(out.dij, 0.2);
    }

    TEST(closest_points, plan_sphere_2d_dispatch_rotation_90_deg)
    {
        constexpr std::size_t dim = 2;
        sphere<dim> s({{0.0, 0.0}}, 0.1);
        plan<dim> p({{0., -0.2}}, PI/2.);

        scopi_container<dim> particles;
        particles.push_back(p);
        particles.push_back(s);

        auto out = closest_points_dispatcher<dim>::dispatch(*particles[0], *particles[1]);

        EXPECT_NEAR(out.pi(0), 0., 1e-7);
        EXPECT_DOUBLE_EQ(out.pi(1), -0.2);
        EXPECT_NEAR(out.pj(0), 0., 1e-7);
        EXPECT_DOUBLE_EQ(out.pj(1), -0.1);
        EXPECT_NEAR(out.nij(0), 0., 1e-7);
        EXPECT_DOUBLE_EQ(out.nij(1), -1.);
        EXPECT_DOUBLE_EQ(out.dij, 0.1);
    }

    TEST(closest_points, plan_sphere_3d)
    {
        constexpr std::size_t dim = 3;
        sphere<dim> s({{0.0, 0.0, 0.0}}, 0.1);
        plan<dim> p({{ 0.3, 0.0, 0.0}}, 0.);

        auto out = closest_points(p, s);

        EXPECT_EQ(out.pi(0), 0.3);
        EXPECT_EQ(out.pi(1), 0.);
        EXPECT_EQ(out.pi(2), 0.);
        EXPECT_EQ(out.pj(0), 0.1);
        EXPECT_EQ(out.pj(1), 0.);
        EXPECT_EQ(out.pj(2), 0.);
        EXPECT_EQ(out.nij(0), 1.);
        EXPECT_EQ(out.nij(1), 0.);
        EXPECT_EQ(out.nij(2), 0.);
        EXPECT_DOUBLE_EQ(out.dij, 0.2);
    }

    TEST(closest_points, plan_sphere_3d_rotation_30_deg)
    {
        constexpr std::size_t dim = 3;
        sphere<dim> s({{0.0, 0.0, 0.0}}, 0.1);
        double dist = 0.3;
        double cosRot = std::sqrt(3.)/2.;
        double sinRot = 1./2.;
        plan<dim> p({{dist*cosRot, dist*sinRot, 0.}}, PI/6.);

        auto out = closest_points(p, s);

        EXPECT_DOUBLE_EQ(out.pi(0), 0.3*cosRot);
        EXPECT_DOUBLE_EQ(out.pi(1), 0.3*sinRot);
        EXPECT_DOUBLE_EQ(out.pi(2), 0.);
        EXPECT_DOUBLE_EQ(out.pj(0), 0.1*cosRot);
        EXPECT_DOUBLE_EQ(out.pj(1), 0.1*sinRot);
        EXPECT_DOUBLE_EQ(out.pj(2), 0.);
        EXPECT_DOUBLE_EQ(out.nij(0), cosRot);
        EXPECT_DOUBLE_EQ(out.nij(1), sinRot);
        EXPECT_DOUBLE_EQ(out.nij(2), 0.);
        EXPECT_DOUBLE_EQ(out.dij, 0.2);
    }

    TEST(closest_points, plan_sphere_3d_rotation_90_deg)
    {
        constexpr std::size_t dim = 3;
        sphere<dim> s({{0.0, 0.0, 0,}}, 0.1);
        plan<dim> p({{0., -0.2, 0.}}, PI/2.);

        auto out = closest_points(p, s);

        EXPECT_NEAR(out.pi(0), 0., 1e-7);
        EXPECT_DOUBLE_EQ(out.pi(1), -0.2);
        EXPECT_DOUBLE_EQ(out.pi(2), 0.);
        EXPECT_NEAR(out.pj(0), 0., 1e-7);
        EXPECT_DOUBLE_EQ(out.pj(1), -0.1);
        EXPECT_DOUBLE_EQ(out.pj(2), 0.);
        EXPECT_NEAR(out.nij(0), 0., 1e-7);
        EXPECT_DOUBLE_EQ(out.nij(1), -1.);
        EXPECT_DOUBLE_EQ(out.nij(2), 0.);
        EXPECT_DOUBLE_EQ(out.dij, 0.1);
    }

    TEST(closest_points, plan_sphere_3d_dispatch)
    {
        constexpr std::size_t dim = 3;
        sphere<dim> s({{0.0, 0.0, 0.0}}, 0.1);
        plan<dim> p({{ 0.3, 0.0, 0.0}}, 0.);

        scopi_container<dim> particles;
        particles.push_back(p);
        particles.push_back(s);

        auto out = closest_points_dispatcher<dim>::dispatch(*particles[0], *particles[1]);

        EXPECT_EQ(out.pi(0), 0.3);
        EXPECT_EQ(out.pi(1), 0.);
        EXPECT_EQ(out.pi(2), 0.);
        EXPECT_EQ(out.pj(0), 0.1);
        EXPECT_EQ(out.pj(1), 0.);
        EXPECT_EQ(out.pj(2), 0.);
        EXPECT_EQ(out.nij(0), 1.);
        EXPECT_EQ(out.nij(1), 0.);
        EXPECT_EQ(out.nij(2), 0.);
        EXPECT_DOUBLE_EQ(out.dij, 0.2);
    }

    TEST(closest_points, plan_sphere_3d_dispatch_rotation_30_deg)
    {
        constexpr std::size_t dim = 3;
        sphere<dim> s({{0.0, 0.0, 0.0}}, 0.1);
        double dist = 0.3;
        double cosRot = std::sqrt(3.)/2.;
        double sinRot = 1./2.;
        plan<dim> p({{dist*cosRot, dist*sinRot, 0.}}, PI/6.);

        scopi_container<dim> particles;
        particles.push_back(p);
        particles.push_back(s);

        auto out = closest_points_dispatcher<dim>::dispatch(*particles[0], *particles[1]);

        EXPECT_DOUBLE_EQ(out.pi(0), 0.3*cosRot);
        EXPECT_DOUBLE_EQ(out.pi(1), 0.3*sinRot);
        EXPECT_DOUBLE_EQ(out.pi(2), 0.);
        EXPECT_DOUBLE_EQ(out.pj(0), 0.1*cosRot);
        EXPECT_DOUBLE_EQ(out.pj(1), 0.1*sinRot);
        EXPECT_DOUBLE_EQ(out.pj(2), 0.);
        EXPECT_DOUBLE_EQ(out.nij(0), cosRot);
        EXPECT_DOUBLE_EQ(out.nij(1), sinRot);
        EXPECT_DOUBLE_EQ(out.nij(2), 0.);
        EXPECT_DOUBLE_EQ(out.dij, 0.2);
    }

    TEST(closest_points, plan_sphere_3d_dispatch_rotation_90_deg)
    {
        constexpr std::size_t dim = 3;
        sphere<dim> s({{0.0, 0.0, 0,}}, 0.1);
        plan<dim> p({{0., -0.2, 0.}}, PI/2.);

        scopi_container<dim> particles;
        particles.push_back(p);
        particles.push_back(s);

        auto out = closest_points_dispatcher<dim>::dispatch(*particles[0], *particles[1]);

        EXPECT_NEAR(out.pi(0), 0., 1e-7);
        EXPECT_DOUBLE_EQ(out.pi(1), -0.2);
        EXPECT_DOUBLE_EQ(out.pi(2), 0.);
        EXPECT_NEAR(out.pj(0), 0., 1e-7);
        EXPECT_DOUBLE_EQ(out.pj(1), -0.1);
        EXPECT_DOUBLE_EQ(out.pj(2), 0.);
        EXPECT_NEAR(out.nij(0), 0., 1e-7);
        EXPECT_DOUBLE_EQ(out.nij(1), -1.);
        EXPECT_DOUBLE_EQ(out.nij(2), 0.);
        EXPECT_DOUBLE_EQ(out.dij, 0.1);
    }

    // distance sphere - superellipsoid
    TEST(closest_points, sphere_superellipsoid_2d)
    {
        constexpr std::size_t dim = 2;
        sphere<dim> s({{-0.2, 0.0}}, 0.1);
        superellipsoid<dim> e({{0.2, 0.0}}, {quaternion(0.)}, {{0.1, 0.2}}, 1);

        auto out = closest_points(s, e);

        EXPECT_NEAR(out.pi(0), -0.1, 1e-7);
        EXPECT_NEAR(out.pi(1), 0., 1e-5);
        EXPECT_NEAR(out.pj(0), 0.1, 1e-7);
        EXPECT_NEAR(out.pj(1), 0., 1e-5);
        EXPECT_NEAR(out.nij(0), -1., 1e-7);
        EXPECT_NEAR(out.nij(1), 0., 1e-5);
        EXPECT_NEAR(out.dij, 0.2, 1e-7);
    }

    TEST(closest_points, sphere_superellipsoid_2d_rotation_30_deg)
    {
        constexpr std::size_t dim = 2;
        double dist = 0.4;
        double cosRot = std::sqrt(3.)/2.;
        double sinRot = 1./2.;
        sphere<dim> s({{dist*cosRot, -dist*sinRot}}, 0.1);
        superellipsoid<dim> e({{-dist*cosRot, dist*sinRot}}, {quaternion(PI-PI/6.)}, {{0.1, 0.2}}, 1);

        auto out = closest_points(s, e);

        EXPECT_NEAR(out.pi(0), 0.3*cosRot, 1e-5);
        EXPECT_NEAR(out.pi(1), -0.3*sinRot, 1e-5);
        EXPECT_NEAR(out.pj(0), -0.3*cosRot, 1e-5);
        EXPECT_NEAR(out.pj(1), 0.3*sinRot, 1e-5);
        EXPECT_NEAR(out.nij(0), cosRot, 1e-5);
        EXPECT_NEAR(out.nij(1), -sinRot, 1e-5);
        EXPECT_NEAR(out.dij, 0.6, 1e-7);
    }

    TEST(closest_points, sphere_superellipsoid_2d_dispatch)
    {
        constexpr std::size_t dim = 2;
        sphere<dim> s({{-0.2, 0.0}}, 0.1);
        superellipsoid<dim> e({{0.2, 0.0}}, {quaternion(0.)}, {{0.1, 0.2}}, 1);

        scopi_container<dim> particles;
        particles.push_back(s);
        particles.push_back(e);

        auto out = closest_points_dispatcher<dim>::dispatch(*particles[0], *particles[1]);

        EXPECT_NEAR(out.pi(0), -0.1, 1e-7);
        EXPECT_NEAR(out.pi(1), 0., 1e-5);
        EXPECT_NEAR(out.pj(0), 0.1, 1e-7);
        EXPECT_NEAR(out.pj(1), 0., 1e-5);
        EXPECT_NEAR(out.nij(0), -1., 1e-7);
        EXPECT_NEAR(out.nij(1), 0., 1e-5);
        EXPECT_NEAR(out.dij, 0.2, 1e-7);
    }

    TEST(closest_points, sphere_superellipsoid_2d_dispatch_rotation_30_deg)
    {
        constexpr std::size_t dim = 2;
        double dist = 0.4;
        double cosRot = std::sqrt(3.)/2.;
        double sinRot = 1./2.;
        sphere<dim> s({{dist*cosRot, -dist*sinRot}}, 0.1);
        superellipsoid<dim> e({{-dist*cosRot, dist*sinRot}}, {quaternion(PI-PI/6.)}, {{0.1, 0.2}}, 1);

        scopi_container<dim> particles;
        particles.push_back(s);
        particles.push_back(e);

        auto out = closest_points_dispatcher<dim>::dispatch(*particles[0], *particles[1]);

        EXPECT_NEAR(out.pi(0), 0.3*cosRot, 1e-5);
        EXPECT_NEAR(out.pi(1), -0.3*sinRot, 1e-5);
        EXPECT_NEAR(out.pj(0), -0.3*cosRot, 1e-5);
        EXPECT_NEAR(out.pj(1), 0.3*sinRot, 1e-5);
        EXPECT_NEAR(out.nij(0), cosRot, 1e-5);
        EXPECT_NEAR(out.nij(1), -sinRot, 1e-5);
        EXPECT_NEAR(out.dij, 0.6, 1e-7);
    }

    TEST(closest_points, sphere_superellipsoid_3d)
    {
        constexpr std::size_t dim = 3;
        sphere<dim> s({{0.2, 0.0, 0.0}}, 0.1);
        superellipsoid<dim> e({{-0.2, 0.0, 0.0}}, {quaternion(0.)}, {{0.1, 0.2, 0.3}}, {1, 1});

        auto out = closest_points(s, e);

        EXPECT_NEAR(out.pi(0), -0.1, 1e-7);
        EXPECT_NEAR(out.pi(1), 0., 1e-5);
        EXPECT_NEAR(out.pi(2), 0., 1e-4);
        EXPECT_NEAR(out.pj(0), 0.1, 1e-7);
        EXPECT_NEAR(out.pj(1), 0., 1e-4);
        EXPECT_NEAR(out.pj(2), 0., 1e-4);
        EXPECT_NEAR(out.nij(0), -1., 1e-6);
        EXPECT_NEAR(out.nij(1), 0., 1e-3);
        EXPECT_NEAR(out.nij(2), 0., 1e-3);
        EXPECT_NEAR(out.dij, 0.2, 1e-7);
    }

    TEST(closest_points, sphere_superellipsoid_3d_rotation_30_deg)
    {
        // FIXME Newton does not converge
        constexpr std::size_t dim = 3;
        double dist = 0.4;
        double cosRot = std::sqrt(3.)/2.;
        double sinRot = 1./2.;
        sphere<dim> s({{dist*cosRot, -dist*sinRot, 0.}}, 0.1);
        superellipsoid<dim> e({{-dist*cosRot, dist*sinRot, 0.}}, {quaternion(PI-PI/6.)}, {{0.1, 0.2, 0.3}},  {1, 1});

        auto out = closest_points(s, e);

        EXPECT_NEAR(out.pi(0), 0.3*cosRot, 1e-5);
        EXPECT_NEAR(out.pi(1), -0.3*sinRot, 1e-5);
        EXPECT_DOUBLE_EQ(out.pi(2), 0.);
        EXPECT_NEAR(out.pj(0), -0.3*cosRot, 1e-5);
        EXPECT_NEAR(out.pj(1), 0.3*sinRot, 1e-5);
        EXPECT_DOUBLE_EQ(out.pj(2), 0.);
        EXPECT_NEAR(out.nij(0), cosRot, 1e-5);
        EXPECT_NEAR(out.nij(1), -sinRot, 1e-5);
        EXPECT_DOUBLE_EQ(out.nij(2), 0.);
        EXPECT_NEAR(out.dij, 0.6, 1e-7);
    }

    TEST(closest_points, sphere_superellipsoid_3d_dispatch)
    {
        constexpr std::size_t dim = 3;
        sphere<dim> s({{0.2, 0.0, 0.0}}, 0.1);
        superellipsoid<dim> e({{-0.2, 0.0, 0.0}}, {quaternion(0.)}, {{0.1, 0.2, 0.3}},  {1, 1});

        scopi_container<dim> particles;
        particles.push_back(s);
        particles.push_back(e);

        auto out = closest_points_dispatcher<dim>::dispatch(*particles[0], *particles[1]);

        EXPECT_NEAR(out.pi(0), -0.1, 1e-7);
        EXPECT_NEAR(out.pi(1), 0., 1e-5);
        EXPECT_NEAR(out.pi(2), 0., 1e-4);
        EXPECT_NEAR(out.pj(0), 0.1, 1e-7);
        EXPECT_NEAR(out.pj(1), 0., 1e-4);
        EXPECT_NEAR(out.pj(2), 0., 1e-4);
        EXPECT_NEAR(out.nij(0), -1., 1e-6);
        EXPECT_NEAR(out.nij(1), 0., 1e-3);
        EXPECT_NEAR(out.nij(2), 0., 1e-3);
        EXPECT_NEAR(out.dij, 0.2, 1e-7);
    }

    TEST(closest_points, sphere_superellipsoid_3d_dispatch_rotation_30_deg)
    {
        // FIXME Newton does not converge
        constexpr std::size_t dim = 3;
        double dist = 0.4;
        double cosRot = std::sqrt(3.)/2.;
        double sinRot = 1./2.;
        sphere<dim> s({{dist*cosRot, -dist*sinRot, 0.}}, 0.1);
        superellipsoid<dim> e({{-dist*cosRot, dist*sinRot, 0.}}, {quaternion(PI-PI/6.)}, {{0.1, 0.2, 0.3}},  {1, 1});

        scopi_container<dim> particles;
        particles.push_back(s);
        particles.push_back(e);

        auto out = closest_points_dispatcher<dim>::dispatch(*particles[0], *particles[1]);

        EXPECT_NEAR(out.pi(0), 0.3*cosRot, 1e-5);
        EXPECT_NEAR(out.pi(1), -0.3*sinRot, 1e-5);
        EXPECT_DOUBLE_EQ(out.pi(2), 0.);
        EXPECT_NEAR(out.pj(0), -0.3*cosRot, 1e-5);
        EXPECT_NEAR(out.pj(1), 0.3*sinRot, 1e-5);
        EXPECT_DOUBLE_EQ(out.pj(2), 0.);
        EXPECT_NEAR(out.nij(0), cosRot, 1e-5);
        EXPECT_NEAR(out.nij(1), -sinRot, 1e-5);
        EXPECT_DOUBLE_EQ(out.nij(2), 0.);
        EXPECT_NEAR(out.dij, 0.6, 1e-7);
    }

    // distance superellipsoid - sphere
    TEST(closest_points, superellipsoid_sphere_2d)
    {
        constexpr std::size_t dim = 2;
        sphere<dim> s({{-0.2, 0.0}}, 0.1);
        superellipsoid<dim> e({{0.2, 0.0}}, {quaternion(0.)}, {{0.1, 0.2}}, 1);

        auto out = closest_points(e, s);

        EXPECT_NEAR(out.pi(0), 0.1, 1e-7);
        EXPECT_NEAR(out.pi(1), 0., 1e-5);
        EXPECT_NEAR(out.pj(0), -0.1, 1e-7);
        EXPECT_NEAR(out.pj(1), 0., 1e-5);
        EXPECT_NEAR(out.nij(0), 1., 1e-7);
        EXPECT_NEAR(out.nij(1), 0., 1e-5);
        EXPECT_NEAR(out.dij, 0.2, 1e-7);
    }

    TEST(closest_points, superellipsoid_sphere_2d_rotation_30_deg)
    {
        constexpr std::size_t dim = 2;
        double dist = 0.4;
        double cosRot = std::sqrt(3.)/2.;
        double sinRot = 1./2.;
        sphere<dim> s({{dist*cosRot, -dist*sinRot}}, 0.1);
        superellipsoid<dim> e({{-dist*cosRot, dist*sinRot}}, {quaternion(PI-PI/6.)}, {{0.1, 0.2}}, 1);

        auto out = closest_points(e, s);

        EXPECT_NEAR(out.pi(0), -0.3*cosRot, 1e-5);
        EXPECT_NEAR(out.pi(1), 0.3*sinRot, 1e-5);
        EXPECT_NEAR(out.pj(0), 0.3*cosRot, 1e-5);
        EXPECT_NEAR(out.pj(1), -0.3*sinRot, 1e-5);
        EXPECT_NEAR(out.nij(0), -cosRot, 1e-5);
        EXPECT_NEAR(out.nij(1), sinRot, 1e-5);
        EXPECT_NEAR(out.dij, 0.6, 1e-7);
    }

    TEST(closest_points, superellipsoid_sphere_2d_dispatch)
    {
        constexpr std::size_t dim = 2;
        sphere<dim> s({{-0.2, 0.0}}, 0.1);
        superellipsoid<dim> e({{0.2, 0.0}}, {quaternion(0.)}, {{0.1, 0.2}}, 1);

        scopi_container<dim> particles;
        particles.push_back(e);
        particles.push_back(s);

        auto out = closest_points_dispatcher<dim>::dispatch(*particles[0], *particles[1]);

        EXPECT_NEAR(out.pi(0), 0.1, 1e-7);
        EXPECT_NEAR(out.pi(1), 0., 1e-5);
        EXPECT_NEAR(out.pj(0), -0.1, 1e-7);
        EXPECT_NEAR(out.pj(1), 0., 1e-5);
        EXPECT_NEAR(out.nij(0), 1., 1e-7);
        EXPECT_NEAR(out.nij(1), 0., 1e-5);
        EXPECT_NEAR(out.dij, 0.2, 1e-7);
    }

    TEST(closest_points, superellipsoid_sphere_2d_dispatch_rotation_30_deg)
    {
        constexpr std::size_t dim = 2;
        double dist = 0.4;
        double cosRot = std::sqrt(3.)/2.;
        double sinRot = 1./2.;
        sphere<dim> s({{dist*cosRot, -dist*sinRot}}, 0.1);
        superellipsoid<dim> e({{-dist*cosRot, dist*sinRot}}, {quaternion(PI-PI/6.)}, {{0.1, 0.2}}, 1);

        scopi_container<dim> particles;
        auto prop = property<dim>().desired_velocity({{0.25, 0}});
        particles.push_back(e, prop);
        particles.push_back(s, prop.desired_velocity({{-0.25, 0}}));

        auto out = closest_points_dispatcher<dim>::dispatch(*particles[0], *particles[1]);

        EXPECT_NEAR(out.pi(0), -0.3*cosRot, 1e-5);
        EXPECT_NEAR(out.pi(1), 0.3*sinRot, 1e-5);
        EXPECT_NEAR(out.pj(0), 0.3*cosRot, 1e-5);
        EXPECT_NEAR(out.pj(1), -0.3*sinRot, 1e-5);
        EXPECT_NEAR(out.nij(0), -cosRot, 1e-5);
        EXPECT_NEAR(out.nij(1), sinRot, 1e-5);
        EXPECT_NEAR(out.dij, 0.6, 1e-7);
    }

    TEST(closest_points, superellipsoid_sphere_3d)
    {
        constexpr std::size_t dim = 3;
        sphere<dim> s({{0.2, 0.0, 0.0}}, 0.1);
        superellipsoid<dim> e({{-0.2, 0.0, 0.0}}, {quaternion(0.)}, {{0.1, 0.2, 0.3}}, {1, 1});

        auto out = closest_points(e, s);

        EXPECT_NEAR(out.pi(0), 0.1, 1e-7);
        EXPECT_NEAR(out.pi(1), 0., 1e-4);
        EXPECT_NEAR(out.pi(2), 0., 1e-4);
        EXPECT_NEAR(out.pj(0), -0.1, 1e-7);
        EXPECT_NEAR(out.pj(1), 0., 1e-5);
        EXPECT_NEAR(out.pj(2), 0., 1e-4);
        EXPECT_NEAR(out.nij(0), 1., 1e-6);
        EXPECT_NEAR(out.nij(1), 0., 1e-3);
        EXPECT_NEAR(out.nij(2), 0., 1e-3);
        EXPECT_NEAR(out.dij, 0.2, 1e-7);
    }

    TEST(closest_points, superellipsoid_sphere_3d_rotation_30_deg)
    {
        // FIXME Newton does not converge
        constexpr std::size_t dim = 3;
        double dist = 0.4;
        double cosRot = std::sqrt(3.)/2.;
        double sinRot = 1./2.;
        sphere<dim> s({{dist*cosRot, -dist*sinRot, 0.}}, 0.1);
        superellipsoid<dim> e({{-dist*cosRot, dist*sinRot, 0.}}, {quaternion(PI-PI/6.)}, {{0.1, 0.2, 0.3}},  {1, 1});

        auto out = closest_points(e, s);

        EXPECT_NEAR(out.pi(0), 0.3*cosRot, 1e-5);
        EXPECT_NEAR(out.pi(1), -0.3*sinRot, 1e-5);
        EXPECT_DOUBLE_EQ(out.pi(2), 0.);
        EXPECT_NEAR(out.pj(0), -0.3*cosRot, 1e-5);
        EXPECT_NEAR(out.pj(1), 0.3*sinRot, 1e-5);
        EXPECT_DOUBLE_EQ(out.pj(2), 0.);
        EXPECT_NEAR(out.nij(0), cosRot, 1e-5);
        EXPECT_NEAR(out.nij(1), -sinRot, 1e-5);
        EXPECT_DOUBLE_EQ(out.nij(2), 0.);
        EXPECT_NEAR(out.dij, 0.6, 1e-7);
    }

    TEST(closest_points, superellipsoid_sphere_3d_dispatch)
    {
        constexpr std::size_t dim = 3;
        sphere<dim> s({{0.2, 0.0, 0.0}}, 0.1);
        superellipsoid<dim> e({{-0.2, 0.0, 0.0}}, {quaternion(0.)}, {{0.1, 0.2, 0.3}}, {1, 1});

        scopi_container<dim> particles;
        particles.push_back(e);
        particles.push_back(s);

        auto out = closest_points_dispatcher<dim>::dispatch(*particles[0], *particles[1]);

        EXPECT_NEAR(out.pi(0), 0.1, 1e-7);
        EXPECT_NEAR(out.pi(1), 0., 1e-4);
        EXPECT_NEAR(out.pi(2), 0., 1e-4);
        EXPECT_NEAR(out.pj(0), -0.1, 1e-7);
        EXPECT_NEAR(out.pj(1), 0., 1e-5);
        EXPECT_NEAR(out.pj(2), 0., 1e-4);
        EXPECT_NEAR(out.nij(0), 1., 1e-6);
        EXPECT_NEAR(out.nij(1), 0., 1e-3);
        EXPECT_NEAR(out.nij(2), 0., 1e-3);
        EXPECT_NEAR(out.dij, 0.2, 1e-7);
    }

    TEST(closest_points, superellipsoid_sphere_3d_dispatch_rotation_30_deg)
    {
        // FIXME Newton does not converge
        constexpr std::size_t dim = 3;
        double dist = 0.4;
        double cosRot = std::sqrt(3.)/2.;
        double sinRot = 1./2.;
        sphere<dim> s({{dist*cosRot, -dist*sinRot, 0.}}, 0.1);
        superellipsoid<dim> e({{-dist*cosRot, dist*sinRot, 0.}}, {quaternion(PI-PI/6.)}, {{0.1, 0.2, 0.3}},  {1, 1});

        scopi_container<dim> particles;
        particles.push_back(e);
        particles.push_back(s);

        auto out = closest_points_dispatcher<dim>::dispatch(*particles[0], *particles[1]);

        EXPECT_NEAR(out.pi(0), 0.3*cosRot, 1e-5);
        EXPECT_NEAR(out.pi(1), -0.3*sinRot, 1e-5);
        EXPECT_DOUBLE_EQ(out.pi(2), 0.);
        EXPECT_NEAR(out.pj(0), -0.3*cosRot, 1e-5);
        EXPECT_NEAR(out.pj(1), 0.3*sinRot, 1e-5);
        EXPECT_DOUBLE_EQ(out.pj(2), 0.);
        EXPECT_NEAR(out.nij(0), cosRot, 1e-5);
        EXPECT_NEAR(out.nij(1), -sinRot, 1e-5);
        EXPECT_DOUBLE_EQ(out.nij(2), 0.);
        EXPECT_NEAR(out.dij, 0.2, 1e-7);
    }

    // distance superellipsoid - superellipsoid
    TEST(closest_points, superellipsoid_superellipsoid_2d)
    {
        constexpr std::size_t dim = 2;
        superellipsoid<dim> s1({{-0.2, 0.0}}, {quaternion(0.)}, {{0.1, 0.2}}, 1);
        superellipsoid<dim> s2({{ 0.2, 0.0}}, {quaternion(0.)}, {{0.1, 0.3}}, 1);

        auto out = closest_points(s1, s2);

        EXPECT_DOUBLE_EQ(out.pi(0), -0.1);
        EXPECT_NEAR(out.pi(1), 0., 1e-8);
        EXPECT_DOUBLE_EQ(out.pj(0), 0.1);
        EXPECT_NEAR(out.pj(1), 0., 1e-8);
        EXPECT_DOUBLE_EQ(out.nij(0), -1.);
        EXPECT_NEAR(out.nij(1), 0., 1e-8);
        EXPECT_NEAR(out.dij, 0.2, 1e-7);
    }

    TEST(closest_points, superellipsoid_superellipsoid_2d_rotation_30_deg)
    {
        constexpr std::size_t dim = 2;
        double dist = 0.4;
        double cosRot = std::sqrt(3.)/2.;
        double sinRot = 1./2.;
        superellipsoid<dim> s1({{-dist*cosRot,  dist*sinRot}}, {quaternion(-PI/6.)}, {{0.1, 0.4}}, 1);
        superellipsoid<dim> s2({{ dist*cosRot, -dist*sinRot}}, {quaternion(PI-PI/6.)}, {{0.2, 0.3}}, 1);

        auto out = closest_points(s1, s2);

        EXPECT_NEAR(out.pi(0), -0.3*cosRot, 1e-7);
        EXPECT_NEAR(out.pi(1), 0.3*sinRot, 1e-7);
        EXPECT_NEAR(out.pj(0), 0.2*cosRot, 1e-7);
        EXPECT_NEAR(out.pj(1), -0.2*sinRot, 1e-7);
        EXPECT_NEAR(out.nij(0), -cosRot, 1e-7);
        EXPECT_NEAR(out.nij(1), sinRot, 1e-7);
        EXPECT_NEAR(out.dij, 0.5, 1e-7);
    }

    TEST(closest_points, superellipsoid_superellipsoid_2d_dispatch)
    {
        constexpr std::size_t dim = 2;
        superellipsoid<dim> s1({{-0.2, 0.0}}, {quaternion(0.)}, {{0.1, 0.2}}, 1);
        superellipsoid<dim> s2({{ 0.2, 0.0}}, {quaternion(0.)}, {{0.1, 0.3}}, 1);

        scopi_container<dim> particles;
        particles.push_back(s1);
        particles.push_back(s2);

        auto out = closest_points_dispatcher<dim>::dispatch(*particles[0], *particles[1]);

        EXPECT_DOUBLE_EQ(out.pi(0), -0.1);
        EXPECT_NEAR(out.pi(1), 0., 1e-8);
        EXPECT_DOUBLE_EQ(out.pj(0), 0.1);
        EXPECT_NEAR(out.pj(1), 0., 1e-8);
        EXPECT_DOUBLE_EQ(out.nij(0), -1.);
        EXPECT_NEAR(out.nij(1), 0., 1e-8);
        EXPECT_NEAR(out.dij, 0.2, 1e-7);
    }

    TEST(closest_points, superellipsoid_superellipsoid_2d_dispatch_rotation_30_deg)
    {
        constexpr std::size_t dim = 2;
        double dist = 0.4;
        double cosRot = std::sqrt(3.)/2.;
        double sinRot = 1./2.;
        superellipsoid<dim> s1({{-dist*cosRot,  dist*sinRot}}, {quaternion(-PI/6.)}, {{0.1, 0.4}}, 1);
        superellipsoid<dim> s2({{ dist*cosRot, -dist*sinRot}}, {quaternion(PI-PI/6.)}, {{0.2, 0.3}}, 1);

        scopi_container<dim> particles;
        particles.push_back(s1);
        particles.push_back(s2);

        auto out = closest_points_dispatcher<dim>::dispatch(*particles[0], *particles[1]);

        EXPECT_NEAR(out.pi(0), -0.3*cosRot, 1e-7);
        EXPECT_NEAR(out.pi(1), 0.3*sinRot, 1e-7);
        EXPECT_NEAR(out.pj(0), 0.2*cosRot, 1e-7);
        EXPECT_NEAR(out.pj(1), -0.2*sinRot, 1e-7);
        EXPECT_NEAR(out.nij(0), -cosRot, 1e-7);
        EXPECT_NEAR(out.nij(1), sinRot, 1e-7);
        EXPECT_NEAR(out.dij, 0.5, 1e-7);
    }

    TEST(closest_points, superellipsoid_superellipsoid_3d)
    {
        constexpr std::size_t dim = 3;
        superellipsoid<dim> s1({{-0.2, 0.0, 0.0}}, {quaternion(0.)}, {{0.1, 0.2, 0.3}},  {1, 1});
        superellipsoid<dim> s2({{ 0.2, 0.0, 0.0}}, {quaternion(0.)}, {{0.1, 0.2, 0.3}},  {1, 1});

        auto out = closest_points(s1, s2);

        EXPECT_DOUBLE_EQ(out.pi(0), -0.1);
        EXPECT_NEAR(out.pi(1), 0., 1e-8);
        EXPECT_NEAR(out.pi(2), 0., 1e-8);
        EXPECT_DOUBLE_EQ(out.pj(0), 0.1);
        EXPECT_NEAR(out.pj(1), 0., 1e-8);
        EXPECT_NEAR(out.pj(2), 0., 1e-8);
        EXPECT_DOUBLE_EQ(out.nij(0), -1.);
        EXPECT_NEAR(out.nij(1), 0., 1e-8);
        EXPECT_NEAR(out.nij(2), 0., 1e-8);
        EXPECT_NEAR(out.dij, 0.2, 1e-7);
    }

    TEST(closest_points, superellipsoid_superellipsoid_3d_rotation_30_deg)
    {
        constexpr std::size_t dim = 3;
        double dist = 0.4;
        double cosRot = std::sqrt(3.)/2.;
        double sinRot = 1./2.;
        superellipsoid<dim> s1({{-dist*cosRot,  dist*sinRot, 0.}}, {quaternion(-PI/6.)}, {{0.1, 0.4, 0.5}}, {1, 1});
        superellipsoid<dim> s2({{ dist*cosRot, -dist*sinRot, 0.}}, {quaternion(PI-PI/6.)}, {{0.2, 0.3, 0.6}}, {1, 1});

        auto out = closest_points(s1, s2);

        EXPECT_NEAR(out.pi(0), -0.3*cosRot, 1e-7);
        EXPECT_NEAR(out.pi(1), 0.3*sinRot, 1e-7);
        EXPECT_NEAR(out.pi(2), 0., 1e-16);
        EXPECT_NEAR(out.pj(0), 0.2*cosRot, 1e-7);
        EXPECT_NEAR(out.pj(1), -0.2*sinRot, 1e-7);
        EXPECT_NEAR(out.pj(2), 0., 1e-16);
        EXPECT_NEAR(out.nij(0), -cosRot, 1e-7);
        EXPECT_NEAR(out.nij(1), sinRot, 1e-7);
        EXPECT_NEAR(out.nij(2), 0., 1e-16);
        EXPECT_NEAR(out.dij, 0.5, 1e-7);
    }

    TEST(closest_points, superellipsoid_superellipsoid_3d_dispatch)
    {
        constexpr std::size_t dim = 3;
        superellipsoid<dim> s1({{-0.2, 0.0, 0.0}}, {quaternion(0.)}, {{0.1, 0.2, 0.3}},  {1, 1});
        superellipsoid<dim> s2({{ 0.2, 0.0, 0.0}}, {quaternion(0.)}, {{0.1, 0.2, 0.3}},  {1, 1});

        scopi_container<dim> particles;
        particles.push_back(s1);
        particles.push_back(s2);

        auto out = closest_points_dispatcher<dim>::dispatch(*particles[0], *particles[1]);

        EXPECT_DOUBLE_EQ(out.pi(0), -0.1);
        EXPECT_NEAR(out.pi(1), 0., 1e-8);
        EXPECT_NEAR(out.pi(2), 0., 1e-8);
        EXPECT_DOUBLE_EQ(out.pj(0), 0.1);
        EXPECT_NEAR(out.pj(1), 0., 1e-8);
        EXPECT_NEAR(out.pj(2), 0., 1e-8);
        EXPECT_DOUBLE_EQ(out.nij(0), -1.);
        EXPECT_NEAR(out.nij(1), 0., 1e-8);
        EXPECT_NEAR(out.nij(2), 0., 1e-8);
        EXPECT_NEAR(out.dij, 0.2, 1e-7);
    }

    TEST(closest_points, superellipsoid_superellipsoid_3d_dispatch_rotation_30_deg)
    {
        constexpr std::size_t dim = 3;
        double dist = 0.4;
        double cosRot = std::sqrt(3.)/2.;
        double sinRot = 1./2.;
        superellipsoid<dim> s1({{-dist*cosRot,  dist*sinRot, 0.}}, {quaternion(-PI/6.)}, {{0.1, 0.4, 0.5}}, {1, 1});
        superellipsoid<dim> s2({{ dist*cosRot, -dist*sinRot, 0.}}, {quaternion(PI-PI/6.)}, {{0.2, 0.3, 0.6}}, {1, 1});

        scopi_container<dim> particles;
        particles.push_back(s1);
        particles.push_back(s2);

        auto out = closest_points_dispatcher<dim>::dispatch(*particles[0], *particles[1]);

        EXPECT_NEAR(out.pi(0), -0.3*cosRot, 1e-7);
        EXPECT_NEAR(out.pi(1), 0.3*sinRot, 1e-7);
        EXPECT_NEAR(out.pi(2), 0., 1e-16);
        EXPECT_NEAR(out.pj(0), 0.2*cosRot, 1e-7);
        EXPECT_NEAR(out.pj(1), -0.2*sinRot, 1e-7);
        EXPECT_NEAR(out.pj(2), 0., 1e-16);
        EXPECT_NEAR(out.nij(0), -cosRot, 1e-7);
        EXPECT_NEAR(out.nij(1), sinRot, 1e-7);
        EXPECT_NEAR(out.nij(2), 0., 1e-16);
        EXPECT_NEAR(out.dij, 0.5, 1e-7);
    }

    // distance superellipsoid - plan
    TEST(closest_points, superellipsoid_plan_2d)
    {
        // FIXME Newton does not converge
        constexpr std::size_t dim = 2;
        superellipsoid<dim> s({{ 0.0, 0.0}}, {quaternion(0.)}, {{0.1, 0.2}}, 1);
        plan<dim> p({{ 0.3, 0.0}}, 0.);

        auto out = closest_points(s, p);

        EXPECT_DOUBLE_EQ(out.pi(0), 0.1);
        EXPECT_DOUBLE_EQ(out.pi(1), 0.);
        EXPECT_DOUBLE_EQ(out.pj(0), 0.3);
        EXPECT_DOUBLE_EQ(out.pj(1), 0.);
        EXPECT_DOUBLE_EQ(out.nij(0), -1.);
        EXPECT_DOUBLE_EQ(out.nij(1), 0.);
        EXPECT_DOUBLE_EQ(out.dij, 0.2);
    }

    TEST(closest_points, superellipsoid_plan_2d_rotation_30_deg)
    {
        // FIXME Newton does not converge
        constexpr std::size_t dim = 2;
        superellipsoid<dim> s({{ 0., 0.}}, {quaternion(0.)}, {{0.1, 0.2}}, 1);
        double dist = 0.3;
        double cosRot = std::sqrt(3.)/2.;
        double sinRot = 1./2.;
        plan<dim> p({{dist*cosRot, dist*sinRot}}, PI/6.);

        auto out = closest_points(s, p);

        EXPECT_DOUBLE_EQ(out.pi(0), 0.1*cosRot);
        EXPECT_DOUBLE_EQ(out.pi(1), 0.1*sinRot);
        EXPECT_DOUBLE_EQ(out.pj(0), 0.3*cosRot);
        EXPECT_DOUBLE_EQ(out.pj(1), 0.3*sinRot);
        EXPECT_DOUBLE_EQ(out.nij(0), -cosRot);
        EXPECT_DOUBLE_EQ(out.nij(1), -sinRot);
        EXPECT_DOUBLE_EQ(out.dij, 0.2);
    }

    TEST(closest_points, superellipsoid_plan_2d_dispatch)
    {
        // FIXME Newton does not converge
        constexpr std::size_t dim = 2;
        superellipsoid<dim> s({{ 0.0, 0.0}}, {quaternion(0.)}, {{0.1, 0.2}}, 1);
        plan<dim> p({{ 0.3, 0.0}}, 0.);

        scopi_container<dim> particles;
        particles.push_back(s);
        particles.push_back(p);

        auto out = closest_points_dispatcher<dim>::dispatch(*particles[0], *particles[1]);

        EXPECT_DOUBLE_EQ(out.pi(0), 0.1);
        EXPECT_DOUBLE_EQ(out.pi(1), 0.);
        EXPECT_DOUBLE_EQ(out.pj(0), 0.3);
        EXPECT_DOUBLE_EQ(out.pj(1), 0.);
        EXPECT_DOUBLE_EQ(out.nij(0), -1.);
        EXPECT_DOUBLE_EQ(out.nij(1), 0.);
        EXPECT_DOUBLE_EQ(out.dij, 0.2);
    }

    TEST(closest_points, superellipsoid_plan_2d_dispatch_rotation_30_deg)
    {
        // FIXME Newton does not converge
        constexpr std::size_t dim = 2;
        superellipsoid<dim> s({{ 0., 0.}}, {quaternion(0.)}, {{0.1, 0.2}}, 1);
        double dist = 0.3;
        double cosRot = std::sqrt(3.)/2.;
        double sinRot = 1./2.;
        plan<dim> p({{dist*cosRot, dist*sinRot}}, PI/6.);

        scopi_container<dim> particles;
        particles.push_back(s);
        particles.push_back(p);

        auto out = closest_points_dispatcher<dim>::dispatch(*particles[0], *particles[1]);

        EXPECT_DOUBLE_EQ(out.pi(0), 0.1*cosRot);
        EXPECT_DOUBLE_EQ(out.pi(1), 0.1*sinRot);
        EXPECT_DOUBLE_EQ(out.pj(0), 0.3*cosRot);
        EXPECT_DOUBLE_EQ(out.pj(1), 0.3*sinRot);
        EXPECT_DOUBLE_EQ(out.nij(0), -cosRot);
        EXPECT_DOUBLE_EQ(out.nij(1), -sinRot);
        EXPECT_DOUBLE_EQ(out.dij, 0.2);
    }

    TEST(closest_points, superellipsoid_plan_3d)
    {
        // FIXME Newton does not converge
        constexpr std::size_t dim = 3;
        superellipsoid<dim> s({{ 0.0, 0.0, 0.0}}, {quaternion(0.)}, {{0.1, 0.2, 0.3}}, {1, 1});
        plan<dim> p({{ 0.3, 0.0, 0.0}}, 0.);

        auto out = closest_points(s, p);

        EXPECT_DOUBLE_EQ(out.pi(0), 0.1);
        EXPECT_DOUBLE_EQ(out.pi(1), 0.);
        EXPECT_DOUBLE_EQ(out.pi(2), 0.);
        EXPECT_DOUBLE_EQ(out.pj(0), 0.3);
        EXPECT_DOUBLE_EQ(out.pj(1), 0.);
        EXPECT_DOUBLE_EQ(out.pj(2), 0.);
        EXPECT_DOUBLE_EQ(out.nij(0), -1.);
        EXPECT_DOUBLE_EQ(out.nij(1), 0.);
        EXPECT_DOUBLE_EQ(out.nij(2), 0.);
        EXPECT_DOUBLE_EQ(out.dij, 0.2);
    }

    TEST(closest_points, superellipsoid_plan_3d_rotation_30_deg)
    {
        // FIXME Newton does not converge
        constexpr std::size_t dim = 3;
        superellipsoid<dim> s({{ 0., 0., 0.}}, {quaternion(0.)}, {{0.1, 0.2, 0.3}}, {1, 1});
        double dist = 0.3;
        double cosRot = std::sqrt(3.)/2.;
        double sinRot = 1./2.;
        plan<dim> p({{dist*cosRot, dist*sinRot, 0.}}, PI/6.);

        auto out = closest_points(s, p);

        EXPECT_DOUBLE_EQ(out.pi(0), 0.1*cosRot);
        EXPECT_DOUBLE_EQ(out.pi(1), 0.1*sinRot);
        EXPECT_DOUBLE_EQ(out.pi(2), 0.);
        EXPECT_DOUBLE_EQ(out.pj(0), 0.3*cosRot);
        EXPECT_DOUBLE_EQ(out.pj(1), 0.3*sinRot);
        EXPECT_DOUBLE_EQ(out.pj(2), 0.);
        EXPECT_DOUBLE_EQ(out.nij(0), -cosRot);
        EXPECT_DOUBLE_EQ(out.nij(1), -sinRot);
        EXPECT_DOUBLE_EQ(out.nij(2), 0.);
        EXPECT_DOUBLE_EQ(out.dij, 0.2);
    }

    TEST(closest_points, superellipsoid_plan_3d_dispatch)
    {
        // FIXME Newton does not converge
        constexpr std::size_t dim = 3;
        superellipsoid<dim> s({{ 0.0, 0.0, 0.0}}, {quaternion(0.)}, {{0.1, 0.2, 0.3}}, {1, 1});
        plan<dim> p({{ 0.3, 0.0, 0.0}}, 0.);

        scopi_container<dim> particles;
        particles.push_back(s);
        particles.push_back(p);

        auto out = closest_points_dispatcher<dim>::dispatch(*particles[0], *particles[1]);

        EXPECT_DOUBLE_EQ(out.pi(0), 0.1);
        EXPECT_DOUBLE_EQ(out.pi(1), 0.);
        EXPECT_DOUBLE_EQ(out.pi(2), 0.);
        EXPECT_DOUBLE_EQ(out.pj(0), 0.3);
        EXPECT_DOUBLE_EQ(out.pj(1), 0.);
        EXPECT_DOUBLE_EQ(out.pj(2), 0.);
        EXPECT_DOUBLE_EQ(out.nij(0), -1.);
        EXPECT_DOUBLE_EQ(out.nij(1), 0.);
        EXPECT_DOUBLE_EQ(out.nij(2), 0.);
        EXPECT_DOUBLE_EQ(out.dij, 0.2);
    }

    TEST(closest_points, superellipsoid_plan_3d_dispatch_rotation_30_deg)
    {
        // FIXME Newton does not converge
        constexpr std::size_t dim = 3;
        superellipsoid<dim> s({{ 0., 0., 0.}}, {quaternion(0.)}, {{0.1, 0.2, 0.3}}, {1, 1});
        double dist = 0.3;
        double cosRot = std::sqrt(3.)/2.;
        double sinRot = 1./2.;
        plan<dim> p({{dist*cosRot, dist*sinRot, 0.}}, PI/6.);

        scopi_container<dim> particles;
        particles.push_back(s);
        particles.push_back(p);

        auto out = closest_points_dispatcher<dim>::dispatch(*particles[0], *particles[1]);

        EXPECT_DOUBLE_EQ(out.pi(0), 0.1*cosRot);
        EXPECT_DOUBLE_EQ(out.pi(1), 0.1*sinRot);
        EXPECT_DOUBLE_EQ(out.pi(2), 0.);
        EXPECT_DOUBLE_EQ(out.pj(0), 0.3*cosRot);
        EXPECT_DOUBLE_EQ(out.pj(1), 0.3*sinRot);
        EXPECT_DOUBLE_EQ(out.pj(2), 0.);
        EXPECT_DOUBLE_EQ(out.nij(0), -cosRot);
        EXPECT_DOUBLE_EQ(out.nij(1), -sinRot);
        EXPECT_DOUBLE_EQ(out.nij(2), 0.);
        EXPECT_DOUBLE_EQ(out.dij, 0.2);
    }

    // distance plan - superellipsoid
    TEST(closest_points, plan_superellipsoid_2d)
    {
        // FIXME Newton does not converge
        constexpr std::size_t dim = 2;
        superellipsoid<dim> s({{ 0.0, 0.0}}, {quaternion(0.)}, {{0.1, 0.2}}, 1);
        plan<dim> p({{ 0.3, 0.0}}, 0.);

        auto out = closest_points(p, s);

        EXPECT_DOUBLE_EQ(out.pi(0), 0.1);
        EXPECT_DOUBLE_EQ(out.pi(1), 0.);
        EXPECT_DOUBLE_EQ(out.pj(0), 0.3);
        EXPECT_DOUBLE_EQ(out.pj(1), 0.);
        EXPECT_DOUBLE_EQ(out.nij(0), -1.);
        EXPECT_DOUBLE_EQ(out.nij(1), 0.);
        EXPECT_DOUBLE_EQ(out.dij, 0.2);
    }

    TEST(closest_points, plan_superellipsoid_2d_rotation_30_deg)
    {
        // FIXME Newton does not converge
        constexpr std::size_t dim = 2;
        superellipsoid<dim> s({{ 0., 0.}}, {quaternion(0.)}, {{0.1, 0.2}}, 1);
        double dist = 0.3;
        double cosRot = std::sqrt(3.)/2.;
        double sinRot = 1./2.;
        plan<dim> p({{dist*cosRot, dist*sinRot}}, PI/6.);

        auto out = closest_points(p, s);

        EXPECT_DOUBLE_EQ(out.pi(0), 0.1*cosRot);
        EXPECT_DOUBLE_EQ(out.pi(1), 0.1*sinRot);
        EXPECT_DOUBLE_EQ(out.pj(0), 0.3*cosRot);
        EXPECT_DOUBLE_EQ(out.pj(1), 0.3*sinRot);
        EXPECT_DOUBLE_EQ(out.nij(0), -cosRot);
        EXPECT_DOUBLE_EQ(out.nij(1), -sinRot);
        EXPECT_DOUBLE_EQ(out.dij, 0.2);
    }

    TEST(closest_points, plan_superellipsoid_2d_dispatch)
    {
        // FIXME Newton does not converge
        constexpr std::size_t dim = 2;
        superellipsoid<dim> s({{ 0.0, 0.0}}, {quaternion(0.)}, {{0.1, 0.2}}, 1);
        plan<dim> p({{ 0.3, 0.0}}, 0.);

        scopi_container<dim> particles;
        particles.push_back(p);
        particles.push_back(s);

        auto out = closest_points_dispatcher<dim>::dispatch(*particles[0], *particles[1]);

        EXPECT_DOUBLE_EQ(out.pi(0), 0.1);
        EXPECT_DOUBLE_EQ(out.pi(1), 0.);
        EXPECT_DOUBLE_EQ(out.pj(0), 0.3);
        EXPECT_DOUBLE_EQ(out.pj(1), 0.);
        EXPECT_DOUBLE_EQ(out.nij(0), -1.);
        EXPECT_DOUBLE_EQ(out.nij(1), 0.);
        EXPECT_DOUBLE_EQ(out.dij, 0.2);
    }

    TEST(closest_points, plan_superellipsoid_2d_dispatch_rotation_30_deg)
    {
        // FIXME Newton does not converge
        constexpr std::size_t dim = 2;
        superellipsoid<dim> s({{ 0., 0.}}, {quaternion(0.)}, {{0.1, 0.2}}, 1);
        double dist = 0.3;
        double cosRot = std::sqrt(3.)/2.;
        double sinRot = 1./2.;
        plan<dim> p({{dist*cosRot, dist*sinRot}}, PI/6.);

        scopi_container<dim> particles;
        particles.push_back(p);
        particles.push_back(s);

        auto out = closest_points_dispatcher<dim>::dispatch(*particles[0], *particles[1]);

        EXPECT_DOUBLE_EQ(out.pi(0), 0.1*cosRot);
        EXPECT_DOUBLE_EQ(out.pi(1), 0.1*sinRot);
        EXPECT_DOUBLE_EQ(out.pj(0), 0.3*cosRot);
        EXPECT_DOUBLE_EQ(out.pj(1), 0.3*sinRot);
        EXPECT_DOUBLE_EQ(out.nij(0), -cosRot);
        EXPECT_DOUBLE_EQ(out.nij(1), -sinRot);
        EXPECT_DOUBLE_EQ(out.dij, 0.2);
    }

    TEST(closest_points, plan_superellipsoid_3d)
    {
        // FIXME Newton does not converge
        constexpr std::size_t dim = 3;
        superellipsoid<dim> s({{ 0.0, 0.0, 0.0}}, {quaternion(0.)}, {{0.1, 0.2, 0.3}}, {1, 1});
        plan<dim> p({{ 0.3, 0.0, 0.0}}, 0.);

        auto out = closest_points(p, s);

        EXPECT_DOUBLE_EQ(out.pi(0), 0.1);
        EXPECT_DOUBLE_EQ(out.pi(1), 0.);
        EXPECT_DOUBLE_EQ(out.pi(2), 0.);
        EXPECT_DOUBLE_EQ(out.pj(0), 0.3);
        EXPECT_DOUBLE_EQ(out.pj(1), 0.);
        EXPECT_DOUBLE_EQ(out.pj(2), 0.);
        EXPECT_DOUBLE_EQ(out.nij(0), -1.);
        EXPECT_DOUBLE_EQ(out.nij(1), 0.);
        EXPECT_DOUBLE_EQ(out.nij(2), 0.);
        EXPECT_DOUBLE_EQ(out.dij, 0.2);
    }

    TEST(closest_points, plan_superellipsoid_3d_rotation_30_deg)
    {
        // FIXME Newton does not converge
        constexpr std::size_t dim = 3;
        superellipsoid<dim> s({{ 0., 0., 0.}}, {quaternion(0.)}, {{0.1, 0.2, 0.3}}, {1, 1});
        double dist = 0.3;
        double cosRot = std::sqrt(3.)/2.;
        double sinRot = 1./2.;
        plan<dim> p({{dist*cosRot, dist*sinRot, 0.}}, PI/6.);

        auto out = closest_points(p, s);

        EXPECT_DOUBLE_EQ(out.pi(0), 0.1*cosRot);
        EXPECT_DOUBLE_EQ(out.pi(1), 0.1*sinRot);
        EXPECT_DOUBLE_EQ(out.pi(2), 0.);
        EXPECT_DOUBLE_EQ(out.pj(0), 0.3*cosRot);
        EXPECT_DOUBLE_EQ(out.pj(1), 0.3*sinRot);
        EXPECT_DOUBLE_EQ(out.pj(2), 0.);
        EXPECT_DOUBLE_EQ(out.nij(0), -cosRot);
        EXPECT_DOUBLE_EQ(out.nij(1), -sinRot);
        EXPECT_DOUBLE_EQ(out.nij(2), 0.);
        EXPECT_DOUBLE_EQ(out.dij, 0.2);
    }

    TEST(closest_points, plan_superellipsoid_3d_dispatch)
    {
        // FIXME Newton does not converge
        constexpr std::size_t dim = 3;
        superellipsoid<dim> s({{ 0.0, 0.0, 0.0}}, {quaternion(0.)}, {{0.1, 0.2, 0.3}}, {1, 1});
        plan<dim> p({{ 0.3, 0.0, 0.0}}, 0.);

        scopi_container<dim> particles;
        particles.push_back(p);
        particles.push_back(s);

        auto out = closest_points_dispatcher<dim>::dispatch(*particles[0], *particles[1]);

        EXPECT_DOUBLE_EQ(out.pi(0), 0.1);
        EXPECT_DOUBLE_EQ(out.pi(1), 0.);
        EXPECT_DOUBLE_EQ(out.pi(2), 0.);
        EXPECT_DOUBLE_EQ(out.pj(0), 0.3);
        EXPECT_DOUBLE_EQ(out.pj(1), 0.);
        EXPECT_DOUBLE_EQ(out.pj(2), 0.);
        EXPECT_DOUBLE_EQ(out.nij(0), -1.);
        EXPECT_DOUBLE_EQ(out.nij(1), 0.);
        EXPECT_DOUBLE_EQ(out.nij(2), 0.);
        EXPECT_DOUBLE_EQ(out.dij, 0.2);
    }

    TEST(closest_points, plan_superellipsoid_3d_dispatch_rotation_30_deg)
    {
        // FIXME Newton does not converge
        constexpr std::size_t dim = 3;
        superellipsoid<dim> s({{ 0., 0., 0.}}, {quaternion(0.)}, {{0.1, 0.2, 0.3}}, {1, 1});
        double dist = 0.3;
        double cosRot = std::sqrt(3.)/2.;
        double sinRot = 1./2.;
        plan<dim> p({{dist*cosRot, dist*sinRot, 0.}}, PI/6.);

        scopi_container<dim> particles;
        particles.push_back(p);
        particles.push_back(s);

        auto out = closest_points_dispatcher<dim>::dispatch(*particles[0], *particles[1]);

        EXPECT_DOUBLE_EQ(out.pi(0), 0.1*cosRot);
        EXPECT_DOUBLE_EQ(out.pi(1), 0.1*sinRot);
        EXPECT_DOUBLE_EQ(out.pi(2), 0.);
        EXPECT_DOUBLE_EQ(out.pj(0), 0.3*cosRot);
        EXPECT_DOUBLE_EQ(out.pj(1), 0.3*sinRot);
        EXPECT_DOUBLE_EQ(out.pj(2), 0.);
        EXPECT_DOUBLE_EQ(out.nij(0), -cosRot);
        EXPECT_DOUBLE_EQ(out.nij(1), -sinRot);
        EXPECT_DOUBLE_EQ(out.nij(2), 0.);
        EXPECT_DOUBLE_EQ(out.dij, 0.2);
    }
}
