#include "doctest/doctest.h"
#include "utils.hpp"

#include <scopi/objects/types/sphere.hpp>
#include <scopi/objects/types/superellipsoid.hpp>
#include <scopi/container.hpp>
#include <scopi/objects/methods/closest_points.hpp>

namespace scopi
{
    // distance sphere - sphere
    TEST_CASE("sphere_sphere_2d")
    {
        constexpr std::size_t dim = 2;
        sphere<dim> s1({{-0.2, 0.0}}, 0.1);
        sphere<dim> s2({{ 0.2, 0.0}}, 0.1);

        auto out = closest_points(s1, s2)[0];

        REQUIRE(out.pi(0) == doctest::Approx(-0.1));
        REQUIRE(out.pi(1) == doctest::Approx(0.));
        REQUIRE(out.pj(0) == doctest::Approx(0.1));
        REQUIRE(out.pj(1) == doctest::Approx(0.));
        REQUIRE(out.nij(0) == doctest::Approx(-1.));
        REQUIRE(out.nij(1) == doctest::Approx(0.));
        REQUIRE(out.dij == doctest::Approx(0.2));
    }

    TEST_CASE("sphere_sphere_2d_rotation_30_deg")
    {
        constexpr std::size_t dim = 2;
        double dist = 0.3;
        double cosRot = std::sqrt(3.)/2.;
        double sinRot = 1./2.;
        sphere<dim> s1({{-dist*cosRot,  dist*sinRot}}, 0.1);
        sphere<dim> s2({{ dist*cosRot, -dist*sinRot}}, 0.1);

        auto out = closest_points(s1, s2)[0];

        REQUIRE(out.pi(0) == doctest::Approx(-0.2*cosRot));
        REQUIRE(out.pi(1) == doctest::Approx(0.2*sinRot));
        REQUIRE(out.pj(0) == doctest::Approx(0.2*cosRot));
        REQUIRE(out.pj(1) == doctest::Approx(-0.2*sinRot));
        REQUIRE(out.nij(0) == doctest::Approx(-cosRot));
        REQUIRE(out.nij(1) == doctest::Approx(sinRot));
        REQUIRE(out.dij == doctest::Approx(0.4));
    }

    TEST_CASE("sphere_sphere_2d_dispatch")
    {
        constexpr std::size_t dim = 2;
        sphere<dim> s1({{-0.2, 0.0}}, 0.1);
        sphere<dim> s2({{ 0.2, 0.0}}, 0.1);

        scopi_container<dim> particles;
        particles.push_back(s1);
        particles.push_back(s2);

        auto out = closest_points_dispatcher<dim>::dispatch(*particles[0], *particles[1])[0];

        REQUIRE(out.pi(0) == doctest::Approx(-0.1));
        REQUIRE(out.pi(1) == doctest::Approx(0.));
        REQUIRE(out.pj(0) == doctest::Approx(0.1));
        REQUIRE(out.pj(1) == doctest::Approx(0.));
        REQUIRE(out.nij(0) == doctest::Approx(-1.));
        REQUIRE(out.nij(1) == doctest::Approx(0.));
        REQUIRE(out.dij == doctest::Approx(0.2));
    }

    TEST_CASE("sphere_sphere_2d_dispatch_rotation_30_deg")
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

        auto out = closest_points_dispatcher<dim>::dispatch(*particles[0], *particles[1])[0];

        REQUIRE(out.pi(0) == doctest::Approx(-0.2*cosRot));
        REQUIRE(out.pi(1) == doctest::Approx(0.2*sinRot));
        REQUIRE(out.pj(0) == doctest::Approx(0.2*cosRot));
        REQUIRE(out.pj(1) == doctest::Approx(-0.2*sinRot));
        REQUIRE(out.nij(0) == doctest::Approx(-cosRot));
        REQUIRE(out.nij(1) == doctest::Approx(sinRot));
        REQUIRE(out.dij == doctest::Approx(0.4));
    }

    TEST_CASE("sphere_sphere_3d")
    {
        constexpr std::size_t dim = 3;
        sphere<dim> s1({{-0.2, 0.0, 0.0}}, 0.1);
        sphere<dim> s2({{ 0.2, 0.0, 0.0}}, 0.1);

        auto out = closest_points(s1, s2)[0];

        REQUIRE(out.pi(0) == doctest::Approx(-0.1));
        REQUIRE(out.pi(1) == doctest::Approx(0.));
        REQUIRE(out.pi(2) == doctest::Approx(0.));
        REQUIRE(out.pj(0) == doctest::Approx(0.1));
        REQUIRE(out.pj(1) == doctest::Approx(0.));
        REQUIRE(out.pj(2) == doctest::Approx(0.));
        REQUIRE(out.nij(0) == doctest::Approx(-1.));
        REQUIRE(out.nij(1) == doctest::Approx(0.));
        REQUIRE(out.nij(2) == doctest::Approx(0.));
        REQUIRE(out.dij == doctest::Approx(0.2));
    }

    TEST_CASE("sphere_sphere_3d_rotation_30_deg")
    {
        constexpr std::size_t dim = 3;
        double dist = 0.3;
        double cosRot = std::sqrt(3.)/2.;
        double sinRot = 1./2.;
        sphere<dim> s1({{-dist*cosRot,  dist*sinRot, 0.}}, 0.1);
        sphere<dim> s2({{ dist*cosRot, -dist*sinRot, 0.}}, 0.1);

        auto out = closest_points(s1, s2)[0];

        REQUIRE(out.pi(0) == doctest::Approx(-0.2*cosRot));
        REQUIRE(out.pi(1) == doctest::Approx(0.2*sinRot));
        REQUIRE(out.pi(2) == doctest::Approx(0.));
        REQUIRE(out.pj(0) == doctest::Approx(0.2*cosRot));
        REQUIRE(out.pj(1) == doctest::Approx(-0.2*sinRot));
        REQUIRE(out.pj(2) == doctest::Approx(0.));
        REQUIRE(out.nij(0) == doctest::Approx(-cosRot));
        REQUIRE(out.nij(1) == doctest::Approx(sinRot));
        REQUIRE(out.nij(2) == doctest::Approx(0.));
        REQUIRE(out.dij == doctest::Approx(0.4));
    }

    TEST_CASE("sphere_sphere_3d_dispatch")
    {
        constexpr std::size_t dim = 3;
        sphere<dim> s1({{-0.2, 0.0, 0.0}}, 0.1);
        sphere<dim> s2({{ 0.2, 0.0, 0.0}}, 0.1);

        scopi_container<dim> particles;
        particles.push_back(s1);
        particles.push_back(s2);

        auto out = closest_points_dispatcher<dim>::dispatch(*particles[0], *particles[1])[0];

        REQUIRE(out.pi(0) == doctest::Approx(-0.1));
        REQUIRE(out.pi(1) == doctest::Approx(0.));
        REQUIRE(out.pi(2) == doctest::Approx(0.));
        REQUIRE(out.pj(0) == doctest::Approx(0.1));
        REQUIRE(out.pj(1) == doctest::Approx(0.));
        REQUIRE(out.pj(2) == doctest::Approx(0.));
        REQUIRE(out.nij(0) == doctest::Approx(-1.));
        REQUIRE(out.nij(1) == doctest::Approx(0.));
        REQUIRE(out.nij(2) == doctest::Approx(0.));
        REQUIRE(out.dij == doctest::Approx(0.2));
    }

    TEST_CASE("sphere_sphere_3d_dispatch_rotation_30_deg")
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

        auto out = closest_points_dispatcher<dim>::dispatch(*particles[0], *particles[1])[0];

        REQUIRE(out.pi(0) == doctest::Approx(-0.2*cosRot));
        REQUIRE(out.pi(1) == doctest::Approx(0.2*sinRot));
        REQUIRE(out.pi(2) == doctest::Approx(0.));
        REQUIRE(out.pj(0) == doctest::Approx(0.2*cosRot));
        REQUIRE(out.pj(1) == doctest::Approx(-0.2*sinRot));
        REQUIRE(out.pj(2) == doctest::Approx(0.));
        REQUIRE(out.nij(0) == doctest::Approx(-cosRot));
        REQUIRE(out.nij(1) == doctest::Approx(sinRot));
        REQUIRE(out.nij(2) == doctest::Approx(0.));
        REQUIRE(out.dij == doctest::Approx(0.4));
    }

    // distance sphere - plan
    TEST_CASE("sphere_plan_2d")
    {
        constexpr std::size_t dim = 2;
        sphere<dim> s({{0.0, 0.0}}, 0.1);
        plan<dim> p({{ 0.3, 0.0}}, 0.);

        auto out = closest_points(s, p)[0];

        REQUIRE(out.pi(0) == doctest::Approx(0.1));
        REQUIRE(out.pi(1) == doctest::Approx(0.));
        REQUIRE(out.pj(0) == doctest::Approx(0.3));
        REQUIRE(out.pj(1) == doctest::Approx(0.));
        REQUIRE(out.nij(0) == doctest::Approx(-1.));
        REQUIRE(out.nij(1) == doctest::Approx(0.));
        REQUIRE(out.dij == doctest::Approx(0.2));
    }

    TEST_CASE("sphere_plan_2d_rotation_30_deg")
    {
        constexpr std::size_t dim = 2;
        sphere<dim> s({{0.0, 0.0}}, 0.1);
        double dist = 0.3;
        double cosRot = std::sqrt(3.)/2.;
        double sinRot = 1./2.;
        plan<dim> p({{dist*cosRot, dist*sinRot}}, PI/6.);

        auto out = closest_points(s, p)[0];

        REQUIRE(out.pi(0) == doctest::Approx(0.1*cosRot));
        REQUIRE(out.pi(1) == doctest::Approx(0.1*sinRot));
        REQUIRE(out.pj(0) == doctest::Approx(0.3*cosRot));
        REQUIRE(out.pj(1) == doctest::Approx(0.3*sinRot));
        REQUIRE(out.nij(0) == doctest::Approx(-cosRot));
        REQUIRE(out.nij(1) == doctest::Approx(-sinRot));
        REQUIRE(out.dij == doctest::Approx(0.2));
    }

    TEST_CASE("sphere_plan_2d_rotation_90_deg")
    {
        constexpr std::size_t dim = 2;
        sphere<dim> s({{0.0, 0.0}}, 0.1);
        plan<dim> p({{0., -0.2}}, PI/2.);

        auto out = closest_points(s, p)[0];

        REQUIRE(out.pi(0) == doctest::Approx(0.));
        REQUIRE(out.pi(1) == doctest::Approx(-0.1));
        REQUIRE(out.pj(0) == doctest::Approx(0.));
        REQUIRE(out.pj(1) == doctest::Approx(-0.2));
        REQUIRE(out.nij(0) == doctest::Approx(0.));
        REQUIRE(out.nij(1) == doctest::Approx(1.));
        REQUIRE(out.dij == doctest::Approx(0.1));
    }

    TEST_CASE("sphere_plan_2d_dispatch")
    {
        constexpr std::size_t dim = 2;
        sphere<dim> s({{0.0, 0.0}}, 0.1);
        plan<dim> p({{ 0.3, 0.0}}, 0.);

        scopi_container<dim> particles;
        particles.push_back(s);
        particles.push_back(p);

        auto out = closest_points_dispatcher<dim>::dispatch(*particles[0], *particles[1])[0];

        REQUIRE(out.pi(0) == doctest::Approx(0.1));
        REQUIRE(out.pi(1) == doctest::Approx(0.));
        REQUIRE(out.pj(0) == doctest::Approx(0.3));
        REQUIRE(out.pj(1) == doctest::Approx(0.));
        REQUIRE(out.nij(0) == doctest::Approx(-1.));
        REQUIRE(out.nij(1) == doctest::Approx(0.));
        REQUIRE(out.dij == doctest::Approx(0.2));
    }

    TEST_CASE("sphere_plan_2d_dispatch_rotation_30_deg")
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

        auto out = closest_points_dispatcher<dim>::dispatch(*particles[0], *particles[1])[0];

        REQUIRE(out.pi(0) == doctest::Approx(0.1*cosRot));
        REQUIRE(out.pi(1) == doctest::Approx(0.1*sinRot));
        REQUIRE(out.pj(0) == doctest::Approx(0.3*cosRot));
        REQUIRE(out.pj(1) == doctest::Approx(0.3*sinRot));
        REQUIRE(out.nij(0) == doctest::Approx(-cosRot));
        REQUIRE(out.nij(1) == doctest::Approx(-sinRot));
        REQUIRE(out.dij == doctest::Approx(0.2));
    }

    TEST_CASE("sphere_plan_2d_dispatch_rotation_90_deg")
    {
        constexpr std::size_t dim = 2;
        sphere<dim> s({{0.0, 0.0}}, 0.1);
        plan<dim> p({{0., -0.2}}, PI/2.);

        scopi_container<dim> particles;
        particles.push_back(s);
        particles.push_back(p);

        auto out = closest_points_dispatcher<dim>::dispatch(*particles[0], *particles[1])[0];

        REQUIRE(out.pi(0) == doctest::Approx(0.));
        REQUIRE(out.pi(1) == doctest::Approx(-0.1));
        REQUIRE(out.pj(0) == doctest::Approx(0.));
        REQUIRE(out.pj(1) == doctest::Approx(-0.2));
        REQUIRE(out.nij(0) == doctest::Approx(0.));
        REQUIRE(out.nij(1) == doctest::Approx(1.));
        REQUIRE(out.dij == doctest::Approx(0.1));
    }

    TEST_CASE("sphere_plan_3d")
    {
        constexpr std::size_t dim = 3;
        sphere<dim> s({{0.0, 0.0, 0.0}}, 0.1);
        plan<dim> p({{ 0.3, 0.0, 0.0}}, 0.);

        auto out = closest_points(s, p)[0];

        REQUIRE(out.pi(0) == doctest::Approx(0.1));
        REQUIRE(out.pi(1) == doctest::Approx(0.));
        REQUIRE(out.pi(2) == doctest::Approx(0.));
        REQUIRE(out.pj(0) == doctest::Approx(0.3));
        REQUIRE(out.pj(1) == doctest::Approx(0.));
        REQUIRE(out.pj(2) == doctest::Approx(0.));
        REQUIRE(out.nij(0) == doctest::Approx(-1.));
        REQUIRE(out.nij(1) == doctest::Approx(0.));
        REQUIRE(out.nij(2) == doctest::Approx(0.));
        REQUIRE(out.dij == doctest::Approx(0.2));
    }

    TEST_CASE("sphere_plan_3d_rotation_30_deg")
    {
        constexpr std::size_t dim = 3;
        sphere<dim> s({{0.0, 0.0, 0.0}}, 0.1);
        double dist = 0.3;
        double cosRot = std::sqrt(3.)/2.;
        double sinRot = 1./2.;
        plan<dim> p({{dist*cosRot, dist*sinRot, 0.}}, PI/6.);

        auto out = closest_points(s, p)[0];

        REQUIRE(out.pi(0) == doctest::Approx(0.1*cosRot));
        REQUIRE(out.pi(1) == doctest::Approx(0.1*sinRot));
        REQUIRE(out.pi(2) == doctest::Approx(0.));
        REQUIRE(out.pj(0) == doctest::Approx(0.3*cosRot));
        REQUIRE(out.pj(1) == doctest::Approx(0.3*sinRot));
        REQUIRE(out.pj(2) == doctest::Approx(0.));
        REQUIRE(out.nij(0) == doctest::Approx(-cosRot));
        REQUIRE(out.nij(1) == doctest::Approx(-sinRot));
        REQUIRE(out.nij(2) == doctest::Approx(0.));
        REQUIRE(out.dij == doctest::Approx(0.2));
    }

    TEST_CASE("sphere_plan_3d_rotation_90_deg")
    {
        constexpr std::size_t dim = 3;
        sphere<dim> s({{0.0, 0.0, 0,}}, 0.1);
        plan<dim> p({{0., -0.2, 0.}}, PI/2.);

        auto out = closest_points(s, p)[0];

        REQUIRE(out.pi(0) == doctest::Approx(0.));
        REQUIRE(out.pi(1) == doctest::Approx(-0.1));
        REQUIRE(out.pi(2) == doctest::Approx(0.));
        REQUIRE(out.pj(0) == doctest::Approx(0.));
        REQUIRE(out.pj(1) == doctest::Approx(-0.2));
        REQUIRE(out.pj(2) == doctest::Approx(0.));
        REQUIRE(out.nij(0) == doctest::Approx(0.));
        REQUIRE(out.nij(1) == doctest::Approx(1.));
        REQUIRE(out.nij(2) == doctest::Approx(0.));
        REQUIRE(out.dij == doctest::Approx(0.1));
    }

    TEST_CASE("sphere_plan_3d_dispatch")
    {
        constexpr std::size_t dim = 3;
        sphere<dim> s({{0.0, 0.0, 0.0}}, 0.1);
        plan<dim> p({{ 0.3, 0.0, 0.0}}, 0.);

        scopi_container<dim> particles;
        particles.push_back(s);
        particles.push_back(p);

        auto out = closest_points_dispatcher<dim>::dispatch(*particles[0], *particles[1])[0];

        REQUIRE(out.pi(0) == doctest::Approx(0.1));
        REQUIRE(out.pi(1) == doctest::Approx(0.));
        REQUIRE(out.pi(2) == doctest::Approx(0.));
        REQUIRE(out.pj(0) == doctest::Approx(0.3));
        REQUIRE(out.pj(1) == doctest::Approx(0.));
        REQUIRE(out.pj(2) == doctest::Approx(0.));
        REQUIRE(out.nij(0) == doctest::Approx(-1.));
        REQUIRE(out.nij(1) == doctest::Approx(0.));
        REQUIRE(out.nij(2) == doctest::Approx(0.));
        REQUIRE(out.dij == doctest::Approx(0.2));
    }

    TEST_CASE("sphere_plan_3d_dispatch_rotation_30_deg")
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

        auto out = closest_points_dispatcher<dim>::dispatch(*particles[0], *particles[1])[0];

        REQUIRE(out.pi(0) == doctest::Approx(0.1*cosRot));
        REQUIRE(out.pi(1) == doctest::Approx(0.1*sinRot));
        REQUIRE(out.pi(2) == doctest::Approx(0.));
        REQUIRE(out.pj(0) == doctest::Approx(0.3*cosRot));
        REQUIRE(out.pj(1) == doctest::Approx(0.3*sinRot));
        REQUIRE(out.pj(2) == doctest::Approx(0.));
        REQUIRE(out.nij(0) == doctest::Approx(-cosRot));
        REQUIRE(out.nij(1) == doctest::Approx(-sinRot));
        REQUIRE(out.nij(2) == doctest::Approx(0.));
        REQUIRE(out.dij == doctest::Approx(0.2));
    }

    TEST_CASE("sphere_plan_3d_dispatch_rotation_90_deg")
    {
        constexpr std::size_t dim = 3;
        sphere<dim> s({{0.0, 0.0, 0,}}, 0.1);
        plan<dim> p({{0., -0.2, 0.}}, PI/2.);

        scopi_container<dim> particles;
        particles.push_back(s);
        particles.push_back(p);

        auto out = closest_points_dispatcher<dim>::dispatch(*particles[0], *particles[1])[0];

        REQUIRE(out.pi(0) == doctest::Approx(0.));
        REQUIRE(out.pi(1) == doctest::Approx(-0.1));
        REQUIRE(out.pi(2) == doctest::Approx(0.));
        REQUIRE(out.pj(0) == doctest::Approx(0.));
        REQUIRE(out.pj(1) == doctest::Approx(-0.2));
        REQUIRE(out.pj(2) == doctest::Approx(0.));
        REQUIRE(out.nij(0) == doctest::Approx(0.));
        REQUIRE(out.nij(1) == doctest::Approx(1.));
        REQUIRE(out.nij(2) == doctest::Approx(0.));
        REQUIRE(out.dij == doctest::Approx(0.1));
    }

    // distance plan - sphere
    TEST_CASE("plan_sphere_2d")
    {
        constexpr std::size_t dim = 2;
        sphere<dim> s({{0.0, 0.0}}, 0.1);
        plan<dim> p({{ 0.3, 0.0}}, 0.);

        auto out = closest_points(p, s)[0];

        REQUIRE(out.pi(0) == doctest::Approx(0.3));
        REQUIRE(out.pi(1) == doctest::Approx(0.));
        REQUIRE(out.pj(0) == doctest::Approx(0.1));
        REQUIRE(out.pj(1) == doctest::Approx(0.));
        REQUIRE(out.nij(0) == doctest::Approx(1.));
        REQUIRE(out.nij(1) == doctest::Approx(0.));
        REQUIRE(out.dij == doctest::Approx(0.2));
    }

    TEST_CASE("plan_sphere_2d_rotation_30_deg")
    {
        constexpr std::size_t dim = 2;
        sphere<dim> s({{0.0, 0.0}}, 0.1);
        double dist = 0.3;
        double cosRot = std::sqrt(3.)/2.;
        double sinRot = 1./2.;
        plan<dim> p({{dist*cosRot, dist*sinRot}}, PI/6.);

        auto out = closest_points(p, s)[0];

        REQUIRE(out.pi(0) == doctest::Approx(0.3*cosRot));
        REQUIRE(out.pi(1) == doctest::Approx(0.3*sinRot));
        REQUIRE(out.pj(0) == doctest::Approx(0.1*cosRot));
        REQUIRE(out.pj(1) == doctest::Approx(0.1*sinRot));
        REQUIRE(out.nij(0) == doctest::Approx(cosRot));
        REQUIRE(out.nij(1) == doctest::Approx(sinRot));
        REQUIRE(out.dij == doctest::Approx(0.2));
    }

    TEST_CASE("plan_sphere_2d_rotation_90_deg")
    {
        constexpr std::size_t dim = 2;
        sphere<dim> s({{0.0, 0.0}}, 0.1);
        plan<dim> p({{0., -0.2}}, PI/2.);

        auto out = closest_points(p, s)[0];

        REQUIRE(out.pi(0) == doctest::Approx(0.));
        REQUIRE(out.pi(1) == doctest::Approx(-0.2));
        REQUIRE(out.pj(0) == doctest::Approx(0.));
        REQUIRE(out.pj(1) == doctest::Approx(-0.1));
        REQUIRE(out.nij(0) == doctest::Approx(0.));
        REQUIRE(out.nij(1) == doctest::Approx(-1.));
        REQUIRE(out.dij == doctest::Approx(0.1));
    }

    TEST_CASE("plan_sphere_2d_dispatch")
    {
        constexpr std::size_t dim = 2;
        sphere<dim> s({{0.0, 0.0}}, 0.1);
        plan<dim> p({{ 0.3, 0.0}}, 0.);

        scopi_container<dim> particles;
        particles.push_back(p);
        particles.push_back(s);

        auto out = closest_points_dispatcher<dim>::dispatch(*particles[0], *particles[1])[0];

        REQUIRE(out.pi(0) == doctest::Approx(0.3));
        REQUIRE(out.pi(1) == doctest::Approx(0.));
        REQUIRE(out.pj(0) == doctest::Approx(0.1));
        REQUIRE(out.pj(1) == doctest::Approx(0.));
        REQUIRE(out.nij(0) == doctest::Approx(1.));
        REQUIRE(out.nij(1) == doctest::Approx(0.));
        REQUIRE(out.dij == doctest::Approx(0.2));
    }

    TEST_CASE("plan_sphere_2d_dispatch_rotation_30_deg")
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

        auto out = closest_points_dispatcher<dim>::dispatch(*particles[0], *particles[1])[0];

        REQUIRE(out.pi(0) == doctest::Approx(0.3*cosRot));
        REQUIRE(out.pi(1) == doctest::Approx(0.3*sinRot));
        REQUIRE(out.pj(0) == doctest::Approx(0.1*cosRot));
        REQUIRE(out.pj(1) == doctest::Approx(0.1*sinRot));
        REQUIRE(out.nij(0) == doctest::Approx(cosRot));
        REQUIRE(out.nij(1) == doctest::Approx(sinRot));
        REQUIRE(out.dij == doctest::Approx(0.2));
    }

    TEST_CASE("plan_sphere_2d_dispatch_rotation_90_deg")
    {
        constexpr std::size_t dim = 2;
        sphere<dim> s({{0.0, 0.0}}, 0.1);
        plan<dim> p({{0., -0.2}}, PI/2.);

        scopi_container<dim> particles;
        particles.push_back(p);
        particles.push_back(s);

        auto out = closest_points_dispatcher<dim>::dispatch(*particles[0], *particles[1])[0];

        REQUIRE(out.pi(0) == doctest::Approx(0.));
        REQUIRE(out.pi(1) == doctest::Approx(-0.2));
        REQUIRE(out.pj(0) == doctest::Approx(0.));
        REQUIRE(out.pj(1) == doctest::Approx(-0.1));
        REQUIRE(out.nij(0) == doctest::Approx(0.));
        REQUIRE(out.nij(1) == doctest::Approx(-1.));
        REQUIRE(out.dij == doctest::Approx(0.1));
    }

    TEST_CASE("plan_sphere_3d")
    {
        constexpr std::size_t dim = 3;
        sphere<dim> s({{0.0, 0.0, 0.0}}, 0.1);
        plan<dim> p({{ 0.3, 0.0, 0.0}}, 0.);

        auto out = closest_points(p, s)[0];

        REQUIRE(out.pi(0) == doctest::Approx(0.3));
        REQUIRE(out.pi(1) == doctest::Approx(0.));
        REQUIRE(out.pi(2) == doctest::Approx(0.));
        REQUIRE(out.pj(0) == doctest::Approx(0.1));
        REQUIRE(out.pj(1) == doctest::Approx(0.));
        REQUIRE(out.pj(2) == doctest::Approx(0.));
        REQUIRE(out.nij(0) == doctest::Approx(1.));
        REQUIRE(out.nij(1) == doctest::Approx(0.));
        REQUIRE(out.nij(2) == doctest::Approx(0.));
        REQUIRE(out.dij == doctest::Approx(0.2));
    }

    TEST_CASE("plan_sphere_3d_rotation_30_deg")
    {
        constexpr std::size_t dim = 3;
        sphere<dim> s({{0.0, 0.0, 0.0}}, 0.1);
        double dist = 0.3;
        double cosRot = std::sqrt(3.)/2.;
        double sinRot = 1./2.;
        plan<dim> p({{dist*cosRot, dist*sinRot, 0.}}, PI/6.);

        auto out = closest_points(p, s)[0];

        REQUIRE(out.pi(0) == doctest::Approx(0.3*cosRot));
        REQUIRE(out.pi(1) == doctest::Approx(0.3*sinRot));
        REQUIRE(out.pi(2) == doctest::Approx(0.));
        REQUIRE(out.pj(0) == doctest::Approx(0.1*cosRot));
        REQUIRE(out.pj(1) == doctest::Approx(0.1*sinRot));
        REQUIRE(out.pj(2) == doctest::Approx(0.));
        REQUIRE(out.nij(0) == doctest::Approx(cosRot));
        REQUIRE(out.nij(1) == doctest::Approx(sinRot));
        REQUIRE(out.nij(2) == doctest::Approx(0.));
        REQUIRE(out.dij == doctest::Approx(0.2));
    }

    TEST_CASE("plan_sphere_3d_rotation_90_deg")
    {
        constexpr std::size_t dim = 3;
        sphere<dim> s({{0.0, 0.0, 0,}}, 0.1);
        plan<dim> p({{0., -0.2, 0.}}, PI/2.);

        auto out = closest_points(p, s)[0];

        REQUIRE(out.pi(0) == doctest::Approx(0.));
        REQUIRE(out.pi(1) == doctest::Approx(-0.2));
        REQUIRE(out.pi(2) == doctest::Approx(0.));
        REQUIRE(out.pj(0) == doctest::Approx(0.));
        REQUIRE(out.pj(1) == doctest::Approx(-0.1));
        REQUIRE(out.pj(2) == doctest::Approx(0.));
        REQUIRE(out.nij(0) == doctest::Approx(0.));
        REQUIRE(out.nij(1) == doctest::Approx(-1.));
        REQUIRE(out.nij(2) == doctest::Approx(0.));
        REQUIRE(out.dij == doctest::Approx(0.1));
    }

    TEST_CASE("plan_sphere_3d_dispatch")
    {
        constexpr std::size_t dim = 3;
        sphere<dim> s({{0.0, 0.0, 0.0}}, 0.1);
        plan<dim> p({{ 0.3, 0.0, 0.0}}, 0.);

        scopi_container<dim> particles;
        particles.push_back(p);
        particles.push_back(s);

        auto out = closest_points_dispatcher<dim>::dispatch(*particles[0], *particles[1])[0];

        REQUIRE(out.pi(0) == doctest::Approx(0.3));
        REQUIRE(out.pi(1) == doctest::Approx(0.));
        REQUIRE(out.pi(2) == doctest::Approx(0.));
        REQUIRE(out.pj(0) == doctest::Approx(0.1));
        REQUIRE(out.pj(1) == doctest::Approx(0.));
        REQUIRE(out.pj(2) == doctest::Approx(0.));
        REQUIRE(out.nij(0) == doctest::Approx(1.));
        REQUIRE(out.nij(1) == doctest::Approx(0.));
        REQUIRE(out.nij(2) == doctest::Approx(0.));
        REQUIRE(out.dij == doctest::Approx(0.2));
    }

    TEST_CASE("plan_sphere_3d_dispatch_rotation_30_deg")
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

        auto out = closest_points_dispatcher<dim>::dispatch(*particles[0], *particles[1])[0];

        REQUIRE(out.pi(0) == doctest::Approx(0.3*cosRot));
        REQUIRE(out.pi(1) == doctest::Approx(0.3*sinRot));
        REQUIRE(out.pi(2) == doctest::Approx(0.));
        REQUIRE(out.pj(0) == doctest::Approx(0.1*cosRot));
        REQUIRE(out.pj(1) == doctest::Approx(0.1*sinRot));
        REQUIRE(out.pj(2) == doctest::Approx(0.));
        REQUIRE(out.nij(0) == doctest::Approx(cosRot));
        REQUIRE(out.nij(1) == doctest::Approx(sinRot));
        REQUIRE(out.nij(2) == doctest::Approx(0.));
        REQUIRE(out.dij == doctest::Approx(0.2));
    }

    TEST_CASE("plan_sphere_3d_dispatch_rotation_90_deg")
    {
        constexpr std::size_t dim = 3;
        sphere<dim> s({{0.0, 0.0, 0,}}, 0.1);
        plan<dim> p({{0., -0.2, 0.}}, PI/2.);

        scopi_container<dim> particles;
        particles.push_back(p);
        particles.push_back(s);

        auto out = closest_points_dispatcher<dim>::dispatch(*particles[0], *particles[1])[0];

        REQUIRE(out.pi(0) == doctest::Approx(0.));
        REQUIRE(out.pi(1) == doctest::Approx(-0.2));
        REQUIRE(out.pi(2) == doctest::Approx(0.));
        REQUIRE(out.pj(0) == doctest::Approx(0.));
        REQUIRE(out.pj(1) == doctest::Approx(-0.1));
        REQUIRE(out.pj(2) == doctest::Approx(0.));
        REQUIRE(out.nij(0) == doctest::Approx(0.));
        REQUIRE(out.nij(1) == doctest::Approx(-1.));
        REQUIRE(out.nij(2) == doctest::Approx(0.));
        REQUIRE(out.dij == doctest::Approx(0.1));
    }

    // distance sphere - superellipsoid
    TEST_CASE("sphere_superellipsoid_2d")
    {
        // FIXME
        constexpr std::size_t dim = 2;
        sphere<dim> s({{-0.2, 0.0}}, 0.1);
        superellipsoid<dim> e({{0.2, 0.0}}, {quaternion(0.)}, {{0.1, 0.2}}, 1);

        auto out = closest_points(s, e)[0];

        REQUIRE(out.pi(0) == doctest::Approx(-0.1).epsilon(1e-7));
        REQUIRE(out.pi(1) == doctest::Approx(0.));
        REQUIRE(out.pj(0) == doctest::Approx(0.1));
        REQUIRE(out.pj(1) == doctest::Approx(0.));
        REQUIRE(out.nij(0) == doctest::Approx(-1.));
        REQUIRE(out.nij(1) == doctest::Approx(0.));
        REQUIRE(out.dij == doctest::Approx(0.2));
    }

    TEST_CASE("sphere_superellipsoid_2d_rotation_30_deg")
    {
        // FIXME
        constexpr std::size_t dim = 2;
        double dist = 0.4;
        double cosRot = std::sqrt(3.)/2.;
        double sinRot = 1./2.;
        sphere<dim> s({{dist*cosRot, -dist*sinRot}}, 0.1);
        superellipsoid<dim> e({{-dist*cosRot, dist*sinRot}}, {quaternion(PI-PI/6.)}, {{0.1, 0.2}}, 1);

        auto out = closest_points(s, e)[0];

        REQUIRE(out.pi(0) == doctest::Approx(0.3*cosRot).epsilon(1e-5));
        REQUIRE(out.pi(1) == doctest::Approx(-0.3*sinRot));
        REQUIRE(out.pj(0) == doctest::Approx(-0.3*cosRot));
        REQUIRE(out.pj(1) == doctest::Approx(0.3*sinRot));
        REQUIRE(out.nij(0) == doctest::Approx(cosRot));
        REQUIRE(out.nij(1) == doctest::Approx(-sinRot));
        REQUIRE(out.dij == doctest::Approx(0.6));
    }

    TEST_CASE("sphere_superellipsoid_2d_dispatch")
    {
        // FIXME
        constexpr std::size_t dim = 2;
        sphere<dim> s({{-0.2, 0.0}}, 0.1);
        superellipsoid<dim> e({{0.2, 0.0}}, {quaternion(0.)}, {{0.1, 0.2}}, 1);

        scopi_container<dim> particles;
        particles.push_back(s);
        particles.push_back(e);

        auto out = closest_points_dispatcher<dim>::dispatch(*particles[0], *particles[1])[0];

        REQUIRE(out.pi(0) == doctest::Approx(-0.1));
        REQUIRE(out.pi(1) == doctest::Approx(0.));
        REQUIRE(out.pj(0) == doctest::Approx(0.1));
        REQUIRE(out.pj(1) == doctest::Approx(0.));
        REQUIRE(out.nij(0) == doctest::Approx(-1.));
        REQUIRE(out.nij(1) == doctest::Approx(0.));
        REQUIRE(out.dij == doctest::Approx(0.2));
    }

    TEST_CASE("sphere_superellipsoid_2d_dispatch_rotation_30_deg")
    {
        // FIXME
        constexpr std::size_t dim = 2;
        double dist = 0.4;
        double cosRot = std::sqrt(3.)/2.;
        double sinRot = 1./2.;
        sphere<dim> s({{dist*cosRot, -dist*sinRot}}, 0.1);
        superellipsoid<dim> e({{-dist*cosRot, dist*sinRot}}, {quaternion(PI-PI/6.)}, {{0.1, 0.2}}, 1);

        scopi_container<dim> particles;
        particles.push_back(s);
        particles.push_back(e);

        auto out = closest_points_dispatcher<dim>::dispatch(*particles[0], *particles[1])[0];

        REQUIRE(out.pi(0) == doctest::Approx(0.3*cosRot));
        REQUIRE(out.pi(1) == doctest::Approx(-0.3*sinRot));
        REQUIRE(out.pj(0) == doctest::Approx(-0.3*cosRot));
        REQUIRE(out.pj(1) == doctest::Approx(0.3*sinRot));
        REQUIRE(out.nij(0) == doctest::Approx(cosRot));
        REQUIRE(out.nij(1) == doctest::Approx(-sinRot));
        REQUIRE(out.dij == doctest::Approx(0.6));
    }

    TEST_CASE("sphere_superellipsoid_3d")
    {
        // FIXME
        constexpr std::size_t dim = 3;
        sphere<dim> s({{0.2, 0.0, 0.0}}, 0.1);
        superellipsoid<dim> e({{-0.2, 0.0, 0.0}}, {quaternion(0.)}, {{0.1, 0.2, 0.3}}, {1, 1});

        auto out = closest_points(s, e)[0];

        REQUIRE(out.pi(0) == doctest::Approx(-0.1));
        REQUIRE(out.pi(1) == doctest::Approx(0.));
        REQUIRE(out.pi(2) == doctest::Approx(0.));
        REQUIRE(out.pj(0) == doctest::Approx(0.1));
        REQUIRE(out.pj(1) == doctest::Approx(0.));
        REQUIRE(out.pj(2) == doctest::Approx(0.));
        REQUIRE(out.nij(0) == doctest::Approx(-1.));
        REQUIRE(out.nij(1) == doctest::Approx(0.));
        REQUIRE(out.nij(2) == doctest::Approx(0.));
        REQUIRE(out.dij == doctest::Approx(0.2));
    }

    TEST_CASE("sphere_superellipsoid_3d_rotation_30_deg")
    {
        // FIXME Newton does not converge
        constexpr std::size_t dim = 3;
        double dist = 0.4;
        double cosRot = std::sqrt(3.)/2.;
        double sinRot = 1./2.;
        sphere<dim> s({{dist*cosRot, -dist*sinRot, 0.}}, 0.1);
        superellipsoid<dim> e({{-dist*cosRot, dist*sinRot, 0.}}, {quaternion(PI-PI/6.)}, {{0.1, 0.2, 0.3}},  {1, 1});

        auto out = closest_points(s, e)[0];

        REQUIRE(out.pi(0) == doctest::Approx(0.3*cosRot));
        REQUIRE(out.pi(1) == doctest::Approx(-0.3*sinRot));
        REQUIRE(out.pi(2) == doctest::Approx(0.));
        REQUIRE(out.pj(0) == doctest::Approx(-0.3*cosRot));
        REQUIRE(out.pj(1) == doctest::Approx(0.3*sinRot));
        REQUIRE(out.pj(2) == doctest::Approx(0.));
        REQUIRE(out.nij(0) == doctest::Approx(cosRot));
        REQUIRE(out.nij(1) == doctest::Approx(-sinRot));
        REQUIRE(out.nij(2) == doctest::Approx(0.));
        REQUIRE(out.dij == doctest::Approx(0.6));
    }

    TEST_CASE("sphere_superellipsoid_3d_dispatch")
    {
        // FIXME
        constexpr std::size_t dim = 3;
        sphere<dim> s({{0.2, 0.0, 0.0}}, 0.1);
        superellipsoid<dim> e({{-0.2, 0.0, 0.0}}, {quaternion(0.)}, {{0.1, 0.2, 0.3}},  {1, 1});

        scopi_container<dim> particles;
        particles.push_back(s);
        particles.push_back(e);

        auto out = closest_points_dispatcher<dim>::dispatch(*particles[0], *particles[1])[0];

        REQUIRE(out.pi(0) == doctest::Approx(-0.1));
        REQUIRE(out.pi(1) == doctest::Approx(0.));
        REQUIRE(out.pi(2) == doctest::Approx(0.));
        REQUIRE(out.pj(0) == doctest::Approx(0.1));
        REQUIRE(out.pj(1) == doctest::Approx(0.));
        REQUIRE(out.pj(2) == doctest::Approx(0.));
        REQUIRE(out.nij(0) == doctest::Approx(-1.));
        REQUIRE(out.nij(1) == doctest::Approx(0.));
        REQUIRE(out.nij(2) == doctest::Approx(0.));
        REQUIRE(out.dij == doctest::Approx(0.2));
    }

    TEST_CASE("sphere_superellipsoid_3d_dispatch_rotation_30_deg")
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

        auto out = closest_points_dispatcher<dim>::dispatch(*particles[0], *particles[1])[0];

        REQUIRE(out.pi(0) == doctest::Approx(0.3*cosRot));
        REQUIRE(out.pi(1) == doctest::Approx(-0.3*sinRot));
        REQUIRE(out.pi(2) == doctest::Approx(0.));
        REQUIRE(out.pj(0) == doctest::Approx(-0.3*cosRot));
        REQUIRE(out.pj(1) == doctest::Approx(0.3*sinRot));
        REQUIRE(out.pj(2) == doctest::Approx(0.));
        REQUIRE(out.nij(0) == doctest::Approx(cosRot));
        REQUIRE(out.nij(1) == doctest::Approx(-sinRot));
        REQUIRE(out.nij(2) == doctest::Approx(0.));
        REQUIRE(out.dij == doctest::Approx(0.6));
    }

    // distance superellipsoid - sphere
    TEST_CASE("superellipsoid_sphere_2d")
    {
        // FIXME
        constexpr std::size_t dim = 2;
        sphere<dim> s({{-0.2, 0.0}}, 0.1);
        superellipsoid<dim> e({{0.2, 0.0}}, {quaternion(0.)}, {{0.1, 0.2}}, 1);

        auto out = closest_points(e, s)[0];

        REQUIRE(out.pi(0) == doctest::Approx(0.1));
        REQUIRE(out.pi(1) == doctest::Approx(0.));
        REQUIRE(out.pj(0) == doctest::Approx(-0.1));
        REQUIRE(out.pj(1) == doctest::Approx(0.));
        REQUIRE(out.nij(0) == doctest::Approx(-1.));
        REQUIRE(out.nij(1) == doctest::Approx(0.));
        REQUIRE(out.dij == doctest::Approx(0.2));
    }

    TEST_CASE("superellipsoid_sphere_2d_rotation_30_deg")
    {
        // FIXME
        constexpr std::size_t dim = 2;
        double dist = 0.4;
        double cosRot = std::sqrt(3.)/2.;
        double sinRot = 1./2.;
        sphere<dim> s({{dist*cosRot, -dist*sinRot}}, 0.1);
        superellipsoid<dim> e({{-dist*cosRot, dist*sinRot}}, {quaternion(PI-PI/6.)}, {{0.1, 0.2}}, 1);

        auto out = closest_points(e, s)[0];

        REQUIRE(out.pi(0) == doctest::Approx(-0.3*cosRot));
        REQUIRE(out.pi(1) == doctest::Approx(0.3*sinRot));
        REQUIRE(out.pj(0) == doctest::Approx(0.3*cosRot));
        REQUIRE(out.pj(1) == doctest::Approx(-0.3*sinRot));
        REQUIRE(out.nij(0) == doctest::Approx(-cosRot));
        REQUIRE(out.nij(1) == doctest::Approx(sinRot));
        REQUIRE(out.dij == doctest::Approx(0.6));
    }

    TEST_CASE("superellipsoid_sphere_2d_dispatch")
    {
        // FIXME
        constexpr std::size_t dim = 2;
        sphere<dim> s({{-0.2, 0.0}}, 0.1);
        superellipsoid<dim> e({{0.2, 0.0}}, {quaternion(0.)}, {{0.1, 0.2}}, 1);

        scopi_container<dim> particles;
        particles.push_back(e);
        particles.push_back(s);

        auto out = closest_points_dispatcher<dim>::dispatch(*particles[0], *particles[1])[0];

        REQUIRE(out.pi(0) == doctest::Approx(0.1));
        REQUIRE(out.pi(1) == doctest::Approx(0.));
        REQUIRE(out.pj(0) == doctest::Approx(-0.1));
        REQUIRE(out.pj(1) == doctest::Approx(0.));
        REQUIRE(out.nij(0) == doctest::Approx(-1.));
        REQUIRE(out.nij(1) == doctest::Approx(0.));
        REQUIRE(out.dij == doctest::Approx(0.2));
    }

    TEST_CASE("superellipsoid_sphere_2d_dispatch_rotation_30_deg")
    {
        // FIXME
        constexpr std::size_t dim = 2;
        double dist = 0.4;
        double cosRot = std::sqrt(3.)/2.;
        double sinRot = 1./2.;
        sphere<dim> s({{dist*cosRot, -dist*sinRot}}, 0.1);
        superellipsoid<dim> e({{-dist*cosRot, dist*sinRot}}, {quaternion(PI-PI/6.)}, {{0.1, 0.2}}, 1);

        scopi_container<dim> particles;
        particles.push_back(e);
        particles.push_back(s);

        auto out = closest_points_dispatcher<dim>::dispatch(*particles[0], *particles[1])[0];

        REQUIRE(out.pi(0) == doctest::Approx(-0.3*cosRot));
        REQUIRE(out.pi(1) == doctest::Approx(0.3*sinRot));
        REQUIRE(out.pj(0) == doctest::Approx(0.3*cosRot));
        REQUIRE(out.pj(1) == doctest::Approx(-0.3*sinRot));
        REQUIRE(out.nij(0) == doctest::Approx(-cosRot));
        REQUIRE(out.nij(1) == doctest::Approx(sinRot));
        REQUIRE(out.dij == doctest::Approx(0.6));
    }

    TEST_CASE("superellipsoid_sphere_3d")
    {
        // FIXME
        constexpr std::size_t dim = 3;
        sphere<dim> s({{0.2, 0.0, 0.0}}, 0.1);
        superellipsoid<dim> e({{-0.2, 0.0, 0.0}}, {quaternion(0.)}, {{0.1, 0.2, 0.3}}, {1, 1});

        auto out = closest_points(e, s)[0];

        REQUIRE(out.pi(0) == doctest::Approx(0.1));
        REQUIRE(out.pi(1) == doctest::Approx(0.));
        REQUIRE(out.pi(2) == doctest::Approx(0.));
        REQUIRE(out.pj(0) == doctest::Approx(-0.1));
        REQUIRE(out.pj(1) == doctest::Approx(0.));
        REQUIRE(out.pj(2) == doctest::Approx(0.));
        REQUIRE(out.nij(0) == doctest::Approx(1.));
        REQUIRE(out.nij(1) == doctest::Approx(0.));
        REQUIRE(out.nij(2) == doctest::Approx(0.));
        REQUIRE(out.dij == doctest::Approx(0.2));
    }

    TEST_CASE("superellipsoid_sphere_3d_rotation_30_deg")
    {
        // FIXME Newton does not converge
        constexpr std::size_t dim = 3;
        double dist = 0.4;
        double cosRot = std::sqrt(3.)/2.;
        double sinRot = 1./2.;
        sphere<dim> s({{dist*cosRot, -dist*sinRot, 0.}}, 0.1);
        superellipsoid<dim> e({{-dist*cosRot, dist*sinRot, 0.}}, {quaternion(PI-PI/6.)}, {{0.1, 0.2, 0.3}},  {1, 1});

        auto out = closest_points(e, s)[0];

        REQUIRE(out.pi(0) == doctest::Approx(0.3*cosRot));
        REQUIRE(out.pi(1) == doctest::Approx(-0.3*sinRot));
        REQUIRE(out.pi(2) == doctest::Approx(0.));
        REQUIRE(out.pj(0) == doctest::Approx(-0.3*cosRot));
        REQUIRE(out.pj(1) == doctest::Approx(0.3*sinRot));
        REQUIRE(out.pj(2) == doctest::Approx(0.));
        REQUIRE(out.nij(0) == doctest::Approx(cosRot));
        REQUIRE(out.nij(1) == doctest::Approx(-sinRot));
        REQUIRE(out.nij(2) == doctest::Approx(0.));
        REQUIRE(out.dij == doctest::Approx(0.6));
    }

    TEST_CASE("superellipsoid_sphere_3d_dispatch")
    {
        // FIXME
        constexpr std::size_t dim = 3;
        sphere<dim> s({{0.2, 0.0, 0.0}}, 0.1);
        superellipsoid<dim> e({{-0.2, 0.0, 0.0}}, {quaternion(0.)}, {{0.1, 0.2, 0.3}}, {1, 1});

        scopi_container<dim> particles;
        particles.push_back(e);
        particles.push_back(s);

        auto out = closest_points_dispatcher<dim>::dispatch(*particles[0], *particles[1])[0];

        REQUIRE(out.pi(0) == doctest::Approx(0.1));
        REQUIRE(out.pi(1) == doctest::Approx(0.));
        REQUIRE(out.pi(2) == doctest::Approx(0.));
        REQUIRE(out.pj(0) == doctest::Approx(-0.1));
        REQUIRE(out.pj(1) == doctest::Approx(0.));
        REQUIRE(out.pj(2) == doctest::Approx(0.));
        REQUIRE(out.nij(0) == doctest::Approx(1.));
        REQUIRE(out.nij(1) == doctest::Approx(0.));
        REQUIRE(out.nij(2) == doctest::Approx(0.));
        REQUIRE(out.dij == doctest::Approx(0.2));
    }

    TEST_CASE("superellipsoid_sphere_3d_dispatch_rotation_30_deg")
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

        auto out = closest_points_dispatcher<dim>::dispatch(*particles[0], *particles[1])[0];

        REQUIRE(out.pi(0) == doctest::Approx(0.3*cosRot));
        REQUIRE(out.pi(1) == doctest::Approx(-0.3*sinRot));
        REQUIRE(out.pi(2) == doctest::Approx(0.));
        REQUIRE(out.pj(0) == doctest::Approx(-0.3*cosRot));
        REQUIRE(out.pj(1) == doctest::Approx(0.3*sinRot));
        REQUIRE(out.pj(2) == doctest::Approx(0.));
        REQUIRE(out.nij(0) == doctest::Approx(cosRot));
        REQUIRE(out.nij(1) == doctest::Approx(-sinRot));
        REQUIRE(out.nij(2) == doctest::Approx(0.));
        REQUIRE(out.dij == doctest::Approx(0.2));
    }

    // distance superellipsoid - superellipsoid
    TEST_CASE("superellipsoid_superellipsoid_2d")
    {
        constexpr std::size_t dim = 2;
        superellipsoid<dim> s1({{-0.2, 0.0}}, {quaternion(0.)}, {{0.1, 0.2}}, 1);
        superellipsoid<dim> s2({{ 0.2, 0.0}}, {quaternion(0.)}, {{0.1, 0.3}}, 1);

        auto out = closest_points(s1, s2)[0];

        REQUIRE(out.pi(0) == doctest::Approx(-0.1));
        REQUIRE(out.pi(1) == doctest::Approx(0.));
        REQUIRE(out.pj(0) == doctest::Approx(0.1));
        REQUIRE(out.pj(1) == doctest::Approx(0.));
        REQUIRE(out.nij(0) == doctest::Approx(-1.));
        REQUIRE(out.nij(1) == doctest::Approx(0.));
        REQUIRE(out.dij == doctest::Approx(0.2));
    }

    TEST_CASE("superellipsoid_superellipsoid_2d_rotation_30_deg")
    {
        constexpr std::size_t dim = 2;
        double dist = 0.4;
        double cosRot = std::sqrt(3.)/2.;
        double sinRot = 1./2.;
        superellipsoid<dim> s1({{-dist*cosRot,  dist*sinRot}}, {quaternion(-PI/6.)}, {{0.1, 0.4}}, 1);
        superellipsoid<dim> s2({{ dist*cosRot, -dist*sinRot}}, {quaternion(PI-PI/6.)}, {{0.2, 0.3}}, 1);

        auto out = closest_points(s1, s2)[0];

        REQUIRE(out.pi(0) == doctest::Approx(-0.3*cosRot));
        REQUIRE(out.pi(1) == doctest::Approx(0.3*sinRot));
        REQUIRE(out.pj(0) == doctest::Approx(0.2*cosRot));
        REQUIRE(out.pj(1) == doctest::Approx(-0.2*sinRot));
        REQUIRE(out.nij(0) == doctest::Approx(-cosRot));
        REQUIRE(out.nij(1) == doctest::Approx(sinRot));
        REQUIRE(out.dij == doctest::Approx(0.5));
    }

    TEST_CASE("superellipsoid_superellipsoid_2d_dispatch")
    {
        constexpr std::size_t dim = 2;
        superellipsoid<dim> s1({{-0.2, 0.0}}, {quaternion(0.)}, {{0.1, 0.2}}, 1);
        superellipsoid<dim> s2({{ 0.2, 0.0}}, {quaternion(0.)}, {{0.1, 0.3}}, 1);

        scopi_container<dim> particles;
        particles.push_back(s1);
        particles.push_back(s2);

        auto out = closest_points_dispatcher<dim>::dispatch(*particles[0], *particles[1])[0];

        REQUIRE(out.pi(0) == doctest::Approx(-0.1));
        REQUIRE(out.pi(1) == doctest::Approx(0.));
        REQUIRE(out.pj(0) == doctest::Approx(0.1));
        REQUIRE(out.pj(1) == doctest::Approx(0.));
        REQUIRE(out.nij(0) == doctest::Approx(-1.));
        REQUIRE(out.nij(1) == doctest::Approx(0.));
        REQUIRE(out.dij == doctest::Approx(0.2));
    }

    TEST_CASE("superellipsoid_superellipsoid_2d_dispatch_rotation_30_deg")
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

        auto out = closest_points_dispatcher<dim>::dispatch(*particles[0], *particles[1])[0];

        REQUIRE(out.pi(0) == doctest::Approx(-0.3*cosRot));
        REQUIRE(out.pi(1) == doctest::Approx(0.3*sinRot));
        REQUIRE(out.pj(0) == doctest::Approx(0.2*cosRot));
        REQUIRE(out.pj(1) == doctest::Approx(-0.2*sinRot));
        REQUIRE(out.nij(0) == doctest::Approx(-cosRot));
        REQUIRE(out.nij(1) == doctest::Approx(sinRot));
        REQUIRE(out.dij == doctest::Approx(0.5));
    }

    TEST_CASE("superellipsoid_superellipsoid_3d")
    {
        constexpr std::size_t dim = 3;
        superellipsoid<dim> s1({{-0.2, 0.0, 0.0}}, {quaternion(0.)}, {{0.1, 0.2, 0.3}},  {1, 1});
        superellipsoid<dim> s2({{ 0.2, 0.0, 0.0}}, {quaternion(0.)}, {{0.1, 0.2, 0.3}},  {1, 1});

        auto out = closest_points(s1, s2)[0];

        REQUIRE(out.pi(0) == doctest::Approx(-0.1));
        REQUIRE(out.pi(1) == doctest::Approx(0.));
        REQUIRE(out.pi(2) == doctest::Approx(0.));
        REQUIRE(out.pj(0) == doctest::Approx(0.1));
        REQUIRE(out.pj(1) == doctest::Approx(0.));
        REQUIRE(out.pj(2) == doctest::Approx(0.));
        REQUIRE(out.nij(0) == doctest::Approx(-1.));
        REQUIRE(out.nij(1) == doctest::Approx(0.));
        REQUIRE(out.nij(2) == doctest::Approx(0.));
        REQUIRE(out.dij == doctest::Approx(0.2));
    }

    TEST_CASE("superellipsoid_superellipsoid_3d_rotation_30_deg")
    {
        constexpr std::size_t dim = 3;
        double dist = 0.4;
        double cosRot = std::sqrt(3.)/2.;
        double sinRot = 1./2.;
        superellipsoid<dim> s1({{-dist*cosRot,  dist*sinRot, 0.}}, {quaternion(-PI/6.)}, {{0.1, 0.4, 0.5}}, {1, 1});
        superellipsoid<dim> s2({{ dist*cosRot, -dist*sinRot, 0.}}, {quaternion(PI-PI/6.)}, {{0.2, 0.3, 0.6}}, {1, 1});

        auto out = closest_points(s1, s2)[0];

        REQUIRE(out.pi(0) == doctest::Approx(-0.3*cosRot));
        REQUIRE(out.pi(1) == doctest::Approx(0.3*sinRot));
        REQUIRE(out.pi(2) == doctest::Approx(0.));
        REQUIRE(out.pj(0) == doctest::Approx(0.2*cosRot));
        REQUIRE(out.pj(1) == doctest::Approx(-0.2*sinRot));
        REQUIRE(out.pj(2) == doctest::Approx(0.));
        REQUIRE(out.nij(0) == doctest::Approx(-cosRot));
        REQUIRE(out.nij(1) == doctest::Approx(sinRot));
        REQUIRE(out.nij(2) == doctest::Approx(0.));
        REQUIRE(out.dij == doctest::Approx(0.5));
    }

    TEST_CASE("superellipsoid_superellipsoid_3d_dispatch")
    {
        constexpr std::size_t dim = 3;
        superellipsoid<dim> s1({{-0.2, 0.0, 0.0}}, {quaternion(0.)}, {{0.1, 0.2, 0.3}},  {1, 1});
        superellipsoid<dim> s2({{ 0.2, 0.0, 0.0}}, {quaternion(0.)}, {{0.1, 0.2, 0.3}},  {1, 1});

        scopi_container<dim> particles;
        particles.push_back(s1);
        particles.push_back(s2);

        auto out = closest_points_dispatcher<dim>::dispatch(*particles[0], *particles[1])[0];

        REQUIRE(out.pi(0) == doctest::Approx(-0.1));
        REQUIRE(out.pi(1) == doctest::Approx(0.));
        REQUIRE(out.pi(2) == doctest::Approx(0.));
        REQUIRE(out.pj(0) == doctest::Approx(0.1));
        REQUIRE(out.pj(1) == doctest::Approx(0.));
        REQUIRE(out.pj(2) == doctest::Approx(0.));
        REQUIRE(out.nij(0) == doctest::Approx(-1.));
        REQUIRE(out.nij(1) == doctest::Approx(0.));
        REQUIRE(out.nij(2) == doctest::Approx(0.));
        REQUIRE(out.dij == doctest::Approx(0.2));
    }

    TEST_CASE("superellipsoid_superellipsoid_3d_dispatch_rotation_30_deg")
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

        auto out = closest_points_dispatcher<dim>::dispatch(*particles[0], *particles[1])[0];

        REQUIRE(out.pi(0) == doctest::Approx(-0.3*cosRot));
        REQUIRE(out.pi(1) == doctest::Approx(0.3*sinRot));
        REQUIRE(out.pi(2) == doctest::Approx(0.));
        REQUIRE(out.pj(0) == doctest::Approx(0.2*cosRot));
        REQUIRE(out.pj(1) == doctest::Approx(-0.2*sinRot));
        REQUIRE(out.pj(2) == doctest::Approx(0.));
        REQUIRE(out.nij(0) == doctest::Approx(-cosRot));
        REQUIRE(out.nij(1) == doctest::Approx(sinRot));
        REQUIRE(out.nij(2) == doctest::Approx(0.));
        REQUIRE(out.dij == doctest::Approx(0.5));
    }

    // distance superellipsoid - plan
    TEST_CASE("superellipsoid_plan_2d")
    {
        // FIXME Newton does not converge
        constexpr std::size_t dim = 2;
        superellipsoid<dim> s({{ 0.0, 0.0}}, {quaternion(0.)}, {{0.1, 0.2}}, 1);
        plan<dim> p({{ 0.3, 0.0}}, 0.);

        auto out = closest_points(s, p)[0];

        REQUIRE(out.pi(0) == doctest::Approx(0.1));
        REQUIRE(out.pi(1) == doctest::Approx(0.));
        REQUIRE(out.pj(0) == doctest::Approx(0.3));
        REQUIRE(out.pj(1) == doctest::Approx(0.));
        REQUIRE(out.nij(0) == doctest::Approx(-1.));
        REQUIRE(out.nij(1) == doctest::Approx(0.));
        REQUIRE(out.dij == doctest::Approx(0.2));
    }

    TEST_CASE("superellipsoid_plan_2d_rotation_30_deg")
    {
        // FIXME Newton does not converge
        constexpr std::size_t dim = 2;
        superellipsoid<dim> s({{ 0., 0.}}, {quaternion(0.)}, {{0.1, 0.2}}, 1);
        double dist = 0.3;
        double cosRot = std::sqrt(3.)/2.;
        double sinRot = 1./2.;
        plan<dim> p({{dist*cosRot, dist*sinRot}}, PI/6.);

        auto out = closest_points(s, p)[0];

        REQUIRE(out.pi(0) == doctest::Approx(0.1*cosRot));
        REQUIRE(out.pi(1) == doctest::Approx(0.1*sinRot));
        REQUIRE(out.pj(0) == doctest::Approx(0.3*cosRot));
        REQUIRE(out.pj(1) == doctest::Approx(0.3*sinRot));
        REQUIRE(out.nij(0) == doctest::Approx(-cosRot));
        REQUIRE(out.nij(1) == doctest::Approx(-sinRot));
        REQUIRE(out.dij == doctest::Approx(0.2));
    }

    TEST_CASE("superellipsoid_plan_2d_dispatch")
    {
        // FIXME Newton does not converge
        constexpr std::size_t dim = 2;
        superellipsoid<dim> s({{ 0.0, 0.0}}, {quaternion(0.)}, {{0.1, 0.2}}, 1);
        plan<dim> p({{ 0.3, 0.0}}, 0.);

        scopi_container<dim> particles;
        particles.push_back(s);
        particles.push_back(p);

        auto out = closest_points_dispatcher<dim>::dispatch(*particles[0], *particles[1])[0];

        REQUIRE(out.pi(0) == doctest::Approx(0.1));
        REQUIRE(out.pi(1) == doctest::Approx(0.));
        REQUIRE(out.pj(0) == doctest::Approx(0.3));
        REQUIRE(out.pj(1) == doctest::Approx(0.));
        REQUIRE(out.nij(0) == doctest::Approx(-1.));
        REQUIRE(out.nij(1) == doctest::Approx(0.));
        REQUIRE(out.dij == doctest::Approx(0.2));
    }

    TEST_CASE("superellipsoid_plan_2d_dispatch_rotation_30_deg")
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

        auto out = closest_points_dispatcher<dim>::dispatch(*particles[0], *particles[1])[0];

        REQUIRE(out.pi(0) == doctest::Approx(0.1*cosRot));
        REQUIRE(out.pi(1) == doctest::Approx(0.1*sinRot));
        REQUIRE(out.pj(0) == doctest::Approx(0.3*cosRot));
        REQUIRE(out.pj(1) == doctest::Approx(0.3*sinRot));
        REQUIRE(out.nij(0) == doctest::Approx(-cosRot));
        REQUIRE(out.nij(1) == doctest::Approx(-sinRot));
        REQUIRE(out.dij == doctest::Approx(0.2));
    }

    TEST_CASE("superellipsoid_plan_3d")
    {
        // FIXME Newton does not converge
        constexpr std::size_t dim = 3;
        superellipsoid<dim> s({{ 0.0, 0.0, 0.0}}, {quaternion(0.)}, {{0.1, 0.2, 0.3}}, {1, 1});
        plan<dim> p({{ 0.3, 0.0, 0.0}}, 0.);

        auto out = closest_points(s, p)[0];

        REQUIRE(out.pi(0) == doctest::Approx(0.1));
        REQUIRE(out.pi(1) == doctest::Approx(0.));
        REQUIRE(out.pi(2) == doctest::Approx(0.));
        REQUIRE(out.pj(0) == doctest::Approx(0.3));
        REQUIRE(out.pj(1) == doctest::Approx(0.));
        REQUIRE(out.pj(2) == doctest::Approx(0.));
        REQUIRE(out.nij(0) == doctest::Approx(-1.));
        REQUIRE(out.nij(1) == doctest::Approx(0.));
        REQUIRE(out.nij(2) == doctest::Approx(0.));
        REQUIRE(out.dij == doctest::Approx(0.2));
    }

    TEST_CASE("superellipsoid_plan_3d_rotation_30_deg")
    {
        // FIXME Newton does not converge
        constexpr std::size_t dim = 3;
        superellipsoid<dim> s({{ 0., 0., 0.}}, {quaternion(0.)}, {{0.1, 0.2, 0.3}}, {1, 1});
        double dist = 0.3;
        double cosRot = std::sqrt(3.)/2.;
        double sinRot = 1./2.;
        plan<dim> p({{dist*cosRot, dist*sinRot, 0.}}, PI/6.);

        auto out = closest_points(s, p)[0];

        REQUIRE(out.pi(0) == doctest::Approx(0.1*cosRot));
        REQUIRE(out.pi(1) == doctest::Approx(0.1*sinRot));
        REQUIRE(out.pi(2) == doctest::Approx(0.));
        REQUIRE(out.pj(0) == doctest::Approx(0.3*cosRot));
        REQUIRE(out.pj(1) == doctest::Approx(0.3*sinRot));
        REQUIRE(out.pj(2) == doctest::Approx(0.));
        REQUIRE(out.nij(0) == doctest::Approx(-cosRot));
        REQUIRE(out.nij(1) == doctest::Approx(-sinRot));
        REQUIRE(out.nij(2) == doctest::Approx(0.));
        REQUIRE(out.dij == doctest::Approx(0.2));
    }

    TEST_CASE("superellipsoid_plan_3d_dispatch")
    {
        // FIXME Newton does not converge
        constexpr std::size_t dim = 3;
        superellipsoid<dim> s({{ 0.0, 0.0, 0.0}}, {quaternion(0.)}, {{0.1, 0.2, 0.3}}, {1, 1});
        plan<dim> p({{ 0.3, 0.0, 0.0}}, 0.);

        scopi_container<dim> particles;
        particles.push_back(s);
        particles.push_back(p);

        auto out = closest_points_dispatcher<dim>::dispatch(*particles[0], *particles[1])[0];

        REQUIRE(out.pi(0) == doctest::Approx(0.1));
        REQUIRE(out.pi(1) == doctest::Approx(0.));
        REQUIRE(out.pi(2) == doctest::Approx(0.));
        REQUIRE(out.pj(0) == doctest::Approx(0.3));
        REQUIRE(out.pj(1) == doctest::Approx(0.));
        REQUIRE(out.pj(2) == doctest::Approx(0.));
        REQUIRE(out.nij(0) == doctest::Approx(-1.));
        REQUIRE(out.nij(1) == doctest::Approx(0.));
        REQUIRE(out.nij(2) == doctest::Approx(0.));
        REQUIRE(out.dij == doctest::Approx(0.2));
    }

    TEST_CASE("superellipsoid_plan_3d_dispatch_rotation_30_deg")
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

        auto out = closest_points_dispatcher<dim>::dispatch(*particles[0], *particles[1])[0];

        REQUIRE(out.pi(0) == doctest::Approx(0.1*cosRot));
        REQUIRE(out.pi(1) == doctest::Approx(0.1*sinRot));
        REQUIRE(out.pi(2) == doctest::Approx(0.));
        REQUIRE(out.pj(0) == doctest::Approx(0.3*cosRot));
        REQUIRE(out.pj(1) == doctest::Approx(0.3*sinRot));
        REQUIRE(out.pj(2) == doctest::Approx(0.));
        REQUIRE(out.nij(0) == doctest::Approx(-cosRot));
        REQUIRE(out.nij(1) == doctest::Approx(-sinRot));
        REQUIRE(out.nij(2) == doctest::Approx(0.));
        REQUIRE(out.dij == doctest::Approx(0.2));
    }

    // distance plan - superellipsoid
    TEST_CASE("plan_superellipsoid_2d")
    {
        // FIXME Newton does not converge
        constexpr std::size_t dim = 2;
        superellipsoid<dim> s({{ 0.0, 0.0}}, {quaternion(0.)}, {{0.1, 0.2}}, 1);
        plan<dim> p({{ 0.3, 0.0}}, 0.);

        auto out = closest_points(p, s)[0];

        REQUIRE(out.pi(0) == doctest::Approx(0.1));
        REQUIRE(out.pi(1) == doctest::Approx(0.));
        REQUIRE(out.pj(0) == doctest::Approx(0.3));
        REQUIRE(out.pj(1) == doctest::Approx(0.));
        REQUIRE(out.nij(0) == doctest::Approx(-1.));
        REQUIRE(out.nij(1) == doctest::Approx(0.));
        REQUIRE(out.dij == doctest::Approx(0.2));
    }

    TEST_CASE("plan_superellipsoid_2d_rotation_30_deg")
    {
        // FIXME Newton does not converge
        constexpr std::size_t dim = 2;
        superellipsoid<dim> s({{ 0., 0.}}, {quaternion(0.)}, {{0.1, 0.2}}, 1);
        double dist = 0.3;
        double cosRot = std::sqrt(3.)/2.;
        double sinRot = 1./2.;
        plan<dim> p({{dist*cosRot, dist*sinRot}}, PI/6.);

        auto out = closest_points(p, s)[0];

        REQUIRE(out.pi(0) == doctest::Approx(0.1*cosRot));
        REQUIRE(out.pi(1) == doctest::Approx(0.1*sinRot));
        REQUIRE(out.pj(0) == doctest::Approx(0.3*cosRot));
        REQUIRE(out.pj(1) == doctest::Approx(0.3*sinRot));
        REQUIRE(out.nij(0) == doctest::Approx(-cosRot));
        REQUIRE(out.nij(1) == doctest::Approx(-sinRot));
        REQUIRE(out.dij == doctest::Approx(0.2));
    }

    TEST_CASE("plan_superellipsoid_2d_dispatch")
    {
        // FIXME Newton does not converge
        constexpr std::size_t dim = 2;
        superellipsoid<dim> s({{ 0.0, 0.0}}, {quaternion(0.)}, {{0.1, 0.2}}, 1);
        plan<dim> p({{ 0.3, 0.0}}, 0.);

        scopi_container<dim> particles;
        particles.push_back(p);
        particles.push_back(s);

        auto out = closest_points_dispatcher<dim>::dispatch(*particles[0], *particles[1])[0];

        REQUIRE(out.pi(0) == doctest::Approx(0.1));
        REQUIRE(out.pi(1) == doctest::Approx(0.));
        REQUIRE(out.pj(0) == doctest::Approx(0.3));
        REQUIRE(out.pj(1) == doctest::Approx(0.));
        REQUIRE(out.nij(0) == doctest::Approx(-1.));
        REQUIRE(out.nij(1) == doctest::Approx(0.));
        REQUIRE(out.dij == doctest::Approx(0.2));
    }

    TEST_CASE("plan_superellipsoid_2d_dispatch_rotation_30_deg")
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

        auto out = closest_points_dispatcher<dim>::dispatch(*particles[0], *particles[1])[0];

        REQUIRE(out.pi(0) == doctest::Approx(0.1*cosRot));
        REQUIRE(out.pi(1) == doctest::Approx(0.1*sinRot));
        REQUIRE(out.pj(0) == doctest::Approx(0.3*cosRot));
        REQUIRE(out.pj(1) == doctest::Approx(0.3*sinRot));
        REQUIRE(out.nij(0) == doctest::Approx(-cosRot));
        REQUIRE(out.nij(1) == doctest::Approx(-sinRot));
        REQUIRE(out.dij == doctest::Approx(0.2));
    }

    TEST_CASE("plan_superellipsoid_3d")
    {
        // FIXME Newton does not converge
        constexpr std::size_t dim = 3;
        superellipsoid<dim> s({{ 0.0, 0.0, 0.0}}, {quaternion(0.)}, {{0.1, 0.2, 0.3}}, {1, 1});
        plan<dim> p({{ 0.3, 0.0, 0.0}}, 0.);

        auto out = closest_points(p, s)[0];

        REQUIRE(out.pi(0) == doctest::Approx(0.1));
        REQUIRE(out.pi(1) == doctest::Approx(0.));
        REQUIRE(out.pi(2) == doctest::Approx(0.));
        REQUIRE(out.pj(0) == doctest::Approx(0.3));
        REQUIRE(out.pj(1) == doctest::Approx(0.));
        REQUIRE(out.pj(2) == doctest::Approx(0.));
        REQUIRE(out.nij(0) == doctest::Approx(-1.));
        REQUIRE(out.nij(1) == doctest::Approx(0.));
        REQUIRE(out.nij(2) == doctest::Approx(0.));
        REQUIRE(out.dij == doctest::Approx(0.2));
    }

    TEST_CASE("plan_superellipsoid_3d_rotation_30_deg")
    {
        // FIXME Newton does not converge
        constexpr std::size_t dim = 3;
        superellipsoid<dim> s({{ 0., 0., 0.}}, {quaternion(0.)}, {{0.1, 0.2, 0.3}}, {1, 1});
        double dist = 0.3;
        double cosRot = std::sqrt(3.)/2.;
        double sinRot = 1./2.;
        plan<dim> p({{dist*cosRot, dist*sinRot, 0.}}, PI/6.);

        auto out = closest_points(p, s)[0];

        REQUIRE(out.pi(0) == doctest::Approx(0.1*cosRot));
        REQUIRE(out.pi(1) == doctest::Approx(0.1*sinRot));
        REQUIRE(out.pi(2) == doctest::Approx(0.));
        REQUIRE(out.pj(0) == doctest::Approx(0.3*cosRot));
        REQUIRE(out.pj(1) == doctest::Approx(0.3*sinRot));
        REQUIRE(out.pj(2) == doctest::Approx(0.));
        REQUIRE(out.nij(0) == doctest::Approx(-cosRot));
        REQUIRE(out.nij(1) == doctest::Approx(-sinRot));
        REQUIRE(out.nij(2) == doctest::Approx(0.));
        REQUIRE(out.dij == doctest::Approx(0.2));
    }

    TEST_CASE("plan_superellipsoid_3d_dispatch")
    {
        // FIXME Newton does not converge
        constexpr std::size_t dim = 3;
        superellipsoid<dim> s({{ 0.0, 0.0, 0.0}}, {quaternion(0.)}, {{0.1, 0.2, 0.3}}, {1, 1});
        plan<dim> p({{ 0.3, 0.0, 0.0}}, 0.);

        scopi_container<dim> particles;
        particles.push_back(p);
        particles.push_back(s);

        auto out = closest_points_dispatcher<dim>::dispatch(*particles[0], *particles[1])[0];

        REQUIRE(out.pi(0) == doctest::Approx(0.1));
        REQUIRE(out.pi(1) == doctest::Approx(0.));
        REQUIRE(out.pi(2) == doctest::Approx(0.));
        REQUIRE(out.pj(0) == doctest::Approx(0.3));
        REQUIRE(out.pj(1) == doctest::Approx(0.));
        REQUIRE(out.pj(2) == doctest::Approx(0.));
        REQUIRE(out.nij(0) == doctest::Approx(-1.));
        REQUIRE(out.nij(1) == doctest::Approx(0.));
        REQUIRE(out.nij(2) == doctest::Approx(0.));
        REQUIRE(out.dij == doctest::Approx(0.2));
    }

    TEST_CASE("plan_superellipsoid_3d_dispatch_rotation_30_deg")
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

        auto out = closest_points_dispatcher<dim>::dispatch(*particles[0], *particles[1])[0];

        REQUIRE(out.pi(0) == doctest::Approx(0.1*cosRot));
        REQUIRE(out.pi(1) == doctest::Approx(0.1*sinRot));
        REQUIRE(out.pi(2) == doctest::Approx(0.));
        REQUIRE(out.pj(0) == doctest::Approx(0.3*cosRot));
        REQUIRE(out.pj(1) == doctest::Approx(0.3*sinRot));
        REQUIRE(out.pj(2) == doctest::Approx(0.));
        REQUIRE(out.nij(0) == doctest::Approx(-cosRot));
        REQUIRE(out.nij(1) == doctest::Approx(-sinRot));
        REQUIRE(out.nij(2) == doctest::Approx(0.));
        REQUIRE(out.dij == doctest::Approx(0.2));
    }

    // distance globule - globule
    void check_neigh_globule_globule(neighbor<2>& out, double pi, double pj, double dij)
    {
        REQUIRE(out.pi(0) == doctest::Approx(pi));
        REQUIRE(out.pi(1) == doctest::Approx(0.));
        REQUIRE(out.pj(0) == doctest::Approx(pj));
        REQUIRE(out.pj(1) == doctest::Approx(0.));
        REQUIRE(out.nij(0) == doctest::Approx(1.));
        REQUIRE(out.nij(1) == doctest::Approx(0.));
        REQUIRE(out.dij == doctest::Approx(dij));
    }

    TEST_CASE("globule_globule_2d")
    {
        constexpr std::size_t dim = 2;
        globule<dim> g1({{3., 0.}, {5., 0.}, {7., 0.}, {9., 0.}, {11., 0.}, {13., 0.}}, 1.);
        globule<dim> g2({{-3., 0.}, {-5., 0.}, {-7., 0.}, {-9., 0.}, {-11., 0.}, {-13., 0.}}, 1.);

        auto out = closest_points(g1, g2);

        check_neigh_globule_globule(out[0], 2., -2., 4.);
        check_neigh_globule_globule(out[1], 2., -4., 6.);
        check_neigh_globule_globule(out[2], 2., -6., 8.);
        check_neigh_globule_globule(out[3], 2., -8., 10.);
        check_neigh_globule_globule(out[4], 2., -10., 12.);
        check_neigh_globule_globule(out[5], 2., -12., 14.);

        check_neigh_globule_globule(out[6], 4., -2., 6.);
        check_neigh_globule_globule(out[7], 4., -4., 8.);
        check_neigh_globule_globule(out[8], 4., -6., 10.);
        check_neigh_globule_globule(out[9], 4., -8., 12.);
        check_neigh_globule_globule(out[10], 4., -10., 14.);
        check_neigh_globule_globule(out[11], 4., -12., 16.);

        check_neigh_globule_globule(out[12], 6., -2., 8.);
        check_neigh_globule_globule(out[13], 6., -4., 10.);
        check_neigh_globule_globule(out[14], 6., -6., 12.);
        check_neigh_globule_globule(out[15], 6., -8., 14.);
        check_neigh_globule_globule(out[16], 6., -10., 16.);
        check_neigh_globule_globule(out[17], 6., -12., 18.);

        check_neigh_globule_globule(out[18], 8., -2., 10.);
        check_neigh_globule_globule(out[19], 8., -4., 12.);
        check_neigh_globule_globule(out[20], 8., -6., 14.);
        check_neigh_globule_globule(out[21], 8., -8., 16.);
        check_neigh_globule_globule(out[22], 8., -10., 18.);
        check_neigh_globule_globule(out[23], 8., -12., 20.);

        check_neigh_globule_globule(out[24], 10., -2., 12.);
        check_neigh_globule_globule(out[25], 10., -4., 14.);
        check_neigh_globule_globule(out[26], 10., -6., 16.);
        check_neigh_globule_globule(out[27], 10., -8., 18.);
        check_neigh_globule_globule(out[28], 10., -10., 20.);
        check_neigh_globule_globule(out[29], 10., -12., 22.);

        check_neigh_globule_globule(out[30], 12., -2., 14.);
        check_neigh_globule_globule(out[31], 12., -4., 16.);
        check_neigh_globule_globule(out[32], 12., -6., 18.);
        check_neigh_globule_globule(out[33], 12., -8., 20.);
        check_neigh_globule_globule(out[34], 12., -10., 22.);
        check_neigh_globule_globule(out[35], 12., -12., 24.);
    }

    TEST_CASE("globule_globule_2d_dispatch")
    {
        constexpr std::size_t dim = 2;
        globule<dim> g1({{3., 0.}, {5., 0.}, {7., 0.}, {9., 0.}, {11., 0.}, {13., 0.}}, 1.);
        globule<dim> g2({{-3., 0.}, {-5., 0.}, {-7., 0.}, {-9., 0.}, {-11., 0.}, {-13., 0.}}, 1.);

        scopi_container<dim> particles;
        particles.push_back(g1);
        particles.push_back(g2);

        auto out = closest_points_dispatcher<dim>::dispatch(*particles[0], *particles[1]);

        check_neigh_globule_globule(out[0], 2., -2., 4.);
        check_neigh_globule_globule(out[1], 2., -4., 6.);
        check_neigh_globule_globule(out[2], 2., -6., 8.);
        check_neigh_globule_globule(out[3], 2., -8., 10.);
        check_neigh_globule_globule(out[4], 2., -10., 12.);
        check_neigh_globule_globule(out[5], 2., -12., 14.);

        check_neigh_globule_globule(out[6], 4., -2., 6.);
        check_neigh_globule_globule(out[7], 4., -4., 8.);
        check_neigh_globule_globule(out[8], 4., -6., 10.);
        check_neigh_globule_globule(out[9], 4., -8., 12.);
        check_neigh_globule_globule(out[10], 4., -10., 14.);
        check_neigh_globule_globule(out[11], 4., -12., 16.);

        check_neigh_globule_globule(out[12], 6., -2., 8.);
        check_neigh_globule_globule(out[13], 6., -4., 10.);
        check_neigh_globule_globule(out[14], 6., -6., 12.);
        check_neigh_globule_globule(out[15], 6., -8., 14.);
        check_neigh_globule_globule(out[16], 6., -10., 16.);
        check_neigh_globule_globule(out[17], 6., -12., 18.);

        check_neigh_globule_globule(out[18], 8., -2., 10.);
        check_neigh_globule_globule(out[19], 8., -4., 12.);
        check_neigh_globule_globule(out[20], 8., -6., 14.);
        check_neigh_globule_globule(out[21], 8., -8., 16.);
        check_neigh_globule_globule(out[22], 8., -10., 18.);
        check_neigh_globule_globule(out[23], 8., -12., 20.);

        check_neigh_globule_globule(out[24], 10., -2., 12.);
        check_neigh_globule_globule(out[25], 10., -4., 14.);
        check_neigh_globule_globule(out[26], 10., -6., 16.);
        check_neigh_globule_globule(out[27], 10., -8., 18.);
        check_neigh_globule_globule(out[28], 10., -10., 20.);
        check_neigh_globule_globule(out[29], 10., -12., 22.);

        check_neigh_globule_globule(out[30], 12., -2., 14.);
        check_neigh_globule_globule(out[31], 12., -4., 16.);
        check_neigh_globule_globule(out[32], 12., -6., 18.);
        check_neigh_globule_globule(out[33], 12., -8., 20.);
        check_neigh_globule_globule(out[34], 12., -10., 22.);
        check_neigh_globule_globule(out[35], 12., -12., 24.);
    }

}
