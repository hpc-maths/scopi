#include "utils.hpp"
#include <doctest/doctest.h>

#include <scopi/container.hpp>
#include <scopi/objects/types/sphere.hpp>
#include <scopi/objects/types/superellipsoid.hpp>

#include <scopi/vap/vap_fpd.hpp>

namespace scopi
{
    TEST_CASE("Container 2d")
    {
        static constexpr std::size_t dim = 2;
        superellipsoid<dim> s1(
            {
                {-0.2, 0.1}
        },
            {quaternion(PI / 3)},
            {{.2, .05}},
            1);
        sphere<dim> s2(
            {
                {0.2, 0.05}
        },
            {quaternion(PI / 2)},
            0.1);
        auto p = property<dim>().omega(PI / 3).desired_omega(PI / 12);
        scopi_container<dim> particles;
        particles.push_back(s1,
                            p.velocity({
                                           {0.1, 0.2}
        })
                                .desired_velocity({{0.01, 0.02}})
                                .force({{1., 2.}})
                                .mass(1.)
                                .moment_inertia(0.1));
        particles.push_back(s2,
                            p.velocity({
                                           {0.4, 0.5}
        })
                                .desired_velocity({{0.04, 0.05}})
                                .force({{4., 5.}})
                                .mass(2.)
                                .moment_inertia(0.3));

        SUBCASE("size")
        {
            CHECK(particles.size() == 2);
        }

        SUBCASE("pos")
        {
            auto pos = particles.pos();
            REQUIRE(pos(0)(0) == doctest::Approx(-0.2));
            REQUIRE(pos(0)(1) == doctest::Approx(0.1));
            REQUIRE(pos(1)(0) == doctest::Approx(0.2));
            REQUIRE(pos(1)(1) == doctest::Approx(0.05));
        }

        SUBCASE("q")
        {
            auto q = particles.q();
            REQUIRE(q(0)(0) == doctest::Approx(std::sqrt(3.) / 2.));
            REQUIRE(q(0)(1) == doctest::Approx(0.));
            REQUIRE(q(0)(2) == doctest::Approx(0.));
            REQUIRE(q(0)(3) == doctest::Approx(1. / 2.));
            REQUIRE(q(1)(0) == doctest::Approx(std::sqrt(2.) / 2.));
            REQUIRE(q(1)(1) == doctest::Approx(0.));
            REQUIRE(q(1)(2) == doctest::Approx(0.));
            REQUIRE(q(1)(3) == doctest::Approx(std::sqrt(2.) / 2.));
        }

        SUBCASE("f")
        {
            auto f = particles.f();
            REQUIRE(f(0)(0) == doctest::Approx(1.));
            REQUIRE(f(0)(1) == doctest::Approx(2.));
            REQUIRE(f(1)(0) == doctest::Approx(4.));
            REQUIRE(f(1)(1) == doctest::Approx(5.));
        }

        SUBCASE("m")
        {
            auto m = particles.m();
            REQUIRE(m(0) == doctest::Approx(1.));
            REQUIRE(m(1) == doctest::Approx(2.));
        }

        SUBCASE("j")
        {
            auto j = particles.j();
            REQUIRE(j(0) == doctest::Approx(0.1));
        }

        SUBCASE("v")
        {
            auto v = particles.v();
            REQUIRE(v(0)(0) == doctest::Approx(0.1));
            REQUIRE(v(0)(1) == doctest::Approx(0.2));
            REQUIRE(v(1)(0) == doctest::Approx(0.4));
            REQUIRE(v(1)(1) == doctest::Approx(0.5));
        }

        SUBCASE("omega")
        {
            auto omega = particles.omega();
            REQUIRE(omega(0) == doctest::Approx(PI / 3.));
            REQUIRE(omega(1) == doctest::Approx(PI / 3.));
        }

        SUBCASE("desired_omega")
        {
            auto desired_omega = particles.desired_omega();
            REQUIRE(desired_omega(0) == doctest::Approx(PI / 12.));
            REQUIRE(desired_omega(1) == doctest::Approx(PI / 12.));
        }

        SUBCASE("vd")
        {
            auto vd = particles.vd();
            REQUIRE(vd(0)(0) == doctest::Approx(0.01));
            REQUIRE(vd(0)(1) == doctest::Approx(0.02));
            REQUIRE(vd(1)(0) == doctest::Approx(0.04));
            REQUIRE(vd(1)(1) == doctest::Approx(0.05));
        }

        SUBCASE("moments fpd sphere")
        {
            REQUIRE(cross_product_vap_fpd(particles, 1) == doctest::Approx(0.));
        }

        SUBCASE("moments fpd superellipsoid")
        {
            REQUIRE(cross_product_vap_fpd(particles, 0) == doctest::Approx(0.));
        }
    }

    TEST_CASE("Container 3d")
    {
        static constexpr std::size_t dim = 3;
        superellipsoid<dim> s1(
            {
                {-0.2, 0.1, 0.}
        },
            {quaternion(PI / 3)},
            {{.2, .05, 0.01}},
            {{1, 1}});
        sphere<dim> s2(
            {
                {0.2, 0.05, 0.}
        },
            {quaternion(PI / 2)},
            0.1);
        auto p = property<dim>()
                     .omega({
                         {PI / 3, PI / 4, PI / 5}
        })
                     .desired_omega({{PI / 12, PI / 15, PI / 18}});
        scopi_container<dim> particles;
        particles.push_back(s1,
                            p.velocity({
                                           {0.1, 0.2, 0.3}
        })
                                .desired_velocity({{0.01, 0.02, 0.03}})
                                .force({{
                                    1.,
                                    2.,
                                    3,
                                }})
                                .mass(1.)
                                .moment_inertia({{0.1, 0.2, 0.3}}));

        particles.push_back(s2,
                            p.velocity({
                                           {0.4, 0.5, 0.6}
        })
                                .desired_velocity({{0.04, 0.05, 0.06}})
                                .force({{
                                    4.,
                                    5.,
                                    6,
                                }})
                                .mass(2.)
                                .moment_inertia({0.1, 0.1, 0.1}));

        SUBCASE("size")
        {
            CHECK(particles.size() == 2);
        }

        SUBCASE("pos")
        {
            auto pos = particles.pos();
            REQUIRE(pos(0)(0) == doctest::Approx(-0.2));
            REQUIRE(pos(0)(1) == doctest::Approx(0.1));
            REQUIRE(pos(0)(2) == doctest::Approx(0.));
            REQUIRE(pos(1)(0) == doctest::Approx(0.2));
            REQUIRE(pos(1)(1) == doctest::Approx(0.05));
            REQUIRE(pos(1)(2) == doctest::Approx(0.));
        }

        SUBCASE("q")
        {
            auto q = particles.q();
            REQUIRE(q(0)(0) == doctest::Approx(std::sqrt(3.) / 2.));
            REQUIRE(q(0)(1) == doctest::Approx(0.));
            REQUIRE(q(0)(2) == doctest::Approx(0.));
            REQUIRE(q(0)(3) == doctest::Approx(1. / 2.));
            REQUIRE(q(1)(0) == doctest::Approx(std::sqrt(2.) / 2.));
            REQUIRE(q(1)(1) == doctest::Approx(0.));
            REQUIRE(q(1)(2) == doctest::Approx(0.));
            REQUIRE(q(1)(3) == doctest::Approx(std::sqrt(2.) / 2.));
        }

        SUBCASE("f")
        {
            auto f = particles.f();
            REQUIRE(f(0)(0) == doctest::Approx(1.));
            REQUIRE(f(0)(1) == doctest::Approx(2.));
            REQUIRE(f(0)(2) == doctest::Approx(3.));
            REQUIRE(f(1)(0) == doctest::Approx(4.));
            REQUIRE(f(1)(1) == doctest::Approx(5.));
            REQUIRE(f(1)(2) == doctest::Approx(6.));
        }

        SUBCASE("m")
        {
            auto m = particles.m();
            REQUIRE(m(0) == doctest::Approx(1.));
            REQUIRE(m(1) == doctest::Approx(2.));
        }

        SUBCASE("j")
        {
            auto j = particles.j();
            REQUIRE(j(0)(0) == doctest::Approx(0.1));
            REQUIRE(j(0)(1) == doctest::Approx(0.2));
            REQUIRE(j(0)(2) == doctest::Approx(0.3));
            REQUIRE(j(1)(0) == doctest::Approx(0.1));
            REQUIRE(j(1)(1) == doctest::Approx(0.1));
            REQUIRE(j(1)(2) == doctest::Approx(0.1));
        }

        SUBCASE("v")
        {
            auto v = particles.v();
            REQUIRE(v(0)(0) == doctest::Approx(0.1));
            REQUIRE(v(0)(1) == doctest::Approx(0.2));
            REQUIRE(v(0)(2) == doctest::Approx(0.3));
            REQUIRE(v(1)(0) == doctest::Approx(0.4));
            REQUIRE(v(1)(1) == doctest::Approx(0.5));
            REQUIRE(v(1)(2) == doctest::Approx(0.6));
        }

        SUBCASE("omega")
        {
            auto omega = particles.omega();
            REQUIRE(omega(0)[0] == doctest::Approx(PI / 3.));
            REQUIRE(omega(0)[1] == doctest::Approx(PI / 4.));
            REQUIRE(omega(0)[2] == doctest::Approx(PI / 5.));
            REQUIRE(omega(1)[0] == doctest::Approx(PI / 3.));
            REQUIRE(omega(1)[1] == doctest::Approx(PI / 4.));
            REQUIRE(omega(1)[2] == doctest::Approx(PI / 5.));
        }

        SUBCASE("desired_omega")
        {
            auto desired_omega = particles.desired_omega();
            REQUIRE(desired_omega(0)[0] == doctest::Approx(PI / 12.));
            REQUIRE(desired_omega(0)[1] == doctest::Approx(PI / 15.));
            REQUIRE(desired_omega(0)[2] == doctest::Approx(PI / 18.));
            REQUIRE(desired_omega(1)[0] == doctest::Approx(PI / 12.));
            REQUIRE(desired_omega(1)[1] == doctest::Approx(PI / 15.));
            REQUIRE(desired_omega(1)[2] == doctest::Approx(PI / 18.));
        }

        SUBCASE("vd")
        {
            auto vd = particles.vd();
            REQUIRE(vd(0)(0) == doctest::Approx(0.01));
            REQUIRE(vd(0)(1) == doctest::Approx(0.02));
            REQUIRE(vd(0)(2) == doctest::Approx(0.03));
            REQUIRE(vd(1)(0) == doctest::Approx(0.04));
            REQUIRE(vd(1)(1) == doctest::Approx(0.05));
            REQUIRE(vd(1)(2) == doctest::Approx(0.06));
        }

        SUBCASE("moments fpd sphere")
        {
            auto cross_product_sphere = cross_product_vap_fpd(particles, 1);
            REQUIRE(cross_product_sphere(0) == doctest::Approx(0.));
            REQUIRE(cross_product_sphere(1) == doctest::Approx(0.));
            REQUIRE(cross_product_sphere(2) == doctest::Approx(0.));
        }

        SUBCASE("moments fpd superellipsoid")
        {
            auto cross_product_superellipsoid = cross_product_vap_fpd(particles, 0);
            REQUIRE(cross_product_superellipsoid(0) == doctest::Approx(PI * PI / 20. * 0.1));
            REQUIRE(cross_product_superellipsoid(1) == doctest::Approx(-PI * PI / 15. * 0.2));
            REQUIRE(cross_product_superellipsoid(2) == doctest::Approx(PI * PI / 12. * 0.1));
        }
    }
}
