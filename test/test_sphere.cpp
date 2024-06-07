#include <doctest/doctest.h>
#include <random>

#include "test_common.hpp"
#include "utils.hpp"

#include <scopi/container.hpp>
#include <scopi/objects/types/sphere.hpp>
#include <scopi/property.hpp>
#include <scopi/solver.hpp>
#include <scopi/vap/vap_fpd.hpp>

namespace scopi
{
    TEST_CASE("Sphere 2D")
    {
        static constexpr std::size_t dim = 2;
        sphere<dim> s(
            {
                {-0.2, 0.3}
        },
            0.1);
        property<dim> p(property<dim>().desired_velocity({
            {0.25, 0}
        }));

        scopi_container<dim> particles;
        particles.push_back(s, p);

        SUBCASE("pos")
        {
            REQUIRE(s.pos()(0) == doctest::Approx(-0.2));
            REQUIRE(s.pos()(1) == doctest::Approx(0.3));
        }

        SUBCASE("pos index")
        {
            REQUIRE(s.pos(0)(0) == doctest::Approx(-0.2));
            REQUIRE(s.pos(0)(1) == doctest::Approx(0.3));
        }

        SUBCASE("pos container")
        {
            REQUIRE(particles[0]->pos()(0) == doctest::Approx(-0.2));
            REQUIRE(particles[0]->pos()(1) == doctest::Approx(0.3));
        }

        SUBCASE("pos index container")
        {
            REQUIRE(particles[0]->pos(0)(0) == doctest::Approx(-0.2));
            REQUIRE(particles[0]->pos(0)(1) == doctest::Approx(0.3));
        }

        SUBCASE("q")
        {
            REQUIRE(s.q()(0) == doctest::Approx(1.));
            REQUIRE(s.q()(1) == doctest::Approx(0.));
            REQUIRE(s.q()(2) == doctest::Approx(0.));
            REQUIRE(s.q()(3) == doctest::Approx(0.));
        }

        SUBCASE("q index")
        {
            REQUIRE(s.q(0)(0) == doctest::Approx(1.));
            REQUIRE(s.q(0)(1) == doctest::Approx(0.));
            REQUIRE(s.q(0)(2) == doctest::Approx(0.));
            REQUIRE(s.q(0)(3) == doctest::Approx(0.));
        }

        SUBCASE("q container")
        {
            REQUIRE(particles[0]->q()(0) == doctest::Approx(1.));
            REQUIRE(particles[0]->q()(1) == doctest::Approx(0.));
            REQUIRE(particles[0]->q()(2) == doctest::Approx(0.));
            REQUIRE(particles[0]->q()(3) == doctest::Approx(0.));
        }

        SUBCASE("q index container")
        {
            REQUIRE(particles[0]->q(0)(0) == doctest::Approx(1.));
            REQUIRE(particles[0]->q(0)(1) == doctest::Approx(0.));
            REQUIRE(particles[0]->q(0)(2) == doctest::Approx(0.));
            REQUIRE(particles[0]->q(0)(3) == doctest::Approx(0.));
        }

        SUBCASE("radius")
        {
            REQUIRE(s.radius() == doctest::Approx(0.1));
        }

        SUBCASE("point x")
        {
            auto point = s.point(0.);
            REQUIRE(point(0) == doctest::Approx(-0.1));
            REQUIRE(point(1) == doctest::Approx(0.3));
        }

        SUBCASE("point y")
        {
            auto point = s.point(PI / 2.);
            REQUIRE(point(0) == doctest::Approx(-0.2));
            REQUIRE(point(1) == doctest::Approx(0.4));
        }

        SUBCASE("normal")
        {
            auto normal = s.normal(0.);
            REQUIRE(normal(0) == doctest::Approx(1.));
            REQUIRE(normal(1) == doctest::Approx(0.));
        }
    }

    TEST_CASE("Sphere 2D const")
    {
        static constexpr std::size_t dim = 2;
        const sphere<dim> s(
            {
                {-0.2, 0.3}
        },
            0.1);
        const property<dim> p(property<dim>().desired_velocity({
            {0.25, 0}
        }));
        scopi_container<dim> particles;
        particles.push_back(s, p);

        SUBCASE("pos")
        {
            REQUIRE(s.pos()(0) == doctest::Approx(-0.2));
            REQUIRE(s.pos()(1) == doctest::Approx(0.3));
        }

        SUBCASE("pos index")
        {
            REQUIRE(s.pos(0)(0) == doctest::Approx(-0.2));
            REQUIRE(s.pos(0)(1) == doctest::Approx(0.3));
        }

        SUBCASE("pos container")
        {
            REQUIRE(particles[0]->pos()(0) == doctest::Approx(-0.2));
            REQUIRE(particles[0]->pos()(1) == doctest::Approx(0.3));
        }

        SUBCASE("pos index container")
        {
            REQUIRE(particles[0]->pos(0)(0) == doctest::Approx(-0.2));
            REQUIRE(particles[0]->pos(0)(1) == doctest::Approx(0.3));
        }

        SUBCASE("q")
        {
            REQUIRE(s.q()(0) == doctest::Approx(1.));
            REQUIRE(s.q()(1) == doctest::Approx(0.));
            REQUIRE(s.q()(2) == doctest::Approx(0.));
            REQUIRE(s.q()(3) == doctest::Approx(0.));
        }

        SUBCASE("q index")
        {
            REQUIRE(s.q(0)(0) == doctest::Approx(1.));
            REQUIRE(s.q(0)(1) == doctest::Approx(0.));
            REQUIRE(s.q(0)(2) == doctest::Approx(0.));
            REQUIRE(s.q(0)(3) == doctest::Approx(0.));
        }

        SUBCASE("q container")
        {
            REQUIRE(particles[0]->q()(0) == doctest::Approx(1.));
            REQUIRE(particles[0]->q()(1) == doctest::Approx(0.));
            REQUIRE(particles[0]->q()(2) == doctest::Approx(0.));
            REQUIRE(particles[0]->q()(3) == doctest::Approx(0.));
        }

        SUBCASE("q index container")
        {
            REQUIRE(particles[0]->q(0)(0) == doctest::Approx(1.));
            REQUIRE(particles[0]->q(0)(1) == doctest::Approx(0.));
            REQUIRE(particles[0]->q(0)(2) == doctest::Approx(0.));
            REQUIRE(particles[0]->q(0)(3) == doctest::Approx(0.));
        }
    }

    TEST_CASE("Sphere 2D rotation")
    {
        static constexpr std::size_t dim = 2;
        sphere<dim> s(
            {
                {-0.2, 0.3}
        },
            {quaternion(PI / 3)},
            0.1);

        auto matrix = s.rotation();
        REQUIRE(matrix(0, 0) == doctest::Approx(1. / 2.));
        REQUIRE(matrix(0, 1) == doctest::Approx(-std::sqrt(3.) / 2.));
        REQUIRE(matrix(1, 0) == doctest::Approx(std::sqrt(3.) / 2.));
        REQUIRE(matrix(1, 1) == doctest::Approx(1. / 2.));
    }

    TEST_CASE("Sphere 3D")
    {
        static constexpr std::size_t dim = 3;
        sphere<dim> s(
            {
                {-0.2, 0.3, 0.1}
        },
            0.1);
        property<dim> p(property<dim>().desired_velocity({
            {0.25, 0, 0}
        }));
        scopi_container<dim> particles;
        particles.push_back(s, p);

        SUBCASE("pos")
        {
            REQUIRE(s.pos()(0) == doctest::Approx(-0.2));
            REQUIRE(s.pos()(1) == doctest::Approx(0.3));
            REQUIRE(s.pos()(2) == doctest::Approx(0.1));
        }

        SUBCASE("pos index")
        {
            REQUIRE(s.pos(0)(0) == doctest::Approx(-0.2));
            REQUIRE(s.pos(0)(1) == doctest::Approx(0.3));
            REQUIRE(s.pos(0)(2) == doctest::Approx(0.1));
        }

        SUBCASE("pos container")
        {
            REQUIRE(particles[0]->pos()(0) == doctest::Approx(-0.2));
            REQUIRE(particles[0]->pos()(1) == doctest::Approx(0.3));
            REQUIRE(particles[0]->pos()(2) == doctest::Approx(0.1));
        }

        SUBCASE("pos index container")
        {
            REQUIRE(particles[0]->pos(0)(0) == doctest::Approx(-0.2));
            REQUIRE(particles[0]->pos(0)(1) == doctest::Approx(0.3));
            REQUIRE(particles[0]->pos(0)(2) == doctest::Approx(0.1));
        }

        SUBCASE("point x")
        {
            auto point = s.point(0., 0.);
            REQUIRE(point(0) == doctest::Approx(-0.1));
            REQUIRE(point(1) == doctest::Approx(0.3));
            REQUIRE(point(2) == doctest::Approx(0.1));
        }

        SUBCASE("point y")
        {
            auto point = s.point(0., PI / 2.);
            REQUIRE(point(0) == doctest::Approx(-0.2));
            REQUIRE(point(1) == doctest::Approx(0.4));
            REQUIRE(point(2) == doctest::Approx(0.1));
        }

        SUBCASE("point z")
        {
            auto point = s.point(PI / 2., 0.);
            REQUIRE(point(0) == doctest::Approx(-0.2));
            REQUIRE(point(1) == doctest::Approx(0.3));
            REQUIRE(point(2) == doctest::Approx(0.2));
        }

        SUBCASE("normal")
        {
            auto normal = s.normal(0., 0.);
            REQUIRE(normal(0) == doctest::Approx(1.));
            REQUIRE(normal(1) == doctest::Approx(0.));
            REQUIRE(normal(2) == doctest::Approx(0.));
        }
    }

    TEST_CASE("Sphere 3D const")
    {
        static constexpr std::size_t dim = 3;
        const sphere<dim> s(
            {
                {-0.2, 0.3, 0.1}
        },
            0.1);
        const property<dim> p(property<dim>().desired_velocity({
            {0.25, 0, 0}
        }));
        scopi_container<dim> particles;
        particles.push_back(s, p);

        SUBCASE("pos")
        {
            REQUIRE(s.pos()(0) == doctest::Approx(-0.2));
            REQUIRE(s.pos()(1) == doctest::Approx(0.3));
            REQUIRE(s.pos()(2) == doctest::Approx(0.1));
        }

        SUBCASE("pos index")
        {
            REQUIRE(s.pos(0)(0) == doctest::Approx(-0.2));
            REQUIRE(s.pos(0)(1) == doctest::Approx(0.3));
            REQUIRE(s.pos(0)(2) == doctest::Approx(0.1));
        }

        SUBCASE("pos container")
        {
            REQUIRE(particles[0]->pos()(0) == doctest::Approx(-0.2));
            REQUIRE(particles[0]->pos()(1) == doctest::Approx(0.3));
            REQUIRE(particles[0]->pos()(2) == doctest::Approx(0.1));
        }

        SUBCASE("pos index container")
        {
            REQUIRE(particles[0]->pos(0)() == doctest::Approx(-0.2));
            REQUIRE(particles[0]->pos(1)() == doctest::Approx(0.3));
            REQUIRE(particles[0]->pos(2)() == doctest::Approx(0.1));
        }
    }

    TEST_CASE("Sphere 3D rotation")
    {
        static constexpr std::size_t dim = 3;
        sphere<dim> s(
            {
                {-0.2, 0.3, 0.1}
        },
            {quaternion(PI / 3)},
            0.1);

        auto matrix = s.rotation();
        REQUIRE(matrix(0, 0) == doctest::Approx(1. / 2.));
        REQUIRE(matrix(0, 1) == doctest::Approx(-std::sqrt(3.) / 2.));
        REQUIRE(matrix(0, 2) == doctest::Approx(0.));
        REQUIRE(matrix(1, 0) == doctest::Approx(std::sqrt(3.) / 2.));
        REQUIRE(matrix(1, 1) == doctest::Approx(1. / 2.));
        REQUIRE(matrix(1, 2) == doctest::Approx(0.));
        REQUIRE(matrix(2, 0) == doctest::Approx(0.));
        REQUIRE(matrix(2, 1) == doctest::Approx(0.));
        REQUIRE(matrix(2, 2) == doctest::Approx(1.));
    }

    TEST_CASE_TEMPLATE_DEFINE("two spheres asymetrical", SolverType, two_spheres_asymetrical)
    {
        static constexpr std::size_t dim = 2;
        double dt                        = .005;
        std::size_t total_it             = 1000;

        sphere<dim> s1(
            {
                {-0.2, -0.05}
        },
            0.1);
        sphere<dim> s2(
            {
                {0.2, 0.05}
        },
            0.1);
        auto p = property<dim>().mass(1.).moment_inertia(0.1);

        scopi_container<dim> particles;
        particles.push_back(s1,
                            p.desired_velocity({
                                {0.25, 0}
        }));
        particles.push_back(s2,
                            p.desired_velocity({
                                {-0.25, 0}
        }));

        SolverType solver(particles);
        auto params                           = solver.get_params();
        params.solver_params.output_frequency = 1; // total_it-1;
        solver.run(dt, total_it);

        CHECK(diffFile("./Results/scopi_objects_0999.json", "../test/references/two_spheres_asymmetrical.json", tolerance));
    }

    TEST_CASE_TEMPLATE_DEFINE("two spheres symetrical", SolverType, two_spheres_symetrical)
    {
        static constexpr std::size_t dim = 2;
        double dt                        = .005;
        std::size_t total_it             = 1000;

        sphere<dim> s1(
            {
                {-0.2, 0.}
        },
            0.1);
        sphere<dim> s2(
            {
                {0.2, 0.}
        },
            0.1);
        auto p = property<dim>().mass(1.).moment_inertia(0.1);

        scopi_container<dim> particles;
        particles.push_back(s1,
                            p.desired_velocity({
                                {0.25, 0}
        }));
        particles.push_back(s2,
                            p.desired_velocity({
                                {-0.25, 0}
        }));

        SolverType solver(particles);
        auto params                           = solver.get_params();
        params.solver_params.output_frequency = 1; // total_it-1;
        solver.run(dt, total_it);

        CHECK(diffFile("./Results/scopi_objects_0999.json", "../test/references/two_spheres_symmetrical.json", tolerance));
    }

    TEST_CASE_TEMPLATE_DEFINE("critical 2d spheres", SolverType, two_spheres_critical)
    {
        static constexpr std::size_t dim = 2;
        double dt                        = .01;
        std::size_t total_it             = 20;
        scopi_container<dim> particles;

        int n = 3; // 2*n*n particles
        std::minstd_rand0 generator(123);
        std::uniform_real_distribution<double> distrib_r(0.2, 0.4);
        std::uniform_real_distribution<double> distrib_move_x(-0.1, 0.1);
        std::uniform_real_distribution<double> distrib_move_y(-0.1, 0.1);
        std::uniform_real_distribution<double> distrib_velocity(2., 5.);
        auto prop = property<dim>().mass(1.).moment_inertia(0.1);

        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < n; ++j)
            {
                double r        = distrib_r(generator);
                double x        = (i + 0.5) + distrib_move_x(generator);
                double y        = (j + 0.5) + distrib_move_y(generator);
                double velocity = distrib_velocity(generator);
                sphere<dim> s1(
                    {
                        {x, y}
                },
                    r);
                particles.push_back(s1,
                                    prop.desired_velocity({
                                        {velocity, 0.}
                }));

                r        = distrib_r(generator);
                x        = (n + i + 0.5) + distrib_move_x(generator);
                y        = (j + 0.5) + distrib_move_y(generator);
                velocity = distrib_velocity(generator);
                sphere<dim> s2(
                    {
                        {x, y}
                },
                    r);
                particles.push_back(s2,
                                    prop.desired_velocity({
                                        {-velocity, 0.}
                }));
            }
        }

        SolverType solver(particles);
        auto params                           = solver.get_params();
        params.solver_params.output_frequency = 1; // total_it-1;

        solver.run(dt, total_it);

        CHECK(diffFile("./Results/scopi_objects_0019.json", "../test/references/2d_case_spheres.json", tolerance));
    }

    TEST_CASE_TEMPLATE_APPLY(two_spheres_asymetrical, solver_dry_without_friction_t<2>);
    TEST_CASE_TEMPLATE_APPLY(two_spheres_symetrical, solver_dry_without_friction_t<2>);
    TEST_CASE_TEMPLATE_APPLY(two_spheres_critical, solver_dry_without_friction_t<2>);

}
