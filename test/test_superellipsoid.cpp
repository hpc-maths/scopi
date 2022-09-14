#include "doctest/doctest.h"
#include <random>

#include "test_common.hpp"
#include "utils.hpp"

#include <scopi/objects/types/superellipsoid.hpp>
#include <scopi/container.hpp>
#include <scopi/solver.hpp>
#include <scopi/property.hpp>

namespace scopi
{
    TEST_CASE("Superellipsoid 2D")
    {
        static constexpr std::size_t dim = 2;
        superellipsoid<dim> s({{-0.2, 0.3}}, {quaternion(0)}, {{0.1, 0.2}}, 1.5);
        property<dim> p(property<dim>().desired_velocity({{0.25, 0}}));
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
            auto radius = s.radius();
            REQUIRE(radius(0) == doctest::Approx(0.1));
            REQUIRE(radius(1) == doctest::Approx(0.2));
        }

        SUBCASE("squareness")
        {
            auto squareness = s.squareness();
            REQUIRE(squareness(0) == doctest::Approx(1.5));
        }

        SUBCASE("point x")
        {
            auto point = s.point(0.);
            REQUIRE(point(0) == doctest::Approx(-0.1));
            REQUIRE(point(1) == doctest::Approx(0.3));
        }

        SUBCASE("point y")
        {
            auto point = s.point(PI/2.);
            REQUIRE(point(0) == doctest::Approx(-0.2));
            REQUIRE(point(1) == doctest::Approx(0.5));
        }

        SUBCASE("normal x")
        {
            auto normal = s.normal(0.);
            REQUIRE(normal(0) == doctest::Approx(1.));
            REQUIRE(normal(1) == doctest::Approx(0.));
        }

        SUBCASE("normal y")
        {
            auto normal = s.normal(PI/2.);
            REQUIRE(normal(0) == doctest::Approx(0.));
            REQUIRE(normal(1) == doctest::Approx(1.));
        }

        SUBCASE("tangent x")
        {
            auto tangent = s.tangent(0.);
            REQUIRE(tangent(0) == doctest::Approx(0.));
            REQUIRE(tangent(1) == doctest::Approx(1.));
        }

        SUBCASE("tangent y")
        {
            auto tangent = s.tangent(PI/2.);
            REQUIRE(tangent(0) == doctest::Approx(-1.));
            REQUIRE(tangent(1) == doctest::Approx(0.));
        }
    }

    TEST_CASE("Superellipsoid 2D const")
    {
        static constexpr std::size_t dim = 2;
        const superellipsoid<dim> s({{-0.2, 0.3}}, {quaternion(0)}, {{0.1, 0.2}}, 1.5);
        const property<dim> p(property<dim>().desired_velocity({{0.25, 0}}));
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

    TEST_CASE("Superellipsoid 2D rotation")
    {
        static constexpr std::size_t dim = 2;
        superellipsoid<dim> s({{-0.2, 0.3}}, {quaternion(PI/3)}, {{0.1, 0.2}}, 1.5);
        property<dim> p(property<dim>().desired_velocity({{0.25, 0}}));

        auto rotation_matrix = s.rotation();
        REQUIRE(rotation_matrix(0, 0) == doctest::Approx(1./2.));
        REQUIRE(rotation_matrix(0, 1) == doctest::Approx(-std::sqrt(3.)/2.));
        REQUIRE(rotation_matrix(1, 0) == doctest::Approx(std::sqrt(3.)/2.));
        REQUIRE(rotation_matrix(1, 1) == doctest::Approx(1./2.));
    }

    TEST_CASE("Superellipsoid 3D")
    {
        static constexpr std::size_t dim = 3;
        superellipsoid<dim> s({{-0.2, 0.3, 0.1}}, {quaternion(0)}, {{0.1, 0.2, 0.3}}, {{0.5, 1.5}});
        property<dim> p(property<dim>().desired_velocity({{0.25, 0, 0}}));
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

        SUBCASE("radius")
        {
            auto radius = s.radius();
            REQUIRE(radius(0) == doctest::Approx(0.1));
            REQUIRE(radius(1) == doctest::Approx(0.2));
            REQUIRE(radius(2) == doctest::Approx(0.3));
        }

        SUBCASE("squareness")
        {
            auto squareness = s.squareness();
            REQUIRE(squareness(0) == doctest::Approx(0.5));
            REQUIRE(squareness(1) == doctest::Approx(1.5));
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
            auto point = s.point(0., PI/2.);
            REQUIRE(point(0) == doctest::Approx(-0.2));
            REQUIRE(point(1) == doctest::Approx(0.5));
            REQUIRE(point(2) == doctest::Approx(0.1));
        }

        SUBCASE("point z")
        {
            auto point = s.point(PI/2., 0.);
            REQUIRE(point(0) == doctest::Approx(-0.2));
            REQUIRE(point(1) == doctest::Approx(0.3));
            REQUIRE(point(2) == doctest::Approx(0.4));
        }

        SUBCASE("normal x")
        {
            auto normal = s.normal(0., 0.);
            REQUIRE(normal(0) == doctest::Approx(1.));
            REQUIRE(normal(1) == doctest::Approx(0.));
            REQUIRE(normal(2) == doctest::Approx(0.));
        }

        SUBCASE("normal y")
        {
            auto normal = s.normal(0., PI/2.);
            REQUIRE(normal(0) == doctest::Approx(0.));
            REQUIRE(normal(1) == doctest::Approx(1.));
            REQUIRE(normal(2) == doctest::Approx(0.));
        }

        SUBCASE("normal z")
        {
            auto normal = s.normal(PI/2., 0.);
            REQUIRE(normal(0) == doctest::Approx(0.));
            REQUIRE(normal(1) == doctest::Approx(0.));
            REQUIRE(normal(2) == doctest::Approx(1.));
        }

        SUBCASE("tangent x")
        {
            auto tangent = s.tangents(0., 0.);
            REQUIRE(tangent.first(0) == doctest::Approx(0.));
            REQUIRE(tangent.first(1) == doctest::Approx(1.));
            REQUIRE(tangent.first(2) == doctest::Approx(0.));

            REQUIRE(tangent.second(0) == doctest::Approx(0.));
            REQUIRE(tangent.second(1) == doctest::Approx(0.));
            REQUIRE(tangent.second(2) == doctest::Approx(1.));
        }

        SUBCASE("tangent y")
        {
            auto tangent = s.tangents(0., PI/2.);
            REQUIRE(tangent.first(0) == doctest::Approx(-1.));
            REQUIRE(tangent.first(1) == doctest::Approx(0.));
            REQUIRE(tangent.first(2) == doctest::Approx(0.));

            REQUIRE(tangent.second(0) == doctest::Approx(0.));
            REQUIRE(tangent.second(1) == doctest::Approx(0.));
            REQUIRE(tangent.second(2) == doctest::Approx(1.));
        }

        SUBCASE("tangent z")
        {
            auto tangent = s.tangents(PI/2., 0.);
            REQUIRE(tangent.first(0) == doctest::Approx(0.));
            REQUIRE(tangent.first(1) == doctest::Approx(1.));
            REQUIRE(tangent.first(2) == doctest::Approx(0.));

            REQUIRE(tangent.second(0) == doctest::Approx(-1.));
            REQUIRE(tangent.second(1) == doctest::Approx(0.));
            REQUIRE(tangent.second(2) == doctest::Approx(0.));
        }
    }

    TEST_CASE("Superellipsoid 3D const")
    {
        static constexpr std::size_t dim = 3;
        const superellipsoid<dim> s({{-0.2, 0.3, 0.1}}, {quaternion(0)}, {{0.1, 0.2, 0.3}}, {{0.5, 1.5}});
        const property<dim> p(property<dim>().desired_velocity({{0.25, 0, 0}}));
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
    }

    TEST_CASE("Superellipsoid 3D rotation")
    {
        static constexpr std::size_t dim = 3;
        superellipsoid<dim> s({{-0.2, 0.3, 0.1}}, {quaternion(PI/3)}, {{0.1, 0.2, 0.3}}, {{0.5, 1.5}});
        property<dim> p(property<dim>().desired_velocity({{0.25, 0, 0}}));

        auto rotation_matrix = s.rotation();
        REQUIRE(rotation_matrix(0, 0) == doctest::Approx(1./2.));
        REQUIRE(rotation_matrix(0, 1) == doctest::Approx(-std::sqrt(3.)/2.));
        REQUIRE(rotation_matrix(0, 2) == doctest::Approx(0.));
        REQUIRE(rotation_matrix(1, 0) == doctest::Approx(std::sqrt(3.)/2.));
        REQUIRE(rotation_matrix(1, 1) == doctest::Approx(1./2.));
        REQUIRE(rotation_matrix(1, 2) == doctest::Approx(0.));
        REQUIRE(rotation_matrix(2, 0) == doctest::Approx(0.));
        REQUIRE(rotation_matrix(2, 1) == doctest::Approx(0.));
        REQUIRE(rotation_matrix(2, 2) == doctest::Approx(1.));
    }

    TEST_CASE_TEMPLATE("two ellispsoids symetrical", SolverType, SOLVER_DRY_WITHOUT_FRICTION(2, contact_kdtree, vap_fixed), SOLVER_DRY_WITHOUT_FRICTION(2, contact_brute_force, vap_fixed))
    {
        using params_t = typename SolverType::params_t;
        static constexpr std::size_t dim = 2;
        double dt = .005;
        std::size_t total_it = 200;

        superellipsoid<dim> s1({{-0.2, 0.}}, {quaternion(PI/4)}, {{.1, .05}}, 1);
        superellipsoid<dim> s2({{0.2, 0.}}, {quaternion(-PI/4)}, {{.1, .05}}, 1);
        auto p = property<dim>().desired_velocity({{0.25, 0}}).mass(1.).moment_inertia(0.1);

        scopi_container<dim> particles;
        particles.push_back(s1, p);
        particles.push_back(s2, p.desired_velocity({{-0.25, 0}}));

        params_t params;
        params.scopi_params.output_frequency = total_it-1;

        SolverType solver(particles, dt, params);
        solver.run(total_it);

        CHECK(diffFile("./Results/scopi_objects_0199.json", "../test/references/two_ellipsoids_symmetrical.json", tolerance));
    }

    TEST_CASE_TEMPLATE("two ellispsoids spheres symetrical", SolverType, SOLVER_DRY_WITHOUT_FRICTION(2, contact_kdtree, vap_fixed), SOLVER_DRY_WITHOUT_FRICTION(2, contact_brute_force, vap_fixed))
    {
        using params_t = typename SolverType::params_t;
        static constexpr std::size_t dim = 2;
        double dt = .005;
        std::size_t total_it = 50;

        superellipsoid<dim> s1({{-0.2, 0.}}, {quaternion(PI/4)}, {{.1, .1}}, 1);
        superellipsoid<dim> s2({{0.2, 0.}}, {quaternion(-PI/4)}, {{.1, .1}}, 1);
        auto p = property<2>().mass(1.).moment_inertia(0.1);

        scopi_container<dim> particles;
        particles.push_back(s1, p.desired_velocity({{0.25, 0}}));
        particles.push_back(s2, p.desired_velocity({{-0.25, 0}}));

        params_t params;
        params.scopi_params.output_frequency = total_it-1;

        SolverType solver(particles, dt, params);
        solver.run(total_it);

        CHECK(diffFile("./Results/scopi_objects_0049.json", "../test/references/two_ellipsoids_spheres_symmetrical.json", tolerance));
    }

    TEST_CASE_TEMPLATE("two ellispsoids asymetrical", SolverType, SOLVER_DRY_WITHOUT_FRICTION(2, contact_kdtree, vap_fixed), SOLVER_DRY_WITHOUT_FRICTION(2, contact_brute_force, vap_fixed))
    {
        using params_t = typename SolverType::params_t;
        static constexpr std::size_t dim = 2;
        double dt = .005;
        std::size_t total_it = 1000;

        superellipsoid<dim> s1({{-0.2, -0.05}}, {quaternion(PI/4)}, {{.1, .05}}, 1);
        superellipsoid<dim> s2({{0.2, 0.05}}, {quaternion(-PI/4)}, {{.1, .05}}, 1);
        auto p = property<dim>().mass(1.).moment_inertia(0.1);

        scopi_container<dim> particles;
        particles.push_back(s1, p.desired_velocity({{0.25, 0}}));
        particles.push_back(s2, p.desired_velocity({{-0.25, 0}}));

        params_t params;
        params.scopi_params.output_frequency = total_it-1;

        SolverType solver(particles, dt, params);
        solver.run(total_it);

        CHECK(diffFile("./Results/scopi_objects_0999.json", "../test/references/two_ellipsoids_asymmetrical.json", tolerance));
    }

    TEST_CASE_TEMPLATE("two ellispsoids spheres asymetrical", SolverType, SOLVER_DRY_WITHOUT_FRICTION(2, contact_kdtree, vap_fixed), SOLVER_DRY_WITHOUT_FRICTION(2, contact_brute_force, vap_fixed))
    {
        using params_t = typename SolverType::params_t;
        static constexpr std::size_t dim = 2;
        double dt = .005;
        std::size_t total_it = 1000;

        superellipsoid<dim> s1({{-0.2, -0.05}}, {quaternion(PI/4)}, {{.1, .1}}, 1);
        superellipsoid<dim> s2({{0.2, 0.05}}, {quaternion(-PI/4)}, {{.1, .1}}, 1);
        auto p = property<dim>().desired_velocity({{0.25, 0}}).mass(1.).moment_inertia(0.1);

        scopi_container<dim> particles;
        particles.push_back(s1, p);
        particles.push_back(s2, p.desired_velocity({{-0.25, 0}}));

        params_t params;
        params.scopi_params.output_frequency = total_it-1;

        SolverType solver(particles, dt, params);
        solver.run(total_it);

        CHECK(diffFile("./Results/scopi_objects_0999.json", "../test/references/two_ellipsoids_spheres_asymmetrical.json", tolerance));
    }

    TEST_CASE_TEMPLATE("critical 2d superellipsoids", SolverType, SOLVER_DRY_WITHOUT_FRICTION(2, contact_kdtree, vap_fixed), SOLVER_DRY_WITHOUT_FRICTION(2, contact_brute_force, vap_fixed))
    {
        using params_t = typename SolverType::params_t;
        static constexpr std::size_t dim = 2;
        double dt = .01;
        std::size_t total_it = 20;
        scopi_container<dim> particles;

        int n = 3; // 2*n*n particles
        std::minstd_rand0 generator(123);
        std::uniform_real_distribution<double> distrib_r(0.2, 0.4);
        std::uniform_real_distribution<double> distrib_r2(0.2, 0.4);
        std::uniform_real_distribution<double> distrib_move_x(-0.1, 0.1);
        std::uniform_real_distribution<double> distrib_move_y(-0.1, 0.1);
        std::uniform_real_distribution<double> distrib_rot(0, PI);
        std::uniform_real_distribution<double> distrib_velocity(2., 5.);
        auto prop = property<dim>().mass(1.).moment_inertia(0.1);

        for(int i = 0; i < n; ++i)
        {
            for(int j = 0; j < n; ++j)
            {
                double rot = distrib_rot(generator);
                double r = distrib_r(generator);
                double r2 = distrib_r2(generator);
                double x = (i + 0.5) + distrib_move_x(generator);
                double y = (j + 0.5) + distrib_move_y(generator);
                double velocity = distrib_velocity(generator);

                superellipsoid<dim> s1({ {x, y}}, {quaternion(rot)}, {{r, r2}}, 1);
                particles.push_back(s1,prop.desired_velocity({{velocity, 0.}}));

                rot = distrib_rot(generator);
                r = distrib_r(generator);
                r2 = distrib_r2(generator);
                x = (n + i + 0.5) + distrib_move_x(generator);
                y = (j + 0.5) + distrib_move_y(generator);
                velocity = distrib_velocity(generator);

                superellipsoid<dim> s2({ {x, y}}, {quaternion(rot)}, {{r, r2}}, 1);
                particles.push_back(s2, prop.desired_velocity({{-velocity, 0.}}));
            }
        }

        params_t params;
        params.scopi_params.output_frequency = total_it-1;

        SolverType solver(particles, dt, params);
        solver.run(total_it);

        CHECK(diffFile("./Results/scopi_objects_0999.json", "../test/references/two_ellipsoids_spheres_asymmetrical.json", tolerance));
    }
}
