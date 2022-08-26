#include "doctest/doctest.h"
#include <random>

#include "test_common.hpp"
#include "utils.hpp"

#include <scopi/objects/types/worm.hpp>
#include <scopi/vap/vap_fpd.hpp>
#include <scopi/container.hpp>
#include <scopi/solver.hpp>
#include <scopi/property.hpp>

namespace scopi
{
    template <class solver_t>
    void set_params_test(OptimParams<solver_t>&)
    {}

    template <>
    void set_params_test<OptimMosek<DryWithoutFriction>>(OptimParams<OptimMosek<DryWithoutFriction>>& params)
    {
        params.change_default_tol_mosek = false;
    }

    TEST_CASE("Worm 2D")
    {
        static constexpr std::size_t dim = 2;
        worm<dim> g({{2., 0.7}, {4., 0.7}, {6., 0.7}, {8., 0.7}, {10., 0.7}, {12., 0.7}},
                {{quaternion(0.)}, {quaternion(0.)}, {quaternion(0.)}, {quaternion(0.)}, {quaternion(0.)}, {quaternion(0.)}},
                1., 6);
        property<dim> p(property<dim>().desired_velocity({{0.25, 0}}));

        scopi_container<dim> particles;
        particles.push_back(g, p);

        SUBCASE("pos index")
        {
            REQUIRE(g.pos(0)(0) == doctest::Approx(2.));
            REQUIRE(g.pos(0)(1) == doctest::Approx(0.7));

            REQUIRE(g.pos(1)(0) == doctest::Approx(4.));
            REQUIRE(g.pos(1)(1) == doctest::Approx(0.7));

            REQUIRE(g.pos(2)(0) == doctest::Approx(6.));
            REQUIRE(g.pos(2)(1) == doctest::Approx(0.7));

            REQUIRE(g.pos(3)(0) == doctest::Approx(8.));
            REQUIRE(g.pos(3)(1) == doctest::Approx(0.7));

            REQUIRE(g.pos(4)(0) == doctest::Approx(10.));
            REQUIRE(g.pos(4)(1) == doctest::Approx(0.7));

            REQUIRE(g.pos(5)(0) == doctest::Approx(12.));
            REQUIRE(g.pos(5)(1) == doctest::Approx(0.7));
        }

        SUBCASE("pos index container")
        {
            REQUIRE(particles[0]->pos(0)(0) == doctest::Approx(2.));
            REQUIRE(particles[0]->pos(0)(1) == doctest::Approx(0.7));

            REQUIRE(particles[0]->pos(1)(0) == doctest::Approx(4.));
            REQUIRE(particles[0]->pos(1)(1) == doctest::Approx(0.7));

            REQUIRE(particles[0]->pos(2)(0) == doctest::Approx(6.));
            REQUIRE(particles[0]->pos(2)(1) == doctest::Approx(0.7));

            REQUIRE(particles[0]->pos(3)(0) == doctest::Approx(8.));
            REQUIRE(particles[0]->pos(3)(1) == doctest::Approx(0.7));

            REQUIRE(particles[0]->pos(4)(0) == doctest::Approx(10.));
            REQUIRE(particles[0]->pos(4)(1) == doctest::Approx(0.7));

            REQUIRE(particles[0]->pos(5)(0) == doctest::Approx(12.));
            REQUIRE(particles[0]->pos(5)(1) == doctest::Approx(0.7));
        }

        SUBCASE("q index")
        {
            REQUIRE(g.q(0)(0) == doctest::Approx(1.));
            REQUIRE(g.q(0)(1) == doctest::Approx(0.));
            REQUIRE(g.q(0)(2) == doctest::Approx(0.));
            REQUIRE(g.q(0)(3) == doctest::Approx(0.));

            REQUIRE(g.q(1)(0) == doctest::Approx(1.));
            REQUIRE(g.q(1)(1) == doctest::Approx(0.));
            REQUIRE(g.q(1)(2) == doctest::Approx(0.));
            REQUIRE(g.q(1)(3) == doctest::Approx(0.));

            REQUIRE(g.q(2)(0) == doctest::Approx(1.));
            REQUIRE(g.q(2)(1) == doctest::Approx(0.));
            REQUIRE(g.q(2)(2) == doctest::Approx(0.));
            REQUIRE(g.q(2)(3) == doctest::Approx(0.));

            REQUIRE(g.q(3)(0) == doctest::Approx(1.));
            REQUIRE(g.q(3)(1) == doctest::Approx(0.));
            REQUIRE(g.q(3)(2) == doctest::Approx(0.));
            REQUIRE(g.q(3)(3) == doctest::Approx(0.));

            REQUIRE(g.q(4)(0) == doctest::Approx(1.));
            REQUIRE(g.q(4)(1) == doctest::Approx(0.));
            REQUIRE(g.q(4)(2) == doctest::Approx(0.));
            REQUIRE(g.q(4)(3) == doctest::Approx(0.));

            REQUIRE(g.q(5)(0) == doctest::Approx(1.));
            REQUIRE(g.q(5)(1) == doctest::Approx(0.));
            REQUIRE(g.q(5)(2) == doctest::Approx(0.));
            REQUIRE(g.q(5)(3) == doctest::Approx(0.));
        }

        SUBCASE("q index container")
        {
            REQUIRE(particles[0]->q(0)(0) == doctest::Approx(1.));
            REQUIRE(particles[0]->q(0)(1) == doctest::Approx(0.));
            REQUIRE(particles[0]->q(0)(2) == doctest::Approx(0.));
            REQUIRE(particles[0]->q(0)(3) == doctest::Approx(0.));

            REQUIRE(particles[0]->q(1)(0) == doctest::Approx(1.));
            REQUIRE(particles[0]->q(1)(1) == doctest::Approx(0.));
            REQUIRE(particles[0]->q(1)(2) == doctest::Approx(0.));
            REQUIRE(particles[0]->q(1)(3) == doctest::Approx(0.));

            REQUIRE(particles[0]->q(2)(0) == doctest::Approx(1.));
            REQUIRE(particles[0]->q(2)(1) == doctest::Approx(0.));
            REQUIRE(particles[0]->q(2)(2) == doctest::Approx(0.));
            REQUIRE(particles[0]->q(2)(3) == doctest::Approx(0.));

            REQUIRE(particles[0]->q(3)(0) == doctest::Approx(1.));
            REQUIRE(particles[0]->q(3)(1) == doctest::Approx(0.));
            REQUIRE(particles[0]->q(3)(2) == doctest::Approx(0.));
            REQUIRE(particles[0]->q(3)(3) == doctest::Approx(0.));

            REQUIRE(particles[0]->q(4)(0) == doctest::Approx(1.));
            REQUIRE(particles[0]->q(4)(1) == doctest::Approx(0.));
            REQUIRE(particles[0]->q(4)(2) == doctest::Approx(0.));
            REQUIRE(particles[0]->q(4)(3) == doctest::Approx(0.));

            REQUIRE(particles[0]->q(5)(0) == doctest::Approx(1.));
            REQUIRE(particles[0]->q(5)(1) == doctest::Approx(0.));
            REQUIRE(particles[0]->q(5)(2) == doctest::Approx(0.));
            REQUIRE(particles[0]->q(5)(3) == doctest::Approx(0.));
        }

        SUBCASE("radius")
        {
            REQUIRE(g.radius() == doctest::Approx(1.));
        }

        SUBCASE("size")
        {
            REQUIRE(g.size() == 6);
        }
    }

    TEST_CASE("Worm 2D const")
    {
        static constexpr std::size_t dim = 2;
        const worm<dim> g({{2., 0.7}, {4., 0.7}, {6., 0.7}, {8., 0.7}, {10., 0.7}, {12., 0.7}},
                {{quaternion(0.)}, {quaternion(0.)}, {quaternion(0.)}, {quaternion(0.)}, {quaternion(0.)}, {quaternion(0.)}},
                1., 6);
        const property<dim> p(property<dim>().desired_velocity({{0.25, 0}}));

        scopi_container<dim> particles;
        particles.push_back(g, p);

        SUBCASE("pos index")
        {
            REQUIRE(g.pos(0)(0) == doctest::Approx(2.));
            REQUIRE(g.pos(0)(1) == doctest::Approx(0.7));

            REQUIRE(g.pos(1)(0) == doctest::Approx(4.));
            REQUIRE(g.pos(1)(1) == doctest::Approx(0.7));

            REQUIRE(g.pos(2)(0) == doctest::Approx(6.));
            REQUIRE(g.pos(2)(1) == doctest::Approx(0.7));

            REQUIRE(g.pos(3)(0) == doctest::Approx(8.));
            REQUIRE(g.pos(3)(1) == doctest::Approx(0.7));

            REQUIRE(g.pos(4)(0) == doctest::Approx(10.));
            REQUIRE(g.pos(4)(1) == doctest::Approx(0.7));

            REQUIRE(g.pos(5)(0) == doctest::Approx(12.));
            REQUIRE(g.pos(5)(1) == doctest::Approx(0.7));
        }

        SUBCASE("pos index container")
        {
            REQUIRE(particles[0]->pos(0)(0) == doctest::Approx(2.));
            REQUIRE(particles[0]->pos(0)(1) == doctest::Approx(0.7));

            REQUIRE(particles[0]->pos(1)(0) == doctest::Approx(4.));
            REQUIRE(particles[0]->pos(1)(1) == doctest::Approx(0.7));

            REQUIRE(particles[0]->pos(2)(0) == doctest::Approx(6.));
            REQUIRE(particles[0]->pos(2)(1) == doctest::Approx(0.7));

            REQUIRE(particles[0]->pos(3)(0) == doctest::Approx(8.));
            REQUIRE(particles[0]->pos(3)(1) == doctest::Approx(0.7));

            REQUIRE(particles[0]->pos(4)(0) == doctest::Approx(10.));
            REQUIRE(particles[0]->pos(4)(1) == doctest::Approx(0.7));

            REQUIRE(particles[0]->pos(5)(0) == doctest::Approx(12.));
            REQUIRE(particles[0]->pos(5)(1) == doctest::Approx(0.7));
        }

        SUBCASE("q index")
        {
            REQUIRE(g.q(0)(0) == doctest::Approx(1.));
            REQUIRE(g.q(0)(1) == doctest::Approx(0.));
            REQUIRE(g.q(0)(2) == doctest::Approx(0.));
            REQUIRE(g.q(0)(3) == doctest::Approx(0.));

            REQUIRE(g.q(1)(0) == doctest::Approx(1.));
            REQUIRE(g.q(1)(1) == doctest::Approx(0.));
            REQUIRE(g.q(1)(2) == doctest::Approx(0.));
            REQUIRE(g.q(1)(3) == doctest::Approx(0.));

            REQUIRE(g.q(2)(0) == doctest::Approx(1.));
            REQUIRE(g.q(2)(1) == doctest::Approx(0.));
            REQUIRE(g.q(2)(2) == doctest::Approx(0.));
            REQUIRE(g.q(2)(3) == doctest::Approx(0.));

            REQUIRE(g.q(3)(0) == doctest::Approx(1.));
            REQUIRE(g.q(3)(1) == doctest::Approx(0.));
            REQUIRE(g.q(3)(2) == doctest::Approx(0.));
            REQUIRE(g.q(3)(3) == doctest::Approx(0.));

            REQUIRE(g.q(4)(0) == doctest::Approx(1.));
            REQUIRE(g.q(4)(1) == doctest::Approx(0.));
            REQUIRE(g.q(4)(2) == doctest::Approx(0.));
            REQUIRE(g.q(4)(3) == doctest::Approx(0.));

            REQUIRE(g.q(5)(0) == doctest::Approx(1.));
            REQUIRE(g.q(5)(1) == doctest::Approx(0.));
            REQUIRE(g.q(5)(2) == doctest::Approx(0.));
            REQUIRE(g.q(5)(3) == doctest::Approx(0.));
        }

        SUBCASE("q index container")
        {
            REQUIRE(particles[0]->q(0)(0) == doctest::Approx(1.));
            REQUIRE(particles[0]->q(0)(1) == doctest::Approx(0.));
            REQUIRE(particles[0]->q(0)(2) == doctest::Approx(0.));
            REQUIRE(particles[0]->q(0)(3) == doctest::Approx(0.));

            REQUIRE(particles[0]->q(1)(0) == doctest::Approx(1.));
            REQUIRE(particles[0]->q(1)(1) == doctest::Approx(0.));
            REQUIRE(particles[0]->q(1)(2) == doctest::Approx(0.));
            REQUIRE(particles[0]->q(1)(3) == doctest::Approx(0.));

            REQUIRE(particles[0]->q(2)(0) == doctest::Approx(1.));
            REQUIRE(particles[0]->q(2)(1) == doctest::Approx(0.));
            REQUIRE(particles[0]->q(2)(2) == doctest::Approx(0.));
            REQUIRE(particles[0]->q(2)(3) == doctest::Approx(0.));

            REQUIRE(particles[0]->q(3)(0) == doctest::Approx(1.));
            REQUIRE(particles[0]->q(3)(1) == doctest::Approx(0.));
            REQUIRE(particles[0]->q(3)(2) == doctest::Approx(0.));
            REQUIRE(particles[0]->q(3)(3) == doctest::Approx(0.));

            REQUIRE(particles[0]->q(4)(0) == doctest::Approx(1.));
            REQUIRE(particles[0]->q(4)(1) == doctest::Approx(0.));
            REQUIRE(particles[0]->q(4)(2) == doctest::Approx(0.));
            REQUIRE(particles[0]->q(4)(3) == doctest::Approx(0.));

            REQUIRE(particles[0]->q(5)(0) == doctest::Approx(1.));
            REQUIRE(particles[0]->q(5)(1) == doctest::Approx(0.));
            REQUIRE(particles[0]->q(5)(2) == doctest::Approx(0.));
            REQUIRE(particles[0]->q(5)(3) == doctest::Approx(0.));
        }

        SUBCASE("radius")
        {
            REQUIRE(g.radius() == doctest::Approx(1.));
        }

        SUBCASE("size")
        {
            REQUIRE(g.size() == 6);
        }
    }

    TEST_CASE("Worm 3D")
    {
        static constexpr std::size_t dim = 3;
        worm<dim> g({{2., 0.7, 0.2}, {4., 0.7, 0.2}, {6., 0.7, 0.2}, {8., 0.7, 0.2}, {10., 0.7, 0.2}, {12., 0.7, 0.2}},
                {{quaternion(0.)}, {quaternion(0.)}, {quaternion(0.)}, {quaternion(0.)}, {quaternion(0.)}, {quaternion(0.)}},
                1., 6);
        property<dim> p(property<dim>().desired_velocity({{0.25, 0, 0}}));
        scopi_container<dim> particles;
        particles.push_back(g, p);

        SUBCASE("pos index")
        {
            REQUIRE(g.pos(0)(0) == doctest::Approx(2.));
            REQUIRE(g.pos(0)(1) == doctest::Approx(0.7));
            REQUIRE(g.pos(0)(2) == doctest::Approx(0.2));

            REQUIRE(g.pos(1)(0) == doctest::Approx(4.));
            REQUIRE(g.pos(1)(1) == doctest::Approx(0.7));
            REQUIRE(g.pos(1)(2) == doctest::Approx(0.2));

            REQUIRE(g.pos(2)(0) == doctest::Approx(6.));
            REQUIRE(g.pos(2)(1) == doctest::Approx(0.7));
            REQUIRE(g.pos(2)(2) == doctest::Approx(0.2));

            REQUIRE(g.pos(3)(0) == doctest::Approx(8.));
            REQUIRE(g.pos(3)(1) == doctest::Approx(0.7));
            REQUIRE(g.pos(3)(2) == doctest::Approx(0.2));

            REQUIRE(g.pos(4)(0) == doctest::Approx(10.));
            REQUIRE(g.pos(4)(1) == doctest::Approx(0.7));
            REQUIRE(g.pos(4)(2) == doctest::Approx(0.2));

            REQUIRE(g.pos(5)(0) == doctest::Approx(12.));
            REQUIRE(g.pos(5)(1) == doctest::Approx(0.7));
            REQUIRE(g.pos(5)(2) == doctest::Approx(0.2));
        }

        SUBCASE("pos index container")
        {
            REQUIRE(particles[0]->pos(0)(0) == doctest::Approx(2.));
            REQUIRE(particles[0]->pos(0)(1) == doctest::Approx(0.7));
            REQUIRE(particles[0]->pos(0)(2) == doctest::Approx(0.2));

            REQUIRE(particles[0]->pos(1)(0) == doctest::Approx(4.));
            REQUIRE(particles[0]->pos(1)(1) == doctest::Approx(0.7));
            REQUIRE(particles[0]->pos(0)(2) == doctest::Approx(0.2));

            REQUIRE(particles[0]->pos(2)(0) == doctest::Approx(6.));
            REQUIRE(particles[0]->pos(2)(1) == doctest::Approx(0.7));
            REQUIRE(particles[0]->pos(0)(2) == doctest::Approx(0.2));

            REQUIRE(particles[0]->pos(3)(0) == doctest::Approx(8.));
            REQUIRE(particles[0]->pos(3)(1) == doctest::Approx(0.7));
            REQUIRE(particles[0]->pos(0)(2) == doctest::Approx(0.2));

            REQUIRE(particles[0]->pos(4)(0) == doctest::Approx(10.));
            REQUIRE(particles[0]->pos(4)(1) == doctest::Approx(0.7));
            REQUIRE(particles[0]->pos(0)(2) == doctest::Approx(0.2));

            REQUIRE(particles[0]->pos(5)(0) == doctest::Approx(12.));
            REQUIRE(particles[0]->pos(5)(1) == doctest::Approx(0.7));
            REQUIRE(particles[0]->pos(0)(2) == doctest::Approx(0.2));
        }

        SUBCASE("radius")
        {
            REQUIRE(g.radius() == doctest::Approx(1.));
        }

        SUBCASE("size")
        {
            REQUIRE(g.size() == 6);
        }
    }

    TEST_CASE("Worm 3D const")
    {
        static constexpr std::size_t dim = 3;
        const worm<dim> g({{2., 0.7, 0.2}, {4., 0.7, 0.2}, {6., 0.7, 0.2}, {8., 0.7, 0.2}, {10., 0.7, 0.2}, {12., 0.7, 0.2}},
                {{quaternion(0.)}, {quaternion(0.)}, {quaternion(0.)}, {quaternion(0.)}, {quaternion(0.)}, {quaternion(0.)}},
                1., 6);
        const property<dim> p(property<dim>().desired_velocity({{0.25, 0, 0}}));
        scopi_container<dim> particles;
        particles.push_back(g, p);

        SUBCASE("pos index")
        {
            REQUIRE(g.pos(0)(0) == doctest::Approx(2.));
            REQUIRE(g.pos(0)(1) == doctest::Approx(0.7));
            REQUIRE(g.pos(0)(2) == doctest::Approx(0.2));

            REQUIRE(g.pos(1)(0) == doctest::Approx(4.));
            REQUIRE(g.pos(1)(1) == doctest::Approx(0.7));
            REQUIRE(g.pos(1)(2) == doctest::Approx(0.2));

            REQUIRE(g.pos(2)(0) == doctest::Approx(6.));
            REQUIRE(g.pos(2)(1) == doctest::Approx(0.7));
            REQUIRE(g.pos(2)(2) == doctest::Approx(0.2));

            REQUIRE(g.pos(3)(0) == doctest::Approx(8.));
            REQUIRE(g.pos(3)(1) == doctest::Approx(0.7));
            REQUIRE(g.pos(3)(2) == doctest::Approx(0.2));

            REQUIRE(g.pos(4)(0) == doctest::Approx(10.));
            REQUIRE(g.pos(4)(1) == doctest::Approx(0.7));
            REQUIRE(g.pos(4)(2) == doctest::Approx(0.2));

            REQUIRE(g.pos(5)(0) == doctest::Approx(12.));
            REQUIRE(g.pos(5)(1) == doctest::Approx(0.7));
            REQUIRE(g.pos(5)(2) == doctest::Approx(0.2));
        }

        SUBCASE("pos index container")
        {
            REQUIRE(particles[0]->pos(0)(0) == doctest::Approx(2.));
            REQUIRE(particles[0]->pos(0)(1) == doctest::Approx(0.7));
            REQUIRE(particles[0]->pos(0)(2) == doctest::Approx(0.2));

            REQUIRE(particles[0]->pos(1)(0) == doctest::Approx(4.));
            REQUIRE(particles[0]->pos(1)(1) == doctest::Approx(0.7));
            REQUIRE(particles[0]->pos(0)(2) == doctest::Approx(0.2));

            REQUIRE(particles[0]->pos(2)(0) == doctest::Approx(6.));
            REQUIRE(particles[0]->pos(2)(1) == doctest::Approx(0.7));
            REQUIRE(particles[0]->pos(0)(2) == doctest::Approx(0.2));

            REQUIRE(particles[0]->pos(3)(0) == doctest::Approx(8.));
            REQUIRE(particles[0]->pos(3)(1) == doctest::Approx(0.7));
            REQUIRE(particles[0]->pos(0)(2) == doctest::Approx(0.2));

            REQUIRE(particles[0]->pos(4)(0) == doctest::Approx(10.));
            REQUIRE(particles[0]->pos(4)(1) == doctest::Approx(0.7));
            REQUIRE(particles[0]->pos(0)(2) == doctest::Approx(0.2));

            REQUIRE(particles[0]->pos(5)(0) == doctest::Approx(12.));
            REQUIRE(particles[0]->pos(5)(1) == doctest::Approx(0.7));
            REQUIRE(particles[0]->pos(0)(2) == doctest::Approx(0.2));
        }

        SUBCASE("radius")
        {
            REQUIRE(g.radius() == doctest::Approx(1.));
        }

        SUBCASE("size")
        {
            REQUIRE(g.size() == 6);
        }
    }

    TEST_CASE_TEMPLATE("two worms", SolverType, SOLVER_WORMS(2, contact_kdtree, vap_fixed), SOLVER_WORMS(2, contact_brute_force, vap_fixed))
    {
        using params_t = typename SolverType::params_t;

        constexpr std::size_t dim = 2;
        double dt = .005;
        std::size_t total_it = 1000;
        scopi_container<dim> particles;
        auto prop = property<dim>().mass(1.).moment_inertia(0.1);

        worm<dim> g1({{1., 0.7}, {3., 0.7}, {5., 0.7}, {7., 0.7}, {9., 0.7}, {11., 0.7}},
                {{quaternion(0.)}, {quaternion(0.)}, {quaternion(0.)}, {quaternion(0.)}, {quaternion(0.)}, {quaternion(0.)}},
                1., 6);
        worm<dim> g2({{-1., -0.7}, {-3., -0.7}, {-5., -0.7}, {-7., -0.7}, {-9., -0.7}, {-11., -0.7}},
                {{quaternion(0.)}, {quaternion(0.)}, {quaternion(0.)}, {quaternion(0.)}, {quaternion(0.)}, {quaternion(0.)}},
                1., 6);
        particles.push_back(g1, prop.desired_velocity({-1., 0.}));
        particles.push_back(g2, prop.desired_velocity({1., 0.}));

        params_t params;
        set_params_test(params.optim_params);
        SolverType solver(particles, dt, params);
        solver.run(total_it);

        CHECK(diffFile("./Results/scopi_objects_0999.json", "../test/references/two_worms.json", tolerance));
    }
}
