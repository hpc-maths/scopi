#include "doctest/doctest.h"
#include "utils.hpp"

#include <xtensor/xadapt.hpp>

#include <scopi/quaternion.hpp>

namespace scopi
{
    TEST_CASE("quaternion")
    {
        auto q = quaternion(PI / 3.);
        REQUIRE(q(0) == doctest::Approx(std::sqrt(3.) / 2.));
        REQUIRE(q(1) == 0.);
        REQUIRE(q(2) == 0.);
        REQUIRE(q(3) == doctest::Approx(1. / 2.));
    }

    TEST_CASE("normalize normalized quaternion")
    {
        auto q = quaternion(PI / 3.);
        normalize(q);
        REQUIRE(q(0) == doctest::Approx(std::sqrt(3.) / 2.));
        REQUIRE(q(1) == 0.);
        REQUIRE(q(2) == 0.);
        REQUIRE(q(3) == doctest::Approx(1. / 2.));
    }

    TEST_CASE("normalize not-normalized quaternion")
    {
        auto q = quaternion(PI / 3.);
        for (auto& elt : q)
        {
            elt *= 2.;
        }
        normalize(q);
        REQUIRE(q(0) == doctest::Approx(std::sqrt(3.) / 2.));
        REQUIRE(q(1) == 0.);
        REQUIRE(q(2) == 0.);
        REQUIRE(q(3) == doctest::Approx(1. / 2.));
    }

    /*
    TEST_CASE("conj")
    {
        auto q = quaternion(PI/3.);
        type::quaternion_t q_conj = conj_scopi(q);
        REQUIRE(q_conj(0) == doctest::Approx(-std::sqrt(3.)/2.));
        REQUIRE(q_conj(1) == 0.);
        REQUIRE(q_conj(2) == 0.);
        REQUIRE(q_conj(3) == doctest::Approx(-1./2.));
    }
    */

    TEST_CASE("mult_quaternion")
    {
        auto q1  = quaternion(PI / 3);
        auto q2  = quaternion(PI / 6);
        auto res = mult_quaternion(q1, q2);

        REQUIRE(res(0) == doctest::Approx(std::sqrt(3.) / 2. * (std::sqrt(6) + std::sqrt(2)) / 4 - (std::sqrt(6) - std::sqrt(2)) / 8));
        REQUIRE(res(1) == doctest::Approx(0.));
        REQUIRE(res(2) == doctest::Approx(0.));
        REQUIRE(res(3) == doctest::Approx(std::sqrt(3.) / 2. * (std::sqrt(6.) - std::sqrt(2.)) / 4. + (std::sqrt(6.) + std::sqrt(2.)) / 8.));
    }

    TEST_CASE("rotation_matrix")
    {
        auto q = xt::adapt(quaternion(PI / 3));
        SUBCASE("2D")
        {
            auto m = rotation_matrix<2>(q);
            REQUIRE(m(0, 0) == doctest::Approx(1. / 2.));
            REQUIRE(m(0, 1) == doctest::Approx(-std::sqrt(3.) / 2.));
            REQUIRE(m(1, 0) == doctest::Approx(std::sqrt(3.) / 2.));
            REQUIRE(m(1, 1) == doctest::Approx(1. / 2.));
        }
        SUBCASE("3D")
        {
            auto m = rotation_matrix<3>(q);
            REQUIRE(m(0, 0) == doctest::Approx(1. / 2.));
            REQUIRE(m(0, 1) == doctest::Approx(-std::sqrt(3.) / 2.));
            REQUIRE(m(0, 2) == doctest::Approx(0.));
            REQUIRE(m(1, 0) == doctest::Approx(std::sqrt(3.) / 2.));
            REQUIRE(m(1, 1) == doctest::Approx(1. / 2.));
            REQUIRE(m(1, 2) == doctest::Approx(0.));
            REQUIRE(m(2, 0) == doctest::Approx(0.));
            REQUIRE(m(2, 1) == doctest::Approx(0.));
            REQUIRE(m(2, 2) == doctest::Approx(1.));
        }
    }
}
