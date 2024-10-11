#include "utils.hpp"
#include <cstddef>
#include <doctest/doctest.h>

#include <scopi/contact/contact_kdtree.hpp>
#include <scopi/contact/property.hpp>
#include <scopi/container.hpp>
#include <scopi/objects/types/sphere.hpp>
#include <scopi/objects/types/superellipsoid.hpp>

namespace scopi
{

    TEST_CASE("Contacts Kd-tree")
    {
        constexpr std::size_t dim = 2;
        scopi_container<dim> particles;

        sphere<dim> s1(
            {
                {0., 0.}
        },
            0.1);
        sphere<dim> s2(
            {
                {1., 1.}
        },
            0.2);
        sphere<dim> s3(
            {
                {5., 10.}
        },
            0.1);

        particles.push_back(s1);
        particles.push_back(s2);
        particles.push_back(s3);

        ContactsParams<contact_kdtree<NoFriction>> params;
        params.kd_tree_radius = 100.;
        contact_kdtree cont(params);
        auto contacts = cont.run(particles);

        SUBCASE("nbContacts")
        {
            CHECK(contacts.size() == 1);
        }

        SUBCASE("particles in contact")
        {
            CHECK(contacts[0].i == 0);
            CHECK(contacts[0].j == 1);
        }

        SUBCASE("distance")
        {
            REQUIRE(contacts[0].dij == doctest::Approx(std::sqrt(2.) - 0.1 - 0.2));
        }

        SUBCASE("normal")
        {
            REQUIRE(contacts[0].nij(0) == doctest::Approx(-1. / std::sqrt(2.)));
            REQUIRE(contacts[0].nij(1) == doctest::Approx(-1. / std::sqrt(2.)));
        }

        SUBCASE("position")
        {
            REQUIRE(contacts[0].pi(0) == doctest::Approx(0.1 * std::sqrt(2.) / 2.));
            REQUIRE(contacts[0].pi(1) == doctest::Approx(0.1 * std::sqrt(2.) / 2.));
            REQUIRE(contacts[0].pj(0) == doctest::Approx(1. - 0.2 * std::sqrt(2.) / 2.));
            REQUIRE(contacts[0].pj(1) == doctest::Approx(1. - 0.2 * std::sqrt(2.) / 2.));
        }
    }
}
