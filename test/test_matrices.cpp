#include "doctest/doctest.h"

#include "xtensor-blas/xlinalg.hpp"
#include <xtensor/xio.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xtensor.hpp>

#include <scopi/contact/contact_brute_force.hpp>
#include <scopi/container.hpp>
#include <scopi/matrix/velocities.hpp>
#include <scopi/objects/types/sphere.hpp>

namespace scopi
{
    TEST_CASE("Matrix A")
    {
        static constexpr std::size_t dim = 3;
        scopi_container<dim> particles;

        sphere<dim> s1({{0., 0., 0.}}, 0.1);
        sphere<dim> s2({{1., 1., 0.}}, 0.2);

        particles.push_back(s1);
        particles.push_back(s2);

        ContactsParams<contact_brute_force> params;
        contact_brute_force cont(params);
        auto contacts = cont.run(particles, 0);

        A a(contacts, particles);
        AT at(contacts, particles);

        xt::xtensor<double, 1> u =
            xt::random::rand<double>({6 * particles.nb_active()});
        xt::xtensor<double, 1> f =
            xt::random::rand<double>({3 * contacts.size()});

        REQUIRE(xt::linalg::dot(a.mat_mult(u), f)[0] ==
                doctest::Approx(xt::linalg::dot(u, at.mat_mult(f))[0]));
    }
} // namespace scopi