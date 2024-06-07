#include <doctest/doctest.h>

#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xtensor.hpp>

#include <scopi/contact/contact_brute_force.hpp>
#include <scopi/container.hpp>
#include <scopi/matrix/velocities.hpp>
#include <scopi/objects/types/sphere.hpp>

#include "utils.hpp"

namespace scopi
{
    TEST_CASE("Matrix A")
    {
        static constexpr std::size_t dim = 3;
        scopi_container<dim> particles;

        sphere<dim> s1(
            {
                {0., 0., 0.}
        },
            0.1);
        sphere<dim> s2(
            {
                {1., 1., 0.}
        },
            0.2);

        particles.push_back(s1);
        particles.push_back(s2);

        ContactsParams<contact_brute_force<NoFriction>> params;
        contact_brute_force cont(params);
        auto contacts = cont.run(particles, 0);

        AMatrix a(contacts, particles);
        ATMatrix at(contacts, particles);

        xt::xtensor<double, 1> u = xt::random::rand<double>({6 * particles.nb_active()});
        xt::xtensor<double, 1> f = xt::random::rand<double>({3 * contacts.size()});

        REQUIRE(xt::linalg::dot(a.mat_mult(u), f)[0] == doctest::Approx(xt::linalg::dot(u, at.mat_mult(f))[0]));
    }

    TEST_CASE("Matrix A case 2")
    {
        static constexpr std::size_t dim = 2;
        scopi_container<dim> particles;

        sphere<dim> sphere(
            {
                {2., 1.}
        },
            0.5);
        scopi::plane<dim> plane(
            {
                {0., 0.}
        },
            PI / 2 - PI / 4);

        particles.push_back(plane, scopi::property<dim>().deactivate());
        particles.push_back(sphere, scopi::property<dim>().mass(1).moment_inertia(0.1));

        ContactsParams<contact_brute_force<NoFriction>> params;
        contact_brute_force<NoFriction> cont(params);
        auto contacts = cont.run(particles, 0);

        AMatrix a(contacts, particles);
        xt::xtensor<double, 1> u{0.187562190766376, -1.184390935327111, 0., 0., 0., -0.00453709103433};
        auto sol = a.mat_mult(u);
        REQUIRE(sol[0] == doctest::Approx(-0.18595809));
        REQUIRE(sol[1] == doctest::Approx(1.18278683));
    }

} // namespace scopi
