#include <gtest/gtest.h>
#include "utils.hpp"

#include <scopi/objects/types/sphere.hpp>
#include <scopi/objects/types/superellipsoid.hpp>
#include <scopi/container.hpp>
#include <scopi/contact/contact_brute_force.hpp>

namespace scopi
{

    class ContactsBruteForceTest  : public ::testing::Test {
        protected:
            ContactsBruteForceTest()
            {
                constexpr std::size_t dim = 2;
                scopi_container<dim> particles;

                sphere<dim> s1({{0., 0.}}, 0.1);
                sphere<dim> s2({{1., 1.}}, 0.2);
                sphere<dim> s3({{5., 10.}}, 0.1);

                particles.push_back(s1, {{0, 0}}, {{0., 0}}, 0, 0, {{0, 0}});
                particles.push_back(s2, {{0, 0}}, {{-0., 0}}, 0, 0, {{0, 0}});
                particles.push_back(s3, {{0, 0}}, {{-0., 0}}, 0, 0, {{0, 0}});

                contact_brute_force cont(2);
                m_contacts = cont.run(particles, 0);
            }
            std::vector<scopi::neighbor<2>> m_contacts;
    };

    TEST_F(ContactsBruteForceTest, nbContacts)
    {
        EXPECT_EQ(m_contacts.size(), 1);
    }

    TEST_F(ContactsBruteForceTest, particlesInContact)
    {
        EXPECT_EQ(m_contacts[0].i, 0);
        EXPECT_EQ(m_contacts[0].j, 1);
    }

    TEST_F(ContactsBruteForceTest, distance)
    {
        EXPECT_DOUBLE_EQ(m_contacts[0].dij, std::sqrt(2.)-0.1-0.2);
    }

    TEST_F(ContactsBruteForceTest, normal)
    {
        EXPECT_DOUBLE_EQ(m_contacts[0].nij(0), -1./std::sqrt(2.));
        EXPECT_DOUBLE_EQ(m_contacts[0].nij(1), -1./std::sqrt(2.)); 
    }

    TEST_F(ContactsBruteForceTest, position)
    {
        EXPECT_DOUBLE_EQ(m_contacts[0].pi(0), 0.1*std::sqrt(2.)/2.);
        EXPECT_DOUBLE_EQ(m_contacts[0].pi(1), 0.1*std::sqrt(2.)/2.);
        EXPECT_DOUBLE_EQ(m_contacts[0].pj(0), 1.-0.2*std::sqrt(2.)/2.);
        EXPECT_DOUBLE_EQ(m_contacts[0].pj(1), 1.-0.2*std::sqrt(2.)/2.);
    }

}
