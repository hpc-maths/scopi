#include <gtest/gtest.h>

#include "test_common.hpp"
#include "utils.hpp"

#include <scopi/objects/types/sphere.hpp>
#include <scopi/vap/vap_fpd.hpp>
#include <scopi/container.hpp>
#include <scopi/solver.hpp>

namespace scopi {
    template <class S>
    class TestTwoSpheresAsymmetricalFriction  : public ::testing::Test {
        static constexpr std::size_t dim = 2;
        protected:
            void SetUp() override {
                sphere<dim> s1({{-0.2, -0.05}}, 0.1);
                sphere<dim> s2({{ 0.2,  0.05}}, 0.1);
                auto p = property<dim>().desired_velocity({{0.25, 0}});
                m_particles.push_back(s1, p);
                m_particles.push_back(s2, p.desired_velocity({{-0.25, 0}}));
            }

            double m_dt = .005;
            std::size_t m_total_it = 1000;
            scopi_container<dim> m_particles;
            double m_mu = 1./2.;
    };

    TYPED_TEST_SUITE(TestTwoSpheresAsymmetricalFriction, solver_with_contact_types_friction<2>);

    TYPED_TEST(TestTwoSpheresAsymmetricalFriction, two_spheres_asymmetrical_friction)
    {
        TypeParam solver(this->m_particles, this->m_dt);
        solver.set_coeff_friction(this->m_mu);
        solver.solve(this->m_total_it);

        EXPECT_PRED3(diffFile, "./Results/scopi_objects_0999.json", "../test/references/two_spheres_asymmetrical_friction.json", tolerance);
    }

    template <class S>
    class Test2dCaseSpheresFriction  : public ::testing::Test {
        static constexpr std::size_t dim = 2;
        protected:
            void SetUp() override {
                int n = 2; // 2*n*n particles
                std::default_random_engine generator(0);
                std::uniform_real_distribution<double> distrib_r(0.2, 0.4);
                std::uniform_real_distribution<double> distrib_move_x(-0.1, 0.1);
                std::uniform_real_distribution<double> distrib_move_y(-0.1, 0.1);
                std::uniform_real_distribution<double> distrib_velocity(2., 5.);

                for(int i = 0; i < n; ++i)
                {
                    for(int j = 0; j < n; ++j)
                    {
                        double r = distrib_r(generator);
                        double x = (i + 0.5) + distrib_move_x(generator);
                        double y = (j + 0.5) + distrib_move_y(generator);
                        double velocity = distrib_velocity(generator);
                        sphere<dim> s1({{x, y}}, r);
                        m_particles.push_back(s1, scopi::property<dim>().desired_velocity({{velocity, 0.}}));

                        r = distrib_r(generator);
                        x = (n + i + 0.5) + distrib_move_x(generator);
                        y = (j + 0.5) + distrib_move_y(generator);
                        velocity = distrib_velocity(generator);
                        sphere<dim> s2({{x, y}}, r);
                        m_particles.push_back(s2, scopi::property<dim>().desired_velocity({{-velocity, 0.}}));
                    }
                }
            }

            double m_dt = .01;
            std::size_t m_total_it = 100;
            scopi_container<dim> m_particles;
            double m_mu = 1./2.;
    };

    TYPED_TEST_SUITE(Test2dCaseSpheresFriction, solver_with_contact_types_friction<2>);

    TYPED_TEST(Test2dCaseSpheresFriction, 2d_case_spheres_friction)
    {
        TypeParam solver(this->m_particles, this->m_dt);
        solver.set_coeff_friction(this->m_mu);
        solver.solve(this->m_total_it);

        EXPECT_PRED3(diffFile, "./Results/scopi_objects_0099.json", "../test/references/2d_case_spheres_friction.json", tolerance);
    }
}
