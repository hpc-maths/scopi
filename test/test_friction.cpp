#include <cstddef>
#include <gtest/gtest.h>
#include <random>

#include "test_common.hpp"
#include "utils.hpp"

#include <scopi/objects/types/sphere.hpp>
#include <scopi/vap/vap_fpd.hpp>
#include <scopi/container.hpp>
#include <scopi/solver.hpp>
#include <tuple>

namespace scopi {
    template <class S>
    class TestTwoSpheresAsymmetricalFriction  : public ::testing::Test {
        static constexpr std::size_t dim = 2;
        protected:
            void SetUp() override {
                sphere<dim> s1({{-0.2, -0.05}}, 0.1);
                sphere<dim> s2({{ 0.2,  0.05}}, 0.1);
                auto p = property<dim>().desired_velocity({{0.25, 0}}).mass(1.).moment_inertia(0.1);
                m_particles.push_back(s1, p);
                m_particles.push_back(s2, p.desired_velocity({{-0.25, 0}}));
            }

            double m_dt = .005;
            std::size_t m_total_it = 1000;
            scopi_container<dim> m_particles;
            double m_mu = 1./2.;
    };

    TYPED_TEST_SUITE(TestTwoSpheresAsymmetricalFriction, solver_with_contact_types_friction<2>, );

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
                std::minstd_rand0 generator(0);
                std::uniform_real_distribution<double> distrib_r(0.2, 0.4);
                std::uniform_real_distribution<double> distrib_move_x(-0.1, 0.1);
                std::uniform_real_distribution<double> distrib_move_y(-0.1, 0.1);
                std::uniform_real_distribution<double> distrib_velocity(2., 5.);
                auto prop = property<dim>().mass(1.).moment_inertia(0.1);

                for(int i = 0; i < n; ++i)
                {
                    for(int j = 0; j < n; ++j)
                    {
                        double r = distrib_r(generator);
                        double x = (i + 0.5) + distrib_move_x(generator);
                        double y = (j + 0.5) + distrib_move_y(generator);
                        double velocity = distrib_velocity(generator);
                        sphere<dim> s1({{x, y}}, r);
                        m_particles.push_back(s1, prop.desired_velocity({{velocity, 0.}}));

                        r = distrib_r(generator);
                        x = (n + i + 0.5) + distrib_move_x(generator);
                        y = (j + 0.5) + distrib_move_y(generator);
                        velocity = distrib_velocity(generator);
                        sphere<dim> s2({{x, y}}, r);
                        m_particles.push_back(s2, prop.desired_velocity({{-velocity, 0.}}));
                    }
                }
            }

            double m_dt = .01;
            std::size_t m_total_it = 100;
            scopi_container<dim> m_particles;
            double m_mu = 1./2.;
    };

    TYPED_TEST_SUITE(Test2dCaseSpheresFriction, solver_with_contact_types_friction<2>, );

    TYPED_TEST(Test2dCaseSpheresFriction, 2d_case_spheres_friction)
    {
        TypeParam solver(this->m_particles, this->m_dt);
        solver.set_coeff_friction(this->m_mu);
        solver.solve(this->m_total_it);

        EXPECT_PRED3(diffFile, "./Results/scopi_objects_0099.json", "../test/references/2d_case_spheres_friction.json", tolerance);
    }

    class TestSphereInclinedPlan :public ::testing::TestWithParam<std::tuple<double, std::size_t, double, double, double, double>>
    {
        protected:
        static constexpr std::size_t dim = 2;
        void SetUp() override {
            m_dt = std::get<0>(GetParam());
            m_total_it = std::get<1>(GetParam());
            m_mu = std::get<2>(GetParam());
            m_alpha = std::get<3>(GetParam());

            auto prop = property<dim>().mass(1.).moment_inertia(1.*m_radius*m_radius/2.);
            plan<dim> p({{-m_radius*std::cos(PI/2.-m_alpha), -m_radius*std::sin(PI/2.-m_alpha)}}, PI/2.-m_alpha);
            sphere<dim> s({{0., 0.}}, m_radius);
            m_particles.push_back(p, property<dim>().deactivate());
            m_particles.push_back(s, prop.force({{0., -m_g}}));
        }

        void check_solution()
        {
            auto pos = m_particles.pos();
            auto omega = m_particles.omega();
            auto sol = scopi::analytical_solution_sphere_plan(m_alpha, m_mu, m_dt*(m_total_it+1), m_radius, m_g);
            auto pos_analytical = sol.first;
            auto omega_analytical = sol.second;
            double err_pos = xt::linalg::norm(pos(1) - pos_analytical) / xt::linalg::norm(pos_analytical);
            double err_omega = std::abs((omega(1)-omega_analytical)/omega_analytical);

            EXPECT_NEAR(err_pos, std::get<4>(GetParam()), tolerance);
            EXPECT_NEAR(err_omega, std::get<5>(GetParam()), tolerance);
        }

        scopi_container<dim> m_particles;
        double m_dt;
        std::size_t m_total_it;
        double m_mu;
        double m_alpha;
        double m_radius = 1.;
        double m_g = 1.;
    };

    TEST_P(TestSphereInclinedPlan, test_sphere_inclined_plan_mosek_kd_tree)
    {
        ScopiSolver<dim, OptimMosek<MatrixOptimSolverFriction>, contact_kdtree, vap_fpd> solver(m_particles, m_dt);
        solver.set_coeff_friction(m_mu);
        solver.solve(m_total_it);
        
        check_solution();
    }

    TEST_P(TestSphereInclinedPlan, test_sphere_inclined_plan_mosek_brute_force)
    {
        ScopiSolver<dim, OptimMosek<MatrixOptimSolverFriction>, contact_brute_force, vap_fpd> solver(m_particles, m_dt);
        solver.set_coeff_friction(m_mu);
        solver.solve(m_total_it);
        
        check_solution();
    }

    INSTANTIATE_TEST_SUITE_P(
        TestSphereInclinedPlan,
        TestSphereInclinedPlan,
        ::testing::Values(
            std::make_tuple(0.005, 2000, 0.1, PI/6., 0.000508866, 0.000485907),
            std::make_tuple(0.005, 2000, 0.1, PI/4., 0.000513437, 0.000464869),
            std::make_tuple(0.005, 2000, 0.1, PI/3., 0.000516023, 0.000428305),
            std::make_tuple(0.005, 2000, 0.5, PI/6., 0.00049975, 0.000499785),
            std::make_tuple(0.005, 2000, 0.5, PI/4., 0.000499728, 0.000500104),
            std::make_tuple(0.005, 2000, 0.5, PI/3., 0.000554791, 0.00044186), 
            std::make_tuple(0.005, 2000, 1., PI/6., 0.000499751, 0.000499752), 
            std::make_tuple(0.005, 2000, 1., PI/4., 0.000499751, 0.00049976)));

}
