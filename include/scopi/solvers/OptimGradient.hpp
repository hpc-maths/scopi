#pragma once

#include <xtensor/xtensor.hpp>

#include "../contact/property.hpp"
#include "../objects/neighbor.hpp"
#include "../utils.hpp"
#include "lagrange_multiplier.hpp"
#include "minimization_problem.hpp"

namespace scopi
{

    template <class problem_t, std::size_t dim>
    inline void update_contact_properties_impl(double, const xt::xtensor<double, 1>&, std::vector<neighbor<dim, problem_t>>&)
    {
    }

    template <std::size_t dim>
    inline void update_contact_properties_impl(double dt, const xt::xtensor<double, 1>& lambda, std::vector<neighbor<dim, Viscous>>& contacts)
    {
        auto lagrange      = make_lagrange_multplier<dim, Viscous>(contacts);
        auto lambda_global = lagrange.local2global(lambda);
        std::size_t row    = 0;

        for (std::size_t i = 0; i < contacts.size(); ++i)
        {
            contacts[i].property.gamma = std::min(
                std::max(contacts[i].property.gamma
                             - dt * xt::linalg::dot(contacts[i].nij, xt::view(lambda_global, xt::range(row, row + dim)))[0],
                         contacts[i].property.gamma_min),
                0.);
            row += 3;
        }
    }

    template <class Method>
    class OptimGradient
    {
      public:

        using method_t = Method;
        using params_t = typename method_t::params_t;

        template <std::size_t dim>
        OptimGradient(std::size_t, double dt, const scopi_container<dim>&)
            : m_dt(dt)
            , m_method(method_t())
        {
        }

        void init_options(CLI::App& app)
        {
            m_method.init_options(app);
        }

        params_t& get_params()
        {
            return m_method.get_params();
        }

        template <class Contacts>
        void extra_steps_before_solve(const Contacts&)
        {
            m_should_solve = true;
        }

        template <class Contacts>
        void extra_steps_after_solve(const Contacts&)
        {
            m_should_solve = false;
        }

        template <std::size_t dim, class problem_t>
        void run(const scopi_container<dim>& particles, const std::vector<neighbor<dim, problem_t>>& contacts, std::size_t)
        {
            tic();
            std::size_t active_offset = particles.nb_inactive();
            m_u.resize({particles.nb_active(), 3});
            m_omega.resize({particles.nb_active(), 3});
            m_omega.fill(0);

            if (contacts.size() != 0)
            {
                auto min_p = make_minimization_problem<problem_t>(m_dt, contacts, particles);

                m_lambda = m_method(min_p);

                auto velocities = min_p.velocities(m_lambda);

                for (std::size_t i = 0; i < particles.nb_active(); ++i)
                {
                    for (std::size_t d = 0; d < dim; ++d)
                    {
                        m_u(i, d) = particles.v()(i + active_offset)(d) + m_dt * velocities(i * 3 + d);

                        if constexpr (dim == 3)
                        {
                            m_omega(i, d) = particles.omega()(i + active_offset)(d) + m_dt * velocities((i + active_offset) * 3 + d);
                        }
                    }
                    if constexpr (dim == 2)
                    {
                        m_omega(i, 2) = particles.omega()(i + active_offset) + m_dt * velocities((i + active_offset) * 3 + 2);
                    }
                }
            }
            else
            {
                for (std::size_t i = 0; i < particles.nb_active(); ++i)
                {
                    for (std::size_t d = 0; d < dim; ++d)
                    {
                        m_u(i, d) = particles.v()(i + active_offset)(d);
                        if constexpr (dim == 3)
                        {
                            m_omega(i, d) = particles.omega()(i + active_offset)(d);
                        }
                    }
                    if constexpr (dim == 2)
                    {
                        m_omega(i, 2) = particles.omega()(i + active_offset);
                    }
                }
            }
            auto duration = toc();
            PLOG_INFO << "----> CPUTIME : solve OptimGradient = " << duration << std::endl;
        }

        template <class Contacts>
        void update_contact_properties(Contacts& contacts)
        {
            update_contact_properties_impl(m_dt, m_lambda, contacts);
        }

        const auto& get_uadapt() const
        {
            return m_u;
        }

        const auto& get_wadapt() const
        {
            return m_omega;
        }

        const auto& lagrange_multiplier() const
        {
            return m_lambda;
        }

        bool should_solve()
        {
            return m_should_solve;
        }

      protected:

        double m_dt;

      private:

        method_t m_method;
        bool m_should_solve;
        xt::xtensor<double, 2> m_u;
        xt::xtensor<double, 2> m_omega;
        xt::xtensor<double, 1> m_lambda;
    };
}