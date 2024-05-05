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
        auto lagrange      = make_lagrange_multplier<dim, Viscous>(contacts, dt);
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

    template <std::size_t dim>
    inline void update_contact_properties_impl(double dt, const xt::xtensor<double, 1>& lambda, std::vector<neighbor<dim, ViscousFriction>>& contacts)
    {
        auto lagrange      = make_lagrange_multplier<dim, ViscousFriction>(contacts, dt);
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

        template  <std::size_t dim, class problem_t>
        void extra_steps_before_solve(std::vector<neighbor<dim, problem_t>>& )
        {
            m_should_solve = true;
        }

        template  <std::size_t dim>
        void extra_steps_before_solve(std::vector<neighbor<dim, FrictionFixedPoint>>& contacts)
        {
            Niter_fixed_point = 0;
            m_should_solve = true;
            for (std::size_t i = 0; i < contacts.size(); ++i)
            {
                contacts[i].sij = contacts[i].property.mu*m_dt;
            }
        }

        template  <std::size_t dim>
        void extra_steps_before_solve(std::vector<neighbor<dim, ViscousFriction>>& contacts)
        {
            Niter_fixed_point = 0;
            m_should_solve = true;
            for (std::size_t i = 0; i < contacts.size(); ++i)
            {
                if(contacts[i].property.gamma == contacts[i].property.gamma_min)
                {
                    contacts[i].sij = 1.;
                }
            }
        }
        
        template  <std::size_t dim, class problem_t>
        void extra_steps_after_solve(std::vector<neighbor<dim, problem_t>>&, const scopi_container<dim>&)
        {
            m_should_solve = false;
        }
        
        template  <std::size_t dim>
        void extra_steps_after_solve(std::vector<neighbor<dim, FrictionFixedPoint>>& contacts, const scopi_container<dim>& particles)
        {
            if(contacts.size()!=0)
            {
                xt::xtensor<double, 1> U = xt::zeros<double>({6 * particles.nb_active()});

                std::size_t offset = 3 * particles.nb_active();
                for (std::size_t i = 0; i < particles.nb_active(); ++i)
                {
                    for (std::size_t d =0; d<dim; ++d)
                    {
                        U[3 * i + d] = m_u(i, d);
                    }
                    if constexpr (dim == 2)
                    {
                        U[offset + 3 * i + 2] = m_omega(i, 2);
                    }
                    else if constexpr (dim == 3)
                    {
                        for (std::size_t d =0; d<dim; ++d)
                        {
                           U[offset + 3 * i + d] = m_omega(i,d);
                        }
                    }
                }
                
                AMatrix A(contacts, particles);
                xt::xtensor<double, 1> AU = A.mat_mult(U);
                double tol_point_fixe = contacts[0].property.fixed_point_tol;
                double Niter_max_fixed_point= contacts[0].property.fixed_point_max_iter;
                double err_point_fixe = 0;
                xt::xtensor<double, 1> new_s = xt::zeros<double>({contacts.size()});
                xt::xtensor<double, 1> old_s = xt::zeros<double>({contacts.size()});
                for (std::size_t i = 0, row = 0; i < contacts.size(); ++i, row += 3)
                {
                    auto AUi= xt::view(AU, xt::range(row, row + dim));
                    auto dotResult = xt::linalg::dot(AUi, contacts[i].nij)[0];

                    auto outerProduct = contacts[i].nij * dotResult;

                    auto TAUi = xt::eval(AUi - contacts[i].nij *(xt::linalg::dot(AUi, contacts[i].nij)[0]));
                    old_s[i] = contacts[i].sij;
                    contacts[i].sij = contacts[i].property.mu*m_dt*xt::norm_l2(TAUi)[0];
                    new_s[i] = contacts[i].sij;
                }
                err_point_fixe = (1/(contacts[0].property.mu*m_dt))*xt::norm_l2(new_s-old_s)[0]/(1+(1/(contacts[0].property.mu*m_dt))*xt::norm_l2(new_s)[0]);
                Niter_fixed_point++;
                if(err_point_fixe<tol_point_fixe || Niter_fixed_point>= Niter_max_fixed_point)
                {
                    m_should_solve = false;
                    //std::cout << "NB ITER PT FIXE = " << Niter_fixed_point << std::endl;
                }
            }
            else
            {
                m_should_solve = false;
            }
            
        }
        template  <std::size_t dim>
        void extra_steps_after_solve(std::vector<neighbor<dim, ViscousFriction>>& contacts, const scopi_container<dim>& particles)
        {
            if(contacts.size()!=0)
            {
                xt::xtensor<double, 1> U = xt::zeros<double>({6 * particles.nb_active()});

                std::size_t offset = 3 * particles.nb_active();
                for (std::size_t i = 0; i < particles.nb_active(); ++i)
                {
                    for (std::size_t d =0; d<dim; ++d)
                    {
                        U[3 * i + d] = m_u(i, d);
                    }
                    if constexpr (dim == 2)
                    {
                        U[offset + 3 * i + 2] = m_omega(i, 2);
                    }
                    else if constexpr (dim == 3)
                    {
                        for (std::size_t d =0; d<dim; ++d)
                        {
                           U[offset + 3 * i + d] = m_omega(i,d);
                        }
                    }
                }
                
                AMatrix A(contacts, particles);
                xt::xtensor<double, 1> AU = A.mat_mult(U);
                double tol_point_fixe = contacts[0].property.fixed_point_tol;
                double Niter_max_fixed_point= contacts[0].property.fixed_point_max_iter;
                double err_point_fixe = 0;
                xt::xtensor<double, 1> new_s = xt::zeros<double>({contacts.size()});
                xt::xtensor<double, 1> old_s = xt::zeros<double>({contacts.size()});
                for (std::size_t i = 0, row = 0; i < contacts.size(); ++i, row += 3)
                {
                    if(contacts[i].property.gamma == contacts[i].property.gamma_min)
                    {
                        auto AUi= xt::view(AU, xt::range(row, row + dim));
                        auto dotResult = xt::linalg::dot(AUi, contacts[i].nij)[0];

                        auto outerProduct = contacts[i].nij * dotResult;

                        auto TAUi = xt::eval(AUi - contacts[i].nij *(xt::linalg::dot(AUi, contacts[i].nij)[0]));
                        old_s[i] = contacts[i].sij;
                        contacts[i].sij = xt::norm_l2(TAUi)[0];
                        new_s[i] = contacts[i].sij;
                    }
                }
                err_point_fixe = xt::norm_l2(new_s-old_s)[0]/(1+xt::norm_l2(new_s)[0]);
                Niter_fixed_point++;
                if(err_point_fixe<tol_point_fixe || Niter_fixed_point>= Niter_max_fixed_point)
                {   
                    double max_contrainte = 0;
                    for (std::size_t i = 0, row = 0; i < contacts.size(); ++i, row += 3)
                     {
                        if(contacts[i].property.gamma == contacts[i].property.gamma_min)
                        {
                            auto AUi= xt::view(AU, xt::range(row, row + dim));
                            auto dotResult = xt::linalg::dot(AUi, contacts[i].nij)[0];

                        double contraintes = contacts[i].dij + m_dt * dotResult;
                        max_contrainte = std::max(std::abs(contraintes),max_contrainte);
                        
                        }
                     }
                    PLOG_INFO << "----> Max Contraintes = " << max_contrainte << std::endl;
                    m_should_solve = false;
                }
            }
            else
            {
                m_should_solve = false;
            }
            
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
                            m_omega(i, d) = particles.omega()(i + active_offset)(d) + m_dt * velocities((i + particles.nb_active()) * 3 + d);
                        }
                    }
                    if constexpr (dim == 2)
                    {
                        m_omega(i, 2) = particles.omega()(i + active_offset) + m_dt * velocities((i + particles.nb_active()) * 3 + 2);
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
            if(contacts.size()!=0)
        {
            update_contact_properties_impl(m_dt, m_lambda, contacts);
            }
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
        std::size_t Niter_fixed_point;
    };
}