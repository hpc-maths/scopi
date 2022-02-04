#pragma once

#include "base.hpp"

namespace scopi
{
    class vap_fpd: public vap_base<vap_fpd>
    {
        public:
            using base_type = vap_base<vap_fpd>;
            template <std::size_t dim>
                void set_a_priori_velocity_impl(scopi_container<dim>& particles);

            template <std::size_t dim>
                void update_velocity_impl(scopi_container<dim>& particles, const xt::xtensor<double, 2>& uadapt, const xt::xtensor<double, 2>& wadapt);

            vap_fpd(std::size_t Nactive, std::size_t active_ptr, double dt);

        private:
            template <std::size_t dim>
                auto f_ext(scopi_container<dim>& particles, std::size_t i);
            double t_ext();

            double _mass;
            // double _moment;

    };

    template <std::size_t dim>
        void vap_fpd::set_a_priori_velocity_impl(scopi_container<dim>& particles)
        {
            for (std::size_t i=0; i<m_Nactive; ++i)
            {
                auto pos = particles.pos()(i + m_active_ptr);
                double dist = xt::linalg::norm(pos);
                auto fExt = - _mass*_mass/(dist*dist)*pos/dist;
                particles.vd()(m_active_ptr + i) = particles.v()(m_active_ptr + i) + m_dt*fExt/_mass; // TODO: add mass into particles
            }
            // TODO should be dt * (R_i * t_i^{ext , n} - omega'_i * (J_i omega'_i)
            particles.desired_omega() = particles.omega();
        }

    template <std::size_t dim>
        void vap_fpd::update_velocity_impl(scopi_container<dim>& particles, const xt::xtensor<double, 2>& uadapt, const xt::xtensor<double, 2>& wadapt)
        {
            for (std::size_t i=0; i<m_Nactive; ++i)
            {
                for (std::size_t d=0; d<dim; ++d)
                {
                    particles.v()(i + m_active_ptr)(d) = uadapt(i, d);
                }
                particles.omega()(i + m_active_ptr) = wadapt(i, 2);
            }
        }

    template <std::size_t dim>
        auto vap_fpd::f_ext(scopi_container<dim>& particles, std::size_t i)
        {
            auto pos = particles.pos()(i + m_active_ptr);
            double dist = xt::linalg::norm(pos);
            auto res = _mass/(dist*dist)*pos/dist;
            return res;
        }
}
