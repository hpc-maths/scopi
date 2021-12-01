#pragma once

#include "base.hpp"

namespace scopi
{
    class vap_pfd: public vap_base<vap_pfd>
    {
        public:
            using base_type = vap_base<vap_pfd>;
            template <std::size_t dim>
                void aPrioriVelocity_impl(scopi_container<dim>& particles);

            template <std::size_t dim>
                void updateVelocity_impl(scopi_container<dim>& particles);

            vap_pfd(std::size_t Nactive, std::size_t active_ptr, double dt);

        private:
            double f_ext();
            double t_ext();

    };

    vap_pfd::vap_pfd(std::size_t Nactive, std::size_t active_ptr, double dt)
        : base_type(Nactive, active_ptr, dt)
    {}

    template <std::size_t dim>
        void vap_pfd::aPrioriVelocity_impl(scopi_container<dim>& particles)
        {
            // TODO what if f_ext and t_ext depends on the particule ?
            particles.vd() = particles.v() + _dt*f_ext(); // TODO mass is missing
            // TODO should be dt * (R_i * t_i^{ext , n} - omega'_i * (J_i omega'_i)
            particles.desired_omega() = particles.omega() + _dt*t_ext(); // TODO momentum is missing
        }

    template <std::size_t dim>
        void vap_pfd::updateVelocity_impl(scopi_container<dim>& particles)
        {
            for (std::size_t i=0; i<_Nactive; ++i)
            {
                for (std::size_t d=0; d<dim; ++d)
                {
                    // particles.v()(i + _active_ptr)(d) = uadapt(i, d);
                    // particles.omega()(i + _active_ptr)(d) = wadapt(i, d);
                }
            }
        }

    double vap_pfd::f_ext()
    {
        return 0.;
    }

    double vap_pfd::t_ext()
    {
        return 0.;
    }
}
