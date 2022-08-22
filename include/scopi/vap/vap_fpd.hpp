#pragma once

#include "base.hpp"
#include <cstddef>
#include <vector>

namespace scopi
{
    class vap_fpd;
     
    template<>
    struct VapParams<vap_fpd>
    {
    };

    class vap_fpd: public vap_base<vap_fpd>
    {
    public:
        using base_type = vap_base<vap_fpd>;
        template <std::size_t dim>
        void set_a_priori_velocity_impl(scopi_container<dim>& particles, const std::vector<neighbor<dim>>& contacts_pos, const std::vector<neighbor<dim>>& contacts_neg);

        vap_fpd(std::size_t Nactive, std::size_t active_ptr, std::size_t nb_parts, double dt, const VapParams<vap_fpd>& params);

    };

    type::moment_t<2> cross_product_vap_fpd(const scopi_container<2>& particles, std::size_t i);
    type::moment_t<3> cross_product_vap_fpd(const scopi_container<3>& particles, std::size_t i);

    template <std::size_t dim>
    void vap_fpd::set_a_priori_velocity_impl(scopi_container<dim>& particles, const std::vector<neighbor<dim>>&, const std::vector<neighbor<dim>>&)
    {
        for (std::size_t i=0; i<m_Nactive; ++i)
        {
            particles.vd()(m_active_ptr + i) = particles.v()(m_active_ptr + i) + m_dt*particles.f()(m_active_ptr + i)/particles.m()(m_active_ptr + i);
            // TODO should be dt * (R_i * t_i^{ext , n} - omega'_i * (J_i omega'_i)
            particles.desired_omega()(m_active_ptr + i) = particles.omega()(m_active_ptr + i) + cross_product_vap_fpd(particles, m_active_ptr + i);
        }
    }

}
