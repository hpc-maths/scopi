#pragma once

#include "base.hpp"
#include <cstddef>
#include <vector>

namespace scopi
{
    class vap_projection;
     
    template<>
    struct VapParams<vap_projection>
    {
    };

    class vap_projection: public vap_base<vap_projection>
    {
    public:
        using base_type = vap_base<vap_projection>;
        template <std::size_t dim>
        void set_a_priori_velocity_impl(scopi_container<dim>& particles, const std::vector<neighbor<dim>>& contacts_pos, const std::vector<neighbor<dim>>& contacts_neg);

        template <std::size_t dim>
        void update_velocity_impl(scopi_container<dim>& particles, const xt::xtensor<double, 2>& uadapt, const xt::xtensor<double, 2>& wadapt);

        vap_projection(std::size_t Nactive, std::size_t active_ptr, std::size_t nb_parts, double dt, const VapParams<vap_projection>& params);

        void set_u_w(const xt::xtensor<double, 2>& u, const xt::xtensor<double, 2>& w);

    private:
        xt::xtensor<double, 2> m_u;
        xt::xtensor<double, 2> m_w;

    };

    template <std::size_t dim>
    void vap_projection::set_a_priori_velocity_impl(scopi_container<dim>& particles, const std::vector<neighbor<dim>>&, const std::vector<neighbor<dim>>&)
    {
        for (std::size_t i=0; i< this->m_Nactive; ++i)
        {
            for (std::size_t d=0; d<dim; ++d)
            {
                particles.vd()(i + this->m_active_ptr)(d) = m_u(i, d);
            }
            particles.omega()(i + this->m_active_ptr) = m_w(i, 2);
        }
    }

    template <std::size_t dim>
    void vap_projection::update_velocity_impl(scopi_container<dim>&, const xt::xtensor<double, 2>&, const xt::xtensor<double, 2>&)
    {}
}
