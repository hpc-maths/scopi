#pragma once

#include "base.hpp"
#include <cstddef>
#include <vector>

namespace scopi
{
    class vap_fixed;
     
    template<>
    struct VapParams<vap_fixed>
    {
    };

    class vap_fixed: public vap_base<vap_fixed>
    {
    public:
        using base_type = vap_base<vap_fixed>;
        template <std::size_t dim>
        void set_a_priori_velocity_impl(scopi_container<dim>& particles, const std::vector<neighbor<dim>>& contacts_pos, const std::vector<neighbor<dim>>& contacts_neg);

        vap_fixed(std::size_t Nactive, std::size_t active_ptr, std::size_t nb_parts, double dt, const VapParams<vap_fixed>& params);

    };

    template <std::size_t dim>
    void vap_fixed::set_a_priori_velocity_impl(scopi_container<dim>&, const std::vector<neighbor<dim>>&, const std::vector<neighbor<dim>>&)
    {}
}
