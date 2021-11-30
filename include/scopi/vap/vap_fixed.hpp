#pragma once

#include "base.hpp"

namespace scopi
{
    class vap_fixed: public vap_base<vap_fixed>
    {
        public:
            using base_type = vap_base<vap_fixed>;
            template <std::size_t dim>
                void aPrioriVelocity_impl(const scopi_container<dim>& particles);

            template <std::size_t dim>
                void updateVelocity_impl(const scopi_container<dim>& particles);

    };

    template <std::size_t dim>
        void vap_fixed::aPrioriVelocity_impl(const scopi_container<dim>& particles)
        {
            std::ignore = particles;
        }

    template <std::size_t dim>
        void vap_fixed::updateVelocity_impl(const scopi_container<dim>& particles)
        {
            std::ignore = particles;
        }
}
