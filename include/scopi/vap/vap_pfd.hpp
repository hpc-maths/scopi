#pragma once

#include "base.hpp"

namespace scopi
{
    class vap_pfd: public vap_base<vap_pfd>
    {
        public:
            using base_type = vap_base<vap_pfd>;
            template <std::size_t dim>
                void aPrioriVelocity_impl(const scopi_container<dim>& particles);

            template <std::size_t dim>
                void updateVelocity_impl(const scopi_container<dim>& particles);

    };

    template <std::size_t dim>
        void vap_pfd::aPrioriVelocity_impl(const scopi_container<dim>& particles)
        {
            std::ignore = particles;
        }

    template <std::size_t dim>
        void vap_pfd::updateVelocity_impl(const scopi_container<dim>& particles)
        {
            std::ignore = particles;
        }
}
