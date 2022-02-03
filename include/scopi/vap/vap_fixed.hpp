#pragma once

#include "base.hpp"

namespace scopi
{
    class vap_fixed: public vap_base<vap_fixed>
    {
        public:
            using base_type = vap_base<vap_fixed>;
            template <std::size_t dim>
                void aPrioriVelocity_impl(scopi_container<dim>& particles);

            template <std::size_t dim>
                void updateVelocity_impl(scopi_container<dim>& particles, const xt::xtensor<double, 2>& uadapt, const xt::xtensor<double, 2>& wadapt);

            vap_fixed(std::size_t Nactive, std::size_t active_ptr, double dt);

    };

    template <std::size_t dim>
        void vap_fixed::aPrioriVelocity_impl(scopi_container<dim>& particles)
        {
            std::ignore = particles;
        }

    template <std::size_t dim>
        void vap_fixed::updateVelocity_impl(scopi_container<dim>& particles, const xt::xtensor<double, 2>& uadapt, const xt::xtensor<double, 2>& wadapt)
        {
            std::ignore = particles;
            std::ignore = uadapt;
            std::ignore = wadapt;
        }

}
