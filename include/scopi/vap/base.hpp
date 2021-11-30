#pragma once 

#include "../container.hpp"

namespace scopi
{
    template <class D>
        class vap_base: public crtp_base<D>
    {
        public:
            template <std::size_t dim>
                void aPrioriVelocity(const scopi_container<dim>& particles);
            template <std::size_t dim>
                void updateVelocity(const scopi_container<dim>& particles);
    };

    template <class D>
        template <std::size_t dim>
        void vap_base<D>::aPrioriVelocity(const scopi_container<dim>& particles)
        {
            this->derived_cast().aPrioriVelocity_impl(particles);
        }

    template <class D>
        template <std::size_t dim>
        void vap_base<D>::updateVelocity(const scopi_container<dim>& particles)
        {
            this->derived_cast().updateVelocity_impl(particles);
        }
}
