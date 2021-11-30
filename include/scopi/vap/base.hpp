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
            vap_base(std::size_t Nactive, std::size_t active_ptr, double dt);
        private:
                std::size_t _Nactive;
                std::size_t _active_ptr;
                double _dt;
    };

    template <class D>
        vap_base<D>::vap_base(std::size_t Nactive, std::size_t active_ptr, double dt)
        : _Nactive(Nactive)
          , _active_ptr(active_ptr),
          _dt(dt)
    {
    }

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
