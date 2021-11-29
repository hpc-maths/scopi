#pragma once 

#include "../container.hpp"

namespace scopi
{
    template <class D>
    class vap_base: public crtp_base<D>
    {
    public:
        template <std::size_t dim>
        void run(const scopi_container<dim>& particles);
    };

    template <class D>
    template <std::size_t dim>
    void vap_base<D>::run(const scopi_container<dim>& particles)
    {
        this->derived_cast().run_impl(particles);
    }

    class vap_case_1: public vap_base<vap_case_1>
    {
    public:
        using base_type = vap_base<vap_case_1>;

        template <std::size_t dim>
        void run_impl(const scopi_container<dim>& particles)
        {
            std::cout << "run implementation" << std::endl;
        }
    };

}
