#pragma once 

#include "../container.hpp"
#include "../objects/methods/closest_points.hpp"
#include "../objects/neighbor.hpp"
#include <nanoflann.hpp>

namespace scopi
{

    template <class D>
    class contact_base: public crtp_base<D>
    {
    public:
        contact_base(double dmax): _dmax(dmax){}
        template <std::size_t dim>
        std::vector<scopi::neighbor<dim>> run(scopi_container<dim>& particles, std::size_t active_ptr);
    protected:
      double _dmax;
    };


    template <class D>
    template <std::size_t dim>
    std::vector<scopi::neighbor<dim>> contact_base<D>::run(scopi_container<dim>& particles, std::size_t active_ptr)
    {
        return this->derived_cast().run_impl(particles, active_ptr);
    }

}
