#pragma once 

#include "../container.hpp"
#include "../objects/methods/closest_points.hpp"
#include "../objects/methods/select.hpp"
#include "../objects/neighbor.hpp"
#include "../params.hpp"
#include <cstddef>
#include <nanoflann.hpp>

namespace scopi
{
    struct ContactsParamsBase
    {
        ContactsParamsBase();
        ContactsParamsBase(const ContactsParamsBase& params);
    };

    template <class D>
    class contact_base: public crtp_base<D>
    {
    public:
        contact_base(double dmax): m_dmax(dmax){}
        template <std::size_t dim>
        std::vector<neighbor<dim>> run(scopi_container<dim>& particles, std::size_t active_ptr);
    protected:
      double m_dmax;
    };

    template <class D>
    template <std::size_t dim>
    std::vector<neighbor<dim>> contact_base<D>::run(scopi_container<dim>& particles, std::size_t active_ptr)
    {
        return this->derived_cast().run_impl(particles, active_ptr);
    }

    template <std::size_t dim>
    void compute_exact_distance(scopi_container<dim>& particles, std::size_t i, std::size_t j, std::vector<neighbor<dim>>& contacts, double dmax)
    {
        std::size_t o1 = particles.object_index(i);
        std::size_t o2 = particles.object_index(j);
        auto neigh = closest_points_dispatcher<dim>::dispatch(*select_object_dispatcher<dim>::dispatch(*particles[o1], index(i-particles.offset(o1))),
                                                              *select_object_dispatcher<dim>::dispatch(*particles[o2], index(j-particles.offset(o2))));
        if (neigh.dij < dmax) {
            neigh.i = i;
            neigh.j = j;
            #pragma omp critical
            contacts.emplace_back(std::move(neigh));
        }
    }

    template <std::size_t dim>
    void sort_contacts(std::vector<neighbor<dim>>& contacts)
    {
        std::sort(contacts.begin(), contacts.end(), [](auto& a, auto& b )
        {
          if (a.i < b.i) {
            return true;
          }
          else {
            if (a.i == b.i) {
              return a.j < b.j;
            }
          }
          return false;
        });
    }

}
