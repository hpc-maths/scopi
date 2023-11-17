#pragma once

#include "../dispatch.hpp"
#include "../neighbor.hpp"
#include "../types/worm.hpp"
#include "closest_points.hpp"

namespace scopi
{

    template <std::size_t dim, class Contacts>
    void add_contact_from_object(const worm<dim, false>& w, std::size_t offset, Contacts& contacts)
    {
        using problem_t = typename Contacts::value_type::problem_t;
        for (std::size_t i = 0; i < w.size() - 1; ++i)
        {
            auto neigh = closest_points_dispatcher<problem_t, dim>::dispatch(*w.get_sphere(i), *w.get_sphere(i + 1));
            neigh.i    = offset + i;
            neigh.j    = offset + i + 1;
            neigh.nij *= -1;
            neigh.dij *= -1;
            contacts.emplace_back(std::move(neigh));
        }
    }

    template <std::size_t dim>
    struct add_contact_from_object_functor
    {
        template <class T1, class Contacts>
        void run(const T1& obj, std::size_t offset, Contacts& contacts) const
        {
            return add_contact_from_object(obj, offset, contacts);
        }

        template <class Contacts>
        void on_error(const object<dim, false>&, std::size_t, Contacts&) const
        {
        }
    };

    template <std::size_t dim>
    using add_contact_from_object_dispatcher = unit_static_dispatcher<add_contact_from_object_functor<dim>,
                                                                      const object<dim, false>,
                                                                      mpl::vector<const worm<dim, false>>>;
}
