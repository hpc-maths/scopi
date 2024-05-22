#pragma once

#include "../box.hpp"
#include "../container.hpp"
#include "../objects/methods/closest_points.hpp"
#include "../objects/methods/select.hpp"
#include "../objects/methods/write_objects.hpp"
#include "../objects/neighbor.hpp"
#include "../params.hpp"
#include <CLI/CLI.hpp>
#include <cstddef>

namespace scopi
{
    /**
     * @brief Base class to compute contacts.
     *
     * @tparam D Class that implements the algorithm for contacts.
     */
    template <class D>
    class contact_base : public crtp_base<D>
    {
      public:

        using params_t = ContactsParams<D>;

        /**
         * @brief Default constructor.
         */
        contact_base(const params_t& params)
            : m_params(params)
        {
        }

        void init_options()
        {
            m_params.init_options();
        }

        /**
         * @brief Compute contacts between particles.
         *
         * @tparam dim Dimension (2 or 3).
         * @param particles [in] Array of particles.
         * @param active_ptr [in] Index of the first active particle.
         *
         * @return Array of neighbors.
         */
        template <std::size_t dim>
        auto run(const BoxDomain<dim>& box, scopi_container<dim>& particles, std::size_t active_ptr);

        /**
         * @brief Compute contacts between particles.
         *
         * @tparam dim Dimension (2 or 3).
         * @param particles [in] Array of particles.
         * @param active_ptr [in] Index of the first active particle.
         *
         * @return Array of neighbors.
         */
        template <std::size_t dim>
        auto run(scopi_container<dim>& particles, std::size_t active_ptr);

        params_t& get_params();

      protected:

        params_t m_params;
    };

    template <class D>
    template <std::size_t dim>
    auto contact_base<D>::run(const BoxDomain<dim>& box, scopi_container<dim>& particles, std::size_t active_ptr)
    {
        return this->derived_cast().run_impl(box, particles, active_ptr);
    }

    template <class D>
    template <std::size_t dim>
    auto contact_base<D>::run(scopi_container<dim>& particles, std::size_t active_ptr)
    {
        return this->derived_cast().run_impl(BoxDomain<dim>(), particles, active_ptr);
    }

    template <class D>
    auto contact_base<D>::get_params() -> params_t&
    {
        return m_params;
    }

    /**
     * @brief Compute the exact distance between two particles.
     *
     * @tparam dim Dimension (2 or 3).
     * @param particles [in] Array of particles.
     * @param contacts [inout] Array of neighbors, if the distance between the two particles is small enough, add a neighbor in this array.
     * @param dmax [in] Maximum distance to consider two particles to be neighbors.
     * @param i [in] Index of the first particle.
     * @param j [in] Index of the second particle.
     * @param default_contact_property [in] Default contact property.
     */
    template <class problem_t, std::size_t dim>
    void compute_exact_distance(const BoxDomain<dim>& box,
                                scopi_container<dim>& particles,
                                std::vector<neighbor<dim, problem_t>>& contacts,
                                double dmax,
                                std::size_t i,
                                std::size_t j,
                                contact_property<problem_t>& default_contact_property)
    {
        std::size_t o1 = particles.object_index(i);
        std::size_t o2 = particles.object_index(j);
        auto neigh     = closest_points_dispatcher<problem_t, dim>::dispatch(
            *select_object_dispatcher<dim>::dispatch(*particles[o1], index(i - particles.offset(o1))),
            *select_object_dispatcher<dim>::dispatch(*particles[o2], index(j - particles.offset(o2))));

        if (neigh.dij < dmax && (i < particles.periodic_ptr() || j < particles.periodic_ptr()))
        {
            neigh.i        = (i < particles.periodic_ptr()) ? i : particles.periodic_index(i - particles.periodic_ptr());
            neigh.j        = (j < particles.periodic_ptr()) ? j : particles.periodic_index(j - particles.periodic_ptr());
            neigh.property = default_contact_property;

            for (std::size_t d = 0; d < dim; ++d)
            {
                if (box.is_periodic(d))
                {
                    if (neigh.pi(d) > box.upper_bound(d) && i >= particles.periodic_ptr())
                    {
                        neigh.pi(d) -= box.upper_bound(d) - box.lower_bound(d);
                    }
                    if (neigh.pj(d) > box.upper_bound(d) && j >= particles.periodic_ptr())
                    {
                        neigh.pj(d) -= box.upper_bound(d) - box.lower_bound(d);
                    }
                }
            }
#pragma omp critical
            contacts.emplace_back(std::move(neigh));
        }
    }

    /**
     * @brief Sort contacts.
     *
     * When the array is sorted, indices i of a contact (i, j) are increasing.
     * For a fixed i, then indices j are increasing.
     * Furthermore, we always have i < j.
     *
     * For example, consider particles (0, 1, 2, 3) and assume all these particles are in contact two by two.
     * Then, the sorted array of neighbors is (0, 1) (0, 2) (0, 3) (1, 2) (1, 3) (2, 3).
     *
     * @tparam dim Dimension (2 or 3).
     * @param contacts [out] Array of contacts.
     */
    template <std::size_t dim, class problem_t>
    void sort_contacts(std::vector<neighbor<dim, problem_t>>& contacts)
    {
        std::sort(contacts.begin(),
                  contacts.end(),
                  [](auto& a, auto& b)
                  {
                      if (a.i < b.i)
                      {
                          return true;
                      }
                      else
                      {
                          if (a.i == b.i)
                          {
                              return a.j < b.j;
                          }
                      }
                      return false;
                  });
    }

}
