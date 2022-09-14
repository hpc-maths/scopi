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
    /**
     * @brief Base class to compute contacts.
     *
     * @tparam D Class that implements the algorithm for contacts.
     */
    template <class D>
    class contact_base: public crtp_base<D>
    {
    public:
        /**
         * @brief Default constructor.
         */
        contact_base() {}

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
        std::vector<neighbor<dim>> run(scopi_container<dim>& particles, std::size_t active_ptr);
    };

    template <class D>
    template <std::size_t dim>
    std::vector<neighbor<dim>> contact_base<D>::run(scopi_container<dim>& particles, std::size_t active_ptr)
    {
        auto contacts = this->derived_cast().run_impl(particles, active_ptr);

        // obstacles
        tic();
        for (std::size_t i = 0; i < active_ptr; ++i)
        {
            for (std::size_t j = active_ptr; j < particles.pos().size(); ++j)
            {
                compute_exact_distance(particles, i, j, contacts, m_params.dmax);
            }
        }
        auto duration = toc();
        PLOG_INFO << "----> CPUTIME : compute contacts with obstacles = " << duration;

        // sort
        tic();
        sort_contacts(contacts);
        duration = toc();
        PLOG_INFO << "----> CPUTIME : sort " << contacts.size() << " contacts = " << duration;

        return contacts;
    }

    /**
     * @brief Compute the exact distance between two particles.
     *
     * @tparam dim Dimension (2 or 3).
     * @param particles [in] Array of particles.
     * @param i [in] Index of the first particle.
     * @param j [in] Index of the second particle.
     * @param contacts [out] Array of neighbors, if the distance between the two particles is small enough, add a neighbor in this array.
     * @param dmax [in] Maximum distance to consider two particles to be neighbors.
     */
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
