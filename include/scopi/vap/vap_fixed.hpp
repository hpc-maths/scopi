#pragma once

#include "base.hpp"
#include <cstddef>
#include <vector>

namespace scopi
{
    class vap_fixed;

    /**
     * @brief Parameters for vap_fixed.
     *
     * Defined for compatibility.
     */
    template<>
    struct VapParams<vap_fixed>
    {
    };

    /**
     * @brief Fixed a priori velocity.
     *
     * The a priori velocity (direction and norm) is fixed at the begining of the simulation and does not change.
     */
    class vap_fixed: public vap_base<vap_fixed>
    {
    public:
        /**
         * @brief Alias for the base class vap_base.
         */
        using base_type = vap_base<vap_fixed>;
        /**
         * @brief Compute the fixed a priori velocity.
         *
         * @tparam dim Dimension (2 or 3).
         * @param particles [out] Array of particles.
         * @param contacts [in] Array of contacts.
         */
        template <std::size_t dim>
        void set_a_priori_velocity_impl(scopi_container<dim>& particles, const std::vector<neighbor<dim>>& contacts);

        /**
         * @brief Constructor.
         *
         * @param Nactive Number of active particles.
         * @param active_ptr Index of the first active particle.
         * @param nb_parts Number of objects.
         * @param dt Time step.
         * @param params Parameters (for compatibility).
         */
        vap_fixed(std::size_t Nactive, std::size_t active_ptr, std::size_t nb_parts, double dt);

    };

    template <std::size_t dim>
    void vap_fixed::set_a_priori_velocity_impl(scopi_container<dim>&, const std::vector<neighbor<dim>>&)
    {

    }
}
