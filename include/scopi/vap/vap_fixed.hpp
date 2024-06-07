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
    template <>
    struct VapParams<vap_fixed>
    {
    };

    /**
     * @brief Fixed a priori velocity.
     *
     * The a priori velocity (direction and norm) is fixed at the begining of the simulation and does not change.
     */
    class vap_fixed : public vap_base<vap_fixed>
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
        template <std::size_t dim, class Contacts>
        void set_a_priori_velocity_impl(double dt, scopi_container<dim>& particles, const Contacts& contacts);
    };

    template <std::size_t dim, class Contacts>
    void vap_fixed::set_a_priori_velocity_impl(double, scopi_container<dim>& particles, const Contacts&)
    {
        auto active_ptr = particles.nb_inactive();
        auto nb_active  = particles.nb_active();
        for (std::size_t i = active_ptr; i < active_ptr + nb_active; ++i)
        {
            particles.v()(i)     = particles.vd()(i);
            particles.omega()(i) = particles.desired_omega()(i);
        }
    }
}
