#pragma once

#include "base.hpp"
#include <cstddef>
#include <vector>

namespace scopi
{
    class vap_projection;

    /**
     * @brief Parameters for vap_projection.
     *
     * Defined for compatibility.
     */
    template <>
    struct VapParams<vap_projection>
    {
    };

    /**
     * @brief A priori velocity for projection used in viscous and friction problem.
     */
    class vap_projection : public vap_base<vap_projection>
    {
      public:

        /**
         * @brief Alias for the base class vap_base.
         */
        using base_type = vap_base<vap_projection>;
        /**
         * @brief Compute the fixed a priori velocity.
         *
         * @tparam dim Dimension (2 or 3).
         * @param particles [out] Array of particles.
         * @param contacts [in] Array of contacts.
         */
        template <std::size_t dim, class Contacts>
        void set_a_priori_velocity_impl(double dt, scopi_container<dim>& particles, const Contacts& contacts_pos);

        /**
         * @brief Update \c u and \c w.
         *
         * @param u [in] New value of \c u.
         * @param w [in] New value of \c w.
         */
        void set_u_w(const xt::xtensor<double, 2>& u, const xt::xtensor<double, 2>& w);

      private:

        /**
         * @brief Value of \c u that will be the a priori velocity.
         *
         * \f$ N \times 3 \f$ array.
         */
        xt::xtensor<double, 2> m_u;
        /**
         * @brief Value of \c w that will be the a priori velocity.
         *
         * \f$ N \times 3 \f$ array.
         */
        xt::xtensor<double, 2> m_w;
    };

    template <std::size_t dim, class Contacts>
    void vap_projection::set_a_priori_velocity_impl(double, scopi_container<dim>& particles, const Contacts&)
    {
        auto active_ptr = particles.nb_inactive();
        auto nb_active  = particles.nb_active();

#pragma omp parallel for
        for (std::size_t i = 0; i < nb_active; ++i)
        {
            for (std::size_t d = 0; d < dim; ++d)
            {
                particles.vd()(i + active_ptr)(d) = m_u(i, d);
            }
            particles.omega()(i + active_ptr) = m_w(i, 2);
        }
    }

}
