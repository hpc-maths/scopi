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
    template<>
    struct VapParams<vap_projection>
    {
    };

    /**
     * @brief A priori velocity for projection used in viscous and friction problem.
     */
    class vap_projection: public vap_base<vap_projection>
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
         * @param contacts_pos [in] Array of neighbors with positive distance.
         * @param contacts_neg [in] Array of neighbors with negative distance.
         */
        template <std::size_t dim>
        void set_a_priori_velocity_impl(scopi_container<dim>& particles, const std::vector<neighbor<dim>>& contacts_pos, const std::vector<neighbor<dim>>& contacts_neg);

        /**
         * @brief Constructor.
         *
         * @param Nactive Number of active particles.
         * @param active_ptr Index of the first active particle.
         * @param nb_parts Number of objects.
         * @param dt Time step.
         * @param params Parameters (for compatibility).
         */
        vap_projection(std::size_t Nactive, std::size_t active_ptr, std::size_t nb_parts, double dt, const VapParams<vap_projection>& params);

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

    template <std::size_t dim>
    void vap_projection::set_a_priori_velocity_impl(scopi_container<dim>& particles, const std::vector<neighbor<dim>>&, const std::vector<neighbor<dim>>&)
    {
        for (std::size_t i=0; i< this->m_Nactive; ++i)
        {
            for (std::size_t d=0; d<dim; ++d)
            {
                particles.vd()(i + this->m_active_ptr)(d) = m_u(i, d);
            }
            particles.omega()(i + this->m_active_ptr) = m_w(i, 2);
        }
    }

}
