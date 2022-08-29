#pragma once

#include "base.hpp"
#include <cstddef>
#include <vector>

namespace scopi
{
    class vap_fpd;
     
    /**
     * @brief Parameters for \c vap_fpd.
     *
     * Defined for compatibility.
     */
    template<>
    struct VapParams<vap_fpd>
    {
    };

    /**
     * @brief A priori velocity for fundamental principle of dynamics (FPD).
     *
     * \f$ \vec{u}^{desired} = \vec{u^n} + \Delta t \mathbb{M}^{-1} \vec{f}^{ext} \f$ 
     * and
     * \f$ \vec{\omega'}^{desired} = \vec{\omega'}^n + \Delta t \vec{\omega'}^n \land ( \mathbb{J} \vec{\omega'}^n ) \f$.
     */
    class vap_fpd: public vap_base<vap_fpd>
    {
    public:
        using base_type = vap_base<vap_fpd>;
        /**
         * @brief Compute the a priori velocity using the fundamental principle of dynamics.
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
        vap_fpd(std::size_t Nactive, std::size_t active_ptr, std::size_t nb_parts, double dt, const VapParams<vap_fpd>& params);

    };

    /**
     * @brief Compute the product \f$\vec{\omega'}^n \land ( \mathbb{J} \vec{\omega'}^n ) \f$.
     *
     * 2D function.
     *
     * @param particles [in] Particles, to access \f$ \vec{\omega'}^n \f$.
     * @param i [in] Index of the particle.
     *
     * @return 
     */
    type::moment_t<2> cross_product_vap_fpd(const scopi_container<2>& particles, std::size_t i);
    /**
     * @brief Compute the product \f$\vec{\omega'}^n \land ( \mathbb{J} \vec{\omega'}^n ) \f$.
     *
     * 3D function.
     *
     * @param particles [in] Particles, to access \f$ \vec{\omega'}^n \f$.
     * @param i [in] Index of the particle.
     *
     * @return 
     */
    type::moment_t<3> cross_product_vap_fpd(const scopi_container<3>& particles, std::size_t i);

    template <std::size_t dim>
    void vap_fpd::set_a_priori_velocity_impl(scopi_container<dim>& particles, const std::vector<neighbor<dim>>&, const std::vector<neighbor<dim>>&)
    {
        for (std::size_t i=0; i<m_Nactive; ++i)
        {
            particles.vd()(m_active_ptr + i) = particles.v()(m_active_ptr + i) + m_dt*particles.f()(m_active_ptr + i)/particles.m()(m_active_ptr + i);
            // TODO should be dt * (R_i * t_i^{ext , n} - omega'_i * (J_i omega'_i)
            particles.desired_omega()(m_active_ptr + i) = particles.omega()(m_active_ptr + i) + cross_product_vap_fpd(particles, m_active_ptr + i);
        }
    }

}
