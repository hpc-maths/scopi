#pragma once

#include "base.hpp"
#include <cstddef>
#include <vector>

namespace scopi
{
    class vap_fpd;

    /**
     * @brief Parameters for vap_fpd.
     *
     * Defined for compatibility.
     */
    template <>
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
    class vap_fpd : public vap_base<vap_fpd>
    {
      public:

        using base_type = vap_base<vap_fpd>;
        /**
         * @brief Compute the a priori velocity using the fundamental principle of dynamics.
         *
         * \todo External momentum is missing.
         *
         * @tparam dim Dimension (2 or 3).
         * @param particles [out] Array of particles.
         * @param contacts [in] Array of contacts.
         */
        template <std::size_t dim, class Contacts>
        void set_a_priori_velocity_impl(scopi_container<dim>& particles, const Contacts& contacts);

        /**
         * @brief Constructor.
         *
         * @param Nactive Number of active particles.
         * @param active_ptr Index of the first active particle.
         * @param nb_parts Number of objects.
         * @param dt Time step.
         * @param params Parameters (for compatibility).
         */
        vap_fpd(std::size_t Nactive, std::size_t active_ptr, std::size_t nb_parts, double dt);
    };

    /**
     * @brief Compute the product \f$\vec{\omega'}^n \land ( \mathbb{J} \vec{\omega'}^n ) \f$.
     *
     * 2D implementation.
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
     * 3D implementation.
     *
     * @param particles [in] Particles, to access \f$ \vec{\omega'}^n \f$.
     * @param i [in] Index of the particle.
     *
     * @return
     */
    type::moment_t<3> cross_product_vap_fpd(const scopi_container<3>& particles, std::size_t i);

    template <std::size_t dim, class Contacts>
    void vap_fpd::set_a_priori_velocity_impl(scopi_container<dim>& particles, const Contacts&)
    {
#pragma omp parallel for
        for (std::size_t i = m_active_ptr; i < m_active_ptr + m_Nactive; ++i)
        {
            particles.v()(i) += m_dt * particles.f()(i) / particles.m()(i);
            particles.omega()(i) += cross_product_vap_fpd(particles, i);
        }
    }

}
