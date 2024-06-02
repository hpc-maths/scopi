#pragma once

#include "../container.hpp"
#include "../objects/neighbor.hpp"
#include "../params.hpp"
#include "../utils.hpp"
#include <vector>

namespace scopi
{
    /**
     * @brief Base class for a priori velocity.
     *
     * @tparam D Class that implements the a priori velocity.
     */
    template <class D>
    class vap_base : public crtp_base<D>
    {
      public:

        using params_t = VapParams<D>;

        /**
         * @brief Compute the a priori velocity.
         *
         * @tparam dim Dimension (2 or 3).
         * @param particles [out] Array of particles.
         * @param contacts [in] Array of contacts.
         */
        template <std::size_t dim, class Contacts>
        void set_a_priori_velocity(scopi_container<dim>& particles, const Contacts& contacts);
        /**
         * @brief Constructor.
         *
         * @param Nactive Number of active particles.
         * @param active_ptr Index of the first active particle.
         * @param dt Time step.
         */
        vap_base(std::size_t Nactive, std::size_t active_ptr, double dt);

        params_t& get_params();

      protected:

        /**
         * @brief Number of active particles.
         */
        std::size_t m_Nactive;
        /**
         * @brief Index of the first active particle.
         */
        std::size_t m_active_ptr;
        /**
         * @brief Time step.
         */
        double m_dt;

      private:

        params_t m_params;
    };

    template <class D>
    vap_base<D>::vap_base(std::size_t Nactive, std::size_t active_ptr, double dt)
        : m_Nactive(Nactive)
        , m_active_ptr(active_ptr)
        , m_dt(dt)
        , m_params()
    {
    }

    template <class D>
    template <std::size_t dim, class Contacts>
    void vap_base<D>::set_a_priori_velocity(scopi_container<dim>& particles, const Contacts& contacts)
    {
        tic();
        this->derived_cast().set_a_priori_velocity_impl(particles, contacts);
        auto duration = toc();
        PLOG_INFO << "----> CPUTIME : set vap = " << duration;
    }

    template <class D>
    auto vap_base<D>::get_params() -> params_t&
    {
        return m_params;
    }

}
