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
        void set_a_priori_velocity(double dt, scopi_container<dim>& particles, const Contacts& contacts);

        params_t& get_params();

      private:

        params_t m_params;
    };

    template <class D>
    template <std::size_t dim, class Contacts>
    void vap_base<D>::set_a_priori_velocity(double dt, scopi_container<dim>& particles, const Contacts& contacts)
    {
        tic();
        this->derived_cast().set_a_priori_velocity_impl(dt, particles, contacts);
        auto duration = toc();
        PLOG_INFO << "----> CPUTIME : set vap = " << duration;
    }

    template <class D>
    auto vap_base<D>::get_params() -> params_t&
    {
        return m_params;
    }

}
