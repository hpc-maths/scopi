#pragma once

#include "base.hpp"

#include <cstddef>
#include <locale>
#include <plog/Log.h>
#include "plog/Initializers/RollingFileInitializer.h"

namespace scopi
{

    class contact_brute_force;

    /**
     * @brief Parameters for contact_brute_force.
     *
     * Specialization of ContactsParams in params.hpp.
     */
    template<>
    struct ContactsParams<contact_brute_force> : ContactsParamsBase
    {
        /**
         * @brief Default constructor.
         */
        ContactsParams();
        /**
         * @brief Copy constructor.
         *
         * @param params Parameters to be copied.
         */
        ContactsParams(const ContactsParams<contact_brute_force>& params);
    };

    /**
     * @brief Brute force contatcs.
     *
     * Contacts between particles are computed using brute force algorithm.
     */
    class contact_brute_force: public contact_base<contact_brute_force>
    {
    public:
        /**
         * @brief Alias for the base class \c contact_base.
         */
        using base_type = contact_base<contact_brute_force>;

        /**
         * @brief Constructor.
         *
         * @param params Parameters, see <tt>ContactsParams<contact_brute_force></tt>.
         */
        contact_brute_force(const ContactsParams<contact_brute_force>& params = ContactsParams<contact_brute_force>());

        /**
         * @brief Compute contacts between particles using brute force algorithm.
         *
         * @tparam dim Dimension (2 or 3).
         * @param particles [in] Array of particles.
         * @param active_ptr [in] Index of the first active particle.
         *
         * @return Array of neighbors.
         */
        template <std::size_t dim>
        std::vector<neighbor<dim>> run_impl(scopi_container<dim>& particles, std::size_t active_ptr);

    protected:
        /**
         * @brief Parameters.
         */
        ContactsParams<contact_brute_force> m_params;

    };

    template <std::size_t dim>
    std::vector<neighbor<dim>> contact_brute_force::run_impl(scopi_container<dim>& particles, std::size_t active_ptr)
    {
        // std::cout << "----> CONTACTS : run implementation contact_brute_force" << std::endl;

        std::vector<neighbor<dim>> contacts;

        tic();

        #pragma omp parallel for //num_threads(1)
        for (std::size_t i = active_ptr; i < particles.pos().size() - 1; ++i)
        {
            for (std::size_t j = i + 1; j < particles.pos().size(); ++j)
            {
                compute_exact_distance(particles, i, j, contacts, m_params.dmax);
            }
        }

        // obstacles
        for (std::size_t i = 0; i < active_ptr; ++i)
        {
            for (std::size_t j = active_ptr; j < particles.pos().size(); ++j)
            {
                compute_exact_distance(particles, i, j, contacts, m_params.dmax);
            }
        }

        auto duration = toc();
        PLOG_INFO << "----> CPUTIME : compute " << contacts.size() << " contacts = " << duration;

        tic();
        sort_contacts(contacts);
        duration = toc();
        PLOG_INFO << "----> CPUTIME : sort " << contacts.size() << " contacts = " << duration;

        // for (std::size_t ic=0; ic<contacts.size(); ++ic)
        // {
        //     std::cout << "----> CONTACTS : i j = " << contacts[ic].i << " " << contacts[ic].j << " d = " <<  contacts[ic].dij << std::endl;
        //     // std::cout << "----> CONTACTS : contact = " << contacts[ic] << std::endl;
        // }

        return contacts;

    }
}
