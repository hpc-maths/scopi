#pragma once

#include "../box.hpp"
#include "../scopi.hpp"
#include "../utils.hpp"
#include "base.hpp"

#include <cstddef>
#include <locale>
#include <plog/Initializers/RollingFileInitializer.h>
#include <plog/Log.h>

namespace scopi
{

    template <class problem_t>
    class contact_brute_force;

    /**
     * @brief Parameters for contact_brute_force.
     *
     * Specialization of ContactsParams.
     */
    template <class problem_t>
    struct ContactsParams<contact_brute_force<problem_t>>
    {
        // /**
        //  * @brief Default constructor.
        //  */
        // ContactsParams() // cppcheck-suppress uninitMemberVar
        //     : dmax(2.)
        // {
        // }

        void init_options()
        {
            auto& app = get_app();
            auto* opt = app.add_option_group("Brute force contact options");
            opt->add_option("--dmax", dmax, "Maximum distance between two neighboring particles")->capture_default_str();
        }

        /**
         * @brief Maximum distance between two neighboring particles.
         *
         * Default value: 2.
         * \note \c dmax > 0
         */
        double dmax{2.};
    };

    /**
     * @brief Brute force contatcs.
     *
     * Contacts between particles are computed using brute force algorithm.
     */
    template <class problem_t>
    class contact_brute_force : public contact_base<contact_brute_force<problem_t>>
    {
      public:

        /**
         * @brief Alias for the base class contact_base.
         */
        using base_type = contact_base<contact_brute_force<problem_t>>;

        /**
         * @brief Constructor.
         *
         * @param params [in] Parameters.
         */
        explicit contact_brute_force(
            const ContactsParams<contact_brute_force<problem_t>>& params = ContactsParams<contact_brute_force<problem_t>>())
            : base_type(params)
        {
        }

        /**
         * @brief Compute contacts between particles using brute force algorithm.
         *
         * Only the contact between particles \c i and \c j is computed, not the contact between \c j and \c i, with \c i < \c j.
         *
         * The returned array of neighbors is sorted.
         * See sort_contacts.
         *
         * @tparam dim Dimension (2 or 3).
         * @param particles [in] Array of particles.
         * @param active_ptr [in] Index of the first active particle.
         *
         * @return Array of neighbors.
         */
        template <std::size_t dim>
        auto run_impl(const BoxDomain<dim>& box, scopi_container<dim>& particles, std::size_t active_ptr);

        contact_property<problem_t> m_default_contact_property;

        auto& default_contact_property()
        {
            return m_default_contact_property;
        }
    };

    template <class problem_t>
    template <std::size_t dim>
    auto contact_brute_force<problem_t>::run_impl(const BoxDomain<dim>& box, scopi_container<dim>& particles, std::size_t active_ptr)
    {
        std::vector<neighbor<dim, problem_t>> contacts;

        add_objects_from_periodicity(box, particles, this->get_params().dmax);

        tic();
#pragma omp parallel for
        for (std::size_t i = active_ptr; i < particles.pos().size() - 1; ++i)
        {
            for (std::size_t j = i + 1; j < particles.pos().size(); ++j)
            {
                compute_exact_distance<problem_t>(box, particles, contacts, this->get_params().dmax, i, j, m_default_contact_property);
            }
        }

        // obstacles
        for (std::size_t i = 0; i < active_ptr; ++i)
        {
            for (std::size_t j = active_ptr; j < particles.pos().size(); ++j)
            {
                compute_exact_distance<problem_t>(box, particles, contacts, this->get_params().dmax, i, j, m_default_contact_property);
            }
        }

        auto duration = toc();
        PLOG_INFO << "----> CPUTIME : compute " << contacts.size() << " contacts = " << duration;

        tic();
        sort_contacts(contacts);
        duration = toc();
        PLOG_INFO << "----> CPUTIME : sort " << contacts.size() << " contacts = " << duration;

        particles.reset_periodic();

        return contacts;
    }
}
