#pragma once

#include "base.hpp"

#include <cstddef>
#include <locale>
#include <plog/Log.h>
#include "plog/Initializers/RollingFileInitializer.h"

namespace scopi
{

    class contact_brute_force;

    template<>
    struct ContactsParams<contact_brute_force> : ContactsParamsBase
    {
        ContactsParams();
        ContactsParams(const ContactsParams<contact_brute_force>& params);
    };

    class contact_brute_force: public contact_base<contact_brute_force>
    {
    public:
        using base_type = contact_base<contact_brute_force>;

        contact_brute_force(const ContactsParams<contact_brute_force>& params = ContactsParams<contact_brute_force>());

        template <std::size_t dim>
        std::vector<neighbor<dim>> run_impl(scopi_container<dim>& particles, std::size_t active_ptr);

    protected:
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
