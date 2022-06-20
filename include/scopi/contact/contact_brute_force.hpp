#pragma once

#include "base.hpp"

#include <plog/Log.h>
#include "plog/Initializers/RollingFileInitializer.h"

namespace scopi
{


    class contact_brute_force: public contact_base<contact_brute_force>
    {
    public:
        using base_type = contact_base<contact_brute_force>;

        contact_brute_force(double dmax): contact_base(dmax){};

        template <std::size_t dim>
        std::vector<neighbor<dim>> run_impl(scopi_container<dim>& particles, std::size_t active_ptr)
        {
            // std::cout << "----> CONTACTS : run implementation contact_brute_force" << std::endl;

            std::vector<neighbor<dim>> contacts;

            tic();

            #pragma omp parallel for num_threads(8)

            for (std::size_t i = active_ptr; i < particles.size() - 1; ++i)
            {
              for (std::size_t j = i + 1; j < particles.size(); ++j)
              {
                if (i < j) 
                {
                    compute_exact_distance(particles, i, j, contacts, m_dmax);
                }

              }
            }

            // obstacles
            for (std::size_t i = 0; i < active_ptr; ++i)
            {
                for (std::size_t j = active_ptr; j < particles.size(); ++j)
                {
                    compute_exact_distance(particles, i, j, contacts, m_dmax);
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

    };

}
