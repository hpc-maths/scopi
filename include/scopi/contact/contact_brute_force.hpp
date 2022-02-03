#pragma once

#include "base.hpp"

namespace scopi
{


    class contact_brute_force: public contact_base<contact_brute_force>
    {
    public:
        using base_type = contact_base<contact_brute_force>;

        contact_brute_force(double dmax): contact_base(dmax){};

        template <std::size_t dim>
        std::vector<scopi::neighbor<dim>> run_impl(scopi_container<dim>& particles, std::size_t active_ptr)
        {
            // std::cout << "----> CONTACTS : run implementation contact_brute_force" << std::endl;

            std::vector<scopi::neighbor<dim>> contacts;

            tic();

            #pragma omp parallel for num_threads(8)

            for(std::size_t i = active_ptr; i < particles.size() - 1; ++i)
            {
              for(std::size_t j = i + 1; j < particles.size(); ++j)
              {
                if (i < j) {
                  auto neigh = scopi::closest_points_dispatcher<dim>::dispatch(*particles[i], *particles[j]);
                  if (neigh.dij < _dmax) {
                      neigh.i = i;
                      neigh.j = j;
                      #pragma omp critical
                      contacts.emplace_back(std::move(neigh));
                      // contacts.back().i = i;
                      // contacts.back().j = j;
                  }
                }

              }
            }

            // obstacles
            for(std::size_t i = 0; i < active_ptr; ++i)
            {
                for(std::size_t j = active_ptr; j < particles.size(); ++j)
                {
                    auto neigh = scopi::closest_points_dispatcher<dim>::dispatch(*particles[i], *particles[j]);
                    if (neigh.dij < _dmax)
                    {
                        neigh.i = i;
                        neigh.j = j;
                        contacts.emplace_back(std::move(neigh));
                    }
                }
            }

            auto duration = toc();
            // std::cout << "----> CPUTIME : compute " << contacts.size() << " contacts = " << duration << std::endl;

            tic();
            std::sort(contacts.begin(), contacts.end(), [](auto& a, auto& b )
            {
              if (a.i < b.i) {
                return true;
              }
              else {
                if (a.i == b.i) {
                  return a.j < b.j;
                }
              }
              return false;
            });
            duration = toc();
            // std::cout << "----> CPUTIME : sort " << contacts.size() << " contacts = " << duration << std::endl;

            // for(std::size_t ic=0; ic<contacts.size(); ++ic)
            // {
            //     std::cout << "----> CONTACTS : i j = " << contacts[ic].i << " " << contacts[ic].j << " d = " <<  contacts[ic].dij << std::endl;
            //     // std::cout << "----> CONTACTS : contact = " << contacts[ic] << std::endl;
            // }

            return contacts;

        }

    };

}
