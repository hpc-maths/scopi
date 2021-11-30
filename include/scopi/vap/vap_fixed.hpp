#pragma once

#include "base.hpp"

namespace scopi
{
    class vap_fixed: public vap_base<vap_fixed>
    {
        public:
            using base_type = vap_base<vap_fixed>;

            template <std::size_t dim>
                void run_impl(const scopi_container<dim>& particles)
                {
                    std::cout << "run implementation" << std::endl;
                }
    };
}
