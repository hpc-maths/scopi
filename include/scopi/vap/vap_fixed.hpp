#pragma once

#include "base.hpp"

namespace scopi
{
    class vap_case_1: public vap_base<vap_case_1>
    {
        public:
            using base_type = vap_base<vap_case_1>;

            template <std::size_t dim>
                void run_impl(const scopi_container<dim>& particles)
                {
                    std::cout << "run implementation" << std::endl;
                }
    };
}
