#pragma once

#include <cstddef>
#include <plog/Log.h>
#include "plog/Initializers/RollingFileInitializer.h"
#include <xtensor/xtensor.hpp>

#include "../container.hpp"
#include "../objects/neighbor.hpp"

namespace scopi
{
    class DryBase
    {
    public:
        template <std::size_t dim>
        void update_gamma(const std::vector<neighbor<dim>>& contacts,
                          xt::xtensor<double, 1> lambda);

        template <std::size_t dim>
        void set_gamma(const std::vector<neighbor<dim>>& contacts);

        std::size_t get_nb_gamma_neg();
        std::size_t get_nb_gamma_min();

    };

    template <std::size_t dim>
    void DryBase::update_gamma(const std::vector<neighbor<dim>>&,
                               xt::xtensor<double, 1>)
    {}

    template <std::size_t dim>
    void DryBase::set_gamma(const std::vector<neighbor<dim>>&)
    {}


}

