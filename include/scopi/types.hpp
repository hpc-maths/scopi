#pragma once

#include <xtensor/xfixed.hpp>

namespace scopi
{
    namespace type
    {
        template<std::size_t dim>
        using position = xt::xtensor_fixed<double, xt::xshape<dim>>;

        template<std::size_t dim>
        using rotation = xt::xtensor_fixed<double, xt::xshape<dim, dim>>;
    }
}