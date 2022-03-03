#pragma once

#include <type_traits>

#include <xtensor/xfixed.hpp>

namespace scopi
{
    namespace type
    {
        template<std::size_t dim>
        using position_t = xt::xtensor_fixed<double, xt::xshape<dim>>;

        template<std::size_t dim>
        using velocity_t = position_t<dim>;

        template<std::size_t dim>
        using force_t = position_t<dim>;

        template<std::size_t dim>
        using moment_t = position_t<dim>;

        template<std::size_t dim>
        using matrix_rotation_t = xt::xtensor_fixed<double, xt::xshape<dim, dim>>;

        template<std::size_t dim>
        using rotation_t = typename std::conditional<dim == 2, double, xt::xtensor_fixed<double, xt::xshape<dim>>>::type;

        using quaternion_t = xt::xtensor_fixed<double, xt::xshape<4>>;
    }
}
