#pragma once

#include <type_traits>

#include <xtensor/xfixed.hpp>

namespace scopi
{
    namespace type
    {
        template<std::size_t dim>
        using position_t = std::array<double, dim>;

        template<std::size_t dim>
        using velocity_t = position_t<dim>;

        template<std::size_t dim>
        using force_t = position_t<dim>;

        template<std::size_t dim>
        using moment_t = typename std::conditional<dim == 2, double, position_t<dim>>::type;

        template<std::size_t dim>
        using matrix_rotation_t = xt::xtensor_fixed<double, xt::xshape<dim, dim>>;
        // using matrix_rotation_t = std::array<std::array<double, dim>, dim>;

        template<std::size_t dim>
        using rotation_t = typename std::conditional<dim == 2, double, position_t<dim>>::type;

        using quaternion_t = std::array<double, 4>;
    }
}
