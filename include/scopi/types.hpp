#pragma once

#include <type_traits>

#include <xtensor/xfixed.hpp>

namespace scopi
{
    namespace type
    {
        /**
         * @brief Type for position.
         *
         * @tparam dim Dimension (2 or 3).
         */
        template<std::size_t dim>
        using position_t = xt::xtensor_fixed<double, xt::xshape<dim>>;

        /**
         * @brief Type for velocity.
         *
         * @tparam dim Dimension (2 or 3).
         */
        template<std::size_t dim>
        using velocity_t = position_t<dim>;

        /**
         * @brief Type for force.
         *
         * @tparam dim Dimension (2 or 3).
         */
        template<std::size_t dim>
        using force_t = position_t<dim>;

        /**
         * @brief Type for momentum.
         *
         * @tparam dim Dimension (2 or 3).
         */
        template<std::size_t dim>
        using moment_t = typename std::conditional<dim == 2, double, xt::xtensor_fixed<double, xt::xshape<dim>>>::type;

        /**
         * @brief Type for rotation matrix.
         *
         * @tparam dim Dimension (2 or 3).
         */
        template<std::size_t dim>
        using matrix_rotation_t = xt::xtensor_fixed<double, xt::xshape<dim, dim>>;

        /**
         * @brief Type of rotation.
         *
         * Scalar in 2D, vector with three elements in 3D.
         *
         * @tparam dim Dimension (2 or 3).
         */
        template<std::size_t dim>
        using rotation_t = typename std::conditional<dim == 2, double, xt::xtensor_fixed<double, xt::xshape<dim>>>::type;

        /**
         * @brief Type for quaternion.
         */
        using quaternion_t = xt::xtensor_fixed<double, xt::xshape<4>>;
    }
}
