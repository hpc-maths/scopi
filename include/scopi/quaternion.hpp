#pragma once

#include <xtensor/xfixed.hpp>
#include <xtensor-blas/xlinalg.hpp>

#include "types.hpp"

using namespace xt::placeholders;

namespace scopi
{
    void normalize(type::quaternion_t& q);

    type::quaternion_t quaternion(double angle, const xt::xtensor_fixed<double, xt::xshape<3>>& axes);
    type::quaternion_t quaternion(double angle, const xt::xtensor_fixed<double, xt::xshape<2>>& axes);
    type::quaternion_t quaternion(double angle=0);
    auto conj(const type::quaternion_t& q);

    namespace detail
    {
        type::matrix_rotation_t<2> rotation_matrix_impl(const type::quaternion_t& q, std::integral_constant<std::size_t, 2>);
        type::matrix_rotation_t<3> rotation_matrix_impl(const type::quaternion_t& q, std::integral_constant<std::size_t, 3>);
    }

    template <std::size_t dim>
    auto rotation_matrix(const type::quaternion_t& q)
    {
        return detail::rotation_matrix_impl(q, std::integral_constant<std::size_t, dim>{});
    }

    type::quaternion_t mult_quaternion(const type::quaternion_t& q1, const type::quaternion_t& q2);
}
