#pragma once

#include <xtensor/xfixed.hpp>
#include <xtensor-blas/xlinalg.hpp>

#include "types.hpp"

using namespace xt::placeholders;

namespace scopi
{
    void normalize(type::quaternion& q);

    type::quaternion quaternion(double angle, const xt::xtensor_fixed<double, xt::xshape<3>>& axes);
    type::quaternion quaternion(double angle, const xt::xtensor_fixed<double, xt::xshape<2>>& axes);
    type::quaternion quaternion(double angle=0);
    auto conj(const type::quaternion& q);

    namespace detail
    {
        type::rotation<2> rotation_matrix_impl(const type::quaternion& q, std::integral_constant<std::size_t, 2>);
        type::rotation<3> rotation_matrix_impl(const type::quaternion& q, std::integral_constant<std::size_t, 3>);
    }

    template <std::size_t dim>
    auto rotation_matrix(const type::quaternion& q)
    {
        return detail::rotation_matrix_impl(q, std::integral_constant<std::size_t, dim>{});
    }

    type::quaternion mult_quaternion(const type::quaternion& q1, const type::quaternion& q2);
}
