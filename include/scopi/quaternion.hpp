#pragma once

#include <xtensor/xfixed.hpp>
#include <xtensor-blas/xlinalg.hpp>

#include "types.hpp"

namespace scopi
{
    void normalize(type::quaternion& q)
    {
        q /= xt::linalg::norm(q, 2);
    }

    type::quaternion quaternion(double angle, const xt::xtensor_fixed<double, xt::xshape<3>>& axes)
    {
        type::quaternion out = { std::cos(angle/2)
                               , std::sin(angle/2)*axes[0]
                               , std::sin(angle/2)*axes[1]
                               , std::sin(angle/2)*axes[2] };
        // type::quaternion out = { std::sin(angle/2)*axes[0]
        //                        , std::sin(angle/2)*axes[1]
        //                        , std::sin(angle/2)*axes[2]
        //                        , std::cos(angle/2)};
        normalize(out);
        return out;
    }

    type::quaternion quaternion(double angle, const xt::xtensor_fixed<double, xt::xshape<2>>& axes)
    {
        xt::xtensor_fixed<double, xt::xshape<3>> new_axes{axes[0], axes[1], 1};
        return quaternion(angle, new_axes);
    }

    type::quaternion quaternion(double angle=0)
    {
        xt::xtensor_fixed<double, xt::xshape<3>> axes{0, 0, 1};
        return quaternion(angle, axes);
    }

    auto conj(const type::quaternion& q)
    {
        type::quaternion out(q);
        out *= -1;
        return out;
    }

    namespace detail
    {
        type::rotation<2> rotation_matrix_impl(const type::quaternion& q, std::integral_constant<std::size_t, 2>)
        {
            auto x = q[0];
            auto y = q[1];    // en 2D on a y = z = 0 et x*x+w*w = 1
            auto z = q[2];    // x = cos(theta/2) et w = sin(theta/2) rotation autour de (0,0,1)
            auto w = q[3];
            return { { 1-2*w*w,  -2*x*w },
                     {   2*x*w, 1-2*w*w } };
            // Loic : return { { (1-2*y*y-2*z*z),   (2*x*y-2*z*w) },
            //               {     (2*x*y+2*z*w), (1-2*x*x-2*z*z) } };
        }

        type::rotation<3> rotation_matrix_impl(const type::quaternion& q, std::integral_constant<std::size_t, 3>)
        {
            auto x = q[0];
            auto y = q[1];  // x*x + y*y + z*z + w*w = 1
            auto z = q[2];
            auto w = q[3];
            // [[ q0**2+q1**2-q2**2-q3**2, 2*q1*q2-2*q0*q3, 2*q1*q3+2*q0*q2 ],
            //  [ 2*q1*q2+2*q0*q3, q0**2-q1**2+q2**2-q3**2, 2*q2*q3-2*q0*q1 ],
            //  [ 2*q1*q3-2*q0*q2, 2*q2*q3+2*q0*q1, q0**2-q1**2-q2**2+q3**2 ]]
            return { { 1-2*z*z-2*w*w,   2*y*z-2*x*w,   2*y*w+2*x*z },
                     {   2*y*z+2*x*w, 1-2*y*y-2*w*w,   2*z*w-2*x*y },
                     {   2*y*w-2*x*z,   2*z*w+2*x*y, 1-2*y*y-2*z*z } };
            // Loic : return { { (1-2*y*y-2*z*z),   (2*x*y-2*z*w),   (2*x*z+2*y*w) },
            //                 {   (2*x*y+2*z*w), (1-2*x*x-2*z*z),   (2*y*z-2*x*w) },
            //                 {   (2*x*z-2*y*w),   (2*y*z+2*x*w), (1-2*x*x-2*y*y) } };
        }
    }

    template <std::size_t dim>
    auto rotation_matrix(const type::quaternion& q)
    {
        return detail::rotation_matrix_impl(q, std::integral_constant<std::size_t, dim>{});
    }
}
