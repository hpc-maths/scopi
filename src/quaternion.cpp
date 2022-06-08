#include "scopi/quaternion.hpp"
#include <xtensor/xadapt.hpp>

namespace scopi
{
    void normalize(type::quaternion_t& q)
    {
        auto qa = xt::adapt(q);
        qa /= xt::linalg::norm(qa, 2);
    }

    type::quaternion_t quaternion(double angle, const xt::xtensor_fixed<double, xt::xshape<3>>& axes)
    {
        type::quaternion_t out = { std::cos(angle/2)
                               , std::sin(angle/2)*axes[0]
                               , std::sin(angle/2)*axes[1]
                               , std::sin(angle/2)*axes[2] };
        // type::quaternion_t out = { std::sin(angle/2)*axes[0]
        //                        , std::sin(angle/2)*axes[1]
        //                        , std::sin(angle/2)*axes[2]
        //                        , std::cos(angle/2)};
        normalize(out);
        return out;
    }

    type::quaternion_t quaternion(double angle, const xt::xtensor_fixed<double, xt::xshape<2>>& axes)
    {
        xt::xtensor_fixed<double, xt::xshape<3>> new_axes{axes[0], axes[1], 1};
        return quaternion(angle, new_axes);
    }

    type::quaternion_t quaternion(double angle)
    {
        xt::xtensor_fixed<double, xt::xshape<3>> axes{0, 0, 1};
        return quaternion(angle, axes);
    }

    auto conj(const type::quaternion_t& q)
    {
        type::quaternion_t out(q);
        auto out_a = xt::adapt(out);
        out_a *= -1;
        return out;
    }

    namespace detail
    {
        type::matrix_rotation_t<2> rotation_matrix_impl(const type::quaternion_t& q, std::integral_constant<std::size_t, 2>)
        {
            auto x = q[0];
            // auto y = q[1];    // en 2D on a y = z = 0 et x*x+w*w = 1
            // auto z = q[2];    // x = cos(theta/2) et w = sin(theta/2) rotation autour de (0,0,1)
            auto w = q[3];
            type::matrix_rotation_t<2> out;
            out[0][0] = 1-2*w*w;
            out[0][1] =  -2*x*w;
            out[1][0] =   2*x*w;
            out[1][1] = 1-2*w*w;
            return out;
            // return { { 1-2*w*w,  -2*x*w },
            //          {   2*x*w, 1-2*w*w } };
        }

        type::matrix_rotation_t<3> rotation_matrix_impl(const type::quaternion_t& q, std::integral_constant<std::size_t, 3>)
        {
            auto x = q[0];
            auto y = q[1];  // x*x + y*y + z*z + w*w = 1
            auto z = q[2];
            auto w = q[3];
            // [[ q0**2+q1**2-q2**2-q3**2, 2*q1*q2-2*q0*q3, 2*q1*q3+2*q0*q2 ],
            //  [ 2*q1*q2+2*q0*q3, q0**2-q1**2+q2**2-q3**2, 2*q2*q3-2*q0*q1 ],
            //  [ 2*q1*q3-2*q0*q2, 2*q2*q3+2*q0*q1, q0**2-q1**2-q2**2+q3**2 ]]
            type::matrix_rotation_t<3> out;
            out[0][0] = 1-2*z*z-2*w*w;
            out[0][1] =   2*y*z-2*x*w;
            out[0][2] =   2*y*w+2*x*z;
            out[1][0] =   2*y*z+2*x*w;
            out[1][1] = 1-2*y*y-2*w*w;
            out[1][2] =   2*z*w-2*x*y;
            out[2][0] =   2*y*w-2*x*z;
            out[2][1] =   2*z*w+2*x*y;
            out[2][2] =   1-2*y*y-2*z*z;
            return out;
            // return { { 1-2*z*z-2*w*w,   2*y*z-2*x*w,   2*y*w+2*x*z },
            //          {   2*y*z+2*x*w, 1-2*y*y-2*w*w,   2*z*w-2*x*y },
            //          {   2*y*w-2*x*z,   2*z*w+2*x*y, 1-2*y*y-2*z*z } };
        }
    }

    type::quaternion_t mult_quaternion(const type::quaternion_t& q1, const type::quaternion_t& q2)
    {
        /*
        type::quaternion_t result;
        auto result_a = xt::adapt(result);

        auto q1a = xt::adapt(q1);
        auto q2a = xt::adapt(q2);

        auto q1_axe = xt::view(q1a, xt::range(1, _));
        auto q2_axe = xt::view(q2a, xt::range(1, _));
        result_a(0) = q1a(0)*q2a(0) - xt::linalg::dot(q1_axe, q2_axe)(0); // xt::linalg::dot(q1_axe, q2_axe) does not compile
        xt::view(result_a, xt::range(1, _)) = q1a(0)*q2_axe + q2a(0)*q1_axe + xt::linalg::cross(q1_axe, q2_axe);
        return result;
        */

        type::quaternion_t result;
        result[0] = q1[0]*q2[0];
        for (std::size_t d = 0; d < 3; ++d)
        {
            result[0] -= q1[d+1] * q2[d+1];
            result[d+1] = q1[0]*q2[d+1] + q2[0]*q1[d+1] ;
        }
        result[1] += (q1[2]*q2[3] - q1[3]*q2[2]);
        result[2] += (q1[3]*q2[1] - q1[1]*q2[3]);
        result[3] += (q1[1]*q2[2] - q1[2]*q2[1]);

        return result;
    }

}
