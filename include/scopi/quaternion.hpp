#pragma once

#include <xtensor/xfixed.hpp>
#include <xtensor-blas/xlinalg.hpp>

#include "types.hpp"

using namespace xt::placeholders;

namespace scopi
{
    /**
     * @brief Normalize quaternion.
     *
     * After calling this function \f$q(0)^2 + q(1)^2 + q(2)^2 + q(3)^2 = 1\f$.
     *
     * @param q [in/out] Quaternion to be normalized.
     */
    void normalize(type::quaternion_t& q);

    /**
     * @brief Rotation of angle "angle" around the axis "axes".
     *
     * Rotation in 3D.
     *
     * @param angle [in] Angle of the rotation.
     * @param axes [in] The rotation is around this axis.
     *
     * @return Quaternion representing the rotation.
     */
    type::quaternion_t quaternion(double angle, const xt::xtensor_fixed<double, xt::xshape<3>>& axes);

    /**
     * @brief Rotation of angle "angle" around the axis "axes".
     *
     * Rotation in 2D.
     *
     * @param angle [in] Angle of the rotation.
     * @param axes [in] The rotation is around this axis.
     *
     * @return Quaternion representing the rotation.
     */
    type::quaternion_t quaternion(double angle, const xt::xtensor_fixed<double, xt::xshape<2>>& axes);

    /**
     * @brief Rotation of angle "angle" around the axis (0, 0, 1)..
     *
     * @param angle [in] Angle of the rotation.
     *
     * @return Quaternion representing the rotation.
     */
    type::quaternion_t quaternion(double angle=0);

    /**
     * @brief Compute the conjugate of \c q.
     *
     * @param q [in] Quaternion.
     *
     * @return Conjuguate of \c q.
     */
    auto conj(const type::quaternion_t& q);

    namespace detail
    {
        template <class q_t>
        type::matrix_rotation_t<2> rotation_matrix_impl(const q_t& q, std::integral_constant<std::size_t, 2>)
        {
            auto x = q(0);
            // auto y = q[1];    // en 2D on a y = z = 0 et x*x+w*w = 1
            // auto z = q[2];    // x = cos(theta/2) et w = sin(theta/2) rotation autour de (0,0,1)
            auto w = q(3);
            /*
            type::matrix_rotation_t<2> out;
            out[0][0] = 1-2*w*w;
            out[0][1] =  -2*x*w;
            out[1][0] =   2*x*w;
            out[1][1] = 1-2*w*w;
            return out;
            */
            return { { 1-2*w*w,  -2*x*w },
                     {   2*x*w, 1-2*w*w } };
        }

        template <class q_t>
        type::matrix_rotation_t<3> rotation_matrix_impl(const q_t& q, std::integral_constant<std::size_t, 3>)
        {
            auto x = q(0);
            auto y = q(1);  // x*x + y*y + z*z + w*w = 1
            auto z = q(2);
            auto w = q(3);
            // [[ q0**2+q1**2-q2**2-q3**2, 2*q1*q2-2*q0*q3, 2*q1*q3+2*q0*q2 ],
            //  [ 2*q1*q2+2*q0*q3, q0**2-q1**2+q2**2-q3**2, 2*q2*q3-2*q0*q1 ],
            //  [ 2*q1*q3-2*q0*q2, 2*q2*q3+2*q0*q1, q0**2-q1**2-q2**2+q3**2 ]]
            /*
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
            */
            return { { 1-2*z*z-2*w*w,   2*y*z-2*x*w,   2*y*w+2*x*z },
                     {   2*y*z+2*x*w, 1-2*y*y-2*w*w,   2*z*w-2*x*y },
                     {   2*y*w-2*x*z,   2*z*w+2*x*y, 1-2*y*y-2*z*z } };
        }
    }

    template <std::size_t dim, class q_t>
    auto rotation_matrix(const q_t& q)
    {
        return detail::rotation_matrix_impl(q, std::integral_constant<std::size_t, dim>{});
    }

    /**
     * @brief Compute the multiplication between two quaternions.
     *
     * If \f$ q_1 = (q_1^0, \vec{q_0}) \f$ and \f$ q_2 = (q_2^0, \vec{q_2}) \f$, with \f$ q_1^0, q_2^0 \in \mathbb{R} \f$ and \f$ \vec{q_1}, \vec{q_2} \in \mathcal{R}^3 \f$, then
     * \f$ q_1 \circ q_2 = (q_1^0 q_2^0 - \vec{q_1} \cdot \vec{q_2}, q_1^0 \vec{q_2} + q_2^0 \vec{q_1} + \vec{q_1} \land \vec{q_2}) \f$.
     *
     * @param q1 [in] Quaternion.
     * @param q2 [in] Quaternion.
     *
     * @return \f$ p = q_1 \circ q_2 \f$.
     */
    type::quaternion_t mult_quaternion(const type::quaternion_t& q1, const type::quaternion_t& q2);
}
