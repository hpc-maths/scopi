#pragma once

#include <mkl_spblas.h>
#include <xtensor/xadapt.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xnoalias.hpp>
#include <plog/Log.h>
#include "plog/Initializers/RollingFileInitializer.h"

#include "../problems/DryWithoutFriction.hpp"
#include "../problems/ViscousWithoutFriction.hpp"

namespace scopi{
    /**
     * @brief Projection \f$ \Pi \f$ on the linear cone for gradients-like algorithm.
     *
     * The projection depends on the problem, template specializations of this class help manage the dependance on the problem.
     *
     * @tparam problem_t Problem to be solved.
     */
    template <class problem_t>
    class projection
    {};

    /**
     * @brief Specialization of \c projection for \c DryWithoutFriction.
     *
     * Projection on \f$ \mathbb{R}^+ \f$.
     */
    template <>
    class projection<DryWithoutFriction>
    {
    protected:
        /**
         * @brief Projection on \f$ \mathbb{R}^+ \f$.
         *
         * @param l [in] Vector to project.
         *
         * @return max (\c l, 0).
         */
        xt::xtensor<double, 1> projection_cone(const xt::xtensor<double, 1>& l);
    };

    /**
     * @brief Specialization of \c projection for \c ViscousWithoutFriction.
     *
     * Projection on \f$ \mathbb{R}^+ \f$.
     */
    template <std::size_t dim>
    class projection<ViscousWithoutFriction<dim>>
    {
    protected:
        /**
         * @brief Projection on \f$ \mathbb{R}^+ \f$.
         *
         * @param l [in] Vector to project.
         *
         * @return max (\c l, 0).
         */
        xt::xtensor<double, 1> projection_cone(const xt::xtensor<double, 1>& l);
    };

    template <std::size_t dim>
    xt::xtensor<double, 1> projection<ViscousWithoutFriction<dim>>::projection_cone(const xt::xtensor<double, 1>& l)
    {
        return xt::maximum( l, 0);
    }
}
