#pragma once

#ifdef SCOPI_USE_MKL
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
     * @brief Projection on the linear cone for gradients-like algorithm.
     */
    template <class problem_t>
    class projection
    {};

    template <>
    class projection<DryWithoutFriction>
    {
    protected:
        /**
         * @brief Projection.
         *
         * @param l [in] Vector to project.
         *
         * @return max (\c l, 0).
         */
        auto projection_cone(const xt::xtensor<double, 1>& l);
    };

    template <std::size_t dim>
    class projection<ViscousWithoutFriction<dim>>
    {
    protected:
        /**
         * @brief Projection.
         *
         * @param l [in] Vector to project.
         *
         * @return max (\c l, 0).
         */
        auto projection_cone(const xt::xtensor<double, 1>& l);
    };

    template <std::size_t dim>
    auto projection<ViscousWithoutFriction<dim>>::projection_cone(const xt::xtensor<double, 1>& l)
    {
        return xt::maximum( l, 0);
    }
}
#endif
