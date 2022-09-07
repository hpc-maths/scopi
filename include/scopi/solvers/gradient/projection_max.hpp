#pragma once

#ifdef SCOPI_USE_MKL
#include <mkl_spblas.h>

#include <xtensor/xadapt.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xnoalias.hpp>
#include <plog/Log.h>
#include "plog/Initializers/RollingFileInitializer.h"

namespace scopi{
    /**
     * @brief Projection on the linear cone for gradients-like algorithm.
     */
    class projection_max
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

    auto projection_max::projection_cone(const xt::xtensor<double, 1>& l)
    {
        return xt::maximum( l, 0);
    }
}
#endif
