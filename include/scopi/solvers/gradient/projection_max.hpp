#pragma once

#ifdef SCOPI_USE_MKL
#include <mkl_spblas.h>

#include <xtensor/xadapt.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xnoalias.hpp>
#include <plog/Log.h>
#include "plog/Initializers/RollingFileInitializer.h"

namespace scopi{
    class projection_max
    {
    protected:
        auto projection_cone(xt::xtensor<double, 1>& l);
    };

    auto projection_max::projection_cone(xt::xtensor<double, 1>& l)
    {
        return xt::maximum( l, 0);
    }
}
#endif
