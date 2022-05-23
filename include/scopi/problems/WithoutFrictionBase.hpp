#pragma once

#include <plog/Log.h>
#include "plog/Initializers/RollingFileInitializer.h"

#include "../container.hpp"
#include "../objects/neighbor.hpp"

namespace scopi
{
    class WithoutFrictionBase
    {
    protected:
        void matrix_free_gemv_inv_P_moment(const scopi_container<2>& particles,
                                           xt::xtensor<double, 1>& U,
                                           std::size_t active_offset,
                                           std::size_t row);
        void matrix_free_gemv_inv_P_moment(const scopi_container<3>& particles,
                                           xt::xtensor<double, 1>& U,
                                           std::size_t active_offset,
                                           std::size_t row);
    };

}

