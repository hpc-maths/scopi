#pragma once

#include <cstddef>
#include <plog/Log.h>
#include "plog/Initializers/RollingFileInitializer.h"

#include "ParamsSolver.hpp"

namespace scopi
{
    template<class solver_t>
    struct Params
    {
        ParamsSolver<solver_t> solver;
        // ParamsModel<problem_t> problem;
    };
}
