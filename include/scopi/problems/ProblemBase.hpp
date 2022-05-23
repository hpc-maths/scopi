#pragma once

#include <cstddef>
#include <plog/Log.h>
#include "plog/Initializers/RollingFileInitializer.h"
#include <xtensor/xtensor.hpp>

namespace scopi
{
    class ProblemBase
    {
    protected:
        ProblemBase(std::size_t nparts, double dt);

        std::size_t m_nparticles;
        double m_dt;

        std::vector<int> m_A_rows;
        std::vector<int> m_A_cols;
        std::vector<double> m_A_values;
        xt::xtensor<double, 1> m_distances;
    };

}
