#pragma once

#include <cstddef>
#include <iostream>
#include <plog/Log.h>
#include "plog/Initializers/RollingFileInitializer.h"

namespace scopi
{
    template<class solver_t>
    class OptimParams
    {
    private:
        OptimParams();
    };

    class OptimParamsUzawaBase
    {
    public:
        OptimParamsUzawaBase();
        template<class solver_t>
        OptimParamsUzawaBase(OptimParams<solver_t>& params);

        double m_tol;
        std::size_t m_max_iter;
        double m_rho;
    };

    template<class solver_t>
    OptimParamsUzawaBase::OptimParamsUzawaBase(OptimParams<solver_t>& params)
    : m_tol(params.m_tol)
    , m_max_iter(params.m_max_iter)
    , m_rho(params.m_rho)
    {}

}

