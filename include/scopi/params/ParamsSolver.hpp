#pragma once

#include <cstddef>
#include <iostream>
#include <plog/Log.h>
#include "plog/Initializers/RollingFileInitializer.h"

namespace scopi
{
    template<template <class> class solver_t>
    class ParamsSolver
    {
    };

    class ParamsSolverUzawaBase
    {
    public:
        ParamsSolverUzawaBase();
        template<template <class> class solver_t>
        ParamsSolverUzawaBase(ParamsSolver<solver_t>& params);

        double m_tol;
        std::size_t m_max_iter;
        double m_rho;
    };

    template<template <class> class solver_t>
    ParamsSolverUzawaBase::ParamsSolverUzawaBase(ParamsSolver<solver_t>& params)
    : m_tol(params.m_tol)
    , m_max_iter(params.m_max_iter)
    , m_rho(params.m_rho)
    {}

}

