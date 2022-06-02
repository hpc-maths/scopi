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
        void test()
        {
            std::cout << "ParamsSolver<OptimUzawa>::test" << std::endl;
        };

        ParamsSolverUzawaBase() {};
        template<template <class> class solver_t>
        ParamsSolverUzawaBase(ParamsSolver<solver_t> params) {std::cout << "ctor with template template parameters" << std::endl;};
    };
}

