#include "scopi/solvers/OptimUzawaBase.hpp"

namespace scopi
{
    OptimParamsUzawaBase::OptimParamsUzawaBase(const OptimParamsUzawaBase& params)
    : tol(params.tol)
    , max_iter(params.max_iter)
    , rho(params.rho)
    {}

    OptimParamsUzawaBase::OptimParamsUzawaBase()
    : tol(1e-9)
    , max_iter(40000)
    , rho(2000.)
    {}
}

