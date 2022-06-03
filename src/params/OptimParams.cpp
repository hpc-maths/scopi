#include "scopi/params/OptimParams.hpp"

#ifdef SCOPI_USE_SCS
#include "scopi/solvers/OptimScs.hpp"
#endif

namespace scopi
{
    OptimParamsUzawaBase::OptimParamsUzawaBase()
    : m_tol(1e-9)
    , m_max_iter(40000)
    , m_rho(2000.)
    {}

#ifdef SCOPI_USE_SCS
    OptimParams<OptimScs>::OptimParams()
    : m_tol(1e-7)
    , m_tol_infeas(1e-10)
    {}

    OptimParams<OptimScs>::OptimParams(OptimParams<OptimScs>& params)
    : m_tol(params.m_tol)
    , m_tol_infeas(params.m_tol_infeas)
    {}
#endif
}
