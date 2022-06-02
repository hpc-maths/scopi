#include "scopi/params/ParamsSolver.hpp"

#ifdef SCOPI_USE_SCS
#include "scopi/solvers/OptimScs.hpp"
#endif

namespace scopi {
    ParamsSolverUzawaBase::ParamsSolverUzawaBase()
    : m_tol(1e-9)
    , m_max_iter(40000)
    , m_rho(2000.)
    {}

#ifdef SCOPI_USE_SCS
    ParamsSolver<OptimScs>::ParamsSolver()
    : m_tol(1e-7)
    , m_tol_infeas(1e-10)
    {}

    ParamsSolver<OptimScs>::ParamsSolver(ParamsSolver<OptimScs>& params)
    : m_tol(params.m_tol)
    , m_tol_infeas(params.m_tol_infeas)
    {}
#endif
}
