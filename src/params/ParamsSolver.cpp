#include "scopi/params/ParamsSolver.hpp"

#ifdef SCOPI_USE_SCS
#include "scopi/solvers/OptimScs.hpp"
#endif

namespace scopi {
#ifdef SCOPI_USE_SCS
    ParamsSolver<OptimScs>::ParamsSolver()
    : m_tol(1e-7)
    , m_tol_infeas(1e-10)
    {};

    ParamsSolver<OptimScs>::ParamsSolver(ParamsSolver<OptimScs>& params)
    : m_tol(params.m_tol)
    , m_tol_infeas(params.m_tol_infeas)
    {};

#endif
}
