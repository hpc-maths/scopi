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
}
