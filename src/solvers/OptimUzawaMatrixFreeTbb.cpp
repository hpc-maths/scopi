#include "scopi/solvers/OptimUzawaMatrixFreeTbb.hpp"

#ifdef SCOPI_USE_TBB

namespace scopi
{
    OptimUzawaMatrixFreeTbb::OptimUzawaMatrixFreeTbb(std::size_t nparts, double dt, double)
    : base_type(nparts, dt)
    {}
}

#endif
