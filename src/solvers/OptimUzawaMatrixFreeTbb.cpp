#include "scopi/solvers/OptimUzawaMatrixFreeTbb.hpp"

namespace scopi
{
    OptimUzawaMatrixFreeTbb::OptimUzawaMatrixFreeTbb(std::size_t nparts, double dt)
    : base_type(nparts, dt)
    {}
}