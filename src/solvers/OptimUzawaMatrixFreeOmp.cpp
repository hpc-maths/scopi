#include "scopi/solvers/OptimUzawaMatrixFreeOmp.hpp"

namespace scopi
{
    OptimUzawaMatrixFreeOmp::OptimUzawaMatrixFreeOmp(std::size_t nparts, double dt)
    : base_type(nparts, dt)
    {}
}