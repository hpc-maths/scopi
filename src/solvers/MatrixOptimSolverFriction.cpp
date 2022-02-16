#include "scopi/solvers/MatrixOptimSolverFriction.hpp"

namespace scopi
{
    MatrixOptimSolverFriction::MatrixOptimSolverFriction(std::size_t, double dt)
    : m_dt(dt)
    , m_mu(1./2.)
    {}
}
