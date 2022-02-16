#include "scopi/solvers/MatrixOptimSolverFriction.hpp"

namespace scopi
{
    MatrixOptimSolverFriction::MatrixOptimSolverFriction(std::size_t, double dt, double mu)
    : m_dt(dt)
    , m_mu(mu)
    {}
}
