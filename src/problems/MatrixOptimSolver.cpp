#include "scopi/problems/MatrixOptimSolver.hpp"

namespace scopi
{
    MatrixOptimSolver::MatrixOptimSolver(std::size_t nparticles, double dt)
    : m_nparticles(nparticles)
    , m_dt(dt)
    {}

}
