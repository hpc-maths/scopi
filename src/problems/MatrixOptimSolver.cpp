#include "scopi/problems/MatrixOptimSolver.hpp"
#include <cstddef>

namespace scopi
{
    MatrixOptimSolver::MatrixOptimSolver(std::size_t nparticles, double dt)
    : m_nparticles(nparticles)
    , m_dt(dt)
    {}

    std::size_t MatrixOptimSolver::get_nb_gamma_neg()
    {
        return 0;
    }

    std::size_t MatrixOptimSolver::get_nb_gamma_min()
    {
        return 0;
    }
}
