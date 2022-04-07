#ifdef SCOPI_USE_MOSEK
#include "scopi/solvers/ConstraintMosek.hpp"

namespace scopi
{
    using namespace monty;

    ConstraintMosek<MatrixOptimSolver>::ConstraintMosek(std::size_t nparticles)
    : m_nparticles(nparticles)
    {}

    std::size_t ConstraintMosek<MatrixOptimSolver>::index_first_col_matrix() const
    {
        return 1;
    }

    std::size_t ConstraintMosek<MatrixOptimSolver>::number_col_matrix() const
    {
        return 1 + 6*m_nparticles + 6*m_nparticles;
    }





    ConstraintMosek<MatrixOptimSolverFriction>::ConstraintMosek(std::size_t nparticles)
    : m_nparticles(nparticles)
    {}

    std::size_t ConstraintMosek<MatrixOptimSolverFriction>::index_first_col_matrix() const
    {
        return 0;
    }

    std::size_t ConstraintMosek<MatrixOptimSolverFriction>::number_col_matrix() const
    {
        return 6*m_nparticles;
    }
}
#endif
