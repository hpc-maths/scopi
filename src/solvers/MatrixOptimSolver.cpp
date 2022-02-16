#include "scopi/solvers/MatrixOptimSolver.hpp"

namespace scopi
{
    using namespace monty;

    MatrixOptimSolver::MatrixOptimSolver(std::size_t nparticles, double dt, double)
    : m_nparticles(nparticles)
    , m_dt(dt)
    {}

#ifdef SCOPI_USE_MOSEK
    std::shared_ptr<ndarray<double, 1>> MatrixOptimSolver::distances_to_mosek_vector(xt::xtensor<double, 1> distances) const
    {
        return std::make_shared<ndarray<double, 1>>(distances.data(), shape_t<1>({distances.shape(0)}));
    }

    std::size_t MatrixOptimSolver::matrix_first_col_index_mosek() const
    {
        return 1;
    }

    std::size_t MatrixOptimSolver::number_col_matrix_mosek() const
    {
        return 1 + 6*m_nparticles + 6*m_nparticles;
    }
#endif

}
