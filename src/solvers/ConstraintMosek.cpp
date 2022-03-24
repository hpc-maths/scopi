#ifdef SCOPI_USE_MOSEK
#include "scopi/solvers/ConstraintMosek.hpp"

namespace scopi
{
    using namespace monty;

    ConstraintMosek<MatrixOptimSolver>::ConstraintMosek(std::size_t nparticles)
    : m_nparticles(nparticles)
    {}

    std::shared_ptr<ndarray<double, 1>> ConstraintMosek<MatrixOptimSolver>::distances_to_vector(xt::xtensor<double, 1> distances) const
    {
        return std::make_shared<ndarray<double, 1>>(distances.data(), shape_t<1>(distances.shape(0)));
    }

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

    std::shared_ptr<ndarray<double, 1>> ConstraintMosek<MatrixOptimSolverFriction>::distances_to_vector(xt::xtensor<double, 1> distances) const
    {
        // TODO clean
        auto D_mosek = std::make_shared<ndarray<double, 1>>(distances.data(), shape_t<1>(4*distances.shape(0)));
        for (std::size_t i = 0; i < distances.size(); ++i)
        {
            (*D_mosek)[4*i] = distances[i];
            (*D_mosek)[4*i + 1] = 0.;
            (*D_mosek)[4*i + 2] = 0.;
            (*D_mosek)[4*i + 3] = 0.;
        }
        return D_mosek;
    }

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
