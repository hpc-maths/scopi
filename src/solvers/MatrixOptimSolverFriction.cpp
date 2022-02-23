#include "scopi/solvers/MatrixOptimSolverFriction.hpp"

namespace scopi
{
#ifdef SCOPI_USE_MOSEK
    using namespace monty;
#endif

    MatrixOptimSolverFriction::MatrixOptimSolverFriction(std::size_t nparticles, double dt, double mu)
    : m_nparticles(nparticles)
    , m_dt(dt)
    , m_mu(mu)
    {}

#ifdef SCOPI_USE_MOSEK
    std::shared_ptr<ndarray<double, 1>> MatrixOptimSolverFriction::distances_to_mosek_vector(xt::xtensor<double, 1> distances) const
    {
        // TODO clean
        auto D_mosek = std::make_shared<ndarray<double, 1>>(distances.data(), shape_t<1>({4*distances.shape(0)}));
        for (std::size_t i = 0; i < distances.size(); ++i)
        {
            (*D_mosek)[4*i] = distances[i];
            (*D_mosek)[4*i + 1] = 0.;
            (*D_mosek)[4*i + 2] = 0.;
            (*D_mosek)[4*i + 3] = 0.;
        }
        return D_mosek;
    }

    std::size_t MatrixOptimSolverFriction::matrix_first_col_index_mosek() const
    {
        return 0;
    }

    std::size_t MatrixOptimSolverFriction::number_col_matrix_mosek() const
    {
        return 6*m_nparticles;
    }
#endif
}
