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

    void MatrixOptimSolver::matrix_free_gemv_inv_P_moment(const scopi_container<2>& particles,
                                                          xt::xtensor<double, 1>& U,
                                                          std::size_t active_offset,
                                                          std::size_t row)
    {
        U(3*m_nparticles + 3*row + 2) /= (-1.*particles.j()(active_offset + row));
    }

    void MatrixOptimSolver::matrix_free_gemv_inv_P_moment(const scopi_container<3>& particles,
                                                          xt::xtensor<double, 1>& U,
                                                          std::size_t active_offset,
                                                          std::size_t row)
    {
        for (std::size_t d = 0; d < 3; ++d)
        {
            U(3*m_nparticles + 3*row + d) /= (-1.*particles.j()(active_offset + row)(d));
        }
    }
}
