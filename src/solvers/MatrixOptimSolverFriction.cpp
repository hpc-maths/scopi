#include "scopi/solvers/MatrixOptimSolverFriction.hpp"

namespace scopi
{
    std::pair<type::position_t<2>, double> analytical_solution_sphere_plan(double alpha, double mu, double t, double r, double g)
    {
        double x_norm, omega;
        if(std::tan(alpha) <= 3*mu)
        {
            x_norm = g*std::sin(alpha)*t*t/3. ;
            omega = -2.*g*std::sin(alpha)*t/(3.*r);
        }
        else
        {
            x_norm = g*(std::sin(alpha) - mu*std::cos(alpha))*t*t/2.;
            omega = -2.*mu*g*std::cos(alpha)*t/r;
        }
        auto x = x_norm*xt::xtensor<double, 1>({std::cos(alpha), -std::sin(alpha)}); 
        return std::make_pair(x, omega);
    }

#ifdef SCOPI_USE_MOSEK
    using namespace monty;
#endif

    MatrixOptimSolverFriction::MatrixOptimSolverFriction(std::size_t nparticles, double dt)
    : m_nparticles(nparticles)
    , m_dt(dt)
    , m_mu(0.)
    {}

    void MatrixOptimSolverFriction::set_coeff_friction(double mu)
    {
        m_mu = mu;
    }

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
