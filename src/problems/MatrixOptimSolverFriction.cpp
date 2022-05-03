#include "scopi/problems/MatrixOptimSolverFriction.hpp"

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


    MatrixOptimSolverFriction::MatrixOptimSolverFriction(std::size_t nparticles, double dt)
    : m_nparticles(nparticles)
    , m_dt(dt)
    , m_mu(0.)
    {}

    void MatrixOptimSolverFriction::set_coeff_friction(double mu)
    {
        m_mu = mu;
    }

    std::size_t MatrixOptimSolverFriction::get_nb_gamma_neg()
    {
        return 0;
    }

    std::size_t MatrixOptimSolverFriction::get_nb_gamma_min()
    {
        return 0;
    }
}
