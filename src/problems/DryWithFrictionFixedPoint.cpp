#include "scopi/problems/DryWithFrictionFixedPoint.hpp"
#include <utility>

namespace scopi
{
    ProblemParams<DryWithFrictionFixedPoint>::ProblemParams()
    : mu(0.)
    , tol_fixed_point(1e-2)
    , max_iter_fixed_point(20)
    {}

    ProblemParams<DryWithFrictionFixedPoint>::ProblemParams(const ProblemParams<DryWithFrictionFixedPoint>& params)
    : mu(params.mu)
    , tol_fixed_point(params.tol_fixed_point)
    , max_iter_fixed_point(params.max_iter_fixed_point)
    {}

    DryWithFrictionFixedPoint::DryWithFrictionFixedPoint(std::size_t nparticles, double dt)
    : DryWithFrictionBase(nparticles, dt)
    {}

    bool DryWithFrictionFixedPoint::should_solve() const
    {
        bool res = (xt::linalg::norm(m_s_old - m_s)/(xt::linalg::norm(m_s)+1.) > m_params.tol_fixed_point && m_nb_iter < m_params.max_iter_fixed_point);
        if (!res)
        {
            PLOG_WARNING << "Number iterations fixed point " << m_nb_iter;
        }
        return res;
    }

}
