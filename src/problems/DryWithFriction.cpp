#include "scopi/problems/DryWithFriction.hpp"
#include <utility>

namespace scopi
{
    ProblemParams<DryWithFriction>::ProblemParams()
    : mu(0.)
    {}

    ProblemParams<DryWithFriction>::ProblemParams(const ProblemParams<DryWithFriction>& params)
    : mu(params.mu)
    {}

    DryWithFriction::DryWithFriction(std::size_t nparticles, double dt, const ProblemParams<DryWithFriction>& params)
    : DryWithFrictionBase(nparticles, dt, params.mu)  
    , m_params(params)
    {}

    bool DryWithFriction::should_solve_optimization_problem()
    {
        return this->m_should_solve;
    }

}
