#include "scopi/problems/DryWithFriction.hpp"
#include <utility>

namespace scopi
{
    ProblemParams<DryWithFriction>::ProblemParams()
        : mu(0.)
    {
    }

    ProblemParams<DryWithFriction>::ProblemParams(const ProblemParams<DryWithFriction>& params)
        : mu(params.mu)
    {
    }

    DryWithFriction::DryWithFriction(std::size_t nparticles, double dt)
        : DryWithFrictionBase(nparticles, dt)
    {
    }

    bool DryWithFriction::should_solve() const
    {
        return this->m_should_solve;
    }

}
