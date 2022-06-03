#include "scopi/params/OptimParams.hpp"

#include "scopi/problems/DryWithFriction.hpp"

namespace scopi
{
    ProblemParams<DryWithFriction>::ProblemParams()
    : m_mu(0.)
    {}

    ProblemParams<DryWithFriction>::ProblemParams(ProblemParams<DryWithFriction>& params)
    : m_mu(params.m_mu)
    {}
}
