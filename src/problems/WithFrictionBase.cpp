#include "scopi/problems/WithFrictionBase.hpp"
#include <cstddef>

namespace scopi
{
    WithFrictionBase::WithFrictionBase()
    : m_mu(0.)
    {}

    void WithFrictionBase::set_coeff_friction(double mu)
    {
        m_mu = mu;
    }
}
