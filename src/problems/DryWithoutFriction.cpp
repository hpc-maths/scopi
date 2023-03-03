#include "scopi/problems/DryWithoutFriction.hpp"
#include <cstddef>

namespace scopi
{
    DryWithoutFriction::DryWithoutFriction(std::size_t nparticles, double dt)
    : ProblemBase(nparticles, dt)
    {}

    bool DryWithoutFriction::should_solve() const
    {
        return this->m_should_solve;
    }

}
