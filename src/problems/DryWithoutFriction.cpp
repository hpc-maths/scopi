#include "scopi/problems/DryWithoutFriction.hpp"
#include <cstddef>

namespace scopi
{
    DryWithoutFriction::DryWithoutFriction(std::size_t nparticles, double dt, const ProblemParams<DryWithoutFriction>&)
    : ProblemBase(nparticles, dt)
    {}

    bool DryWithoutFriction::should_solve_optimization_problem()
    {
        return this->m_should_solve;
    }

}
