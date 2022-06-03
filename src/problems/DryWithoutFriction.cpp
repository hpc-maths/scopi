#include "scopi/problems/DryWithoutFriction.hpp"
#include <cstddef>

namespace scopi
{
    DryWithoutFriction::DryWithoutFriction(std::size_t nparticles, double dt, ProblemParams<DryWithoutFriction>& problem_params)
    : ProblemBase(nparticles, dt)
    , DryBase()
    , m_params(problem_params)
    {}


}
