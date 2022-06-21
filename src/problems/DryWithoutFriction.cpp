#include "scopi/problems/DryWithoutFriction.hpp"
#include <cstddef>

namespace scopi
{
    DryWithoutFriction::DryWithoutFriction(std::size_t nparticles, double dt, const ProblemParams<DryWithoutFriction>& problem_params)
    : ProblemBase(nparticles, dt)
    , m_params(problem_params)
    {}


}
