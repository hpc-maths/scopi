#include "scopi/problems/ViscousGlobule.hpp"
#include <cstddef>

namespace scopi
{
    ViscousGlobule::ViscousGlobule(std::size_t nparticles, double dt, ProblemParams<ViscousGlobule>& problem_params)
    : ProblemBase(nparticles, dt)
    , m_params(problem_params)
    {}

}
