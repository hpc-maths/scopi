#include "scopi/problems/ViscousGlobule.hpp"
#include <cstddef>

namespace scopi
{
    ViscousGlobule::ViscousGlobule(std::size_t nparticles, double dt, const ProblemParams<ViscousGlobule>&)
    : ProblemBase(nparticles, dt)
    {}

}
