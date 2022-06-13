#include "scopi/problems/DryWithoutFriction.hpp"
#include <cstddef>

namespace scopi
{
    DryWithoutFriction::DryWithoutFriction(std::size_t nparticles, double dt, ProblemParams<DryWithoutFriction>&)
    : ProblemBase(nparticles, dt)
    {}


}
