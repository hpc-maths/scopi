#include "scopi/problems/ProblemBase.hpp"
#include <cstddef>

namespace scopi
{
    ProblemBase::ProblemBase(std::size_t nparts, double dt)
    : m_nparticles(nparts)
    , m_dt(dt)
    {}
}

