#include "scopi/params.hpp"

namespace scopi
{
    ScopiParams::ScopiParams(const ScopiParams& params)
    : frequence_output(params.frequence_output)
    {}

    ScopiParams::ScopiParams()
    : frequence_output(std::size_t(-1))
    {}
}
