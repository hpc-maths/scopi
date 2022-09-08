#include "scopi/params.hpp"

namespace scopi
{
    ScopiParams::ScopiParams(const ScopiParams& params)
    : output_frequency(params.output_frequency)
    , filename(params.filename)
    , write_velocity(params.write_velocity)
    {}

    ScopiParams::ScopiParams()
    : output_frequency(std::size_t(-1))
    , filename("./Results/scopi_objects_")
    , write_velocity(false)
    {}
}
