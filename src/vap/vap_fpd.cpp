#include "scopi/vap/vap_fpd.hpp"

namespace scopi
{
    vap_fpd::vap_fpd(std::size_t Nactive, std::size_t active_ptr, double dt)
        : base_type(Nactive, active_ptr, dt)
        //   , _moment(0.1)
    {}
}
