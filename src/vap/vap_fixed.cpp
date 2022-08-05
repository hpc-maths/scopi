#include "scopi/vap/vap_fixed.hpp"

namespace scopi
{
    vap_fixed::vap_fixed(std::size_t Nactive, std::size_t active_ptr, double dt, const VapParams<vap_fixed>&)
        : base_type(Nactive, active_ptr, dt)
    {}
}
