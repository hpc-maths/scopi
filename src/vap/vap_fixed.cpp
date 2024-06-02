#include "scopi/vap/vap_fixed.hpp"
#include <cstddef>

namespace scopi
{
    vap_fixed::vap_fixed(std::size_t Nactive, std::size_t active_ptr, std::size_t, double dt)
        : base_type(Nactive, active_ptr, dt)
    {
    }
}
