#include "scopi/vap/vap_fpd.hpp"

namespace scopi
{
    vap_fpd::vap_fpd(std::size_t Nactive, std::size_t active_ptr, double dt)
        : base_type(Nactive, active_ptr, dt)
          , _mass(1.)
        //   , _moment(0.1)
    {}

    double vap_fpd::t_ext()
    {
        return 0.;
    }
}
