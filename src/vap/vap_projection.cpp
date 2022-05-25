#include "scopi/vap/vap_projection.hpp"

namespace scopi
{
    vap_projection::vap_projection(std::size_t Nactive, std::size_t active_ptr, double dt)
        : base_type(Nactive, active_ptr, dt)
    {}


    void vap_projection::set_u_w(xt::xtensor<double, 2> u, xt::xtensor<double, 2> w)
    {
        m_u = u;
        m_w = w;
    }
}
