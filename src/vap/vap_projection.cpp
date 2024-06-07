#include "scopi/vap/vap_projection.hpp"
#include <cstddef>

namespace scopi
{
    void vap_projection::set_u_w(const xt::xtensor<double, 2>& u, const xt::xtensor<double, 2>& w)
    {
        m_u = u;
        m_w = w;
    }
}
