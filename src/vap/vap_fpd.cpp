#include "scopi/vap/vap_fpd.hpp"
#include <cstddef>

namespace scopi
{
    vap_fpd::vap_fpd(std::size_t Nactive, std::size_t active_ptr, std::size_t, double dt, const VapParams<vap_fpd>&)
    : base_type(Nactive, active_ptr, dt)
    {}

    type::moment_t<2> cross_product_vap_fpd(const scopi_container<2>&, std::size_t)
    {
        return 0.;
    }

    type::moment_t<3> cross_product_vap_fpd(const scopi_container<3>& particles, std::size_t i)
    {
        double omega_1 = particles.omega()(i)[0];
        double omega_2 = particles.omega()(i)[1];
        double omega_3 = particles.omega()(i)[2];
        double j1 = particles.j()(i)[0];
        double j2 = particles.j()(i)[1];
        double j3 = particles.j()(i)[2];

        type::moment_t<3> res;
        res[0] = omega_2*omega_3*(j3-j2);
        res[1] = omega_1*omega_3*(j1-j3);
        res[2] = omega_1*omega_2*(j2-j1);

        return res;
    }

    template<>
    void vap_fpd::update_omega(scopi_container<2>& particles, std::size_t i, const xt::xtensor<double, 2>& wadapt)
    {
        particles.omega()(i + m_active_ptr) = wadapt(i, 2);
    }

    template<>
    void vap_fpd::update_omega(scopi_container<3>& particles, std::size_t i, const xt::xtensor<double, 2>& wadapt)
    {
        for (std::size_t d = 0; d < 3; ++d)
        {
            particles.omega()(i + m_active_ptr)(d) = wadapt(i, d);
        }
    }
}
