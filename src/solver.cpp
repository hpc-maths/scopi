#include "scopi/solver.hpp"

namespace scopi
{
    template<>
    void update_velocity_omega(scopi_container<2>& particles, std::size_t i, const xt::xtensor<double, 2>& wadapt)
    {
        particles.omega()(i + particles.nb_inactive()) = wadapt(i, 2);
    }

    template<>
    void update_velocity_omega(scopi_container<3>& particles, std::size_t i, const xt::xtensor<double, 2>& wadapt)
    {
        for (std::size_t d = 0; d < 3; ++d)
        {
            particles.omega()(i + particles.nb_inactive())(d) = wadapt(i, d);
        }
    }
}
