#include "scopi/solvers/OptimUzawaMatrixFreeTbb.hpp"

#ifdef SCOPI_USE_TBB

namespace scopi
{
    void OptimUzawaMatrixFreeTbb::gemv_inv_P_moment_impl(const scopi_container<2>& particles,
                                                  std::size_t active_offset,
                                                  std::size_t i)
    {
        this->m_U(3*this->m_nparts + 3*i + 2) /= (-1.*particles.j()(active_offset + i));
    }

    void OptimUzawaMatrixFreeTbb::gemv_inv_P_moment_impl(const scopi_container<3>& particles,
                                                  std::size_t active_offset,
                                                  std::size_t i)
    {
        for (std::size_t d = 0; d < 3; ++d)
        {
            this->m_U(3*this->m_nparts + 3*i + d) /= (-1.*particles.j()(active_offset + i)(d));
        }
    }
}

#endif
