#ifdef SCOPI_USE_MOSEK
#include "scopi/solvers/ConstraintMosek.hpp"

namespace scopi
{
    using namespace monty;

    ConstraintMosek<DryWithoutFriction>::ConstraintMosek(std::size_t nparticles)
    : m_nparticles(nparticles)
    {}

    std::size_t ConstraintMosek<DryWithoutFriction>::index_first_col_matrix() const
    {
        return 1;
    }

    std::size_t ConstraintMosek<DryWithoutFriction>::number_col_matrix() const
    {
        return 1 + 6*m_nparticles + 6*m_nparticles;
    }

    void ConstraintMosek<DryWithoutFriction>::update_dual(std::size_t,
                                                         std::size_t,
                                                         std::size_t,
                                                         std::size_t)
    {
        m_dual = m_qc1->dual();
    }





    ConstraintMosek<DryWithFriction>::ConstraintMosek(std::size_t nparticles)
    : m_nparticles(nparticles)
    {}

    std::size_t ConstraintMosek<DryWithFriction>::index_first_col_matrix() const
    {
        return 0;
    }

    std::size_t ConstraintMosek<DryWithFriction>::number_col_matrix() const
    {
        return 6*m_nparticles;
    }

    void ConstraintMosek<DryWithFriction>::update_dual(std::size_t,
                                                                 std::size_t,
                                                                 std::size_t,
                                                                 std::size_t)
    {
        m_dual = m_qc1->dual();
    }
}
#endif
