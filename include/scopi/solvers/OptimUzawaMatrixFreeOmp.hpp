#pragma once

#include "OptimUzawaBase.hpp"
#include <omp.h>

#include <xtensor/xadapt.hpp>
#include <xtensor/xview.hpp>

#include <plog/Log.h>
#include "plog/Initializers/RollingFileInitializer.h"

#include "../quaternion.hpp"

namespace scopi{
    class OptimUzawaMatrixFreeOmp : public OptimUzawaBase<OptimUzawaMatrixFreeOmp>
    {
    public:
        using base_type = OptimUzawaBase<OptimUzawaMatrixFreeOmp>;
        OptimUzawaMatrixFreeOmp(std::size_t nparts, double dt);

        template <std::size_t dim>
        void gemv_inv_P_impl(const scopi_container<dim>& particles);

        template <std::size_t dim>
        void gemv_A_impl(const scopi_container<dim>& particles,
                         const std::vector<neighbor<dim>>& contacts);

        template <std::size_t dim>
        void gemv_transpose_A_impl(const scopi_container<dim>& particles,
                                   const std::vector<neighbor<dim>>& contacts);

        template <std::size_t dim>
        void init_uzawa_impl(const std::vector<neighbor<dim>>& contacts);
    };

    template <std::size_t dim>
    void OptimUzawaMatrixFreeOmp::init_uzawa_impl(const std::vector<neighbor<dim>>&)
    {}

    template<std::size_t dim>
    void OptimUzawaMatrixFreeOmp::gemv_inv_P_impl(const scopi_container<dim>& particles)
    {
        #pragma omp parallel for
        for (std::size_t i = 0; i < particles.nb_active(); ++i)
        {
            for (std::size_t d = 0; d < 3; ++d)
            {
                this->m_U(3*i + d) /= (-1. * this->m_mass); // TODO: add mass into particles
                this->m_U(3*this->m_nparts + 3*i + d) /= (-1. * this->m_moment);
            }
        }
    }

    template <std::size_t dim>
    void OptimUzawaMatrixFreeOmp::gemv_A_impl(const scopi_container<dim>& particles,
                                              const std::vector<neighbor<dim>>& contacts)
    {
        std::size_t active_offset = particles.nb_inactive();
        #pragma omp parallel for
        for (std::size_t ic = 0; ic < contacts.size(); ++ic)
        {
            auto &c = contacts[ic];
            if (c.i >= active_offset)
            {
                for (std::size_t d = 0; d < 3; ++d)
                {
                    this->m_R(ic) -= (-this->m_dt*c.nij[d]) * this->m_U((c.i - active_offset)*3 + d);
                }
            }
            if (c.j >= active_offset)
            {
                for (std::size_t d = 0; d < 3; ++d)
                {
                    this->m_R(ic) -= (this->m_dt*c.nij[d]) * this->m_U((c.j - active_offset)*3 + d);
                }
            }

            auto r_i = c.pi - particles.pos()(c.i);
            auto r_j = c.pj - particles.pos()(c.j);

            xt::xtensor_fixed<double, xt::xshape<3, 3>> ri_cross, rj_cross;

            if (dim == 2)
            {
                ri_cross = {{      0,      0, r_i(1)},
                            {      0,      0, -r_i(0)},
                            {-r_i(1), r_i(0),       0}};

                rj_cross = {{      0,      0,  r_j(1)},
                            {      0,      0, -r_j(0)},
                            {-r_j(1), r_j(0),       0}};
            }
            else
            {
                ri_cross = {{      0, -r_i(2),  r_i(1)},
                            { r_i(2),       0, -r_i(0)},
                            {-r_i(1),  r_i(0),       0}};

                rj_cross = {{      0, -r_j(2),  r_j(1)},
                            { r_j(2),       0, -r_j(0)},
                            {-r_j(1),  r_j(0),       0}};
            }

            auto Ri = rotation_matrix<3>(particles.q()(c.i));
            auto Rj = rotation_matrix<3>(particles.q()(c.j));

            if (c.i >= active_offset)
            {
                std::size_t ind_part = c.i - active_offset;
                auto dot = xt::eval(xt::linalg::dot(ri_cross, Ri));
                for (std::size_t ip = 0; ip < 3; ++ip)
                {
                    this->m_R(ic) -= (this->m_dt*(c.nij[0]*dot(0, ip) + c.nij[1]*dot(1, ip) + c.nij[2]*dot(2, ip)))
                                    * this->m_U(3*particles.nb_active() + 3*ind_part + ip);
                }
            }

            if (c.j >= active_offset)
            {
                std::size_t ind_part = c.j - active_offset;
                auto dot = xt::eval(xt::linalg::dot(rj_cross, Rj));
                for (std::size_t ip = 0; ip < 3; ++ip)
                {
                    this->m_R(ic) -= (-this->m_dt*(c.nij[0]*dot(0, ip) + c.nij[1]*dot(1, ip) + c.nij[2]*dot(2, ip)))
                                     * this->m_U(3*particles.nb_active() + 3*ind_part + ip);
                }
            }

        }
    }

    template <std::size_t dim>
    void OptimUzawaMatrixFreeOmp::gemv_transpose_A_impl(const scopi_container<dim>& particles,
                                                        const std::vector<neighbor<dim>>& contacts)
    {
        std::size_t active_offset = particles.nb_inactive();
        #pragma omp parallel for
        for(std::size_t ic = 0; ic < contacts.size(); ++ic)
        {
            auto &c = contacts[ic];

            if (c.i >= active_offset)
            {
                for (std::size_t d = 0; d < 3; ++d)
                {
                    #pragma omp atomic
                    this->m_U((c.i - active_offset)*3 + d) += -this->m_L(ic) * this->m_dt * c.nij[d];
                }
            }
            if (c.j >= active_offset)
            {
                for (std::size_t d = 0; d < 3; ++d)
                {
                    #pragma omp atomic
                    this->m_U((c.j - active_offset)*3 + d) += this->m_L(ic) * this->m_dt * c.nij[d];
                }
            }

            auto r_i = c.pi - particles.pos()(c.i);
            auto r_j = c.pj - particles.pos()(c.j);

            xt::xtensor_fixed<double, xt::xshape<3, 3>> ri_cross, rj_cross;

            if (dim == 2)
            {
                ri_cross = {{      0,      0, r_i(1)},
                            {      0,      0, -r_i(0)},
                            {-r_i(1), r_i(0),       0}};

                rj_cross = {{      0,      0,  r_j(1)},
                            {      0,      0, -r_j(0)},
                            {-r_j(1), r_j(0),       0}};
            }
            else
            {
                ri_cross = {{      0, -r_i(2),  r_i(1)},
                            { r_i(2),       0, -r_i(0)},
                            {-r_i(1),  r_i(0),       0}};

                rj_cross = {{      0, -r_j(2),  r_j(1)},
                            { r_j(2),       0, -r_j(0)},
                            {-r_j(1),  r_j(0),       0}};
            }

            auto Ri = rotation_matrix<3>(particles.q()(c.i));
            auto Rj = rotation_matrix<3>(particles.q()(c.j));

            if (c.i >= active_offset)
            {
                std::size_t ind_part = c.i - active_offset;
                auto dot = xt::eval(xt::linalg::dot(ri_cross, Ri));
                for (std::size_t ip = 0; ip < 3; ++ip)
                {
                    #pragma omp atomic
                    this->m_U(3*particles.nb_active() + 3*ind_part + ip) += this->m_L(ic) * (this->m_dt*(c.nij[0]*dot(0, ip)+c.nij[1]*dot(1, ip)+c.nij[2]*dot(2, ip)));
                }
            }

            if (c.j >= active_offset)
            {
                std::size_t ind_part = c.j - active_offset;
                auto dot = xt::eval(xt::linalg::dot(rj_cross, Rj));
                for (std::size_t ip = 0; ip < 3; ++ip)
                {
                    #pragma omp atomic
                    this->m_U(3*particles.nb_active() + 3*ind_part + ip) += this->m_L(ic) * (-this->m_dt*(c.nij[0]*dot(0, ip)+c.nij[1]*dot(1, ip)+c.nij[2]*dot(2, ip)));
                }
            }
        }
    }

}
