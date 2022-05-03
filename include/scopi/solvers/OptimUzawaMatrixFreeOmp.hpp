#pragma once

#include "OptimUzawaBase.hpp"
#include <cstddef>
#include <omp.h>

#include <xtensor/xadapt.hpp>
#include <xtensor/xview.hpp>

#include <plog/Log.h>
#include "plog/Initializers/RollingFileInitializer.h"

#include "../quaternion.hpp"
#include "../utils.hpp"

namespace scopi{
    template<class problem_t = MatrixOptimSolver>
    class OptimUzawaMatrixFreeOmp : public OptimUzawaBase<OptimUzawaMatrixFreeOmp<problem_t>, problem_t>
    {
    public:
        using base_type = OptimUzawaBase<OptimUzawaMatrixFreeOmp<problem_t>, problem_t>;
        template <std::size_t dim>
        OptimUzawaMatrixFreeOmp(std::size_t nparts, double dt, const scopi_container<dim>& particles);

        template <std::size_t dim>
        void gemv_inv_P_impl(const scopi_container<dim>& particles);

        template <std::size_t dim>
        void gemv_A_impl(const scopi_container<dim>& particles,
                         const std::vector<neighbor<dim>>& contacts);

        template <std::size_t dim>
        void gemv_transpose_A_impl(const scopi_container<dim>& particles,
                                   const std::vector<neighbor<dim>>& contacts);

        template <std::size_t dim>
        void init_uzawa_impl(const scopi_container<dim>& particles,
                             const std::vector<neighbor<dim>>& contacts);

    private:
        void gemv_inv_P_moment(const scopi_container<2>& particles,
                               std::size_t active_offset,
                               std::size_t i);
        void gemv_inv_P_moment(const scopi_container<3>& particles,
                               std::size_t active_offset,
                               std::size_t i);
    };

    template <class problem_t>
    template <std::size_t dim>
    void OptimUzawaMatrixFreeOmp<problem_t>::init_uzawa_impl(const scopi_container<dim>&,
                                                  const std::vector<neighbor<dim>>&)
    {}

    template <class problem_t>
    template<std::size_t dim>
    void OptimUzawaMatrixFreeOmp<problem_t>::gemv_inv_P_impl(const scopi_container<dim>& particles)
    {
        auto active_offset = particles.nb_inactive();
        #pragma omp parallel for
        for (std::size_t i = 0; i < particles.nb_active(); ++i)
        {
            for (std::size_t d = 0; d < dim; ++d)
            {
                this->m_U(3*i + d) /= (-1.*particles.m()(active_offset + i)); 
            }
            gemv_inv_P_moment(particles, active_offset, i);
        }
    }

    template <class problem_t>
    template <std::size_t dim>
    void OptimUzawaMatrixFreeOmp<problem_t>::gemv_A_impl(const scopi_container<dim>& particles,
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

            auto ri_cross = cross_product<dim>(c.pi - particles.pos()(c.i));
            auto rj_cross = cross_product<dim>(c.pj - particles.pos()(c.j));
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

    template <class problem_t>
    template <std::size_t dim>
    void OptimUzawaMatrixFreeOmp<problem_t>::gemv_transpose_A_impl(const scopi_container<dim>& particles,
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


            auto ri_cross = cross_product<dim>(c.pi - particles.pos()(c.i));
            auto rj_cross = cross_product<dim>(c.pj - particles.pos()(c.j));
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

    template <class problem_t>
    template <std::size_t dim>
    OptimUzawaMatrixFreeOmp<problem_t>::OptimUzawaMatrixFreeOmp(std::size_t nparts, double dt, const scopi_container<dim>&)
    : base_type(nparts, dt)
    {}

    template <class problem_t>
    void OptimUzawaMatrixFreeOmp<problem_t>::gemv_inv_P_moment(const scopi_container<2>& particles,
                                                    std::size_t active_offset,
                                                    std::size_t i)
    {
        this->m_U(3*this->m_nparts + 3*i + 2) /= (-1.*particles.j()(active_offset + i));
    }

    template <class problem_t>
    void OptimUzawaMatrixFreeOmp<problem_t>::gemv_inv_P_moment(const scopi_container<3>& particles,
                                                    std::size_t active_offset,
                                                    std::size_t i)
    {
        for (std::size_t d = 0; d < 3; ++d)
        {
            this->m_U(3*this->m_nparts + 3*i + d) /= (-1.*particles.j()(active_offset + i)(d));
        }
    }
}
