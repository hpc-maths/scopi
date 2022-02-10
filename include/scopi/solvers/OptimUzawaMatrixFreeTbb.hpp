#pragma once

#ifdef SCOPI_USE_TBB
#include "OptimUzawaBase.hpp"
#include <omp.h>
#include <tbb/tbb.h>

#include <xtensor/xadapt.hpp>
#include <xtensor/xview.hpp>
#include <plog/Log.h>
#include "plog/Initializers/RollingFileInitializer.h"

#include "../utils.hpp"

namespace scopi
{
    class OptimUzawaMatrixFreeTbb : public OptimUzawaBase<OptimUzawaMatrixFreeTbb>
    {
    public:
        using base_type = OptimUzawaBase<OptimUzawaMatrixFreeTbb>;
        OptimUzawaMatrixFreeTbb(std::size_t nparts, double dt);

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

    template<std::size_t dim>
    void OptimUzawaMatrixFreeTbb::init_uzawa_impl(const std::vector<neighbor<dim>>&)
    {}

    template<std::size_t dim>
    void OptimUzawaMatrixFreeTbb::gemv_inv_P_impl(const scopi_container<dim>& particles)
    {
        // for loops instead of xtensor functions to control exactly the parallelism
        tbb::parallel_for(std::size_t(0), this->m_nparts, [=](std::size_t i) {
            for (std::size_t d = 0; d < 3; ++d)
            {
                this->m_U(3*i + d) /= (-1. * this->m_mass); // TODO: add mass into particles
                this->m_U(3*this->m_nparts + 3*i + d) /= (-1. * this->m_moment);
            }
        });
    }

    template<std::size_t dim>
    void OptimUzawaMatrixFreeTbb::gemv_A_impl(const scopi_container<dim>& particles,
                                              const std::vector<neighbor<dim>>& contacts)
    {
        std::size_t active_offset = particles.nb_inactive();

        tbb::parallel_for(std::size_t(0), contacts.size(), [&](std::size_t ic)
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
                    this->m_R(ic) -= (this->m_dt*(c.nij[0]*dot(0, ip)+c.nij[1]*dot(1, ip)+c.nij[2]*dot(2, ip))) * this->m_U(3*this->m_nparts + 3*ind_part + ip);
                }
            }

            if (c.j >= active_offset)
            {
                std::size_t ind_part = c.j - active_offset;
                auto dot = xt::eval(xt::linalg::dot(rj_cross, Rj));
                for (std::size_t ip = 0; ip < 3; ++ip)
                {
                    this->m_R(ic) -= (-this->m_dt*(c.nij[0]*dot(0, ip)+c.nij[1]*dot(1, ip)+c.nij[2]*dot(2, ip))) * this->m_U(3*this->m_nparts + 3*ind_part + ip);
                }
            }
        });
    }

    template<std::size_t dim>
    void OptimUzawaMatrixFreeTbb::gemv_transpose_A_impl(const scopi_container<dim>& particles,
                                                        const std::vector<neighbor<dim>>& contacts)
    {
        std::size_t active_offset = particles.nb_inactive();
        this->m_U = this->m_U + tbb::parallel_reduce(tbb::blocked_range<std::size_t>(0, contacts.size()),
            xt::zeros_like(this->m_U),
            [&](tbb::blocked_range<std::size_t>& r, xt::xtensor<double, 1> partialSum) -> xt::xtensor<double, 1>
            {
                for(std::size_t ic=r.begin(); ic!=r.end(); ++ic)
                {
                    auto &c = contacts[ic];

                    if (c.i >= active_offset)
                    {
                        for (std::size_t d = 0; d < 3; ++d)
                        {
                            partialSum((c.i - active_offset)*3 + d) += this->m_L(ic) * (-this->m_dt*c.nij[d]);
                        }
                    }

                    if (c.j >= active_offset)
                    {
                        for (std::size_t d = 0; d < 3; ++d)
                        {
                           partialSum((c.j - active_offset)*3 + d) += this->m_L(ic) * (this->m_dt*c.nij[d]);
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
                            partialSum(3*this->m_nparts + 3*ind_part + ip) += this->m_L(ic) * (this->m_dt*(c.nij[0]*dot(0, ip)+c.nij[1]*dot(1, ip)+c.nij[2]*dot(2, ip)));
                        }
                    }

                    if (c.j >= active_offset)
                    {
                        std::size_t ind_part = c.j - active_offset;
                        auto dot = xt::eval(xt::linalg::dot(rj_cross, Rj));
                        for (std::size_t ip = 0; ip < 3; ++ip)
                        {
                            partialSum(3*this->m_nparts + 3*ind_part + ip) += this->m_L(ic) * (-this->m_dt*(c.nij[0]*dot(0, ip)+c.nij[1]*dot(1, ip)+c.nij[2]*dot(2, ip)));
                        }
                    }
                }
                return partialSum;
            },
            []( xt::xtensor<double, 1> x, xt::xtensor<double, 1> y )-> xt::xtensor<double, 1>
            {
                return x+y;
            }
        );
    }
}
#endif
