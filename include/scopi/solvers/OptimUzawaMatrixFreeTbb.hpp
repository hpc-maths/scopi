#pragma once

#ifdef SCOPI_USE_TBB
#include "OptimUzawaBase.hpp"
#include <omp.h>
#include "tbb/tbb.h"

#include <xtensor/xadapt.hpp>
#include <xtensor/xview.hpp>
#include <plog/Log.h>
#include "plog/Initializers/RollingFileInitializer.h"

namespace scopi{
    template<std::size_t dim>
    class OptimUzawaMatrixFreeTbb : public OptimUzawaBase<OptimUzawaMatrixFreeTbb<dim>, dim>
    {
    public:
        OptimUzawaMatrixFreeTbb(scopi_container<dim>& particles, double dt, std::size_t Nactive, std::size_t active_ptr);

        void gemv_inv_P_impl();
        void gemv_A_impl(const std::vector<neighbor<dim>>& contacts);
        void gemv_transpose_A_impl(const std::vector<neighbor<dim>>& contacts);
        void setup_impl(const std::vector<neighbor<dim>>& contacts);

    };

    template<std::size_t dim>
    OptimUzawaMatrixFreeTbb<dim>::OptimUzawaMatrixFreeTbb(scopi_container<dim>& particles, double dt, std::size_t Nactive, std::size_t active_ptr)
    : OptimUzawaBase<OptimUzawaMatrixFreeTbb<dim>, dim>(particles, dt, Nactive, active_ptr)
    {}

    template<std::size_t dim>
    void OptimUzawaMatrixFreeTbb<dim>::setup_impl(const std::vector<neighbor<dim>>&)
    {}

    template<std::size_t dim>
    void OptimUzawaMatrixFreeTbb<dim>::gemv_inv_P_impl()
    {
        // for loops instead of xtensor functions to control exactly the parallelism
        tbb::parallel_for(std::size_t(0), this->m_Nactive, [=](std::size_t i) {
            for (std::size_t d = 0; d < 3; ++d)
            {
                this->m_U(3*i + d) /= (-1. * this->m_mass); // TODO: add mass into particles
                this->m_U(3*this->m_Nactive + 3*i + d) /= (-1. * this->m_moment);
            }
        });
    }

    template<std::size_t dim>
    void OptimUzawaMatrixFreeTbb<dim>::gemv_A_impl(const std::vector<neighbor<dim>>& contacts)
    {
        tbb::parallel_for(std::size_t(0), contacts.size(), [=](std::size_t ic) {
            auto &c = contacts[ic];
            for (std::size_t d = 0; d < 3; ++d)
            {
                if (c.i >= this->m_active_ptr)
                {
                this->m_R(ic) -= (-this->m_dt*c.nij[d]) * this->m_U((c.i - this->m_active_ptr)*3 + d);
                }
                if (c.j >= this->m_active_ptr)
                {
                this->m_R(ic) -= (this->m_dt*c.nij[d]) * this->m_U((c.j - this->m_active_ptr)*3 + d);
                }
            }

            auto r_i = c.pi - this->m_particles.pos()(c.i);
            auto r_j = c.pj - this->m_particles.pos()(c.j);

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

            auto Ri = rotation_matrix<3>(this->m_particles.q()(c.i));
            auto Rj = rotation_matrix<3>(this->m_particles.q()(c.j));

            if (c.i >= this->m_active_ptr)
            {
                std::size_t ind_part = c.i - this->m_active_ptr;
                auto dot = xt::eval(xt::linalg::dot(ri_cross, Ri));
                for (std::size_t ip = 0; ip < 3; ++ip)
                {
                    this->m_R(ic) -= (this->m_dt*(c.nij[0]*dot(0, ip)+c.nij[1]*dot(1, ip)+c.nij[2]*dot(2, ip))) * this->m_U(3*this->m_Nactive + 3*ind_part + ip);
                }
            }

            if (c.j >= this->m_active_ptr)
            {
                std::size_t ind_part = c.j - this->m_active_ptr;
                auto dot = xt::eval(xt::linalg::dot(rj_cross, Rj));
                for (std::size_t ip = 0; ip < 3; ++ip)
                {
                    this->m_R(ic) -= (-this->m_dt*(c.nij[0]*dot(0, ip)+c.nij[1]*dot(1, ip)+c.nij[2]*dot(2, ip))) * this->m_U(3*this->m_Nactive + 3*ind_part + ip);
                }
            }
        });
    }

    template<std::size_t dim>
    void OptimUzawaMatrixFreeTbb<dim>::gemv_transpose_A_impl(const std::vector<neighbor<dim>>& contacts)
    {
        this->m_U = this->m_U + tbb::parallel_reduce(tbb::blocked_range<std::size_t>(0, contacts.size()),
            xt::zeros_like(this->m_U),
            [=](tbb::blocked_range<std::size_t>& r, xt::xtensor<double, 1> partialSum) -> xt::xtensor<double, 1>
            {
                for(std::size_t ic=r.begin(); ic!=r.end(); ++ic)
                {
                    auto &c = contacts[ic];

                    for (std::size_t d = 0; d < 3; ++d)
                    {
                        if (c.i >= this->m_active_ptr)
                        {
                            partialSum((c.i - this->m_active_ptr)*3 + d) += this->m_L(ic) * (-this->m_dt*c.nij[d]);
                        }
                        if (c.j >= this->m_active_ptr)
                        {
                        partialSum((c.j - this->m_active_ptr)*3 + d) += this->m_L(ic) * (this->m_dt*c.nij[d]);
                        }
                    }

                    auto r_i = c.pi - this->m_particles.pos()(c.i);
                    auto r_j = c.pj - this->m_particles.pos()(c.j);

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

                    auto Ri = rotation_matrix<3>(this->m_particles.q()(c.i));
                    auto Rj = rotation_matrix<3>(this->m_particles.q()(c.j));

                    if (c.i >= this->m_active_ptr)
                    {
                        std::size_t ind_part = c.i - this->m_active_ptr;
                        auto dot = xt::eval(xt::linalg::dot(ri_cross, Ri));
                        for (std::size_t ip = 0; ip < 3; ++ip)
                        {
                            partialSum(3*this->m_Nactive + 3*ind_part + ip) += this->m_L(ic) * (this->m_dt*(c.nij[0]*dot(0, ip)+c.nij[1]*dot(1, ip)+c.nij[2]*dot(2, ip)));
                        }
                    }

                    if (c.j >= this->m_active_ptr)
                    {
                        std::size_t ind_part = c.j - this->m_active_ptr;
                        auto dot = xt::eval(xt::linalg::dot(rj_cross, Rj));
                        for (std::size_t ip = 0; ip < 3; ++ip)
                        {
                            partialSum(3*this->m_Nactive + 3*ind_part + ip) += this->m_L(ic) * (-this->m_dt*(c.nij[0]*dot(0, ip)+c.nij[1]*dot(1, ip)+c.nij[2]*dot(2, ip)));
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
