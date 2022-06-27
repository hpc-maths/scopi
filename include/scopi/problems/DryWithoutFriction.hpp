#pragma once

#include <cstddef>
#include <plog/Log.h>
#include "plog/Initializers/RollingFileInitializer.h"
#include <xtensor/xtensor.hpp>

#include "../container.hpp"
#include "../objects/methods/closest_points.hpp"
#include "../objects/methods/select.hpp"
#include "../quaternion.hpp"
#include "../objects/neighbor.hpp"
#include "../params/ProblemParams.hpp"
#include "ProblemBase.hpp"

namespace scopi
{
    class DryWithoutFriction;

    template<>
    class ProblemParams<DryWithoutFriction>
    {};

    class DryWithoutFriction : public ProblemBase
    {

    public:
        DryWithoutFriction(std::size_t nparts, double dt, const ProblemParams<DryWithoutFriction>& problem_params);

        template <std::size_t dim>
        void create_matrix_constraint_coo(scopi_container<dim>& particles,
                                          const std::vector<neighbor<dim>>& contacts,
                                          std::size_t firstCol);
        template <std::size_t dim>
        std::size_t number_row_matrix(const std::vector<neighbor<dim>>& contacts,
                                      scopi_container<dim>& particles);
        template<std::size_t dim>
        void create_vector_distances(const std::vector<neighbor<dim>>& contacts, scopi_container<dim>& particles);

        template<std::size_t dim>
        void matrix_free_gemv_A(const neighbor<dim>& c,
                                const scopi_container<dim>& particles,
                                const xt::xtensor<double, 1>& U,
                                xt::xtensor<double, 1>& R,
                                std::size_t active_offset,
                                std::size_t row);
        template<std::size_t dim>
        void matrix_free_gemv_transpose_A(const neighbor<dim>& c,
                                          const scopi_container<dim>& particles,
                                          const xt::xtensor<double, 1>& L,
                                          xt::xtensor<double, 1>& U,
                                          std::size_t active_offset,
                                          std::size_t row);

        template<std::size_t dim>
        void extra_setps_before_solve(const std::vector<neighbor<dim>>& contacts);
        template<std::size_t dim>
        void extra_setps_after_solve(const std::vector<neighbor<dim>>& contacts,
                                     xt::xtensor<double, 1> lambda);
    };

    template<std::size_t dim>
    void DryWithoutFriction::create_matrix_constraint_coo(scopi_container<dim>& particles,
                                                          const std::vector<neighbor<dim>>& contacts,
                                                          std::size_t firstCol)
    {
        std::size_t idx = matrix_positive_distance(particles, contacts, firstCol, number_row_matrix(contacts, particles), 1);
        std::size_t ic = contacts.size();
        std::size_t active_offset = particles.nb_inactive();

        for (std::size_t i = 0; i < particles.size(); ++i)
        {
            for (std::size_t j = 0; j < particles[i]->size()-1; ++j)
            {
                auto neigh = closest_points_dispatcher<dim>::dispatch(*select_object_dispatcher<dim>::dispatch(*particles[i], index(j  )),
                                                                      *select_object_dispatcher<dim>::dispatch(*particles[i], index(j+1)));
                for (std::size_t d = 0; d < 3; ++d)
                {
                    this->m_A_rows[idx] = ic;
                    this->m_A_cols[idx] = firstCol + (particles.offset(i) + j - active_offset)*3 + d;
                    this->m_A_values[idx] = this->m_dt*neigh.nij[d];
                    idx++;
                }
                for (std::size_t d = 0; d < 3; ++d)
                {
                    this->m_A_rows[idx] = ic;
                    this->m_A_cols[idx] = firstCol + (particles.offset(i) + j + 1 - active_offset)*3 + d;
                    this->m_A_values[idx] = -this->m_dt*neigh.nij[d];
                    idx++;
                }

                auto ri_cross = cross_product<dim>(neigh.pi - particles.pos()(particles.offset(i)+j  ));
                auto rj_cross = cross_product<dim>(neigh.pj - particles.pos()(particles.offset(i)+j+1));
                auto Ri = rotation_matrix<3>(particles.q()(particles.offset(i)+j  ));
                auto Rj = rotation_matrix<3>(particles.q()(particles.offset(i)+j+1));

                std::size_t ind_part = particles.offset(i) + j - active_offset;
                auto dot = xt::eval(xt::linalg::dot(ri_cross, Ri));
                for (std::size_t ip = 0; ip < 3; ++ip)
                {
                    this->m_A_rows[idx] = ic;
                    this->m_A_cols[idx] = firstCol + 3*particles.nb_active() + 3*ind_part + ip;
                    this->m_A_values[idx] = -this->m_dt*(neigh.nij[0]*dot(0, ip)+neigh.nij[1]*dot(1, ip)+neigh.nij[2]*dot(2, ip));
                    idx++;
                }
                ind_part = particles.offset(i) + j + 1 - active_offset;
                dot = xt::eval(xt::linalg::dot(rj_cross, Rj));
                for (std::size_t ip = 0; ip < 3; ++ip)
                {
                    this->m_A_rows[idx] = ic;
                    this->m_A_cols[idx] = firstCol + 3*particles.nb_active() + 3*ind_part + ip;
                    this->m_A_values[idx] = this->m_dt*(neigh.nij[0]*dot(0, ip)+neigh.nij[1]*dot(1, ip)+neigh.nij[2]*dot(2, ip));
                    idx++;
                }
                ic++;
            }
        }
    }

    template <std::size_t dim>
    std::size_t DryWithoutFriction::number_row_matrix(const std::vector<neighbor<dim>>& contacts,
                                                      scopi_container<dim>& particles)
    {
        std::size_t nb_row = contacts.size();
        for (std::size_t i = 0; i < particles.size(); ++i)
        {
            nb_row += particles[i]->size()-1;
        }
        return nb_row;
    }

    template<std::size_t dim>
    void DryWithoutFriction::create_vector_distances(const std::vector<neighbor<dim>>& contacts, scopi_container<dim>& particles)
    {
        this->m_distances = xt::zeros<double>({number_row_matrix(contacts, particles)});
        for (std::size_t i = 0; i < contacts.size(); ++i)
        {
            this->m_distances[i] = contacts[i].dij;
        }

        std::size_t nb_previous_constraints = contacts.size();
        for (std::size_t i = 0; i < particles.size(); ++i)
        {
            for (std::size_t j = 0; j < particles[i]->size()-1; ++j)
            {
                auto neigh = closest_points_dispatcher<dim>::dispatch(*select_object_dispatcher<dim>::dispatch(*particles[i], index(j  )),
                                                                      *select_object_dispatcher<dim>::dispatch(*particles[i], index(j+1)));
                this->m_distances[nb_previous_constraints + j ] = -neigh.dij;
            }
            nb_previous_constraints += particles[i]->size()-1;
        }
    }

    template<std::size_t dim>
    void DryWithoutFriction::matrix_free_gemv_A(const neighbor<dim>& c,
                                               const scopi_container<dim>& particles,
                                               const xt::xtensor<double, 1>& U,
                                               xt::xtensor<double, 1>& R,
                                               std::size_t active_offset,
                                               std::size_t row)
    {
        if (c.i >= active_offset)
        {
            for (std::size_t d = 0; d < 3; ++d)
            {
                R(row) -= (-this->m_dt*c.nij[d]) * U((c.i - active_offset)*3 + d);
            }
        }
        if (c.j >= active_offset)
        {
            for (std::size_t d = 0; d < 3; ++d)
            {
                R(row) -= (this->m_dt*c.nij[d]) * U((c.j - active_offset)*3 + d);
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
                R(row) -= (this->m_dt*(c.nij[0]*dot(0, ip) + c.nij[1]*dot(1, ip) + c.nij[2]*dot(2, ip)))
                    * U(3*particles.nb_active() + 3*ind_part + ip);
            }
        }

        if (c.j >= active_offset)
        {
            std::size_t ind_part = c.j - active_offset;
            auto dot = xt::eval(xt::linalg::dot(rj_cross, Rj));
            for (std::size_t ip = 0; ip < 3; ++ip)
            {
                R(row) -= (-this->m_dt*(c.nij[0]*dot(0, ip) + c.nij[1]*dot(1, ip) + c.nij[2]*dot(2, ip)))
                    * U(3*particles.nb_active() + 3*ind_part + ip);
            }
        }
    }

    template<std::size_t dim>
    void DryWithoutFriction::matrix_free_gemv_transpose_A(const neighbor<dim>& c,
                                                         const scopi_container<dim>& particles,
                                                         const xt::xtensor<double, 1>& L,
                                                         xt::xtensor<double, 1>& U,
                                                         std::size_t active_offset,
                                                         std::size_t row)
    {
        if (c.i >= active_offset)
        {
            for (std::size_t d = 0; d < 3; ++d)
            {
#pragma omp atomic
                U((c.i - active_offset)*3 + d) += -L(row) * this->m_dt * c.nij[d];
            }
        }
        if (c.j >= active_offset)
        {
            for (std::size_t d = 0; d < 3; ++d)
            {
#pragma omp atomic
                U((c.j - active_offset)*3 + d) += L(row) * this->m_dt * c.nij[d];
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
                U(3*particles.nb_active() + 3*ind_part + ip) += L(row) * (this->m_dt*(c.nij[0]*dot(0, ip)+c.nij[1]*dot(1, ip)+c.nij[2]*dot(2, ip)));
            }
        }

        if (c.j >= active_offset)
        {
            std::size_t ind_part = c.j - active_offset;
            auto dot = xt::eval(xt::linalg::dot(rj_cross, Rj));
            for (std::size_t ip = 0; ip < 3; ++ip)
            {
#pragma omp atomic
                U(3*particles.nb_active() + 3*ind_part + ip) += L(row) * (-this->m_dt*(c.nij[0]*dot(0, ip)+c.nij[1]*dot(1, ip)+c.nij[2]*dot(2, ip)));
            }
        }
    }

    template<std::size_t dim>
    void DryWithoutFriction::extra_setps_before_solve(const std::vector<neighbor<dim>>&)
    {}

    template<std::size_t dim>
    void DryWithoutFriction::extra_setps_after_solve(const std::vector<neighbor<dim>>&,
                                                     xt::xtensor<double, 1>)
    {}

}

