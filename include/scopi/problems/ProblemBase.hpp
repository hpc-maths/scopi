#pragma once

#include <cstddef>
#include <plog/Log.h>
#include "plog/Initializers/RollingFileInitializer.h"
#include <xtensor/xtensor.hpp>

#include "../container.hpp"
#include "../utils.hpp"
#include "../objects/neighbor.hpp"
#include "../quaternion.hpp"

namespace scopi
{
    class ProblemBase
    {
    protected:
        ProblemBase(std::size_t nparts, double dt);

        template<std::size_t dim>
        void matrix_free_gemv_inv_P(const scopi_container<dim>& particles,
                                    xt::xtensor<double, 1>& U,
                                    std::size_t active_offset,
                                    std::size_t row);

    private:
        template<std::size_t dim>
        std::size_t matrix_positive_distance(const scopi_container<dim>& particles,
                                             const std::vector<neighbor<dim>>& contacts,
                                             std::size_t firstCol,
                                             std::size_t nb_row,
                                             std::size_t nb_row_per_contact);


        void matrix_free_gemv_inv_P_moment(const scopi_container<2>& particles,
                                           xt::xtensor<double, 1>& U,
                                           std::size_t active_offset,
                                           std::size_t row);
        void matrix_free_gemv_inv_P_moment(const scopi_container<3>& particles,
                                           xt::xtensor<double, 1>& U,
                                           std::size_t active_offset,
                                           std::size_t row);

    protected:
        std::size_t m_nparticles;
        double m_dt;
        std::vector<int> m_A_rows;
        std::vector<int> m_A_cols;
        std::vector<double> m_A_values;
        xt::xtensor<double, 1> m_distances;
    };

    template<std::size_t dim>
    void ProblemBase::matrix_free_gemv_inv_P(const scopi_container<dim>& particles,
                                             xt::xtensor<double, 1>& U,
                                             std::size_t active_offset,
                                             std::size_t row)
    {
        for (std::size_t d = 0; d < dim; ++d)
        {
            U(3*row + d) /= (-1.*particles.m()(active_offset + row)); 
        }
        matrix_free_gemv_inv_P_moment(particles, U, active_offset, row);
    }

    template<std::size_t dim>
    std::size_t ProblemBase::matrix_positive_distance(const scopi_container<dim>& particles,
                                                      const std::vector<neighbor<dim>>& contacts,
                                                      std::size_t firstCol,
                                                      std::size_t nb_row,
                                                      std::size_t nb_row_per_contact)
    {
        std::size_t active_offset = particles.nb_inactive();
        this->m_A_rows.resize(12*nb_row);
        this->m_A_cols.resize(12*nb_row);
        this->m_A_values.resize(12*nb_row);

        std::size_t ic = 0;
        std::size_t index = 0;
        for (auto &c: contacts)
        {
            if (c.i >= active_offset)
            {
                for (std::size_t d = 0; d < 3; ++d)
                {
                    this->m_A_rows[index] = nb_row_per_contact*ic;
                    this->m_A_cols[index] = firstCol + (c.i - active_offset)*3 + d;
                    this->m_A_values[index] = -this->m_dt*c.nij[d];
                    index++;
                }
            }

            if (c.j >= active_offset)
            {
                for (std::size_t d = 0; d < 3; ++d)
                {
                    this->m_A_rows[index] = nb_row_per_contact*ic;
                    this->m_A_cols[index] = firstCol + (c.j - active_offset)*3 + d;
                    this->m_A_values[index] = this->m_dt*c.nij[d];
                    index++;
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
                    this->m_A_rows[index] = nb_row_per_contact*ic;
                    this->m_A_cols[index] = firstCol + 3*particles.nb_active() + 3*ind_part + ip;
                    this->m_A_values[index] = this->m_dt*(c.nij[0]*dot(0, ip)+c.nij[1]*dot(1, ip)+c.nij[2]*dot(2, ip));
                    index++;
                }
            }

            if (c.j >= active_offset)
            {
                std::size_t ind_part = c.j - active_offset;
                auto dot = xt::eval(xt::linalg::dot(rj_cross, Rj));
                for (std::size_t ip = 0; ip < 3; ++ip)
                {
                    this->m_A_rows[index] = nb_row_per_contact*ic;
                    this->m_A_cols[index] = firstCol + 3*particles.nb_active() + 3*ind_part + ip;
                    this->m_A_values[index] = -this->m_dt*(c.nij[0]*dot(0, ip)+c.nij[1]*dot(1, ip)+c.nij[2]*dot(2, ip));
                    index++;
                }
            }

            ++ic;
        }
        return index;
    }
}
