#pragma once
#include "../crtp.hpp"
#include "../container.hpp"
#include "../objects/neighbor.hpp"
#include <plog/Log.h>
#include "plog/Initializers/RollingFileInitializer.h"

namespace scopi{
    template <class D, std::size_t dim>
    class MatrixOptimSolver: public crtp_base<D, MatrixOptimSolver<D, dim>>
    {
    protected:
        MatrixOptimSolver(scopi_container<dim>& particles, double dt, std::size_t Nactive, std::size_t active_ptr);
        void create_matrix_constraint_coo(const std::vector<neighbor<dim>>& contacts, std::vector<int>& A_rows, std::vector<int>& A_cols, std::vector<double>& A_values, std::size_t firstCol);
        void create_matrix_mass_coo(const std::vector<neighbor<dim>>& contacts, std::vector<int>& A_rows, std::vector<int>& A_cols, std::vector<double>& A_values);

        scopi_container<dim>& m_particles;
        double m_dt;
        std::size_t m_Nactive;
        std::size_t m_active_ptr;
    };


    template<class D, std::size_t dim>
    MatrixOptimSolver<D, dim>::MatrixOptimSolver(scopi_container<dim>& particles, double dt, std::size_t Nactive, std::size_t active_ptr)
    : m_particles(particles)
    , m_dt(dt)
    , m_Nactive(Nactive)
    , m_active_ptr(active_ptr)
    {}

    template<class D, std::size_t dim>
    void MatrixOptimSolver<D, dim>::create_matrix_constraint_coo(const std::vector<neighbor<dim>>& contacts, std::vector<int>& A_rows, std::vector<int>& A_cols, std::vector<double>& A_values, std::size_t firstCol)
    {
        std::size_t u_size = 3*contacts.size()*2;
        std::size_t w_size = 3*contacts.size()*2;
        A_rows.reserve(u_size + w_size);
        A_cols.reserve(u_size + w_size);
        A_values.reserve(u_size + w_size);

        std::size_t ic = 0;
        for (auto &c: contacts)
        {
            for (std::size_t d = 0; d < 3; ++d)
            {
                if (c.i >= m_active_ptr)
                {
                    A_rows.push_back(ic);
                    A_cols.push_back(firstCol + (c.i - m_active_ptr)*3 + d);
                    A_values.push_back(-m_dt*c.nij[d]);
                }
                if (c.j >= m_active_ptr)
                {
                    A_rows.push_back(ic);
                    A_cols.push_back(firstCol + (c.j - m_active_ptr)*3 + d);
                    A_values.push_back(m_dt*c.nij[d]);
                }
            }

            auto r_i = c.pi - m_particles.pos()(c.i);
            auto r_j = c.pj - m_particles.pos()(c.j);

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

            auto Ri = rotation_matrix<3>(m_particles.q()(c.i));
            auto Rj = rotation_matrix<3>(m_particles.q()(c.j));

            if (c.i >= m_active_ptr)
            {
                std::size_t ind_part = c.i - m_active_ptr;
                auto dot = xt::eval(xt::linalg::dot(ri_cross, Ri));
                for (std::size_t ip = 0; ip < 3; ++ip)
                {
                    A_rows.push_back(ic);
                    A_cols.push_back(firstCol + 3*m_Nactive + 3*ind_part + ip);
                    A_values.push_back(m_dt*(c.nij[0]*dot(0, ip)+c.nij[1]*dot(1, ip)+c.nij[2]*dot(2, ip)));
                }
            }

            if (c.j >= m_active_ptr)
            {
                std::size_t ind_part = c.j - m_active_ptr;
                auto dot = xt::eval(xt::linalg::dot(rj_cross, Rj));
                for (std::size_t ip = 0; ip < 3; ++ip)
                {
                    A_rows.push_back(ic);
                    A_cols.push_back(firstCol + 3*m_Nactive + 3*ind_part + ip);
                    A_values.push_back(-m_dt*(c.nij[0]*dot(0, ip)+c.nij[1]*dot(1, ip)+c.nij[2]*dot(2, ip)));
                }
            }

            ++ic;
        }
    }

}

