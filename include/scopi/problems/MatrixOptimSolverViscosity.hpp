#pragma once

#include <plog/Log.h>
#include "plog/Initializers/RollingFileInitializer.h"
#include <vector>
#include <xtensor/xtensor.hpp>

#include "../container.hpp"
#include "../quaternion.hpp"
#include "../objects/neighbor.hpp"
#include "../utils.hpp"

namespace scopi
{
    template<std::size_t dim>
    class MatrixOptimSolverViscosity
    {
    protected:
        MatrixOptimSolverViscosity(std::size_t nparts, double dt, double tol=1e-10);

        void create_matrix_constraint_coo(const scopi_container<dim>& particles,
                                          const std::vector<neighbor<dim>>& contacts,
                                          std::size_t firstCol);
        void set_gamma(const std::vector<neighbor<dim>>& contacts_new);
        void update_gamma(const std::vector<neighbor<dim>>& contacts, xt::xtensor<double, 1> lambda);
        std::size_t number_row_matrix(const std::vector<neighbor<dim>>& contacts);
        void create_vector_distances(const std::vector<neighbor<dim>>& contacts);

        std::size_t m_nparticles;
        double m_dt;

        std::vector<int> m_A_rows;
        std::vector<int> m_A_cols;
        std::vector<double> m_A_values;
        xt::xtensor<double, 1> m_distances;

        std::vector<neighbor<dim>> m_contacts_old;
        std::vector<double> m_gamma;
        std::vector<double> m_gamma_old;
        std::size_t m_nb_gamma_neg;
        double m_tol;
    };

    template<std::size_t dim>
    void MatrixOptimSolverViscosity<dim>::create_matrix_constraint_coo(const scopi_container<dim>& particles,
                                                              const std::vector<neighbor<dim>>& contacts,
                                                              std::size_t firstCol)
    {
        std::size_t active_offset = particles.nb_inactive();
        std::size_t u_size = 3*contacts.size()*2;
        std::size_t w_size = 3*contacts.size()*2;
        m_A_rows.resize(2*(u_size + w_size));
        m_A_cols.resize(2*(u_size + w_size));
        m_A_values.resize(2*(u_size + w_size));

        std::size_t ic = 0;
        std::size_t index = 0;
        for (auto &c: contacts)
        {
            if (c.i >= active_offset)
            {
                for (std::size_t d = 0; d < 3; ++d)
                {
                    m_A_rows[index] = ic;
                    m_A_cols[index] = firstCol + (c.i - active_offset)*3 + d;
                    m_A_values[index] = -m_dt*c.nij[d];
                    index++;
                    if (m_gamma[ic] < -m_tol)
                    {
                        m_A_rows[index] = contacts.size() + ic;
                        m_A_cols[index] = firstCol + (c.i - active_offset)*3 + d;
                        m_A_values[index] = m_dt*c.nij[d];
                        index++;
                    }
                }
            }

            if (c.j >= active_offset)
            {
                for (std::size_t d = 0; d < 3; ++d)
                {
                    m_A_rows[index] = ic;
                    m_A_cols[index] = firstCol + (c.j - active_offset)*3 + d;
                    m_A_values[index] = m_dt*c.nij[d];
                    index++;
                    if (m_gamma[ic] < -m_tol)
                    {
                        m_A_rows[index] = contacts.size() + ic;
                        m_A_cols[index] = firstCol + (c.j - active_offset)*3 + d;
                        m_A_values[index] = -m_dt*c.nij[d];
                        index++;
                    }
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
                    m_A_rows[index] = ic;
                    m_A_cols[index] = firstCol + 3*particles.nb_active() + 3*ind_part + ip;
                    m_A_values[index] = m_dt*(c.nij[0]*dot(0, ip)+c.nij[1]*dot(1, ip)+c.nij[2]*dot(2, ip));
                    index++;
                    if (m_gamma[ic] < -m_tol)
                    {
                        m_A_rows[index] = contacts.size() + ic;
                        m_A_cols[index] = firstCol + 3*particles.nb_active() + 3*ind_part + ip;
                        m_A_values[index] = -m_dt*(c.nij[0]*dot(0, ip)+c.nij[1]*dot(1, ip)+c.nij[2]*dot(2, ip));
                        index++;
                    }
                }
            }

            if (c.j >= active_offset)
            {
                std::size_t ind_part = c.j - active_offset;
                auto dot = xt::eval(xt::linalg::dot(rj_cross, Rj));
                for (std::size_t ip = 0; ip < 3; ++ip)
                {
                    m_A_rows[index] = ic;
                    m_A_cols[index] = firstCol + 3*particles.nb_active() + 3*ind_part + ip;
                    m_A_values[index] = -m_dt*(c.nij[0]*dot(0, ip)+c.nij[1]*dot(1, ip)+c.nij[2]*dot(2, ip));
                    index++;
                    if (m_gamma[ic] < -m_tol)
                    {
                        m_A_rows[index] = contacts.size() + ic;
                        m_A_cols[index] = firstCol + 3*particles.nb_active() + 3*ind_part + ip;
                        m_A_values[index] = m_dt*(c.nij[0]*dot(0, ip)+c.nij[1]*dot(1, ip)+c.nij[2]*dot(2, ip));
                        index++;
                    }
                }
            }

            ++ic;
        }
        m_A_rows.resize(index);
        m_A_cols.resize(index);
        m_A_values.resize(index);
    }

    template<std::size_t dim>
    MatrixOptimSolverViscosity<dim>::MatrixOptimSolverViscosity(std::size_t nparticles, double dt, double tol)
    : m_nparticles(nparticles)
    , m_dt(dt)
    , m_tol(tol)
    {}

    template<std::size_t dim>
    void MatrixOptimSolverViscosity<dim>::set_gamma(const std::vector<neighbor<dim>>& contacts_new)
    {
        m_gamma.resize(contacts_new.size());
        if(m_contacts_old.size() > 0)
        {
            for (std::size_t index_new = 0; index_new < contacts_new.size(); ++index_new)
            {
                auto cn = contacts_new[index_new];
                std::size_t index_old = 0.;
                auto co = m_contacts_old[index_old];
                while (co.i < cn.i && index_old < m_contacts_old.size())
                {
                    index_old ++;
                    co = m_contacts_old[index_old];
                }
                while (co.j < cn.j && co.i == cn.i)
                {
                    index_old ++;
                    co = m_contacts_old[index_old];
                }
                if(co.i == cn.i && co.j == cn.j)
                    m_gamma[index_new] = m_gamma_old[index_old];
                else
                    m_gamma[index_new] = 0.;
            }
        }
        else
        {
            for (auto& g : m_gamma)
                g = 0.;
        }

        m_nb_gamma_neg = 0;
        for (auto& g : m_gamma)
        {
            if (g < -m_tol)
                m_nb_gamma_neg++;
        }
    }

    template<std::size_t dim>
    void MatrixOptimSolverViscosity<dim>::update_gamma(const std::vector<neighbor<dim>>& contacts, xt::xtensor<double, 1> lambda)
    {
        m_contacts_old = contacts;
        m_gamma_old.resize(m_gamma.size());
        std::size_t nb_gamma_neg = 0;
        for (std::size_t i = 0; i < m_gamma_old.size(); ++i)
        {
            double f_contact;
            if (m_gamma[i] < -m_tol)
            {
                f_contact = lambda(i) - lambda(m_gamma.size() + nb_gamma_neg);
                nb_gamma_neg++;
            }
            else
            {
                f_contact = lambda(i);
            }
            m_gamma_old[i] = std::min(0., m_gamma[i] - m_dt * f_contact);
            if(m_gamma_old[i] > -m_tol)
                m_gamma_old[i] = 0.;
            // if (m_gamma_old[i] > -1e-8)
            //     m_gamma_old[i] = 0.;
            PLOG_WARNING << m_gamma[i] << "    " << contacts[0].dij;
        }
    }

    template<std::size_t dim>
    std::size_t MatrixOptimSolverViscosity<dim>::number_row_matrix(const std::vector<neighbor<dim>>& contacts)
    {
        return contacts.size() + m_nb_gamma_neg;
    }

    template<std::size_t dim>
    void MatrixOptimSolverViscosity<dim>::create_vector_distances(const std::vector<neighbor<dim>>& contacts)
    {
        m_distances = xt::zeros<double>({contacts.size() + m_nb_gamma_neg});
        for (std::size_t i = 0; i < contacts.size(); ++i)
        {
            m_distances[i] = contacts[i].dij;
            if(m_gamma[i] < -m_tol)
            {
                m_distances[contacts.size() + i] = -contacts[i].dij;
            }
        }
    }
}

