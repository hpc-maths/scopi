#pragma once

#include <plog/Log.h>
#include "plog/Initializers/RollingFileInitializer.h"
#include <vector>
#include <xtensor/xtensor.hpp>

#include "../container.hpp"
#include "../quaternion.hpp"
#include "../objects/neighbor.hpp"
#include "../utils.hpp"

#include "ProblemBase.hpp"
#include "ViscousBase.hpp"
#include "WithoutFrictionBase.hpp"

namespace scopi
{
    template<std::size_t dim>
    class ViscousWithoutFriction: public ProblemBase
                                , public ViscousBase<ViscousWithoutFriction<dim>, dim>
                                , public WithoutFrictionBase
    {
    public:
        using base_type = ViscousBase<ViscousWithoutFriction, dim>;

        ViscousWithoutFriction(std::size_t nparts, double dt);

        void create_matrix_constraint_coo_impl(const scopi_container<dim>& particles,
                                               const std::vector<neighbor<dim>>& contacts,
                                               std::size_t firstCol);
        void update_gamma_impl(const std::vector<neighbor<dim>>& contacts,
                               xt::xtensor<double, 1> lambda,
                               const scopi_container<dim>& particles,
                               const xt::xtensor<double, 2>& u);
        std::size_t number_row_matrix_impl(const std::vector<neighbor<dim>>& contacts);
        void create_vector_distances_impl(const std::vector<neighbor<dim>>& contacts);

        std::size_t get_nb_gamma_min_impl();

    protected:
        void set_gamma(const std::vector<neighbor<dim>>& contacts_new);
    };

    template<std::size_t dim>
    void ViscousWithoutFriction<dim>::create_matrix_constraint_coo_impl(const scopi_container<dim>& particles,
                                                                        const std::vector<neighbor<dim>>& contacts,
                                                                        std::size_t firstCol)
    {
        std::size_t active_offset = particles.nb_inactive();
        std::size_t u_size = 3*contacts.size()*2;
        std::size_t w_size = 3*contacts.size()*2;
        this->m_A_rows.resize(2*(u_size + w_size));
        this->m_A_cols.resize(2*(u_size + w_size));
        this->m_A_values.resize(2*(u_size + w_size));

        std::size_t ic = 0;
        std::size_t index = 0;
        for (auto &c: contacts)
        {
            if (c.i >= active_offset)
            {
                for (std::size_t d = 0; d < 3; ++d)
                {
                    this->m_A_rows[index] = ic;
                    this->m_A_cols[index] = firstCol + (c.i - active_offset)*3 + d;
                    this->m_A_values[index] = -this->m_dt*c.nij[d];
                    index++;
                    if (this->m_gamma[ic] < -this->m_tol)
                    {
                        this->m_A_rows[index] = contacts.size() + ic;
                        this->m_A_cols[index] = firstCol + (c.i - active_offset)*3 + d;
                        this->m_A_values[index] = this->m_dt*c.nij[d];
                        index++;
                    }
                }
            }

            if (c.j >= active_offset)
            {
                for (std::size_t d = 0; d < 3; ++d)
                {
                    this->m_A_rows[index] = ic;
                    this->m_A_cols[index] = firstCol + (c.j - active_offset)*3 + d;
                    this->m_A_values[index] = this->m_dt*c.nij[d];
                    index++;
                    if (this->m_gamma[ic] < -this->m_tol)
                    {
                        this->m_A_rows[index] = contacts.size() + ic;
                        this->m_A_cols[index] = firstCol + (c.j - active_offset)*3 + d;
                        this->m_A_values[index] = -this->m_dt*c.nij[d];
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
                    this->m_A_rows[index] = ic;
                    this->m_A_cols[index] = firstCol + 3*this->m_nparticles + 3*ind_part + ip;
                    this->m_A_values[index] = this->m_dt*(c.nij[0]*dot(0, ip)+c.nij[1]*dot(1, ip)+c.nij[2]*dot(2, ip));
                    index++;
                    if (this->m_gamma[ic] < -this->m_tol)
                    {
                        this->m_A_rows[index] = contacts.size() + ic;
                        this->m_A_cols[index] = firstCol + 3*this->m_nparticles + 3*ind_part + ip;
                        this->m_A_values[index] = -this->m_dt*(c.nij[0]*dot(0, ip)+c.nij[1]*dot(1, ip)+c.nij[2]*dot(2, ip));
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
                    this->m_A_rows[index] = ic;
                    this->m_A_cols[index] = firstCol + 3*this->m_nparticles + 3*ind_part + ip;
                    this->m_A_values[index] = -this->m_dt*(c.nij[0]*dot(0, ip)+c.nij[1]*dot(1, ip)+c.nij[2]*dot(2, ip));
                    index++;
                    if (this->m_gamma[ic] < -this->m_tol)
                    {
                        this->m_A_rows[index] = contacts.size() + ic;
                        this->m_A_cols[index] = firstCol + 3*this->m_nparticles + 3*ind_part + ip;
                        this->m_A_values[index] = this->m_dt*(c.nij[0]*dot(0, ip)+c.nij[1]*dot(1, ip)+c.nij[2]*dot(2, ip));
                        index++;
                    }
                }
            }
            ++ic;
        }
        this->m_A_rows.resize(index);
        this->m_A_cols.resize(index);
        this->m_A_values.resize(index);
    }


    template<std::size_t dim>
    ViscousWithoutFriction<dim>::ViscousWithoutFriction(std::size_t nparticles, double dt)
    : ProblemBase(nparticles, dt)
    , base_type()
    , WithoutFrictionBase()
    {}

    template<std::size_t dim>
    void ViscousWithoutFriction<dim>::set_gamma(const std::vector<neighbor<dim>>& contacts_new)
    {
        this->set_gamma_base(contacts_new);
        this->m_nb_gamma_neg = 0;
        for (auto& g : this->m_gamma)
        {
            if (g < -this->m_tol)
                this->m_nb_gamma_neg++;
        }
    }

    template<std::size_t dim>
    void ViscousWithoutFriction<dim>::update_gamma_impl(const std::vector<neighbor<dim>>& contacts,
                                                        xt::xtensor<double, 1> lambda,
                                                        const scopi_container<dim>&,
                                                        const xt::xtensor<double, 2>&)
    {
        this->m_contacts_old = contacts;
        this->m_gamma_old.resize(this->m_gamma.size());
        std::size_t ind_gamma_neg = 0;
        for (std::size_t i = 0; i < this->m_gamma_old.size(); ++i)
        {
            double f_contact;
            if (this->m_gamma[i] < -this->m_tol)
            {
                f_contact = lambda(i) - lambda(this->m_gamma.size() + ind_gamma_neg);
                ind_gamma_neg++;
            }
            else
            {
                f_contact = lambda(i);
            }
            this->m_gamma_old[i] = std::min(0., this->m_gamma[i] - this->m_dt * f_contact);
            if (this->m_gamma_old[i] > -this->m_tol)
                this->m_gamma_old[i] = 0.;
            // if (this->m_gamma_old[i] > -this->m_tol)
            //     this->m_gamma_old[i] = 0.;

            PLOG_WARNING << this->m_gamma[i];
        }
    }


    template<std::size_t dim>
    std::size_t ViscousWithoutFriction<dim>::number_row_matrix_impl(const std::vector<neighbor<dim>>& contacts)
    {
        return contacts.size() + this->m_nb_gamma_neg;
    }

    template<std::size_t dim>
    void ViscousWithoutFriction<dim>::create_vector_distances_impl(const std::vector<neighbor<dim>>& contacts)
    {
        this->m_distances = xt::zeros<double>({contacts.size() + this->m_nb_gamma_neg});
        for (std::size_t i = 0; i < contacts.size(); ++i)
        {
            this->m_distances[i] = contacts[i].dij;
            if(this->m_gamma[i] < -this->m_tol)
            {
                this->m_distances[contacts.size() + i] = -contacts[i].dij;
            }
        }
    }

    template<std::size_t dim>
    std::size_t ViscousWithoutFriction<dim>::get_nb_gamma_min_impl()
    {
        return 0;
    }

}

