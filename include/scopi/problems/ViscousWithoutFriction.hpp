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

namespace scopi
{
    template<std::size_t dim>
    class ViscousWithoutFriction;

    template<std::size_t dim>
    class ProblemParams<ViscousWithoutFriction<dim>>
    {
    public:
        ProblemParams();
        ProblemParams(const ProblemParams<ViscousWithoutFriction<dim>>& params);

        double m_tol;
    };

    template<std::size_t dim>
    class ViscousWithoutFriction: public ProblemBase
                                , public ViscousBase<dim>
    {
    public:
        ViscousWithoutFriction(std::size_t nparts, double dt, const ProblemParams<ViscousWithoutFriction<dim>>& problem_params);

        void create_matrix_constraint_coo(const scopi_container<dim>& particles,
                                          const std::vector<neighbor<dim>>& contacts,
                                          std::size_t firstCol);
        void update_gamma(const std::vector<neighbor<dim>>& contacts,
                          xt::xtensor<double, 1> lambda);
        std::size_t number_row_matrix(const std::vector<neighbor<dim>>& contacts,
                                      const scopi_container<dim>& particles);
        void create_vector_distances(const std::vector<neighbor<dim>>& contacts,
                                     const scopi_container<dim>& particles);

        std::size_t get_nb_gamma_min();

        void set_gamma(const std::vector<neighbor<dim>>& contacts_new);

        void matrix_free_gemv_A(const neighbor<dim>& c,
                                const scopi_container<dim>& particles,
                                const xt::xtensor<double, 1>& U,
                                xt::xtensor<double, 1>& R,
                                std::size_t active_offset,
                                std::size_t row);
        void matrix_free_gemv_transpose_A(const neighbor<dim>& c,
                                          const scopi_container<dim>& particles,
                                          const xt::xtensor<double, 1>& L,
                                          xt::xtensor<double, 1>& U,
                                          std::size_t active_offset,
                                          std::size_t row);

    private:
        ProblemParams<ViscousWithoutFriction<dim>> m_params;
    };

    template<std::size_t dim>
    void ViscousWithoutFriction<dim>::create_matrix_constraint_coo(const scopi_container<dim>& particles,
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
                    if (this->m_gamma[ic] < -m_params.m_tol)
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
                    if (this->m_gamma[ic] < -m_params.m_tol)
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
                    if (this->m_gamma[ic] < -m_params.m_tol)
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
                    if (this->m_gamma[ic] < -m_params.m_tol)
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
    ViscousWithoutFriction<dim>::ViscousWithoutFriction(std::size_t nparticles, double dt, const ProblemParams<ViscousWithoutFriction<dim>>& problem_params)
    : ProblemBase(nparticles, dt)
    , ViscousBase<dim>()
    , m_params(problem_params)
    {}

    template<std::size_t dim>
    void ViscousWithoutFriction<dim>::set_gamma(const std::vector<neighbor<dim>>& contacts_new)
    {
        this->set_gamma_base(contacts_new);
        this->m_nb_gamma_neg = 0;
        for (auto& g : this->m_gamma)
        {
            if (g < -m_params.m_tol)
                this->m_nb_gamma_neg++;
        }
    }

    template<std::size_t dim>
    void ViscousWithoutFriction<dim>::update_gamma(const std::vector<neighbor<dim>>& contacts,
                                                   xt::xtensor<double, 1> lambda)
    {
        this->m_contacts_old = contacts;
        this->m_gamma_old.resize(this->m_gamma.size());
        std::size_t ind_gamma_neg = 0;
        for (std::size_t i = 0; i < this->m_gamma_old.size(); ++i)
        {
            double f_contact;
            if (this->m_gamma[i] < -m_params.m_tol)
            {
                f_contact = lambda(i) - lambda(this->m_gamma.size() + ind_gamma_neg);
                ind_gamma_neg++;
            }
            else
            {
                f_contact = lambda(i);
            }
            this->m_gamma_old[i] = std::min(0., this->m_gamma[i] - this->m_dt * f_contact);
            if (this->m_gamma_old[i] > -m_params.m_tol)
                this->m_gamma_old[i] = 0.;
            // if (this->m_gamma_old[i] > -m_params.m_tol)
            //     this->m_gamma_old[i] = 0.;

            PLOG_WARNING << this->m_gamma[i];
        }
    }


    template<std::size_t dim>
    std::size_t ViscousWithoutFriction<dim>::number_row_matrix(const std::vector<neighbor<dim>>& contacts,
                                                               const scopi_container<dim>&)
    {
        return contacts.size() + this->m_nb_gamma_neg;
    }

    template<std::size_t dim>
    void ViscousWithoutFriction<dim>::create_vector_distances(const std::vector<neighbor<dim>>& contacts,
                                                              const scopi_container<dim>&)
    {
        this->m_distances = xt::zeros<double>({contacts.size() + this->m_nb_gamma_neg});
        for (std::size_t i = 0; i < contacts.size(); ++i)
        {
            this->m_distances[i] = contacts[i].dij;
            if(this->m_gamma[i] < -m_params.m_tol)
            {
                this->m_distances[contacts.size() + i] = -contacts[i].dij;
            }
        }
    }

    template<std::size_t dim>
    std::size_t ViscousWithoutFriction<dim>::get_nb_gamma_min()
    {
        return 0;
    }

    template<std::size_t dim>
    void ViscousWithoutFriction<dim>::matrix_free_gemv_A(const neighbor<dim>& c,
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
                if (this->m_gamma[row] < -m_params.m_tol)
                {
                    R(this->m_gamma.size() + row) -= (this->m_dt*c.nij[d]) * U((c.i - active_offset)*3 + d);
                }
            }
        }

        if (c.j >= active_offset)
        {
            for (std::size_t d = 0; d < 3; ++d)
            {
                R(row) -= (this->m_dt*c.nij[d]) * U((c.j - active_offset)*3 + d);
                if (this->m_gamma[row] < -m_params.m_tol)
                {
                    R(this->m_gamma.size() + row) -= (-this->m_dt*c.nij[d]) * U((c.j - active_offset)*3 + d);
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
                R(row) -= (this->m_dt*(c.nij[0]*dot(0, ip)+c.nij[1]*dot(1, ip)+c.nij[2]*dot(2, ip))) * U(3*this->m_nparticles + 3*ind_part + ip);
                if (this->m_gamma[row] < -m_params.m_tol)
                {
                    R(this->m_gamma.size() + row) -= (-this->m_dt*(c.nij[0]*dot(0, ip)+c.nij[1]*dot(1, ip)+c.nij[2]*dot(2, ip))) * U(3*this->m_nparticles + 3*ind_part + ip);
                }
            }
        }

        if (c.j >= active_offset)
        {
            std::size_t ind_part = c.j - active_offset;
            auto dot = xt::eval(xt::linalg::dot(rj_cross, Rj));
            for (std::size_t ip = 0; ip < 3; ++ip)
            {
                R(row) -= (-this->m_dt*(c.nij[0]*dot(0, ip)+c.nij[1]*dot(1, ip)+c.nij[2]*dot(2, ip))) * U(3*this->m_nparticles + 3*ind_part + ip);
                if (this->m_gamma[row] < -m_params.m_tol)
                {
                    R(this->m_gamma.size() + row) -= (this->m_dt*(c.nij[0]*dot(0, ip)+c.nij[1]*dot(1, ip)+c.nij[2]*dot(2, ip))) * U(3*this->m_nparticles + 3*ind_part + ip);
                }
            }
        }
    }

    template<std::size_t dim>
    void ViscousWithoutFriction<dim>::matrix_free_gemv_transpose_A(const neighbor<dim>& c,
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
                U((c.i - active_offset)*3 + d) += L(row) * (-this->m_dt*c.nij[d]);
                if (this->m_gamma[row] < -m_params.m_tol)
                {
#pragma omp atomic
                    U((c.i - active_offset)*3 + d) += L(this->m_gamma.size() + row) * (this->m_dt*c.nij[d]);
                }
            }
        }

        if (c.j >= active_offset)
        {
            for (std::size_t d = 0; d < 3; ++d)
            {
#pragma omp atomic
                U((c.j - active_offset)*3 + d) += L(row) * (this->m_dt*c.nij[d]);
                if (this->m_gamma[row] < -m_params.m_tol)
                {
#pragma omp atomic
                    U((c.j - active_offset)*3 + d) += L(this->m_gamma.size() + row) * (-this->m_dt*c.nij[d]);
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
#pragma omp atomic
                U(3*this->m_nparticles + 3*ind_part + ip) += L(row) * (this->m_dt*(c.nij[0]*dot(0, ip)+c.nij[1]*dot(1, ip)+c.nij[2]*dot(2, ip)));
                if (this->m_gamma[row] < -m_params.m_tol)
                {
#pragma omp atomic
                    U(3*this->m_nparticles + 3*ind_part + ip) += L(this->m_gamma.size() + row) * (-this->m_dt*(c.nij[0]*dot(0, ip)+c.nij[1]*dot(1, ip)+c.nij[2]*dot(2, ip)));
                }
            }
        }

        if (c.j >= active_offset)
        {
            std::size_t ind_part = c.j - active_offset;
            auto dot = xt::eval(xt::linalg::dot(rj_cross, Rj));
            for (std::size_t ip = 0; ip < 3; ++ip)
            {
#pragma omp atomic
                U(3*this->m_nparticles + 3*ind_part + ip) += L(row) * (-this->m_dt*(c.nij[0]*dot(0, ip)+c.nij[1]*dot(1, ip)+c.nij[2]*dot(2, ip)));
                if (this->m_gamma[row] < -m_params.m_tol)
                {
#pragma omp atomic
                    U(3*this->m_nparticles + 3*ind_part + ip) += L(this->m_gamma.size() + row) * (this->m_dt*(c.nij[0]*dot(0, ip)+c.nij[1]*dot(1, ip)+c.nij[2]*dot(2, ip)));
                }
            }
        }
    }


    template<std::size_t dim>
    ProblemParams<ViscousWithoutFriction<dim>>::ProblemParams()
    : m_tol(1e-6)
    {}

    template<std::size_t dim>
    ProblemParams<ViscousWithoutFriction<dim>>::ProblemParams(const ProblemParams<ViscousWithoutFriction<dim>>& params)
    : m_tol(params.m_tol)
    {}

}

