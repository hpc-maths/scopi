#pragma once

#include <cstddef>
#include <memory>
#include <plog/Log.h>
#include "plog/Initializers/RollingFileInitializer.h"
#include <vector>
#include <xtensor/xtensor.hpp>

#include "../container.hpp"
#include "../quaternion.hpp"
#include "../objects/neighbor.hpp"
#include "../utils.hpp"
#include "../objects/methods/matrix_particles.hpp"

#include "../params/ProblemParams.hpp"
#include "ProblemBase.hpp"

namespace scopi
{
    class ViscousGlobule;

    template<>
    class ProblemParams<ViscousGlobule>
    {};

    class ViscousGlobule: public ProblemBase
    {
    public:
        ViscousGlobule(std::size_t nparts, double dt, const ProblemParams<ViscousGlobule>& problem_params);

        template <std::size_t dim>
        void create_matrix_constraint_coo(scopi_container<dim>& particles,
                                          const std::vector<neighbor<dim>>& contacts,
                                          std::size_t firstCol);
        template <std::size_t dim>
        std::size_t number_row_matrix(const std::vector<neighbor<dim>>& contacts,
                                      scopi_container<dim>& particles);
        template <std::size_t dim>
        void create_vector_distances(const std::vector<neighbor<dim>>& contacts,
                                     scopi_container<dim>& particles);

    private:
        template <std::size_t dim>
        std::size_t number_extra_contacts(scopi_container<dim>& particles);
    };

    template<std::size_t dim>
    void ViscousGlobule::create_matrix_constraint_coo(scopi_container<dim>& particles,
                                                      const std::vector<neighbor<dim>>& contacts,
                                                      std::size_t firstCol)
    {
        std::size_t active_offset = particles.nb_inactive();
        std::size_t nb_row = contacts.size() + number_extra_contacts(particles);
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
                    this->m_A_rows[index] = ic;
                    this->m_A_cols[index] = firstCol + (c.i - active_offset)*3 + d;
                    this->m_A_values[index] = -this->m_dt*c.nij[d];
                    index++;
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
                }
            }
            ++ic;
        }

        // TODO obstacles
        std::size_t nb_previous_constraints = contacts.size();
        for (std::size_t i = 0; i < particles.size(); ++i)
        {
            auto mat_loc = matrix_per_particle_dispatcher<dim>::dispatch(*particles[i]);
            for (std::size_t c = 0; c < particles[i]->size()-1; ++c)
            {
                // TODO use xtensor's function instead of a loop
                // D <= 0, cols u
                for (std::size_t j = c*12; j < c*12+6; ++j)
                {
                    this->m_A_rows[index] = nb_previous_constraints;
                    this->m_A_cols[index] = firstCol + 3*i*particles[i]->size() + mat_loc(j, 1);
                    this->m_A_values[index] = this->m_dt * mat_loc(j, 2);
                    index++;
                }
                // D <= 0, cols w
                for (std::size_t j = c*12+6; j < c*12+2*6; ++j)
                {
                    this->m_A_rows[index] = nb_previous_constraints;
                    this->m_A_cols[index] = firstCol + 3*this->m_nparticles + 3*i*particles[i]->size() + mat_loc(j, 1);
                    this->m_A_values[index] = this->m_dt * mat_loc(j, 2);
                    index++;
                }
                nb_previous_constraints ++;
            }
        }
    }

    template<std::size_t dim>
    std::size_t ViscousGlobule::number_row_matrix(const std::vector<neighbor<dim>>& contacts,
                                                  scopi_container<dim>& particles)
    {
        return contacts.size() + number_extra_contacts(particles);
    }

    template<std::size_t dim>
    void ViscousGlobule::create_vector_distances(const std::vector<neighbor<dim>>& contacts,
                                                 scopi_container<dim>& particles)
    {
        this->m_distances = xt::zeros<double>({contacts.size() + number_extra_contacts(particles)});
        for (std::size_t i = 0; i < contacts.size(); ++i)
        {
            this->m_distances[i] = contacts[i].dij;
        }

        std::size_t nb_previous_constraints = contacts.size();
        for (std::size_t i = 0; i < particles.size(); ++i)
        {
            auto dist_loc = distances_per_particle_dispatcher<dim>::dispatch(*particles[i]);
            // TODO use xtensor's function instead of a loop
            for (std::size_t j = 0; j < dist_loc.size(); ++j)
            {
                this->m_distances[nb_previous_constraints + j ] = dist_loc(j);
            }
            nb_previous_constraints += dist_loc.size();
        }
    }

    template <std::size_t dim>
    std::size_t ViscousGlobule::number_extra_contacts(scopi_container<dim>& particles)
    {
        std::size_t nb_extra_contacts = 0;
        for (std::size_t i = 0; i < particles.size(); ++i)
        {
            nb_extra_contacts += particles[i]->size()-1;
        }
        return nb_extra_contacts;
    }
}

