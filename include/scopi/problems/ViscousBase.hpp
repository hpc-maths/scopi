#pragma once

#include <cstddef>
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
    class ViscousBase
    {
    protected:
        ViscousBase();

        void create_matrix_constraint_coo(const scopi_container<dim>& particles,
                                          const std::vector<neighbor<dim>>& contacts,
                                          std::size_t firstCol);
        void set_gamma_base(const std::vector<neighbor<dim>>& contacts_new);
        void update_gamma(const std::vector<neighbor<dim>>& contacts,
                          xt::xtensor<double, 1> lambda);
        std::size_t number_row_matrix(const std::vector<neighbor<dim>>& contacts);
        void create_vector_distances(const std::vector<neighbor<dim>>& contacts);

        std::size_t get_nb_gamma_neg() const;
        std::size_t get_nb_gamma_min();

        std::vector<neighbor<dim>> m_contacts_old;
        std::vector<double> m_gamma;
        std::vector<double> m_gamma_old;
        std::size_t m_nb_gamma_neg;
    };

    template<std::size_t dim>
    ViscousBase<dim>::ViscousBase()
    {}

    template<std::size_t dim>
    void ViscousBase<dim>::set_gamma_base(const std::vector<neighbor<dim>>& contacts_new)
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
    }

    template<std::size_t dim>
    std::size_t ViscousBase<dim>::get_nb_gamma_neg() const
    {
        return m_nb_gamma_neg;
    }
}

