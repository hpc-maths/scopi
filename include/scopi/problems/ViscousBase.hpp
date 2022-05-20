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
    template<class Derived, std::size_t dim>
    class ViscousBase
    {
    protected:
        ViscousBase(std::size_t nparts, double dt);

        void create_matrix_constraint_coo(const scopi_container<dim>& particles,
                                          const std::vector<neighbor<dim>>& contacts,
                                          std::size_t firstCol);
        void set_gamma_base(const std::vector<neighbor<dim>>& contacts_new);
        void update_gamma(const std::vector<neighbor<dim>>& contacts, xt::xtensor<double, 1> lambda);
        std::size_t number_row_matrix(const std::vector<neighbor<dim>>& contacts);
        void create_vector_distances(const std::vector<neighbor<dim>>& contacts);

        std::size_t get_nb_gamma_neg() const;
        std::size_t get_nb_gamma_min();

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

    template<class Derived, std::size_t dim>
    void ViscousBase<Derived, dim>::create_matrix_constraint_coo(const scopi_container<dim>& particles,
                                                              const std::vector<neighbor<dim>>& contacts,
                                                              std::size_t firstCol)
    {
        static_cast<Derived&>(*this).create_matrix_constraint_coo_impl(particles, contacts, firstCol);
    }

    template<class Derived, std::size_t dim>
    ViscousBase<Derived, dim>::ViscousBase(std::size_t nparticles, double dt)
    : m_nparticles(nparticles)
    , m_dt(dt)
    , m_tol(1e-6)
    {}

    template<class Derived, std::size_t dim>
    void ViscousBase<Derived, dim>::set_gamma_base(const std::vector<neighbor<dim>>& contacts_new)
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

    template<class Derived, std::size_t dim>
    void ViscousBase<Derived, dim>::update_gamma(const std::vector<neighbor<dim>>& contacts, xt::xtensor<double, 1> lambda)
    {
        static_cast<Derived&>(*this).update_gamma_impl(contacts, lambda);
    }


    template<class Derived, std::size_t dim>
    std::size_t ViscousBase<Derived, dim>::number_row_matrix(const std::vector<neighbor<dim>>& contacts)
    {
        return static_cast<Derived&>(*this).number_row_matrix_impl(contacts);
    }

    template<class Derived, std::size_t dim>
    void ViscousBase<Derived, dim>::create_vector_distances(const std::vector<neighbor<dim>>& contacts)
    {
        static_cast<Derived&>(*this).create_vector_distances_impl(contacts);
    }

    template<class Derived, std::size_t dim>
    std::size_t ViscousBase<Derived, dim>::get_nb_gamma_neg() const
    {
        return m_nb_gamma_neg;
    }

    template<class Derived, std::size_t dim>
    std::size_t ViscousBase<Derived, dim>::get_nb_gamma_min()
    {
        return static_cast<Derived&>(*this).get_nb_gamma_min_impl();
    }

}

