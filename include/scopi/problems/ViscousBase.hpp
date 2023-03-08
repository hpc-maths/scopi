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
    /**
     * @class ViscousBase
     * @brief Shared methods for models with viscosity.
     *
     * For each contact \f$ ij \f$, there is a new variable \f$ \gamma_{ij} \f$.
     * The value of \f$ \gamma_{ij} \f$ dictates the behavior (problem-depedant).
     * \f$ \gamma_{ij} \f$ depends on the time.
     * When there is no contact between particles \c i and \c j, \f$ \gamma_{ij} = 0 \f$ by convention, the variable is not stored.
     *
     * @tparam dim Dimension (2 or 3).
     */
    template<std::size_t dim>
    class ViscousBase
    {
    public:
        /**
         * @brief Returns the number of contacts with \f$ \gamma_{ij} < 0 \f$.
         */
        std::size_t get_nb_gamma_neg() const;
    protected:
        /**
         * @brief Default constructor.
         */
        ViscousBase();

        /**
         * @brief Set \f$ \gamma_{ij}^n \f$ from the previous time step.
         *
         * Look if particles \c i and \c j were already in contact.
         *
         * @param contacts_new [in] Array of contacts at time \f$ t^{n+1} \f$.
         */
        void set_gamma_base(const std::vector<neighbor<dim>>& contacts_new);

        /**
         * @brief Array of contacts at time \f$ t^n \f$.
         */
        std::vector<neighbor<dim>> m_contacts_old;
        /**
         * @brief Array of \f$ \gamma^{n+1} \f$.
         */
        std::vector<double> m_gamma;
        /**
         * @brief Array of \f$ \gamma^{n} \f$.
         */
        std::vector<double> m_gamma_old;
        /**
         * @brief Number of contacts at time \f$ t^{n+1} \f$ with \f$ \gamma_{ij} < 0 \f$.
         */
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
                while (co.i < cn.i && index_old < m_contacts_old.size()-1)
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

