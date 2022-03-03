#pragma once

#include <plog/Log.h>
#include "plog/Initializers/RollingFileInitializer.h"

#include "../container.hpp"
#include "../crtp.hpp"
#include "../objects/neighbor.hpp"
#include "../utils.hpp"

namespace scopi{
    template <class Derived>
    class OptimBase
    {
    public:
        template<std::size_t dim>
        void run(const scopi_container<dim>& particles, const std::vector<neighbor<dim>>& contacts, const std::size_t nite);
        auto get_uadapt();
        auto get_wadapt();

    protected:
        OptimBase(std::size_t nparts, double dt, std::size_t cSize, std::size_t c_dec);

        std::size_t m_nparts;
        double m_dt;
        xt::xtensor<double, 1> m_c;
        std::size_t m_c_dec;
        xt::xtensor<double, 1> m_distances;

    private:
        template<std::size_t dim>
        void create_vector_distances(const std::vector<neighbor<dim>>& contacts);

        template<std::size_t dim>
        void create_vector_c(const scopi_container<dim>& particles);

        template<std::size_t dim>
        int solve_optimization_problem(const scopi_container<dim>& particles,
                                       const std::vector<neighbor<dim>>& contacts);

        int get_nb_active_contacts() const;
    };

    template<class Derived>
    template<std::size_t dim>
    void OptimBase<Derived>::run(const scopi_container<dim>& particles, const std::vector<neighbor<dim>>& contacts, const std::size_t)
    {
        tic();
        create_vector_c(particles);
        create_vector_distances(contacts);
        auto duration = toc();
        PLOG_INFO << "----> CPUTIME : vectors = " << duration;

        auto nbIter = solve_optimization_problem(particles, contacts);
        PLOG_INFO << "iterations : " << nbIter;
        PLOG_INFO << "Contacts: " << contacts.size() << "  active contacts " << get_nb_active_contacts();
    }


    template<class Derived>
    OptimBase<Derived>::OptimBase(std::size_t nparts, double dt, std::size_t cSize, std::size_t c_dec)
    : m_nparts(nparts)
    , m_dt(dt)
    , m_c(xt::zeros<double>({cSize}))
    , m_c_dec(c_dec)
    {}

    template<class Derived>
    template<std::size_t dim>
    void OptimBase<Derived>::create_vector_distances(const std::vector<neighbor<dim>>& contacts)
    {
        m_distances = xt::zeros<double>({contacts.size()});
        for (std::size_t i = 0; i < contacts.size(); ++i)
        {
            m_distances[i] = contacts[i].dij;
        }
    }

    template<class Derived>
    template<std::size_t dim>
    void OptimBase<Derived>::create_vector_c(const scopi_container<dim>& particles)
    {
        std::size_t mass_dec = m_c_dec;
        std::size_t moment_dec = mass_dec + 3*particles.nb_active();

        auto active_offset = particles.nb_inactive();

        auto desired_velocity = particles.vd();
        auto desired_omega = particles.desired_omega();

        for (std::size_t i = 0; i < particles.nb_active(); ++i)
        {
            for (std::size_t d = 0; d < dim; ++d)
            {
                m_c(mass_dec + 3*i + d) = -particles.m()(active_offset + i)*desired_velocity(i + active_offset)[d];
            }
            auto omega = get_omega(desired_omega(i + active_offset));
            for (std::size_t d = 0; d < 3; ++d)
            {
                m_c(moment_dec + 3*i + d) = -particles.j()(active_offset + i)(d)*omega(d);
            }
        }
    }

    template<class Derived>
    template<std::size_t dim>
    int OptimBase<Derived>::solve_optimization_problem(const scopi_container<dim>& particles,
                                                 const std::vector<neighbor<dim>>& contacts)
    {
        return static_cast<Derived&>(*this).solve_optimization_problem_impl(particles, contacts);
    }

    template<class Derived>
    auto OptimBase<Derived>::get_uadapt()
    {
        auto data = static_cast<Derived&>(*this).uadapt_data();
        return xt::adapt(reinterpret_cast<double*>(data), {this->m_nparts, 3UL});
    }

    template<class Derived>
    auto OptimBase<Derived>::get_wadapt()
    {
        auto data = static_cast<Derived&>(*this).wadapt_data();
        return xt::adapt(reinterpret_cast<double*>(data), {this->m_nparts, 3UL});
    }

    template<class Derived>
    int OptimBase<Derived>::get_nb_active_contacts() const
    {
        return static_cast<const Derived&>(*this).get_nb_active_contacts_impl();
    }
}

