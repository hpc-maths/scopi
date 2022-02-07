#pragma once
#include "../crtp.hpp"
#include "../container.hpp"
#include "../objects/neighbor.hpp"
#include <plog/Log.h>
#include "plog/Initializers/RollingFileInitializer.h"

namespace scopi{
    template <class D, std::size_t dim>
    class OptimBase: public crtp_base<D, OptimBase<D, dim>>
    {
    public:
        void run(const std::vector<neighbor<dim>>& contacts, const std::size_t nite);
        auto get_uadapt();
        auto get_wadapt();

    protected:
        OptimBase(scopi_container<dim>& particles, double dt, std::size_t Nactive, std::size_t active_ptr, std::size_t cSize, std::size_t c_dec);
        void setup(const std::vector<neighbor<dim>>& contacts);
        void tear_down();

        scopi_container<dim>& m_particles;
        double m_dt;
        std::size_t m_Nactive;
        std::size_t m_active_ptr;
        double m_mass;
        double m_moment;
        xt::xtensor<double, 1> m_c;
        std::size_t m_c_dec;
        xt::xtensor<double, 1> m_distances;

    private:
        void create_vector_distances(const std::vector<neighbor<dim>>& contacts);
        void create_vector_c();
        int solve_optimization_problem(const std::vector<neighbor<dim>>& contacts);
        int get_nb_active_contacts();
    };

    template<class D, std::size_t dim>
    void OptimBase<D, dim>::run(const std::vector<neighbor<dim>>& contacts, const std::size_t nite)
    {
        tic();
        create_vector_c();
        create_vector_distances(contacts);
        setup(contacts);
        auto duration4 = toc();
        PLOG_INFO << "----> CPUTIME : matrices = " << duration4;

        PLOG_INFO << "----> Create optimization problem " << nite;
        tic();
        auto nbIter = solve_optimization_problem(contacts);
        auto duration5 = toc();
        PLOG_INFO << "----> CPUTIME : solve = " << duration5;
        PLOG_INFO << "iterations : " << nbIter;
        PLOG_INFO << "Contacts: " << contacts.size() << "  active contacts " << get_nb_active_contacts();
        tear_down();
    }


    template<class D, std::size_t dim>
    OptimBase<D, dim>::OptimBase(scopi_container<dim>& particles, double dt, std::size_t Nactive, std::size_t active_ptr, std::size_t cSize, std::size_t c_dec)
    : m_particles(particles)
    , m_dt(dt)
    , m_Nactive(Nactive)
    , m_active_ptr(active_ptr)
    , m_mass(1.)
    , m_moment(0.1)
    , m_c(xt::zeros<double>({cSize}))
    , m_c_dec(c_dec)
    {}

    template<class D, std::size_t dim>
    void OptimBase<D, dim>::create_vector_distances(const std::vector<neighbor<dim>>& contacts)
    {
        m_distances = xt::zeros<double>({contacts.size()});
        for (std::size_t i = 0; i < contacts.size(); ++i)
        {
            m_distances[i] = contacts[i].dij;
        }
        // std::cout << "distances " << distances << std::endl;
    }

    template<class D, std::size_t dim>
    void OptimBase<D, dim>::create_vector_c()
    {
        std::size_t mass_dec = m_c_dec;
        std::size_t moment_dec = mass_dec + 3*m_Nactive;
        for (std::size_t i = 0; i < m_Nactive; ++i)
        {
            for (std::size_t d = 0; d < dim; ++d)
            {
                m_c(mass_dec + 3*i + d) = -m_mass*m_particles.vd()(m_active_ptr + i)[d]; // TODO: add mass into particles
            }
            m_c(moment_dec + 3*i + 2) = -m_moment*m_particles.desired_omega()(m_active_ptr + i);
        }
    }

    template<class D, std::size_t dim>
    int OptimBase<D, dim>::solve_optimization_problem(const std::vector<neighbor<dim>>& contacts)
    {
        return this->derived_cast().solve_optimization_problem_impl(contacts);
    }

    template<class D, std::size_t dim>
    auto OptimBase<D, dim>::get_uadapt()
    {
        return this->derived_cast().get_uadapt_impl();
    }

    template<class D, std::size_t dim>
    auto OptimBase<D, dim>::get_wadapt()
    {
        return this->derived_cast().get_wadapt_impl();
    }

    template<class D, std::size_t dim>
    void OptimBase<D, dim>::setup(const std::vector<neighbor<dim>>& contacts)
    {
        this->derived_cast().setup_impl(contacts);
    }

    template<class D, std::size_t dim>
    void OptimBase<D, dim>::tear_down()
    {
        this->derived_cast().tear_down_impl();
    }

    template<class D, std::size_t dim>
    int OptimBase<D, dim>::get_nb_active_contacts()
    {
        return this->derived_cast().get_nb_active_contacts_impl();
    }
}

