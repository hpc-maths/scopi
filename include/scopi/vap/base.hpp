#pragma once 

#include "../container.hpp"
#include "../utils.hpp"
#include "../params.hpp"
#include "../objects/neighbor.hpp"
#include <vector>

namespace scopi
{
    template <class D>
    class vap_base: public crtp_base<D>
    {
    public:
        template <std::size_t dim>
        void set_a_priori_velocity(scopi_container<dim>& particles, const std::vector<neighbor<dim>>& contacts_pos, const std::vector<neighbor<dim>>& contacts_neg);
        template <std::size_t dim>
        void update_velocity(scopi_container<dim>& particles, const xt::xtensor<double, 2>& uadapt, const xt::xtensor<double, 2>& wadapt);
        vap_base(std::size_t Nactive, std::size_t active_ptr, double dt);

    protected:
        std::size_t m_Nactive;
        std::size_t m_active_ptr;
        double m_dt;

    };

    template <std::size_t dim>
    void update_omega(scopi_container<dim>& particles, std::size_t i, const xt::xtensor<double, 2>& wadapt);

    template <class D>
    vap_base<D>::vap_base(std::size_t Nactive, std::size_t active_ptr, double dt)
    : m_Nactive(Nactive)
    , m_active_ptr(active_ptr)
    , m_dt(dt)
    {}

    template <class D>
    template <std::size_t dim>
    void vap_base<D>::set_a_priori_velocity(scopi_container<dim>& particles, const std::vector<neighbor<dim>>& contacts_pos, const std::vector<neighbor<dim>>& contacts_neg)
    {
        tic();
        this->derived_cast().set_a_priori_velocity_impl(particles, contacts_pos, contacts_neg);
        auto duration = toc();
        PLOG_INFO << "----> CPUTIME : set vap = " << duration;
    }

    template <class D>
    template <std::size_t dim>
    void vap_base<D>::update_velocity(scopi_container<dim>& particles, const xt::xtensor<double, 2>& uadapt, const xt::xtensor<double, 2>& wadapt)
    {
        tic();
        for (std::size_t i=0; i<m_Nactive; ++i)
        {
            for (std::size_t d=0; d<dim; ++d)
            {
                particles.v()(i + m_active_ptr)(d) = uadapt(i, d);
            }
            update_omega(particles, i, wadapt);
        }
        auto duration = toc();
        PLOG_INFO << "----> CPUTIME : update vap = " << duration;
    }

    template<>
    void update_omega(scopi_container<2>& particles, std::size_t i, const xt::xtensor<double, 2>& wadapt)
    {
        particles.omega()(i + particles.nb_inactive()) = wadapt(i, 2);
    }

    template<>
    void update_omega(scopi_container<3>& particles, std::size_t i, const xt::xtensor<double, 2>& wadapt)
    {
        for (std::size_t d = 0; d < 3; ++d)
        {
            particles.omega()(i + particles.nb_inactive())(d) = wadapt(i, d);
        }
    }
}
