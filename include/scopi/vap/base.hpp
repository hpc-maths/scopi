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
        this->derived_cast().update_velocity_impl(particles, uadapt, wadapt);
        auto duration = toc();
        PLOG_INFO << "----> CPUTIME : update vap = " << duration;
    }
}
