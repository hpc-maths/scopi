#pragma once

#include "types.hpp"

namespace scopi
{
    template<std::size_t dim>
    class property
    {
    public:

        using position_type = type::position_t<dim>;
        using velocity_type = type::velocity_t<dim>;
        using rotation_type = type::rotation_t<dim>;
        using force_type = type::force_t<dim>;

        property& velocity(const velocity_type& v);
        const velocity_type velocity() const;
        property& desired_velocity(const velocity_type& dv);
        const velocity_type desired_velocity() const;

        property& omega(const rotation_type& omega);
        const rotation_type omega() const;
        property& desired_omega(const rotation_type& domega);
        const rotation_type desired_omega() const;

        property& force(const force_type& f);
        const force_type force() const;

        property& deactivate();
        property& activate();
        bool is_active() const;

    private:
        velocity_type m_v;
        velocity_type m_dv;
        rotation_type m_omega;
        rotation_type m_domega;
        force_type m_f;
        bool m_active = true;
    };

    template <std::size_t dim>
    auto property<dim>::velocity(const velocity_type& v) -> property&
    {
        m_v = v;
        return *this;
    }

    template <std::size_t dim>
    auto property<dim>::velocity() const -> const velocity_type
    {
        return m_v;
    }

    template <std::size_t dim>
    auto property<dim>::desired_velocity(const velocity_type& dv) -> property&
    {
        m_dv = dv;
        return *this;
    }

    template <std::size_t dim>
    auto property<dim>::desired_velocity() const -> const velocity_type
    {
        return m_dv;
    }

    template <std::size_t dim>
    auto property<dim>::omega(const rotation_type& omega) -> property&
    {
        m_omega = omega;
        return *this;
    }

    template <std::size_t dim>
    auto property<dim>::omega() const -> const rotation_type
    {
        return m_omega;
    }

    template <std::size_t dim>
    auto property<dim>::desired_omega(const rotation_type& domega) -> property&
    {
        m_domega = domega;
        return *this;
    }

    template <std::size_t dim>
    auto property<dim>::desired_omega() const -> const rotation_type
    {
        return m_domega;
    }

    template <std::size_t dim>
    auto property<dim>::force(const force_type& f) -> property&
    {
        m_f = f;
        return *this;
    }

    template <std::size_t dim>
    auto property<dim>::force() const -> const force_type
    {
        return m_f;
    }

    template <std::size_t dim>
    auto property<dim>::activate() -> property&
    {
        m_active = true;
        return *this;
    }

    template <std::size_t dim>
    auto property<dim>::deactivate() -> property&
    {
        m_active = true;
        return *this;
    }

    template <std::size_t dim>
    bool property<dim>::is_active() const
    {
        return m_active;
    }
}