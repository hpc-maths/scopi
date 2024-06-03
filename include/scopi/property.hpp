#pragma once

#include "types.hpp"

namespace scopi
{
    /**
     * @class property
     * @brief Add properties to the particles.
     *
     * All the methods are applied to one particle, when it is pushed back into the container.
     *
     * @tparam dim Dimension (2 or 3).
     */
    template <std::size_t dim>
    class property
    {
      public:

        /**
         * @brief Alias for the type of the position.
         */
        using position_type = type::position_t<dim>;
        /**
         * @brief Alias for the type of the velocity.
         */
        using velocity_type = type::velocity_t<dim>;
        /**
         * @brief Alias for the type of the rotation.
         */
        using rotation_type = type::rotation_t<dim>;
        /**
         * @brief Alias for the type of the force.
         */
        using force_type = type::force_t<dim>;
        /**
         * @brief Alias for the type of the mass.
         */
        using mass_type = double;
        /**
         * @brief Alias for the type of the moment of inertia.
         */
        using moment_type = type::moment_t<dim>;

        /**
         * @brief Set the velocity.
         *
         * @param v [in] New velocity.
         *
         * @return Property with new velocity.
         */
        property& velocity(const velocity_type& v);
        /**
         * @brief Get the velocity.
         *
         * @return Velocity in the property.
         */
        const velocity_type velocity() const;
        /**
         * @brief Set the desired velocity.
         *
         * @param dv [in] New desired velocity.
         *
         * @return Property with the new desired velocity.
         */
        property& desired_velocity(const velocity_type& dv);
        /**
         * @brief Get the desired velocity.
         *
         * @return Desired velocity in the property.
         */
        const velocity_type desired_velocity() const;

        /**
         * @brief Set the rotation.
         *
         * @param omega [in] New rotation.
         *
         * @return Property with the new rotation.
         */
        property& omega(const rotation_type& omega);
        /**
         * @brief Get the rotation.
         *
         * @return Rotation in the property.
         */
        const rotation_type omega() const;
        /**
         * @brief Set the desired rotation.
         *
         * @param domega [in] New desired rotation.
         *
         * @return Property with the new desired rotation.
         */
        property& desired_omega(const rotation_type& domega);
        /**
         * @brief Get the desired rotation.
         *
         * @return Desired rotation in the property.
         */
        const rotation_type desired_omega() const;

        /**
         * @brief Set the external force.
         *
         * @param f [in] New external force.
         *
         * @return Property with the new external force.
         */
        property& force(const force_type& f);
        /**
         * @brief Get the external force.
         *
         * @return External force in the property.
         */
        const force_type force() const;

        /**
         * @brief Set the mass.
         *
         * @param m [in] New mass.
         *
         * @return Property with the new mass.
         */
        property& mass(const mass_type& m);
        /**
         * @brief Get the mass.
         *
         * @return Mass in the property
         */
        mass_type mass() const;
        /**
         * @brief Set the moment of inertia.
         *
         * @param m [in] New moment of inertia.
         *
         * @return Property with the new moment of inertia.
         */
        property& moment_inertia(const moment_type& m);
        /**
         * @brief Get the moment of inertia.
         *
         * @return Moment of inertia in the property.
         */
        const moment_type moment_inertia() const;

        /**
         * @brief Deactivate a particle (an obstacle).
         *
         * @return Property with deactivated particle.
         */
        property& deactivate();
        /**
         * @brief Activate a particle.
         *
         * @return Property with activated particle.
         */
        property& activate();
        /**
         * @brief Whether the particle is active.
         *
         * @return Whether the particle is active.
         */
        bool is_active() const;

      private:

        /**
         * @brief Velocity.
         */
        velocity_type m_v;
        /**
         * @brief Desired velocity.
         */
        velocity_type m_dv;
        /**
         * @brief Rotation.
         */
        rotation_type m_omega;
        /**
         * @brief Desired rotation.
         */
        rotation_type m_domega;
        /**
         * @brief Force.
         */
        force_type m_f;
        /**
         * @brief Mass.
         */
        mass_type m_m = 1.0;
        /**
         * @brief Moment of inertia
         */
        moment_type m_j;
        /**
         * @brief Whether the particle is active.
         */
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
    auto property<dim>::mass(const mass_type& m) -> property&
    {
        m_m = m;
        return *this;
    }

    template <std::size_t dim>
    auto property<dim>::mass() const -> mass_type
    {
        return m_m;
    }

    template <std::size_t dim>
    auto property<dim>::moment_inertia(const moment_type& j) -> property&
    {
        m_j = j;
        return *this;
    }

    template <std::size_t dim>
    auto property<dim>::moment_inertia() const -> const moment_type
    {
        return m_j;
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
        m_active = false;
        return *this;
    }

    template <std::size_t dim>
    bool property<dim>::is_active() const
    {
        return m_active;
    }
}
