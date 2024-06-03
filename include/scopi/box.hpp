#pragma once

#include <array>

namespace scopi
{
    template <std::size_t dim>
    class BoxDomain
    {
      public:

        BoxDomain();
        BoxDomain(const std::array<double, dim>& min_corner, const std::array<double, dim>& max_corner);

        BoxDomain& with_periodicity();
        BoxDomain& with_periodicity(std::size_t axis);
        BoxDomain& with_periodicity(const std::initializer_list<::size_t>& axis);

        const std::array<bool, dim>& is_periodic() const;
        bool is_periodic(std::size_t axis) const;

        const std::array<double, dim>& min_corner() const;
        const std::array<double, dim>& max_corner() const;

        double lower_bound(std::size_t axis) const;
        double upper_bound(std::size_t axis) const;

      private:

        std::array<double, dim> m_min_corner;
        std::array<double, dim> m_max_corner;
        std::array<bool, dim> m_periodic;
    };

    template <std::size_t dim>
    BoxDomain<dim>::BoxDomain()
    {
        m_min_corner.fill(0.0);
        m_max_corner.fill(0.0);
        m_periodic.fill(false);
    }

    template <std::size_t dim>
    BoxDomain<dim>::BoxDomain(const std::array<double, dim>& min_corner, const std::array<double, dim>& max_corner)
        : m_min_corner(min_corner)
        , m_max_corner(max_corner)
    {
        m_periodic.fill(false);
    }

    template <std::size_t dim>
    BoxDomain<dim>& BoxDomain<dim>::with_periodicity()
    {
        m_periodic.fill(true);
        return *this;
    }

    template <std::size_t dim>
    BoxDomain<dim>& BoxDomain<dim>::with_periodicity(std::size_t axis)
    {
        m_periodic[axis] = true;
        return *this;
    }

    template <std::size_t dim>
    BoxDomain<dim>& BoxDomain<dim>::with_periodicity(const std::initializer_list<::size_t>& axis)
    {
        for (auto a : axis)
        {
            m_periodic[a] = true;
        }
        return *this;
    }

    template <std::size_t dim>
    const std::array<bool, dim>& BoxDomain<dim>::is_periodic() const
    {
        return m_periodic;
    }

    template <std::size_t dim>
    bool BoxDomain<dim>::is_periodic(std::size_t axis) const
    {
        return m_periodic[axis];
    }

    template <std::size_t dim>
    const std::array<double, dim>& BoxDomain<dim>::min_corner() const
    {
        return m_min_corner;
    }

    template <std::size_t dim>
    const std::array<double, dim>& BoxDomain<dim>::max_corner() const
    {
        return m_max_corner;
    }

    template <std::size_t dim>
    double BoxDomain<dim>::lower_bound(std::size_t axis) const
    {
        return m_min_corner[axis];
    }

    template <std::size_t dim>
    double BoxDomain<dim>::upper_bound(std::size_t axis) const
    {
        return m_max_corner[axis];
    }
}
