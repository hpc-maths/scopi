#pragma once

#include <array>
#include <functional>
#include <map>
#include <memory>
#include <vector>

#include <xtensor/xadapt.hpp>

#include "object/base.hpp"

namespace scopi
{

    ////////////////////////////////
    // scopi_container definition //
    ////////////////////////////////
    template<std::size_t dim>
    class scopi_container
    {
    public:

        using default_container_type = std::array<double, dim>;
        using position_type = default_container_type;
        using velocity_type = default_container_type;
        using force_type = default_container_type;
        using desired_velocity_type = default_container_type;

        std::unique_ptr<object<dim, false>> operator[](std::size_t i);

        void push_back(const object<dim>& s,
                       const velocity_type& v,
                       const desired_velocity_type& dv,
                       const force_type& f);

        void reserve(std::size_t size);

        const auto pos() const;
        auto pos();

        const auto f() const;
        auto f();

        const auto v() const;
        auto v();

        const auto vd() const;
        auto vd();

        std::size_t size() const;

    private:

        std::map<std::size_t, std::unique_ptr<base_constructor<dim>>> m_shape_map;
        std::vector<std::array<double, dim>> m_positions;  //pos()
        std::vector<std::array<double, dim>> m_forces;  // f()
        std::vector<std::array<double, dim>> m_velocities;  // v()
        std::vector<std::array<double, dim>> m_desired_velocities;  // vd()
        std::vector<std::size_t> m_shapes_id;
        std::vector<std::size_t> m_offset;
    };

    template<std::size_t dim>
    std::unique_ptr<object<dim, false>> scopi_container<dim>::operator[](std::size_t i)
    {
        return (*m_shape_map[m_shapes_id[i]])(&m_positions[m_offset[i]]);
    }

    template<std::size_t dim>
    void scopi_container<dim>::push_back(const object<dim>& s,
                                         const velocity_type& v,
                                         const desired_velocity_type& dv,
                                         const force_type& f)
    {
        if (m_offset.empty())
        {
            m_offset = {0, s.size()};
        }
        else
        {
            m_offset.push_back(m_offset.back() + s.size());
        }

        for(std::size_t i = 0; i< s.size(); ++i)
        {
            m_positions.push_back(s.pos(i));
            m_velocities.push_back(v);
            m_desired_velocities.push_back(dv);
            m_forces.push_back(f);
        }

        auto it = m_shape_map.find(s.hash());
        if (it == m_shape_map.end())
        {
            m_shape_map.insert(std::make_pair(s.hash(), std::move(s.construct())));
        }

        m_shapes_id.push_back(s.hash());
    }

    template<std::size_t dim>
    void scopi_container<dim>::reserve(std::size_t size)
    {
        m_positions.reserve(size);
        m_velocities.reserve(size);
        m_desired_velocities.reserve(size);
        m_forces.reserve(size);
        m_offset.reserve(size+1);
        m_shapes_id.reserve(size);
    }

    template<std::size_t dim>
    std::size_t scopi_container<dim>::size() const
    {
        return m_shapes_id.size();
    }

    // position

    template<std::size_t dim>
    const auto scopi_container<dim>::pos() const
    {
        return xt::adapt(reinterpret_cast<double*>(m_positions.data()), {m_positions.size(), dim});
    }

    template<std::size_t dim>
    auto scopi_container<dim>::pos()
    {
        return xt::adapt(reinterpret_cast<double*>(m_positions.data()), {m_positions.size(), dim});
    }

    // velocity

    template<std::size_t dim>
    const auto scopi_container<dim>::v() const
    {
        return xt::adapt(reinterpret_cast<double*>(m_velocities.data()), {m_velocities.size(), dim});
    }

    template<std::size_t dim>
    auto scopi_container<dim>::v()
    {
        return xt::adapt(reinterpret_cast<double*>(m_velocities.data()), {m_velocities.size(), dim});
    }

    // desired velocity

    template<std::size_t dim>
    const auto scopi_container<dim>::vd() const
    {
        return xt::adapt(reinterpret_cast<double*>(m_desired_velocities.data()), {m_desired_velocities.size(), dim});
    }

    template<std::size_t dim>
    auto scopi_container<dim>::vd()
    {
        return xt::adapt(reinterpret_cast<double*>(m_desired_velocities.data()), {m_desired_velocities.size(), dim});
    }

    // force

    template<std::size_t dim>
    const auto scopi_container<dim>::f() const
    {
        return xt::adapt(reinterpret_cast<double*>(m_forces.data()), {m_forces.size(), dim});
    }

    template<std::size_t dim>
    auto scopi_container<dim>::f()
    {
        return xt::adapt(reinterpret_cast<double*>(m_forces.data()), {m_forces.size(), dim});
    }

}
