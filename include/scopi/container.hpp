#pragma once

#include <algorithm>
#include <array>
#include <functional>
#include <map>
#include <memory>
#include <vector>

#include <xtensor/xadapt.hpp>

#include "objects/types/base.hpp"
#include "types.hpp"
#include "property.hpp"
#include "crtp.hpp"

namespace scopi
{

    ////////////////////////////////
    // scopi_container definition //
    ////////////////////////////////
    template<std::size_t dim>
    class scopi_container
    {
    public:

        using position_type = type::position_t<dim>;
        using velocity_type = type::velocity_t<dim>;
        using rotation_type = type::rotation_t<dim>;
        using force_type = type::force_t<dim>;
        using quaternion_type = type::quaternion_t;

        scopi_container();

        std::unique_ptr<object<dim, false>> operator[](std::size_t i);

        void push_back(const object<dim>& s, const property<dim>& p = property<dim>());
        void push_back(std::size_t i, const position_type& pos);

        void reserve(std::size_t size);

        auto pos() const;
        auto pos();

        auto q() const;
        auto q();

        auto f() const;
        auto f();

        auto v() const;
        auto v();

        auto omega() const;
        auto omega();

        auto desired_omega() const;
        auto desired_omega();

        auto vd() const;
        auto vd();

        std::size_t size() const;
        std::size_t nb_active() const;
        std::size_t nb_inactive() const;

        void reset_periodic();

    private:

        std::map<std::size_t, std::unique_ptr<base_constructor<dim>>> m_shape_map;
        std::vector<position_type> m_positions;  // pos()
        std::vector<quaternion_type> m_quaternions;  // q()
        std::vector<force_type> m_forces;  // f()
        std::vector<velocity_type> m_velocities;  // v()
        std::vector<velocity_type> m_desired_velocities;  // vd()
        std::vector<rotation_type> m_omega;  // omega()
        std::vector<rotation_type> m_desired_omega;  // desired_omega()
        std::vector<std::size_t> m_shapes_id;
        std::vector<std::size_t> m_offset;
        std::vector<std::size_t> m_periodic_indices;

        std::size_t m_periodic_ptr;

        std::size_t m_nb_inactive_core_objects;

        bool m_periodic_added;
    };

    template<std::size_t dim>
    scopi_container<dim>::scopi_container()
    : m_periodic_ptr(0)
    , m_nb_inactive_core_objects(0)
    , m_periodic_added(false)
    {}

    template<std::size_t dim>
    std::unique_ptr<object<dim, false>> scopi_container<dim>::operator[](std::size_t i)
    {
        return (*m_shape_map[m_shapes_id[i]])(&m_positions[m_offset[i]], &m_quaternions[m_offset[i]]);
    }

    template<std::size_t dim>
    void scopi_container<dim>::push_back(const object<dim>& s, const property<dim>& p)
    {
        assert(!m_periodic_added);

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
            m_quaternions.push_back(s.q(i));
            m_velocities.push_back(p.velocity());
            m_omega.push_back(p.omega());
            m_desired_omega.push_back(p.desired_omega());
            m_desired_velocities.push_back(p.desired_velocity());
            m_forces.push_back(p.force());
        }

        if (!p.is_active())
        {
            m_nb_inactive_core_objects += s.size();
            if (m_nb_inactive_core_objects != m_positions.size())
            {
                throw std::runtime_error("All the obstacles must be pushed before the active particles.");
            }
        }

        auto it = m_shape_map.find(s.hash());
        if (it == m_shape_map.end())
        {
            m_shape_map.insert(std::make_pair(s.hash(), std::move(s.construct())));
        }

        m_shapes_id.push_back(s.hash());
        m_periodic_ptr++;
    }

    template<std::size_t dim>
    void scopi_container<dim>::push_back(std::size_t i,
                                         const position_type& pos)
    {
        assert(i >= 0 && i < m_positions.size());
        m_periodic_added = true;
        m_offset.push_back(m_offset.back() + m_offset[i+1] - m_offset[i]);

        m_positions.push_back(pos);
        m_quaternions.push_back(m_quaternions[i]);
        m_velocities.push_back(m_velocities[i]);
        m_omega.push_back(m_omega[i]);
        m_desired_omega.push_back(m_desired_omega[i]);
        m_desired_velocities.push_back(m_desired_velocities[i]);
        m_forces.push_back(m_forces[i]);

        m_shapes_id.push_back(m_shapes_id[i]);

        m_periodic_indices.push_back(i);
    }

    template<std::size_t dim>
    void scopi_container<dim>::reserve(std::size_t size)
    {
        m_positions.reserve(size);
        m_quaternions.reserve(size);
        m_velocities.reserve(size);
        m_desired_velocities.reserve(size);
        m_omega.reserve(size);
        m_desired_omega.reserve(size);
        m_forces.reserve(size);
        m_offset.reserve(size+1);
        m_shapes_id.reserve(size);
    }

    template<std::size_t dim>
    std::size_t scopi_container<dim>::size() const
    {
        return m_shapes_id.size();
    }

    template<std::size_t dim>
    std::size_t scopi_container<dim>::nb_active() const
    {
        return m_positions.size() - m_nb_inactive_core_objects;
    }

    template<std::size_t dim>
    std::size_t scopi_container<dim>::nb_inactive() const
    {
        return m_nb_inactive_core_objects;
    }

    // position

    template<std::size_t dim>
    auto scopi_container<dim>::pos() const
    {
        return xt::adapt(reinterpret_cast<const position_type*>(m_positions.data()), {m_positions.size()});
    }

    template<std::size_t dim>
    auto scopi_container<dim>::pos()
    {
        return xt::adapt(reinterpret_cast<position_type*>(m_positions.data()), {m_positions.size()});
    }

    // rotation

    template<std::size_t dim>
    auto scopi_container<dim>::q() const
    {
        return xt::adapt(reinterpret_cast<const quaternion_type*>(m_quaternions.data()), {m_quaternions.size()});
    }

    template<std::size_t dim>
    auto scopi_container<dim>::q()
    {
        return xt::adapt(reinterpret_cast<quaternion_type*>(m_quaternions.data()), {m_quaternions.size()});
    }

    // velocity

    template<std::size_t dim>
    auto scopi_container<dim>::v() const
    {
        return xt::adapt(reinterpret_cast<const velocity_type*>(m_velocities.data()), {m_velocities.size()});
    }

    template<std::size_t dim>
    auto scopi_container<dim>::v()
    {
        return xt::adapt(reinterpret_cast<velocity_type*>(m_velocities.data()), {m_velocities.size()});
    }

    // desired velocity

    template<std::size_t dim>
    auto scopi_container<dim>::vd() const
    {
        return xt::adapt(reinterpret_cast<const velocity_type*>(m_desired_velocities.data()), {m_desired_velocities.size()});
    }

    template<std::size_t dim>
    auto scopi_container<dim>::vd()
    {
        return xt::adapt(reinterpret_cast<velocity_type*>(m_desired_velocities.data()), {m_desired_velocities.size()});
    }

    // omega

    template<std::size_t dim>
    auto scopi_container<dim>::omega() const
    {
        return xt::adapt(reinterpret_cast<const rotation_type*>(m_omega.data()), {m_omega.size()});
    }

    template<std::size_t dim>
    auto scopi_container<dim>::omega()
    {
        return xt::adapt(reinterpret_cast<rotation_type*>(m_omega.data()), {m_omega.size()});
    }

    // desired velocity

    template<std::size_t dim>
    auto scopi_container<dim>::desired_omega() const
    {
        return xt::adapt(reinterpret_cast<const rotation_type*>(m_desired_omega.data()), {m_desired_omega.size()});
    }

    template<std::size_t dim>
    auto scopi_container<dim>::desired_omega()
    {
        return xt::adapt(reinterpret_cast<rotation_type*>(m_desired_omega.data()), {m_desired_omega.size()});
    }

    // force

    template<std::size_t dim>
    auto scopi_container<dim>::f() const
    {
        return xt::adapt(reinterpret_cast<const force_type*>(m_forces.data()), {m_forces.size()});
    }

    template<std::size_t dim>
    auto scopi_container<dim>::f()
    {
        return xt::adapt(reinterpret_cast<force_type*>(m_forces.data()), {m_forces.size()});
    }

    template<std::size_t dim>
    void scopi_container<dim>::reset_periodic()
    {
        m_periodic_indices.clear();
        std::size_t size = m_periodic_ptr;
        m_positions.resize(size);
        m_quaternions.resize(size);
        m_velocities.resize(size);
        m_desired_velocities.resize(size);
        m_omega.resize(size);
        m_desired_omega.resize(size);
        m_forces.resize(size);
        m_offset.resize(size+1);
        m_shapes_id.resize(size);

        m_periodic_added = false;
        m_periodic_ptr = size;
    }

}
