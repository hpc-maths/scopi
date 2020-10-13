#pragma once

#include <array>
#include <functional>
#include <map>
#include <memory>
#include <vector>

#include <xtensor/xadapt.hpp>

#include "shape/base.hpp"
#include "shape/globule.hpp"

namespace scopi
{

    ////////////////////////////////
    // scopi_container definition //
    ////////////////////////////////
    template<std::size_t dim>
    class scopi_container
    {
    public:

        auto operator[](std::size_t i);

        void push_back(object<dim>& s);

        void reserve(std::size_t size);

        const auto pos() const;
        auto pos();

        std::size_t size() const;

    private:

        std::map<std::size_t, std::shared_ptr<base_constructor>> m_shape_map;
        std::vector<std::array<double, dim>> m_positions;
        std::vector<std::array<double, dim>> m_forces;
        std::vector<std::size_t> m_shapes_id;
        std::vector<std::size_t> m_offset;
    };

    template<std::size_t dim>
    auto scopi_container<dim>::operator[](std::size_t i)
    {
        return (*m_shape_map[m_shapes_id[i]])(&m_positions[m_offset[i]]);
    }

    template<std::size_t dim>
    void scopi_container<dim>::push_back(object<dim>& s)
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
        }
 
        auto it = m_shape_map.find(s.hash());
        if (it == m_shape_map.end())
        {
            m_shape_map.insert({s.hash(), s.construct()});
        }

        m_shapes_id.push_back(s.hash());
    }

    template<std::size_t dim>
    void scopi_container<dim>::reserve(std::size_t size)
    {
        m_positions.reserve(size);
        m_offset.reserve(size+1);
        m_shapes_id.reserve(size);
    }

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

    template<std::size_t dim>
    std::size_t scopi_container<dim>::size() const
    {
        return m_shapes_id.size();
    }
}