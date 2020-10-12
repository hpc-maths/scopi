#pragma once

#include <array>
#include <functional>
#include <map>
#include <variant>
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

        const auto operator[](std::size_t i) const;
        auto operator[](std::size_t i);

        void push_back(shape<dim>& s);

        void reserve(std::size_t size);

        const auto pos() const;
        auto pos();

    private:

        using my_variant = std::function<shape<dim, false>*(std::array<double, dim>*)>;
        std::map<std::size_t, my_variant> m_shape_map;
        std::vector<std::array<double, dim>> m_positions;
        std::vector<std::array<double, dim>> m_forces;
        std::vector<std::size_t> m_shapes_id;
    };

    template<std::size_t dim>
    const auto scopi_container<dim>::operator[](std::size_t i) const
    {

        return m_shape_map[m_shapes_id[i]](&m_positions[i]);
    }

    template<std::size_t dim>
    auto scopi_container<dim>::operator[](std::size_t i)
    {
        return m_shape_map[m_shapes_id[i]](&m_positions[i]);
    }

    template<std::size_t dim>
    void scopi_container<dim>::push_back(shape<dim>& s)
    {
        m_positions.push_back(s.pos());
        m_shape_map.insert({s.hash(), s.construct()});
        m_shapes_id.push_back(s.hash());
    }

    template<std::size_t dim>
    void scopi_container<dim>::reserve(std::size_t size)
    {
        m_positions.reserve(size);
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
}