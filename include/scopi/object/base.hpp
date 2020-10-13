#pragma once

#include <array>
#include <vector>

#include "constructor.hpp"

namespace scopi
{

    namespace detail
    {
        template<std::size_t dim, class T>
        const auto get_value(const std::vector<std::array<T, dim>>& t, std::size_t size)
        {
            return xt::adapt(reinterpret_cast<double*>(t.data()), {size, dim});
        }

        template<std::size_t dim, class T>
        auto get_value(std::vector<std::array<T, dim>>& t, std::size_t size)
        {
            return xt::adapt(reinterpret_cast<double*>(t.data()), {size, dim});
        }

        template<std::size_t dim, class T>
        const auto get_value(const std::array<T, dim>* t, std::size_t size)
        {
            return xt::adapt(reinterpret_cast<double*>(t), {size, dim});
        }

        template<std::size_t dim, class T>
        auto get_value(std::array<T, dim>* t, std::size_t size)
        {
            return xt::adapt(reinterpret_cast<double*>(t), {size, dim});
        }

        template<std::size_t dim, class T>
        const std::array<T, dim>& get_value(const std::vector<std::array<T, dim>>& t, std::size_t, std::size_t i)
        {
            return t[i];
        }

        template<std::size_t dim, class T>
        std::array<T, dim>& get_value(std::vector<std::array<T, dim>>& t, std::size_t, std::size_t i)
        {
            return t[i];
        }

        template<std::size_t dim, bool owner>
        struct object_inner_type
        {
            using default_type = typename std::vector<std::array<double, dim>>;
            using position_type = default_type;
            using force_type = default_type;
        };

        template<std::size_t dim>
        struct object_inner_type<dim, false>
        {
            using default_type = typename std::array<double, dim>*;
            using position_type = default_type;
            using force_type = default_type;
        };
    }

    /////////////////////////////////
    // object_container definition //
    /////////////////////////////////
    template<std::size_t dim, bool owner>
    class object_container: public detail::object_inner_type<dim, owner>
    {
    public:

        using inner_types = detail::object_inner_type<dim, owner>;
        using position_type = typename inner_types::position_type;
        using force_type = typename inner_types::force_type;

        object_container(position_type pos, std::size_t size);

        const auto pos() const;
        auto pos();
        const auto pos(std::size_t i) const;
        auto pos(std::size_t i);

        auto force() const;
        auto force();

        std::size_t size() const;

    private:

        position_type m_pos;
        force_type m_force;
        std::size_t m_size;
    };

    /////////////////////////////////////
    // object_container implementation //
    /////////////////////////////////////
    template<std::size_t dim, bool owner>
    inline object_container<dim, owner>::object_container(position_type pos, std::size_t size)
    : m_pos(pos), m_size(size)
    {}

    template<std::size_t dim, bool owner>
    inline const auto object_container<dim, owner>::pos() const
    {
        return detail::get_value(m_pos, m_size);
    }

    template<std::size_t dim, bool owner>
    inline auto object_container<dim, owner>::pos()
    {
        return detail::get_value(m_pos, m_size);
    }

    template<std::size_t dim, bool owner>
    inline const auto object_container<dim, owner>::pos(std::size_t i) const
    {
        return detail::get_value(m_pos, m_size, i);
    }

    template<std::size_t dim, bool owner>
    inline auto object_container<dim, owner>::pos(std::size_t i)
    {
        return detail::get_value(m_pos, m_size, i);
    }

    template<std::size_t dim, bool owner>
    inline auto object_container<dim, owner>::force() const
    {
        return detail::get_value(m_force, m_size);
    }

    template<std::size_t dim, bool owner>
    inline auto object_container<dim, owner>::force()
    {
        return detail::get_value(m_force, m_size);
    }

    template<std::size_t dim, bool owner>
    inline std::size_t object_container<dim, owner>::size() const
    {
        return m_size;
    }

    ///////////////////////
    // object definition //
    ///////////////////////
    template<std::size_t Dim, bool owner=true>
    class object: public object_container<Dim, owner>
    {
    public:
        static constexpr std::size_t dim = Dim;
        using base_type = object_container<dim, owner>;
        using position_type = typename base_type::position_type;
        using ref_position_type = typename base_type::ref_position_type;
        using force_type = typename base_type::force_type;

        object() = default;
        object(std::array<double, dim>* pos, std::size_t size);
        object(const std::vector<std::array<double, dim>>& pos);

        virtual std::shared_ptr<base_constructor<dim>> construct() const = 0;
        virtual void print() const = 0;
        virtual std::size_t hash() const = 0;

    };

    ///////////////////////////
    // object implementation //
    ///////////////////////////
    template<std::size_t dim, bool owner>
    object<dim, owner>::object(const std::vector<std::array<double, dim>>& pos)
    : base_type(pos, pos.size())
    {
    }

    template<std::size_t dim, bool owner>
    object<dim, owner>::object(std::array<double, dim>* pos, std::size_t size)
    : base_type(pos, size)
    {
    }

#if defined(__GNUC__) && !defined(__clang__)
    namespace workaround
    {
        // Fixes "undefined symbol" issues
        inline void long_long_allocator()
        {
            std::allocator<long long> a;
            std::allocator<unsigned long long> b;
            std::allocator<double> c;
            std::allocator<std::complex<double>> d;
        }
    }
#endif
}