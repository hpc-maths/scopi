#pragma once

#include <array>
#include <algorithm>
#include <cstddef>
#include <vector>
#include <cmath>

#include <xtl/xmultimethods.hpp>

#include <xtensor/xadapt.hpp>
#include <xtensor/xfixed.hpp>
#include <xtensor/xview.hpp>

#include "constructor.hpp"
#include "../../utils.hpp"
#include "../../types.hpp"

namespace scopi
{
    template<std::size_t dim>
    class object_base
    {
    public:

        virtual ~object_base() = default;

        object_base& operator=(const object_base&) = delete;
        object_base& operator=(object_base&&) = delete;

        virtual std::unique_ptr<base_constructor<dim>> construct() const = 0;
        virtual void print() const = 0;
        virtual std::size_t hash() const = 0;

    protected:

        object_base() = default;

        object_base(const object_base&) = default;
        object_base(object_base&&) = default;
    };

    namespace detail
    {
        // position type
        template<std::size_t dim>
        auto get_value_impl(const std::vector<type::position_t<dim>>& t, std::size_t size)
        {
            return xt::view((xt::adapt(reinterpret_cast<const double*>(t.data()->data()), {size, dim+2})), xt::all(), xt::range(0, dim));
        }

        template<std::size_t dim>
        auto get_value_impl(const type::position_t<dim>* t, std::size_t size)
        {
            std::cout << "get_value_impl position 2" << std::endl;
            return xt::adapt(reinterpret_cast<const double*>(t->data()), {size, dim});
        }

        template<std::size_t dim>
        auto get_value_impl(std::vector<type::position_t<dim>>& t, std::size_t size)
        {
            return xt::view((xt::adapt(reinterpret_cast<double*>(t.data()->data()), {size, dim+2})), xt::all(), xt::range(0, dim));
        }

        template<std::size_t dim>
        auto get_value_impl(type::position_t<dim>* t, std::size_t size)
        {
            return xt::view((xt::adapt(reinterpret_cast<double*>(t->data()), {size, dim+2})), xt::all(), xt::range(0, dim));
        }

        // quaternion type
        template <class object_t = type::quaternion_t>
        auto get_value_impl(const std::vector<object_t>& t, std::size_t)
        {
            std::cout << "get_value_impl quaternion 1" << std::endl;
            return xt::adapt(t);
        }

        template <class object_t = type::quaternion_t>
        auto get_value_impl(const object_t* t, std::size_t size)
        {
            std::cout << "get_value_impl quaternion 2" << std::endl;
            return xt::adapt(reinterpret_cast<const double*>(t->data()), {size, 4UL});
        }

        template <class object_t = type::quaternion_t>
        auto get_value_impl(std::vector<object_t>& t, std::size_t)
        {
            std::cout << "get_value_impl quaternion 3" << std::endl;
            return xt::adapt(t);
        }

        template <class object_t = type::quaternion_t>
        auto get_value_impl(object_t* t, std::size_t size)
        {
            std::cout << "get_value_impl quaternion 4" << std::endl;
            return xt::adapt(reinterpret_cast<double*>(t->data()), {size, 4UL});
        }

        template <class T>
        auto get_value(T& t, std::size_t s)
        {
            return get_value_impl(t, s);
        }

        template<std::size_t dim, bool owner>
        struct object_inner_type
        {
            using position_type = std::vector<type::position_t<dim>>;
            using quaternion_type = std::vector<type::quaternion_t> ;
        };

        template<std::size_t dim>
        struct object_inner_type<dim, false>
        {
            using position_type = type::position_t<dim>*;
            using quaternion_type = type::quaternion_t*;
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
        using quaternion_type = typename inner_types::quaternion_type;

        object_container(position_type pos, quaternion_type q, std::size_t size);

        auto pos() const;
        auto pos();

        auto pos(std::size_t i) const;
        auto pos(std::size_t i);

        auto q() const;
        auto q();

        auto q(std::size_t i) const;
        auto q(std::size_t i);

        std::size_t size() const;

    private:

        position_type m_pos;
        quaternion_type m_q;
        std::size_t m_size;
    };

    /////////////////////////////////////
    // object_container implementation //
    /////////////////////////////////////
    template<std::size_t dim, bool owner>
    inline object_container<dim, owner>::object_container(
      position_type pos,
      quaternion_type q,
      std::size_t size
    )
    : m_pos(pos)
    , m_q(q)
    , m_size(size)
    {}

    template<std::size_t dim, bool owner>
    inline auto object_container<dim, owner>::pos() const
    {
        return detail::get_value(m_pos, m_size);
    }

    template<std::size_t dim, bool owner>
    inline auto object_container<dim, owner>::pos()
    {
        return detail::get_value(m_pos, m_size);
    }

    template<std::size_t dim, bool owner>
    inline auto object_container<dim, owner>::pos(std::size_t i) const
    {
        return xt::view(detail::get_value(m_pos, m_size), i);
    }

    template<std::size_t dim, bool owner>
    inline auto object_container<dim, owner>::pos(std::size_t i)
    {
        return xt::view(detail::get_value(m_pos, m_size), i);
    }

    template<std::size_t dim, bool owner>
    inline auto object_container<dim, owner>::q() const
    {
        return detail::get_value(m_q, m_size);
    }

    template<std::size_t dim, bool owner>
    inline auto object_container<dim, owner>::q()
    {
        return detail::get_value(m_q, m_size);
    }

    template<std::size_t dim, bool owner>
    inline auto object_container<dim, owner>::q(std::size_t i) const
    {
        return xt::view(detail::get_value(m_q, m_size), i);
    }

    template<std::size_t dim, bool owner>
    inline auto object_container<dim, owner>::q(std::size_t i)
    {
        return xt::view(detail::get_value(m_q, m_size), i);
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
    class object: public object_base<Dim>
                , public object_container<Dim, owner>
    {
    public:
        static constexpr std::size_t dim = Dim;
        using base_type = object_container<dim, owner>;
        using position_type = typename base_type::position_type;
        using quaternion_type = typename base_type::quaternion_type;

        virtual ~object() = default;

        object(object&&) = delete;
        object& operator=(const object&) = delete;
        object& operator=(object&&) = delete;

        object(position_type pos, quaternion_type q, std::size_t size);

    protected:
        object() = default;
        object(const object&) = default;
    };

    ///////////////////////////
    // object implementation //
    ///////////////////////////
    template<std::size_t dim, bool owner>
    object<dim, owner>::object(position_type pos, quaternion_type q, std::size_t size)
    : base_type(pos, q, size)
    {}

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
