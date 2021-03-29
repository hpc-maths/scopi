#pragma once

#include <array>
#include <vector>

#include <xtl/xmultimethods.hpp>

#include <xtensor/xadapt.hpp>
#include <xtensor/xfixed.hpp>

#include "constructor.hpp"
#include "../utils.hpp"

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
        template<std::size_t dim, class T>
        const auto get_value(const std::vector<xt::xtensor_fixed<T, xt::xshape<dim, dim>>>& t, std::size_t size)
        {
            return xt::adapt(reinterpret_cast<double*>(t.data()->data()), {size, dim});
        }

        template<std::size_t dim, class T>
        const auto get_value(const std::vector<xt::xtensor_fixed<T, xt::xshape<dim>>>& t, std::size_t size)
        {
            return xt::adapt(reinterpret_cast<double*>(t.data()->data()), {size, dim});
        }

        template<std::size_t dim, class T>
        auto get_value(std::vector<xt::xtensor_fixed<T, xt::xshape<dim, dim>>>& t, std::size_t size)
        {
            return xt::adapt(reinterpret_cast<double*>(t.data()->data()), {size, dim, dim});
        }

        template<std::size_t dim, class T>
        auto get_value(std::vector<xt::xtensor_fixed<T, xt::xshape<dim>>>& t, std::size_t size)
        {
            return xt::adapt(reinterpret_cast<double*>(t.data()->data()), {size, dim});

        }

        template<std::size_t dim, class T>
        const auto get_value(const xt::xtensor_fixed<T, xt::xshape<dim, dim>>* t, std::size_t size)
        {
            return xt::adapt(reinterpret_cast<double*>(t->data()), {size, dim, dim});
        }

        template<std::size_t dim, class T>
        const auto get_value(const xt::xtensor_fixed<T, xt::xshape<dim>>* t, std::size_t size)
        {
            return xt::adapt(reinterpret_cast<double*>(t->data()), {size, dim});
        }

        template<std::size_t dim, class T>
        auto get_value(xt::xtensor_fixed<T, xt::xshape<dim, dim>>* t, std::size_t size)
        {
            return xt::adapt(reinterpret_cast<double*>(t->data()), {size, dim, dim});
        }

        template<std::size_t dim, class T>
        auto get_value(xt::xtensor_fixed<T, xt::xshape<dim>>* t, std::size_t size)
        {
            return xt::adapt(reinterpret_cast<double*>(t->data()), {size, dim});
        }

        template<std::size_t dim, class T>
        const xt::xtensor_fixed<T, xt::xshape<dim, dim>>& get_value(const std::vector<xt::xtensor_fixed<T, xt::xshape<dim, dim>>>& t, std::size_t, std::size_t i)
        {
            return t[i];
        }

        template<std::size_t dim, class T>
        const xt::xtensor_fixed<T, xt::xshape<dim>>& get_value(const std::vector<xt::xtensor_fixed<T, xt::xshape<dim>>>& t, std::size_t, std::size_t i)
        {
            return t[i];
        }

        template<std::size_t dim, class T>
        const xt::xtensor_fixed<T, xt::xshape<dim>>& get_value(const xt::xtensor_fixed<T, xt::xshape<dim>>* t, std::size_t, std::size_t i)
        {
            return *(t + i);
        }

        template<std::size_t dim, class T>
        xt::xtensor_fixed<T, xt::xshape<dim>>& get_value(xt::xtensor_fixed<T, xt::xshape<dim>>* t, std::size_t, std::size_t i)
        {
            return *(t + i);
        }

        template<std::size_t dim, class T>
        xt::xtensor_fixed<T, xt::xshape<dim, dim>>& get_value(std::vector<xt::xtensor_fixed<T, xt::xshape<dim, dim>>>& t, std::size_t, std::size_t i)
        {
            return t[i];
        }

        template<std::size_t dim, class T>
        xt::xtensor_fixed<T, xt::xshape<dim>>& get_value(std::vector<xt::xtensor_fixed<T, xt::xshape<dim>>>& t, std::size_t, std::size_t i)
        {
            return t[i];
        }

        template<std::size_t dim, class T>
        const xt::xtensor_fixed<T, xt::xshape<dim, dim>>& get_value(const xt::xtensor_fixed<T, xt::xshape<dim, dim>>* t, std::size_t, std::size_t i)
        {
            return *(t + i);
        }


        template<std::size_t dim, class T>
        xt::xtensor_fixed<T, xt::xshape<dim, dim>>& get_value(xt::xtensor_fixed<T, xt::xshape<dim, dim>>* t, std::size_t, std::size_t i)
        {
            return *(t + i);
        }

        template<std::size_t dim, bool owner>
        struct object_inner_type
        {
            using default_type = typename std::vector<xt::xtensor_fixed<double, xt::xshape<dim>>>;
            using position_type = default_type;
            using rotation_type = typename std::vector<xt::xtensor_fixed<double, xt::xshape<dim, dim>>> ;
        };

        template<std::size_t dim>
        struct object_inner_type<dim, false>
        {
            using default_type = typename xt::xtensor_fixed<double, xt::xshape<dim>>*;
            using position_type = default_type;
            using rotation_type = typename xt::xtensor_fixed<double, xt::xshape<dim, dim>>*;
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
        using rotation_type = typename inner_types::rotation_type;

        object_container(position_type pos, rotation_type r, std::size_t size);

        const auto pos() const;
        auto pos();

        const auto pos(std::size_t i) const;
        auto pos(std::size_t i);

        const auto R() const;
        auto R();

        const auto R(std::size_t i) const;
        auto R(std::size_t i);

        std::size_t size() const;

    private:

        position_type m_pos;
        rotation_type m_r;
        std::size_t m_size;
    };

    /////////////////////////////////////
    // object_container implementation //
    /////////////////////////////////////
    template<std::size_t dim, bool owner>
    inline object_container<dim, owner>::object_container(
      position_type pos,
      rotation_type r,
      std::size_t size
    )
    : m_pos(pos)
    , m_r(r)
    , m_size(size)
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
    inline const auto object_container<dim, owner>::R() const
    {
        return detail::get_value(m_r, m_size);
    }

    template<std::size_t dim, bool owner>
    inline auto object_container<dim, owner>::R()
    {
        return detail::get_value(m_r, m_size);
    }

    template<std::size_t dim, bool owner>
    inline const auto object_container<dim, owner>::R(std::size_t i) const
    {
        return detail::get_value(m_r, m_size, i);
    }

    template<std::size_t dim, bool owner>
    inline auto object_container<dim, owner>::R(std::size_t i)
    {
        return detail::get_value(m_r, m_size, i);
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
        using rotation_type = typename base_type::rotation_type;

        virtual ~object() = default;

        object(object&&) = delete;
        object& operator=(const object&) = delete;
        object& operator=(object&&) = delete;

        object(position_type pos, rotation_type r, std::size_t size);

    protected:
        object() = default;
        object(const object&) = default;
    };

    ///////////////////////////
    // object implementation //
    ///////////////////////////
    template<std::size_t dim, bool owner>
    object<dim, owner>::object(position_type pos, rotation_type r, std::size_t size)
    : base_type(pos, r, size)
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
