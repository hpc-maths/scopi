#pragma once

#include <array>
#include <vector>

#include "constructor.hpp"
#include <scopi/utils.hpp>

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
            using velocity_type = default_type;
            using desired_velocity_type = default_type;
            using force_type = default_type;
        };

        template<std::size_t dim>
        struct object_inner_type<dim, false>
        {
            using default_type = typename std::array<double, dim>*;
            using position_type = default_type;
            using velocity_type = default_type;
            using desired_velocity_type = default_type;
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
        using velocity_type = typename inner_types::velocity_type;
        using desired_velocity_type = typename inner_types::desired_velocity_type;
        using force_type = typename inner_types::force_type;

        object_container(
          position_type pos,
          velocity_type v,
          desired_velocity_type vd,
          force_type f,
          std::size_t size);

        const auto pos() const;
        auto pos();
        const auto pos(std::size_t i) const;
        auto pos(std::size_t i);

        const auto v() const;
        auto v();
        const auto v(std::size_t i) const;
        auto v(std::size_t i);

        const auto vd() const;
        auto vd();
        const auto vd(std::size_t i) const;
        auto vd(std::size_t i);

        const auto f() const;
        auto f();
        const auto f(std::size_t i) const;
        auto f(std::size_t i);

        std::size_t size() const;

    private:

        position_type m_pos;
        velocity_type m_v;
        desired_velocity_type m_vd;
        force_type m_f;
        std::size_t m_size;
    };

    /////////////////////////////////////
    // object_container implementation //
    /////////////////////////////////////
    template<std::size_t dim, bool owner>
    inline object_container<dim, owner>::object_container(
      position_type pos,
      velocity_type v,
      desired_velocity_type vd,
      force_type f,
      std::size_t size
    )
    : m_pos(pos), m_v(v), m_vd(vd), m_f(f), m_size(size)
    {}

    // position

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

    // velocity

    template<std::size_t dim, bool owner>
    inline const auto object_container<dim, owner>::v() const
    {
        return detail::get_value(m_v, m_size);
    }

    template<std::size_t dim, bool owner>
    inline auto object_container<dim, owner>::v()
    {
        return detail::get_value(m_v, m_size);
    }

    template<std::size_t dim, bool owner>
    inline const auto object_container<dim, owner>::v(std::size_t i) const
    {
        return detail::get_value(m_v, m_size, i);
    }

    template<std::size_t dim, bool owner>
    inline auto object_container<dim, owner>::v(std::size_t i)
    {
        return detail::get_value(m_v, m_size, i);
    }

    // desired velocity

    template<std::size_t dim, bool owner>
    inline const auto object_container<dim, owner>::vd() const
    {
        return detail::get_value(m_vd, m_size);
    }

    template<std::size_t dim, bool owner>
    inline auto object_container<dim, owner>::vd()
    {
        return detail::get_value(m_vd, m_size);
    }

    template<std::size_t dim, bool owner>
    inline const auto object_container<dim, owner>::vd(std::size_t i) const
    {
        return detail::get_value(m_vd, m_size, i);
    }

    template<std::size_t dim, bool owner>
    inline auto object_container<dim, owner>::vd(std::size_t i)
    {
        return detail::get_value(m_vd, m_size, i);
    }

    // force

    template<std::size_t dim, bool owner>
    inline const auto object_container<dim, owner>::f() const
    {
        return detail::get_value(m_f, m_size);
    }

    template<std::size_t dim, bool owner>
    inline auto object_container<dim, owner>::f()
    {
        return detail::get_value(m_f, m_size);
    }

    template<std::size_t dim, bool owner>
    inline const auto object_container<dim, owner>::f(std::size_t i) const
    {
        return detail::get_value(m_f, m_size, i);
    }

    template<std::size_t dim, bool owner>
    inline auto object_container<dim, owner>::f(std::size_t i)
    {
        return detail::get_value(m_f, m_size, i);
    }

    // size

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
        using velocity_type = typename base_type::velocity_type;
        using desired_velocity_type = typename base_type::desired_velocity_type;
        using force_type = typename base_type::force_type;

        object() = default;
        object(
          std::array<double, dim>* pos,
          std::array<double, dim>* v,
          std::array<double, dim>* vd,
          std::array<double, dim>* f,
          std::size_t size);
        object(
          const std::vector<std::array<double, dim>>& pos,
          const std::vector<std::array<double, dim>>& v,
          const std::vector<std::array<double, dim>>& vd,
          const std::vector<std::array<double, dim>>& f
        );

        virtual std::shared_ptr<base_constructor<dim>> construct() const = 0;
        virtual void print() const = 0;
        virtual std::size_t hash() const = 0;

    };

    ///////////////////////////
    // object implementation //
    ///////////////////////////
    template<std::size_t dim, bool owner>
    object<dim, owner>::object(
      const std::vector<std::array<double, dim>>& pos,
      const std::vector<std::array<double, dim>>& v,
      const std::vector<std::array<double, dim>>& vd,
      const std::vector<std::array<double, dim>>& f
    )
    : base_type(pos, v, vd, f, pos.size())
    {
    }

    template<std::size_t dim, bool owner>
    object<dim, owner>::object(
      std::array<double, dim>* pos,
      std::array<double, dim>* v,
      std::array<double, dim>* vd,
      std::array<double, dim>* f,
      std::size_t size
    )
    : base_type(pos, v, vd, f, size)
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
