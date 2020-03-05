#pragma once

namespace scopi
{
    template<class D>
    class shape_base
    {
      public:
        using derived_type = D;

        derived_type &derived_cast() & noexcept;
        const derived_type &derived_cast() const &noexcept;
        derived_type derived_cast() && noexcept;
    //   protected:
    //     shape_base(){};
    //     ~shape_base() = default;

    //     shape_base(const shape_base &) = default;
    //     shape_base &operator=(const shape_base &) = default;

    //     shape_base(shape_base &&) = default;
    //     shape_base &operator=(shape_base &&) = default;
    };

    template<class D>
    inline auto shape_base<D>::derived_cast() & noexcept -> derived_type &
    {
        return *static_cast<derived_type *>(this);
    }

    template<class D>
    inline auto shape_base<D>::derived_cast() const & noexcept -> const derived_type &
    {
        return *static_cast<const derived_type *>(this);
    }

    template<class D>
    inline auto shape_base<D>::derived_cast() && noexcept -> derived_type
    {
        return *static_cast<derived_type *>(this);
    }

    template<class D, class... T>
    class object_base
    {
      public:
        using derived_type = D;
        using tuple_type = std::tuple<T...>;

        derived_type &derived_cast() & noexcept;
        const derived_type &derived_cast() const &noexcept;
        derived_type derived_cast() && noexcept;
      protected:
        object_base(const T&... shape): m_shape{shape...}{};
        ~object_base() = default;

        object_base(const object_base &) = default;
        object_base &operator=(const object_base &) = default;

        object_base(object_base &&) = default;
        object_base &operator=(object_base &&) = default;

        tuple_type m_shape;
    //   private:
    };

    template<class D, class... T>
    inline auto object_base<D, T...>::derived_cast() & noexcept -> derived_type &
    {
        return *static_cast<derived_type *>(this);
    }

    template<class D, class... T>
    inline auto object_base<D, T...>::derived_cast() const & noexcept -> const derived_type &
    {
        return *static_cast<const derived_type *>(this);
    }

    template<class D, class... T>
    inline auto object_base<D, T...>::derived_cast() && noexcept -> derived_type
    {
        return *static_cast<derived_type *>(this);
    }

}