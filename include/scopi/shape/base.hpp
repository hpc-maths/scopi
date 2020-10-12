#pragma once

#include <array>
#include <tuple>
#include <utility>

namespace scopi
{
    namespace detail
    {
        template<std::size_t dim, bool owner>
        struct inner_shape_type
        {
            using position_type = std::array<double, dim>;
            using force_type = std::array<double, dim>;

            template<class T>
            T& get_pointer(T& t)
            {
                return t;
            }

            template<class T>
            const T& get_value(const T& t) const
            {
                return t;
            }

            template<class T>
            T& get_value(T& t) const
            {
                return t;
            }
        };

        template<std::size_t dim>
        struct inner_shape_type<dim, false>
        {
            using position_type = std::array<double, dim>*;
            using force_type = std::array<double, dim>*;

            template<class T>
            T* get_pointer(T*& t)
            {
                return t;
            }

            template<class T>
            const T& get_value(const T* t) const
            {
                return *t;
            }

            template<class T>
            T& get_value(T* t) const
            {
                return *t;
            }
        };
    }

    ////////////////////////////////
    // shape_container definition //
    ////////////////////////////////
    template<std::size_t dim, bool owner>
    class shape_container: public detail::inner_shape_type<dim, owner>
    {
    public:

        using inner_types = detail::inner_shape_type<dim, owner>;
        using position_type = typename inner_types::position_type;
        using force_type = typename inner_types::force_type;

        shape_container();
        template<class T>
        shape_container(T&& pos);

        shape_container(std::array<double, dim>& pos);
        shape_container(std::array<double, dim>&& pos);

        const std::array<double, dim>& pos() const;
        std::array<double, dim>& pos();

        const std::array<double, dim>& force() const;
        std::array<double, dim>& force();

    private:

        position_type m_pos;
        force_type m_force;
    };

    ///////////////////////////////////
    // shape_constructor definition //
    //////////////////////////////////
    template<class T, class... Args>
    class shape_constructor
    {
    public:

        using shape_type = T;
        using tuple_type = std::tuple<Args...>;

        template<class... CTA>
        shape_constructor(CTA&&... args);

        auto operator()() const;
        auto extra_args() const;

    private:

        template<std::size_t... I>
        auto lambda_constructor(std::index_sequence<I...>) const;

        tuple_type m_extra;
    };

    ///////////////////////
    // shape definition //
    //////////////////////
    template<std::size_t dim, bool owner=true>
    class shape: public shape_container<dim, owner>
    {
    public:

        using base_type = shape_container<dim, owner>;

        template<class T>
        shape(T&& pos);

        virtual std::function<shape<dim, false>*(std::array<double, dim>*)> construct() const = 0;
        virtual void print() = 0;
        virtual std::size_t hash() = 0;
    };

    ////////////////////////////////////
    // shape_container implementation //
    ////////////////////////////////////
    template<std::size_t dim, bool owner>
    shape_container<dim, owner>::shape_container()
    : m_pos{nullptr}, m_force{nullptr}
    {}

    template<std::size_t dim, bool owner>
    template<class T>
    shape_container<dim, owner>::shape_container(T&& pos)
    : m_pos(this->get_pointer(pos))
    {}

    template<std::size_t dim, bool owner>
    shape_container<dim, owner>::shape_container(std::array<double, dim>& pos)
    : m_pos(this->get_pointer(pos))
    {}

    template<std::size_t dim, bool owner>
    shape_container<dim, owner>::shape_container(std::array<double, dim>&& pos)
    : m_pos(this->get_pointer(pos))
    {}

    template<std::size_t dim, bool owner>
    const std::array<double, dim>& shape_container<dim, owner>::pos() const
    {
        return this->get_value(m_pos);
    }

    template<std::size_t dim, bool owner>
    std::array<double, dim>& shape_container<dim, owner>::pos()
    {
        return this->get_value(m_pos);
    }

    template<std::size_t dim, bool owner>
    const std::array<double, dim>& shape_container<dim, owner>::force() const
    {
        return this->get_value(m_force);
    }

    template<std::size_t dim, bool owner>
    std::array<double, dim>& shape_container<dim, owner>::force()
    {
        return this->get_value(m_force);
    }

    ///////////////////////////
    // shape implementation //
    //////////////////////////
    template<std::size_t dim, bool owner>
    template<class T>
    shape<dim, owner>::shape(T&& pos)
    : base_type(std::forward<T>(pos))
    {}

    //////////////////////////////////////
    // shape_constructor implementation //
    //////////////////////////////////////
    template<class T, class... Args>
    template<class... CTA>
    shape_constructor<T, Args...>::shape_constructor(CTA&&... args)
    : m_extra(std::forward<CTA>(args)...)
    {}

    template<class T, class... Args>
    auto shape_constructor<T, Args...>::operator()() const
    {
        return lambda_constructor(std::make_index_sequence<sizeof...(Args)>{});
    }

    template<class T, class... Args>
    auto shape_constructor<T, Args...>::extra_args() const
    {
        return m_extra;
    }

    template<class T, class... Args>
    template<std::size_t... I>
    auto shape_constructor<T, Args...>::lambda_constructor(std::index_sequence<I...>) const
    {
        return [&](auto pos){return new shape_type(pos, std::get<I>(m_extra)...);};
    }

    template<class T, class... Args>
    auto make_shape_constructor(Args&&... args)
    {
        using constructor_type = shape_constructor<T, Args...>;
        return constructor_type(args...);
    }
}