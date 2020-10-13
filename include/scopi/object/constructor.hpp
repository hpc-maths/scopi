#pragma once

#include <array>
#include <memory>
#include <tuple>

namespace scopi
{
    ///////////////////////////////////
    // object_constructor definition //
    ///////////////////////////////////
    template<std::size_t dim, bool owner>
    class object;

    template<std::size_t dim>
    struct base_constructor;

    template<std::size_t dim, class T, class... Args>
    class object_constructor;

    // Dim = 2 specialization due to virtual method
    template<>
    struct base_constructor<2>
    {
        virtual std::shared_ptr<object<2, false>> operator()(std::array<double, 2>* pos) const = 0;
    };
    
    template<class T, class... Args>
    class object_constructor<2, T, Args...>: public base_constructor<2>
    {
    public:

        using object_type = T;
        using tuple_type = std::tuple<Args...>;

        template<class... CTA>
        object_constructor(CTA&&... args);

        virtual std::shared_ptr<object<2, false>> operator()(std::array<double, 2>* pos) const override;

    private:

        template<std::size_t... I>
        auto constructor(std::array<double, 2>* pos, std::index_sequence<I...>) const;

        tuple_type m_extra;
    };

    // Dim = 3 specialization due to virtual method
    template<>
    struct base_constructor<3>
    {
        virtual std::shared_ptr<object<3, false>> operator()(std::array<double, 3>* pos) const = 0;
    };

    template<class T, class... Args>
    class object_constructor<3, T, Args...>: public base_constructor<3>
    {
    public:

        using object_type = T;
        using tuple_type = std::tuple<Args...>;

        template<class... CTA>
        object_constructor(CTA&&... args);

        virtual std::shared_ptr<object<3, false>> operator()(std::array<double, 3>* pos) const override;

    private:

        template<std::size_t... I>
        auto constructor(std::array<double, 3>* pos, std::index_sequence<I...>) const;

        tuple_type m_extra;
    };

    ///////////////////////////////////////
    // object_constructor implementation //
    ///////////////////////////////////////
    // Dim = 2
    template<class T, class... Args>
    template<class... CTA>
    object_constructor<2, T, Args...>::object_constructor(CTA&&... args)
    : m_extra(std::forward<CTA>(args)...)
    {}

    template<class T, class... Args>
    typename std::shared_ptr<object<2, false>> object_constructor<2, T, Args...>::operator()(std::array<double, 2>* pos) const
    {
        return constructor(pos, std::make_index_sequence<sizeof...(Args)>{});
    }

    template<class T, class... Args>
    template<std::size_t... I>
    auto object_constructor<2, T, Args...>::constructor(std::array<double, 2>* pos, std::index_sequence<I...>) const
    {
        return std::make_shared<object_type>(pos, std::get<I>(m_extra)...);
    }

    // Dim = 3
    template<class T, class... Args>
    template<class... CTA>
    object_constructor<3, T, Args...>::object_constructor(CTA&&... args)
    : m_extra(std::forward<CTA>(args)...)
    {}

    template<class T, class... Args>
    typename std::shared_ptr<object<3, false>> object_constructor<3, T, Args...>::operator()(std::array<double, 3>* pos) const
    {
        return constructor(pos, std::make_index_sequence<sizeof...(Args)>{});
    }

    template<class T, class... Args>
    template<std::size_t... I>
    auto object_constructor<3, T, Args...>::constructor(std::array<double, 3>* pos, std::index_sequence<I...>) const
    {
        return std::make_shared<object_type>(pos, std::get<I>(m_extra)...);
    }

    template<std::size_t dim, class T, class... Args>
    auto make_object_constructor(Args&&... args)
    {
        using constructor_type = object_constructor<dim, T, Args...>;
        return std::make_shared<constructor_type>(args...);
    } 

}