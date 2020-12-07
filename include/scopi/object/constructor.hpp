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
    struct base_constructor
    {
        virtual ~base_constructor() = default;
        virtual std::unique_ptr<object<dim, false>> operator()(std::array<double, dim>* pos) const = 0;
    };

    template<class T, class... Args>
    class object_constructor: public base_constructor<T::dim>
    {
    public:
        static constexpr std::size_t dim = T::dim;
        using object_type = T;
        using tuple_type = std::tuple<Args...>;

        template<class... CTA>
        object_constructor(CTA&&... args);

        virtual std::unique_ptr<object<dim, false>> operator()(std::array<double, dim>* pos) const override;

    private:

        template<std::size_t... I>
        auto constructor(std::array<double, dim>* pos, std::index_sequence<I...>) const;

        tuple_type m_extra;
    };

     ///////////////////////////////////////
    // object_constructor implementation //
    ///////////////////////////////////////
    template<class T, class... Args>
    template<class... CTA>
    object_constructor<T, Args...>::object_constructor(CTA&&... args)
    : m_extra(std::forward<CTA>(args)...)
    {}

    template<class T, class... Args>
    auto object_constructor<T, Args...>::operator()(std::array<double, dim>* pos) const -> std::unique_ptr<object<dim, false>>
    {
        return constructor(pos, std::make_index_sequence<sizeof...(Args)>{});
    }

    template<class T, class... Args>
    template<std::size_t... I>
    auto object_constructor<T, Args...>::constructor(std::array<double, dim>* pos, std::index_sequence<I...>) const
    {
        return std::make_unique<object_type>(pos, std::get<I>(m_extra)...);
    }

    template<class T, class... Args>
    auto make_object_constructor(Args&&... args)
    {
        using constructor_type = object_constructor<T, Args...>;
        return std::make_unique<constructor_type>(args...);
    }
}
